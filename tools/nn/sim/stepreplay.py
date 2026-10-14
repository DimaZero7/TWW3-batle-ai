"""The step replay (docs/en/training/simulator.md, "The step replay"): a measuring tool without drift. Every H seconds
the simulator is put back into the RECORDED state of a game battle and plays H seconds on the recorded orders; what it
predicts for each second of the window is kept beside what the game recorded (tools/nn/sim/stepcells.py compares them).

At each window start the simulator gets:
* the observed state: place, bearing, men, health, morale points (MoralePercent x leadership: both sides are recorded),
  rout / shatter / waver / melee flags, ammunition, kills, fatigue level (points: its own if at that level, else a
  quarter into it), the target;
* the hidden clocks rebuilt from the recording (tools/nn/sim/stepclock.py): the casualty windows (the last 4 s / 60 s of
  HP lost, from the recorded health), the fight balance, the waver timer, routs so far, the rout and rally clocks, the
  melee clock, the run-up and the charge, the velocity (only where the game's moving flag is on), the rout's exit, under
  fire, the lord's fall, a unit gone off the map (never a loss);
* the abilities' timers from the recorded casts (the run's nn_effects: a lord's ability coming on); in the window the
  recorded casts are replayed as orders (Orders.ability) and nobody fires by the game-AI rule;
* a shooter: the target the game shows it firing at, with the aim of the seconds it stood; a random place in its reload
  cycle (the game's fire is a trickle of men, no unit volley to read); then the target it takes from these places
  without the retarget's cost (a probe step on a copy).
What it carries from its own last window (the recording has nothing for it): the order clocks (order_s, exit_s, brk_s,
leave_s), the charge's morale blocks, a shooter's reload and aim when the game shows it not firing, the contact effects
and timed effects, the skirmish run, stuck men. What is left of the reset's jump: a shooter fires ~0.7 of its later rate
in a window's first 3-5 s (stepcells reports the loss after SETTLE_S seconds apart).

Options (ablations and the forced modes): force pos - the game's places, bearings and velocity put back every second
(the damage at the game's own geometry); force hp - its health, men, casualty windows and fight balance every second
(morale and routs on the game's own losses); facing sim - a formation in contact faces its nearest opponent's centre at
the start; keep - fields left as the simulator has them (an ablation of the reset).

Orders: tools/nn/sim/replay.recorded_orders as tools/nn/sim/check.py builds them (the network's holds, the engine's far
points, the leavers), clock-indexed (no contact-phase latch: the clock is the recording's).

Output, one npz per battle (out_dir/<run>.npz): starts [W], sim_<f> [W, H, C, n] (per second of each window, each copy:
hp, men, mp, r, s, w, m, x, z, gone, mel / mis = HP taken in melee / from missiles that second, dl / sdl = HP dealt), the
first step's contact (t_n: standing enemies touching, t_sec: the worst side touched 0 front 1 flank 2 rear, t_gap: the
nearest standing enemy's edge gap) [W, C, n], the morale terms at the window's first and last step (copy mean)
[K, W, n] and what morale.step saw besides (SEEN: x0_<name> / x1_<name> [W, n]), the rebuilt clocks at the starts c_<name> [W, n], the game arrays g_<f> [T, n] (g_dl / g_sdl: HP dealt a row by stepclock's
attribution), the recorded orders o_kind / o_target [T, n] and the units' passport numbers.
Needs torch: python -m tools.nn.sim.stepreplay --sets it1 ... --out build/stepreplay/<tag> (the container, GPU when free).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from tools.nn.sim import stepclock

SIM_SEC = ("hp", "men", "mp", "r", "s", "w", "m", "x", "z", "gone", "mel", "mis", "dl", "sdl")
GAME_F = ("x", "z", "b", "men", "hp", "mp", "ms", "a", "k", "fat", "phr")
GAME_B = ("r", "s", "w", "m", "mv", "f", "fire", "lf", "rf", "bf")
# what morale.step saw besides the terms, at a window's first and last step (x0_<name> / x1_<name> [W, n], copy mean):
# the start bonus (the effects' leadership, on the state only inside the step), in a fight / in melee, the share lost by
# the enemies it fights, the fight balance (HP dealt + 1) / (taken + 1), routing enemies counted, the side struck from
SEEN = ("bonus", "in_combat", "in_melee", "foe_lost", "ratio", "routing_enemies", "flank_hit")
TERMS = ("aura", "lord", "secure", "casualties", "recent", "extended", "combat", "attacked", "exposed",
         "routing_friends", "routing_enemies", "under_fire", "fatigue", "strong_enemy", "charge", "collapse")
CLOCKS = ("taken", "shot_taken", "dealt", "shot_dealt", "waver_s", "rout_count", "rout_s", "rally_s", "contact_s",
          "out_s", "melee_s", "ran_in", "runup", "charge", "vx", "vz", "rpin_s", "rpin_d", "under_fire_s", "low_s",
          "dead_s", "gone", "gone_s", "unready", "recent", "extended", "aim", "aim_tgt")


class _Last:
    """morale.TRACE stand-in: keeps the last step's terms (and the start bonus, morale_bonus, as morale.step saw it:
    the effects' leadership is on the state only inside the step, effects.restore takes it off)."""
    def __init__(self):
        self.parts = None
        self.seen = None

    def append(self, p):
        self.parts = p


def _seen(step):
    """morale.step wrapped to keep what it sees (SEEN, into morale.TRACE.seen)."""
    def wrapped(u, ctx, params, dt):
        import torch
        from tools.nn.sim import morale
        if isinstance(morale.TRACE, _Last):
            z = torch.zeros_like(u["morale"])
            morale.TRACE.seen = {
                "bonus": u["morale_bonus"] + z, "in_combat": ctx.get("in_combat", ctx["in_melee"]).float(),
                "in_melee": ctx["in_melee"].float(), "foe_lost": ctx.get("foe_lost", z) + z,
                "ratio": (u["dealt"] + u["shot_dealt"] + 1.0) / (u["taken"] + u["shot_taken"] + 1.0),
                "routing_enemies": ctx["routing_enemies"] + z, "flank_hit": u["flank_hit"] + z}
        return step(u, ctx, params, dt)
    wrapped.inner = step
    return wrapped


def _params(sets=()):
    from tools.nn.sim.params import load as load_params
    params = load_params()
    for item in sets:
        path, val = item.split("=", 1)
        try:
            val = json.loads(val)
        except ValueError:
            pass
        sec, key = path.split(".", 1)
        if "." in key:          # nested: section.a.b=value
            import copy
            top, rest = key.split(".", 1)
            d = copy.deepcopy(params.sim[sec][top])
            node = d
            ks = rest.split(".")
            for k in ks[:-1]:
                node = node[k]
            node[ks[-1]] = val
            params = params.with_cal(sec, **{top: d})
        else:
            params = params.with_cal(sec, **{key: val})
    return params.with_cal("noise", blows=True).with_cal("battle_limit_s", value=1e6)


def _orders(g, army, params, slot_of, N):
    """check._simulate_batch's order rows for one recorded battle, without the contact-phase keys (clock-indexed)."""
    from tools.nn.sim import check, replay
    width = {x["name"]: x.get("width") for side in (1, 2) for x in army["sides"][side]["units"]}
    missile = [bool((params.units.get(k) or {}).get("missile")) for k in g.keys] if g.keys else [False] * len(g.names)
    leavers = [bool(m) or (g.own_ai == check.NET and int(s) == 1) for m, s in zip(missile, g.side)]
    nearest = [check.FIGHT_NEAREST and not (g.own_ai == check.NET and int(s) == 1) for s in g.side]
    spacing = [params.spacing_of(k) for k in g.keys] if g.keys else params.sim["formation"]["spacing_m"]
    holds = check.network_holds(g) if check.NET_HOLDS and g.own_ai == check.NET else None
    far_m = check.ENGINE_FAR_M if (params.sim.get("skirmish") or {}).get("on") else None
    rows = replay.recorded_orders(g, slot_of, N, [width.get(n) for n in g.names], spacing, nearest,
                                  params.sim["contact"].get("leave_m", 0.0), leavers, holds=holds, engine_far_m=far_m)
    for k in ("phase", "end", "phase_target"):
        rows.pop(k, None)
    return rows


def window_starts(T, H):
    """Window start rows of a recording of T rows: 0, H, 2H ... while the window ends inside it."""
    return list(range(0, max(T - H, 0), H))


def _abilities(g, run_dir, slot_of, ab_idx, params, starts, T, N):
    """The recorded casts and the abilities' timers: (ability order rows [T, N] slot or -1, on [W, 3, n], cd [W, 3, n],
    snap_s [W, n], snap_ab [W, n]). ab_idx [3, n]: each unit's ability index per slot (-1 none)."""
    from tools.nn.sim import abilities as A
    keys = A.keys(params)
    fx = stepclock.effect_rows(run_dir, g)
    n = len(g.names)
    W = len(starts)
    use = np.full((T, N), -1, dtype=np.int64)
    on = np.zeros((W, A.SLOTS, n))
    cd = np.zeros((W, A.SLOTS, n))
    snap_s = np.zeros((W, n))
    snap_ab = np.full((W, n), -1, dtype=np.int64)
    cast_rows = {}
    for i in range(n):
        for k in range(A.SLOTS):
            a = int(ab_idx[k, i])
            if a < 0:
                continue
            key = keys[a]
            p = params.abilities[key]
            if p.get("passive") or p.get("auto"):
                continue
            init = float(p.get("initial_s", 0.0) or 0.0)
            if key in fx:
                c = stepclock.casts(fx[key][:, [i]])[:, 0]
            else:
                c = np.zeros(T, bool)
            cast_rows.setdefault(key, []).append((i, np.flatnonzero(c)))
            use[np.flatnonzero(c), slot_of[i]] = k
            o, d = stepclock.ability_timers(c[:, None], starts, float(p["active_s"]), float(p["recharge_s"]), init)
            on[:, k, i], cd[:, k, i] = o[:, 0], d[:, 0]
    # an ability laid at the cast on the friends in range (update_targets false: Stand Your Ground) - on a friend at a
    # window start: the seconds left from its side's last cast
    for key, owners in cast_rows.items():
        p = params.abilities[key]
        if p.get("update_targets", True) or key not in fx:
            continue
        for i, rows in owners:
            for w, s0 in enumerate(starts):
                before = rows[rows <= s0]
                if not len(before):
                    continue
                left = float(p["active_s"]) - (s0 - before[-1])
                if left <= 0:
                    continue
                on_f = fx[key][s0] & (np.asarray(g.side) == g.side[i])
                on_f[i] = False
                snap_s[w, on_f] = left
                snap_ab[w, on_f] = keys.index(key)
    return use, on, cd, snap_s, snap_ab


def _force(u, force, g_row, per_row, row, known, K, N, tens):
    """Put the game's row into the simulator at the end of a second of the window (force): pos - places, bearings and
    the velocity; hp - health and men, the second's loss as the game's in the casualty windows (the newest two steps of
    lost_hist) and the fight balance sums rebuilt from the recording (stepclock.clocks)."""
    import torch
    gx = tens(g_row["x"][:, row])
    ok = known & torch.isfinite(gx)
    if "pos" in force:
        px, pz = tens(g_row["x"][:, row - 1]), tens(g_row["z"][:, row - 1])
        gz = tens(g_row["z"][:, row])
        u["x"] = torch.where(ok, gx, u["x"])
        u["z"] = torch.where(ok, gz, u["z"])
        gb = tens(g_row["b"][:, row])
        u["b"] = torch.where(ok & torch.isfinite(gb), torch.nan_to_num(gb), u["b"])
        moved = ok & torch.isfinite(px) & torch.isfinite(pz) & (torch.nan_to_num(tens(g_row["mv"][:, row])) > 0.5)
        u["vx"] = torch.where(ok, torch.where(moved, torch.nan_to_num(gx - px), torch.zeros_like(u["vx"])), u["vx"])
        u["vz"] = torch.where(ok, torch.where(moved, torch.nan_to_num(gz - pz), torch.zeros_like(u["vz"])), u["vz"])
    if "hp" in force:
        hp = torch.nan_to_num(tens(g_row["hp"][:, row]), nan=1.0).clamp(0, 1)
        men = torch.nan_to_num(tens(g_row["men"][:, row])).clamp(min=0)
        dead = ok & (hp <= 0.003)
        keep = ok & ~u["gone"]
        u["hp"] = torch.where(keep, hp, u["hp"])
        u["hp_abs"] = torch.where(keep, hp * u["hp0"], u["hp_abs"])
        u["men"] = torch.where(keep & ~dead, torch.minimum(torch.maximum(men, (hp > 0.003).float()), u["men0"]),
                               torch.where(dead, torch.zeros_like(u["men"]), u["men"]))
        for f in ("taken", "shot_taken", "dealt", "shot_dealt"):
            u[f] = torch.where(keep, tens(per_row[f][:, row]), u[f])
        if K:
            B = u["men"].shape[0]
            hist = u["lost_hist"].reshape(B, K, N)
            half = tens(per_row["loss"][:, row]) / 2
            new = torch.where(keep[:, None, :], half[:, None, :].expand(-1, 2, -1), hist[:, :2])
            hist = torch.cat([new, hist[:, 2:]], 1)
            u["lost_hist"] = hist.reshape(B, K * N)
            # (morale.casualty_windows: recent = the newest casualties_s, extended = all; recomputed at the next step)


def _batch(job):
    """One batch of recorded battles -> {run: npz path}."""
    import torch
    from tools.nn.sim import abilities as A, battle, check, fatigue as F, geometry, morale, replay, scenario
    run_dirs, H, out_dir, sets, copies, device, threads = job[:7]
    force = tuple(job[7]) if len(job) > 7 else ()
    facing = job[8] if len(job) > 8 else "game"
    keep = set(job[9]) if len(job) > 9 else set()      # fields left as the simulator has them (an ablation)
    if device == "cpu":
        torch.set_num_threads(threads)
    params = _params(sets)
    morale.TRACE = _Last()
    if not hasattr(morale.step, "inner"):
        morale.step = _seen(morale.step)
    torch.manual_seed(sum(ord(c) for d in run_dirs for c in Path(d).name) % (2 ** 31))
    gen = torch.Generator(device="cpu").manual_seed(sum(ord(c) for d in run_dirs for c in Path(d).name) % (2 ** 31) + 1)
    games, armies = [], []
    for d in run_dirs:
        g = check.load(d)
        fac = check.factions_of(d, g.arena)
        a = scenario.from_recording(g, d)
        a["sides"][1]["faction"] = fac.get("own")
        a["sides"][2]["faction"] = fac.get("enemy")
        games.append(g)
        armies.append(a)
    HS = max(check.units_a_side(d) for d in run_dirs)
    C = copies
    st = scenario.build([a for a in armies for _ in range(C)], params, device=device, per_side=HS)
    B, N = st.B, st.N
    u = st.u
    u["ai"] = torch.zeros_like(u["ai"])              # the recorded casts only (below)
    dt = params.dt
    per_s = int(round(1.0 / dt))
    cal = {"dt": dt, "fade_s": params.sim["morale"]["recent_s"], "hist_s": max(params.sim["morale"]["casualties_s"],
           params.sim["morale"].get("extended_s") or 0.0), "recent_window_s": params.sim["morale"]["casualties_s"],
           "reset_s": params.sim["contact"].get("reset_s", 0.0), "min_share": params.sim["charge"]["min_speed_share"],
           "min_runup": params.sim["charge"]["min_runup_m"], "decay": params.battle["charge_decay_duration"],
           "rout_pin_s": params.sim["contact"].get("rout_pin_s") or 0.0,
           "blocked_side": params.sim["charge"].get("blocked_side", 0), "front_deg": params.sim["contact"]["front_deg"],
           "rear_deg": params.sim["contact"]["rear_deg"]}
    T = np.array([len(g.t) for g in games])
    starts = window_starts(int(T.max()), H)
    W = len(starts)
    K = morale.history_steps(params, dt)
    # the game state and the clocks at the window starts, in slot layout [B, W, N]
    put_f = {k: np.full((B, W, N), np.nan, np.float32) for k in GAME_F + ("target",) + CLOCKS}
    put_b = {k: np.zeros((B, W, N), bool) for k in GAME_B}
    hist = np.zeros((B, W, K, N), np.float32)
    lord_dead = np.full((B, W, 2), -1.0, np.float32)
    ab_on = np.zeros((B, W, A.SLOTS, N), np.float32)
    ab_cd = np.zeros((B, W, A.SLOTS, N), np.float32)
    snap_s = np.zeros((B, W, N), np.float32)
    snap_ab = np.full((B, W, N), -1, np.int64)
    use = np.full((B, int(T.max()) + 1, N), -1, np.int64)
    # forced each second (force): the game's places (pos) and health with its casualty windows and fight balance (hp)
    g_row = {f: np.full((B, int(T.max()) + 1, N), np.nan, np.float32) for f in ("x", "z", "b", "hp", "men", "mv")} if force else {}
    per_row = ({f: np.zeros((B, int(T.max()) + 1, N), np.float32) for f in ("taken", "shot_taken", "dealt", "shot_dealt",
                                                                            "loss")} if "hp" in force else {})
    rows, metas, clk_all = [], [], []
    u_np = {k: u[k].detach().cpu().numpy() for k in ("hp0", "run", "men0", "reload", "range", "lord", "cost", "large",
                                                         "ab0", "ab1", "ab2", "leadership")}
    for k, (g, a, d) in enumerate(zip(games, armies, run_dirs)):
        m = scenario.slots(a, HS)
        so = np.array([m[nm] for nm in g.names])
        b0 = k * C
        info = {"hp0": u_np["hp0"][b0, so], "run": u_np["run"][b0, so], "men0": u_np["men0"][b0, so],
                "reload": np.where(u_np["range"][b0, so] > 0, u_np["reload"][b0, so], 0.0),
                "lord": u_np["lord"][b0, so]}
        st_k = [s for s in starts if s + H < T[k]]
        clk = stepclock.clocks(g, st_k, info, cal)
        clk_all.append(clk)
        every = stepclock.clocks(g, list(range(len(g.t))), info, cal, hist=False)
        fr = math.exp(-1.0 / float(cal["fade_s"]))
        # the game's HP dealt a row, melee / missile (stepclock's attribution; increments of its decaying sums)
        g_dl = np.vstack([every["dealt"][:1], every["dealt"][1:] - every["dealt"][:-1] * fr])
        g_sdl = np.vstack([every["shot_dealt"][:1], every["shot_dealt"][1:] - every["shot_dealt"][:-1] * fr])
        if "hp" in force:
            for f in ("taken", "shot_taken", "dealt", "shot_dealt"):
                for c in range(C):
                    per_row[f][k * C + c, :len(g.t)][:, so] = every[f]
            loss = stepclock.losses(g, info["hp0"])
            for c in range(C):
                per_row["loss"][k * C + c, :len(g.t)][:, so] = loss
        ab_idx = np.stack([u_np[f"ab{j}"][b0, so] for j in range(A.SLOTS)])
        a_use, a_on, a_cd, a_ss, a_sa = _abilities(g, d, list(so), ab_idx, params, st_k, len(g.t), N)
        r = _orders(g, a, params, list(so), N)
        lut = np.full(N, -1)
        lut[so] = np.arange(len(so))
        metas.append({"so": so, "info": info, "lut": lut, "starts": st_k, "ab_idx": ab_idx, "g_dl": g_dl, "g_sdl": g_sdl,
                      "o_kind": r["kind"][:, so], "o_target": lut[np.clip(r["target"][:, so], -1, N - 1)]})
        nw = len(st_k)
        for c in range(C):
            b = b0 + c
            rows.append(r)
            use[b, :len(g.t)] = a_use
            for f in g_row:
                g_row[f][b, :len(g.t)][:, so] = (stepclock._ffill(g.f[f]) if f in ("hp", "men") else
                                                 np.asarray(g.f[f], np.float32))
            for w, i0 in enumerate(st_k):
                for f in GAME_F:
                    put_f[f][b, w, so] = g.f[f][i0]
                tg = np.asarray(g.target[i0], int)
                put_f["target"][b, w, so] = np.where(tg >= 0, so[np.clip(tg, 0, None)], -1)
                for f in GAME_B:
                    put_b[f][b, w, so] = np.asarray(g.f[f][i0], bool)
            for f in CLOCKS:
                v = clk[f]
                if f == "aim_tgt":
                    v = np.where(v >= 0, so[np.clip(v, 0, None).astype(int)], -1)
                put_f[f][b, :nw][:, so] = v
            hist[b, :nw][:, :, so] = clk["lost_hist"][:, :K]
            lord_dead[b, :nw] = clk["lord_dead_s"]
            ab_on[b, :nw][:, :, so] = a_on
            ab_cd[b, :nw][:, :, so] = a_cd
            snap_s[b, :nw][:, so] = a_ss
            snap_ab[b, :nw][:, so] = a_sa
    policy = replay.Replay(rows, device=device, grace_s=1e9,
                           router_release=params.sim["missile"].get("replay_router_release", False), params=params)
    use_t = torch.as_tensor(use, device=device)
    th = torch.tensor([params.fatigue[k] for k in F.LEVELS] + [params.fatigue["threshold_max"]], device=device)
    Tt = torch.as_tensor(T, device=device).repeat_interleave(C)
    sim = {k: np.zeros((B, W, H, N), np.float32) for k in SIM_SEC}
    t_n = np.zeros((B, W, N), np.int8)
    t_sec = np.zeros((B, W, N), np.int8)
    t_gap = np.zeros((B, W, N), np.float32)
    terms0 = np.zeros((len(TERMS), B, W, N), np.float32)
    terms1 = np.zeros((len(TERMS), B, W, N), np.float32)
    seen0 = np.zeros((len(SEEN), B, W, N), np.float32)
    seen1 = np.zeros((len(SEEN), B, W, N), np.float32)
    fade = math.exp(-dt / params.sim["morale"]["recent_s"])
    present = u["side"] > 0
    spacing = params.sim["formation"]["spacing_m"]
    ccal = params.sim["contact"]
    tens = lambda a: torch.as_tensor(a, device=device)
    for w, i0 in enumerate(starts):
        active = (i0 + H) < Tt
        if not bool(active.any()):
            break
        gx = tens(put_f["x"][:, w])
        known = present & active[:, None] & torch.isfinite(gx) & torch.isfinite(tens(put_f["men"][:, w]))

        def put(name, val, mask=known):
            if name in keep:
                return
            u[name] = torch.where(mask, val.to(u[name].dtype), u[name])

        def gf(name, nan=0.0):
            return torch.nan_to_num(tens(put_f[name][:, w]), nan=nan)

        gone = known & (gf("gone") > 0)
        put("gone", gone, known)
        live = known & ~gone
        put("x", gx)
        put("z", gf("z"))
        put("b", gf("b"), known & torch.isfinite(tens(put_f["b"][:, w])))
        men = gf("men").clamp(min=0)
        hp = gf("hp", 1.0).clamp(0, 1)
        hp = torch.where((men > 0) | gone, hp, torch.zeros_like(hp))
        put("men", torch.minimum(men, u["men0"]))
        put("hp", hp)
        put("hp_abs", hp * u["hp0"])
        mpk = known & torch.isfinite(tens(put_f["mp"][:, w]))
        put("mp", gf("mp"), mpk)
        put("morale", gf("mp") * u["leadership"], mpk)
        put("ms", gf("ms"), known & torch.isfinite(tens(put_f["ms"][:, w])))
        for f in ("r", "s", "w", "m", "lf", "rf", "bf"):
            put(f, tens(put_b[f][:, w]))
        put("a", gf("a"), known & torch.isfinite(tens(put_f["a"][:, w])))
        put("k", gf("k"), known & torch.isfinite(tens(put_f["k"][:, w])))
        put("target", gf("target", -1).long())
        lv = tens(put_f["fat"][:, w])
        lvi = torch.nan_to_num(lv, nan=0).long().clamp(0, len(F.LEVELS) - 1)
        fk = known & torch.isfinite(lv) & (lvi.float() != u["fat"])
        put("fatigue", th[lvi] + 0.25 * (th[lvi + 1] - th[lvi]), fk)
        put("fat", lvi.float(), known & torch.isfinite(lv))
        # the clocks
        for f in ("taken", "shot_taken", "dealt", "shot_dealt", "waver_s", "rout_count", "rout_s", "rally_s",
                  "contact_s", "out_s", "melee_s", "runup", "charge", "vx", "vz", "rpin_s", "rpin_d", "under_fire_s",
                  "low_s", "dead_s", "gone_s", "recent", "extended"):
            if f in u:
                put(f, gf(f))
        put("ran_in", gf("ran_in") > 0)
        # A shooter keeps its own reload and aim (the game's men fire as each is loaded, not by unit volleys: a reload
        # rebuilt from its ammunition put every shooter at 'just fired' and the windows' fire in step - the simulator's
        # missile HP 6k/s in a window's 2nd second, 150k/s in its 9th, the game's flat); only the target the game
        # shows it firing at goes in (else the reset's new places made it pick again: the 3 s retarget in step), with
        # the aim of the seconds it stood. --keep game_shooters (an ablation): the rebuilt reload and aim instead.
        shooter = known & (u["range"] > 0)
        g_at = gf("aim_tgt", -1).long()
        firing_g = shooter & (g_at >= 0)
        put("aim_tgt", g_at, firing_g)
        put("aim", torch.maximum(u["aim"], gf("aim")), firing_g)
        # its place in the reload cycle: a random one (the game's fire is a trickle of men, no unit volley to read; a
        # shooter loaded but idle in the simulator's own last window would get its target back at once and every such
        # one fired in the window's first second, then reloaded in step - volleys 118 / 243 a second in its 2nd / 4th)
        put("unready", torch.rand(u["unready"].shape, generator=gen, device="cpu").to(device), firing_g)
        if "game_shooters" in keep:
            put("unready", gf("unready"), shooter)
            put("aim", gf("aim"), shooter)
            put("aim_tgt", g_at, shooter)
        if K and "lost_hist" not in keep:
            hk = tens(hist[:, w]).reshape(B, K * N)
            mask = known.repeat(1, K)
            u["lost_hist"] = torch.where(mask, hk, u["lost_hist"])
        # stuck men only on a router
        if "stk_n" in u:
            put("stk_n", torch.zeros_like(u["stk_n"]), known & ~u["r"])
        if "lord_dead_s" not in keep:
            st.lord_dead_s = torch.where(active[:, None], tens(lord_dead[:, w]), st.lord_dead_s)
        for j in range(A.SLOTS):
            put(f"ab{j}_on", tens(ab_on[:, w, j]))
            put(f"ab{j}_cd", tens(ab_cd[:, w, j]))
        if "snap_s" in u:
            put("snap_s", tens(snap_s[:, w]))
            put("snap_ab", tens(snap_ab[:, w]))
        st.t = torch.full((B,), float(i0), device=device)
        st.done = ~active
        st.winner = torch.zeros_like(st.winner)
        if "aim_tgt" not in keep:
            # the target each shooter takes from these places, without the retarget's cost: a probe step on a copy (the
            # reset's jump made shooters pick anew and wait missile.retarget_s - the shots in a window's 2nd-3rd second
            # 0.55 of its later ones - a switch the game never made)
            probe = st.clone()
            battle.step(probe, policy(probe), params, dt)
            took = shooter & (probe.u["aim_tgt"] >= 0)
            put("aim_tgt", probe.u["aim_tgt"], took)
            del probe
        with torch.no_grad():
            pw = geometry.pairwise(u, spacing, rect=bool(ccal.get("rect_gap")),
                                    melee_depth=geometry.melee_depth(params.sim))
            alive = present & (u["men"] > 0) & ~u["gone"]
            standing = alive & ~u["r"]
            lim = geometry.contact_limit(u, pw, ccal, params.battle, spacing)
            foe = pw["enemy"] & alive[:, :, None] & standing[:, None, :]
            touch = foe & (pw["gap"] <= lim)
            sec = geometry.sector(pw["rel_i"], ccal["front_deg"], ccal["rear_deg"]) if hasattr(geometry, "sector") else None
            t_n[:, w] = touch.sum(2).cpu().numpy()
            if sec is not None:
                t_sec[:, w] = torch.where(touch, sec.float(), torch.zeros_like(pw["gap"])).amax(2).cpu().numpy()
            t_gap[:, w] = torch.where(foe, pw["gap"], torch.full_like(pw["gap"], 1e9)).amin(2).clamp(max=1e4).cpu().numpy()
            if facing == "sim":
                # a formation the simulator sees in contact faces its nearest standing opponent's centre, as its own
                # units do (battle.py's facing in melee), instead of the game's bearing
                near = torch.where(touch, pw["dist"], torch.full_like(pw["dist"], 1e9))
                opp = near.argmin(2)
                turn = known & standing & touch.any(2) & (u["men0"] > 1)
                bx = u["x"].gather(1, opp) - u["x"]
                bz = u["z"].gather(1, opp) - u["z"]
                u["b"] = torch.where(turn, torch.rad2deg(torch.atan2(bx, bz)) % 360, u["b"])
        for h in range(H):
            mel = torch.zeros_like(u["hp"])
            mis = torch.zeros_like(u["hp"])
            dl = torch.zeros_like(u["hp"])
            sdl = torch.zeros_like(u["hp"])
            for s in range(per_s):
                o = policy(st)
                o.ability = use_t[:, min(i0 + h, use_t.shape[1] - 1)]
                tk, stk = u["taken"].clone(), u["shot_taken"].clone()
                dk, sdk = u["dealt"].clone(), u["shot_dealt"].clone()
                battle.step(st, o, params, dt)
                u = st.u
                mel = mel + (u["taken"] - tk * fade).clamp(min=0)
                mis = mis + (u["shot_taken"] - stk * fade).clamp(min=0)
                dl = dl + (u["dealt"] - dk * fade).clamp(min=0)
                sdl = sdl + (u["shot_dealt"] - sdk * fade).clamp(min=0)
                if h == 0 and s == 0 and morale.TRACE.parts is not None:
                    for q, nm in enumerate(TERMS):
                        terms0[q, :, w] = morale.TRACE.parts[nm].float().cpu().numpy()
                    for q, nm in enumerate(SEEN):
                        seen0[q, :, w] = morale.TRACE.seen[nm].float().cpu().numpy()
                st.done = ~active
                st.winner = torch.zeros_like(st.winner)
            st.t = torch.full((B,), float(i0 + h + 1), device=device)
            if force:
                _force(u, force, g_row, per_row, i0 + h + 1, known & active[:, None], K, N, tens)
            vals = {"mel": mel, "mis": mis, "dl": dl, "sdl": sdl}
            for f in SIM_SEC:
                v = vals.get(f, u.get(f))
                sim[f][:, w, h] = v.float().cpu().numpy()
        if morale.TRACE.parts is not None:
            for q, nm in enumerate(TERMS):
                terms1[q, :, w] = morale.TRACE.parts[nm].float().cpu().numpy()
            for q, nm in enumerate(SEEN):
                seen1[q, :, w] = morale.TRACE.seen[nm].float().cpu().numpy()
    out = {}
    for k, (g, meta, clk) in enumerate(zip(games, metas, clk_all)):
        so, nw = meta["so"], len(meta["starts"])
        bs = slice(k * C, (k + 1) * C)
        d = {f"sim_{f}": np.moveaxis(sim[f][bs, :nw][..., so], 0, 2) for f in SIM_SEC}      # [W, H, C, n]
        for f in ("hp", "men", "mp", "mel", "mis", "x", "z", "dl", "sdl"):
            d[f"sim_{f}"] = d[f"sim_{f}"].astype(np.float32)
        for f in ("r", "s", "w", "m", "gone"):
            d[f"sim_{f}"] = d[f"sim_{f}"] > 0.5
        d["t_n"] = np.moveaxis(t_n[bs, :nw][..., so], 0, 1)                                    # [W, C, n]
        d["t_sec"] = np.moveaxis(t_sec[bs, :nw][..., so], 0, 1)
        d["t_gap"] = np.moveaxis(t_gap[bs, :nw][..., so], 0, 1)
        d["terms0"] = terms0[:, bs, :nw][..., so].mean(1)                                       # [K, W, n]
        d["terms1"] = terms1[:, bs, :nw][..., so].mean(1)
        for q, nm in enumerate(SEEN):
            d[f"x0_{nm}"] = seen0[q, bs, :nw][..., so].mean(0)                                 # [W, n]
            d[f"x1_{nm}"] = seen1[q, bs, :nw][..., so].mean(0)
        for f in CLOCKS:
            d[f"c_{f}"] = clk[f].astype(np.float32)
        d["starts"] = np.array(meta["starts"])
        d["H"] = np.array(H)
        d["copies"] = np.array(C)
        d["force"] = np.array(",".join(force))
        d["facing"] = np.array(facing)
        d["t"] = np.asarray(g.t, float)
        for f in GAME_F:
            d[f"g_{f}"] = np.asarray(g.f[f], np.float32)
        for f in GAME_B:
            d[f"g_{f}"] = np.asarray(g.f[f], bool)
        d["g_target"] = np.asarray(g.target, np.int16)
        d["side"] = np.asarray(g.side, np.int8)
        d["keys"] = np.array(g.keys)
        d["names"] = np.array(g.names)
        d["own_ai"] = np.array(g.own_ai)
        for f in ("hp0", "men0", "run", "reload", "lord"):
            d[f"u_{f}"] = np.asarray(meta["info"][f])
        b0 = k * C
        for f in ("cost", "range", "large", "leadership"):
            d[f"u_{f}"] = u_np[f][b0, so]
        d["casts"] = (use[b0, :len(g.t)][:, so] >= 0)
        d["g_dl"], d["g_sdl"] = meta["g_dl"].astype(np.float32), meta["g_sdl"].astype(np.float32)
        d["o_kind"], d["o_target"] = meta["o_kind"].astype(np.int8), meta["o_target"].astype(np.int16)
        p = Path(out_dir) / f"{g.run}.npz"
        np.savez_compressed(p, **d)
        out[g.run] = str(p)
    return out


def run(run_dirs, out_dir, H=10, copies=4, device="cpu", chunk=12, workers=1, sets=(), force=(), facing="game",
        keep=(), log=print):
    """Step-replay every recorded battle not yet in out_dir; batches of similar size."""
    from tools.nn.sim import check
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [Path(d) for d in run_dirs if not (out_dir / f"{Path(d).name}.npz").exists()]
    log(f"step replay H={H} C={copies}: {len(todo)} battles to do, {len(run_dirs) - len(todo)} cached in {out_dir}")
    if not todo:
        return
    hs = {d: check.units_a_side(d) for d in todo}
    size = {d: (d / "events.jsonl").stat().st_size / max(hs[d], 1) for d in todo}
    order = sorted(todo, key=lambda d: (hs[d], size[d]))
    parts = [order[k:k + chunk] for k in range(0, len(order), chunk)]
    jobs = [([str(d) for d in p], H, str(out_dir), list(sets), copies, device, 1 if workers > 1 else 4, tuple(force),
             facing, tuple(keep)) for p in parts]
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=1)
        it = pool.imap_unordered(_batch, jobs)
    else:
        it = map(_batch, jobs)
    for n, got in enumerate(it, 1):
        log(f"  batch {n} of {len(jobs)} done ({len(got)} battles), {time.time() - t0:.0f} s")
    if workers > 1:
        pool.close()
        pool.join()


def gate_runs(sets):
    """The recorded battles of gate sets (build/steps/watch_set8_<set>.log 'Results:' lines)."""
    import re
    from tools import config as project
    out = []
    for it in sets:
        for line in (project.BUILD / "steps" / f"watch_set8_{it}.log").read_text(encoding="utf-8",
                                                                                   errors="replace").splitlines():
            if "Results:" in line:
                m = re.findall(r"(\d{8}-\d{6})", line)
                if m and m[-1] not in out:
                    out.append(m[-1])
    return out


def main(argv=None):
    from tools import config as project
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", nargs="*", default=[], help="gate sets (it1 ... it9: the 72 battles)")
    ap.add_argument("--lists", nargs="*", default=[], help="json files with a 'runs' list (build/compare/itS24.json)")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="section.key=json over config/nn/sim.json")
    ap.add_argument("--facing", default="game", help="game: the recorded bearings at a window start; sim: a formation "
                    "in contact faces its nearest opponent's centre (the simulator's own rule)")
    ap.add_argument("--keep", default="", help="fields not put back at a window start (comma list; 'clocks': every "
                    "rebuilt clock; 'game_shooters': the shooters' reload and aim rebuilt from the recording): an ablation")
    ap.add_argument("--force", default="", help="pos, hp or pos,hp: the game's places / health put back every second")
    a = ap.parse_args(argv)
    names = list(a.runs) + gate_runs(a.sets)
    for lst in a.lists:
        names += json.loads(Path(lst).read_text(encoding="utf-8"))["runs"]
    names = list(dict.fromkeys(names))
    runs = [project.BUILD / "nn-arena" / "runs" / n for n in names]
    run(runs, a.out, H=a.H, copies=a.copies, device=a.device, chunk=a.chunk, workers=a.workers, sets=a.set,
        force=[f for f in a.force.split(",") if f], facing=a.facing,
        keep=[k for f in a.keep.split(",") if f for k in (CLOCKS + ("lost_hist", "lord_dead_s", "ab0_on", "ab0_cd",
              "ab1_on", "ab1_cd", "ab2_on", "ab2_cd", "snap_s", "snap_ab", "quiet", "late", "loaded_s", "turn_on",
              "stk_n") if f == "clocks" else (f,))])


if __name__ == "__main__":
    main()
