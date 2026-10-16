"""Magic in the game before the simulator knows it (build/magic/plan.md 6-7): the Gold Wizard (Lore of Metal,
Searing Doom) and the Warlock Engineer (Lore of Ruin, Warp Lightning), each casting his level-1 spell on a held,
fearless target at a set second. The lanes run on the melee probe (tools/nn/charge_probe.py plan 'magic',
src/entries/charge_probe.lua lane mode 'cast'); far apart, the caster gap_m in front of his target (inside the
spells' 200 m), nobody else ordered.

  battle 1  the game's own winds (P0, P1): each side's caster casts at 40 s and tries again every second from 41 s to
            150 s (when the second and third casts go: the cooldown from the cast or from the end, the cost); a second
            caster each side without <unit_capabilities> (the game's default abilities, never casting): what the
            list changes (probe_mage). Targets clanrats and Empire spearmen;
  battle 2  scripted winds (P1): the Empire side +30 current +40 reserve (are the current and the reserve capped?), the
            Skaven side +30 current -60 reserve (does the flow slow with a small reserve?); two casters a side, casting
            at 10 / 20 s and again from 35 / 45 s: Searing Doom on stormvermin (armour) and skavenslaves (none), Warp
            Lightning on swordsmen and greatswords (P2, P3 damage by armour);
  battle 3  the cast's rules and a moving target (P2, P4): Searing Doom on clanrats walking at the caster, Warp
            Lightning ordered on spearmen out of range (260 m: does the caster walk in?), the stormvermin and the
            swordsmen again (a second sample);
  battle 4  friendly hits: each target (held, 45 m wide: a shallow block) has a unit of the caster's own side right
            behind it (extras: halted, fearless, never ordered), its front FRIEND_GAPS_M (2 / 5 m) from the target's
            back; two casts a lane (10 s, then from 50 s); both sides +30 current winds. What a bombardment's projectiles
            and explosions take from the caster's own men standing near its target (the passports' can_damage_allies /
            affects_allies: true for both spells).

Every lane samples both units every 0.5 s (men, health), the soldiers' places every 1 s and each side's winds every
1 s (probe_winds). The casters' abilities are exactly the spell and their innate ones (tools/nn/scenario.py
capabilities).

    python -m tools.nn.charge_probe plan --plan magic             # the battles
    python -m tools.nn.charge_probe run --plan magic --speed 3    # build + launch each, in turn
    python -m tools.nn.magic_probe [runs...] [--out build/magic/probe.json]   # the casts, the winds, the damage
    python -m tools.nn.magic_probe runs... --sim [--copies 8] [--set wounds='"even"']   # + the simulator's twin
                                    # (tools/nn/sim/magic.py; needs torch: bash tools/nn/dock.sh tools.nn.magic_probe ...)
    python -m tools.nn.magic_probe <battle-4 run> --friendly [--copies 8]   # the friendly hits, game against twin
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp

SEARING_DOOM = "wh_main_spell_metal_searing_doom"
WARP_LIGHTNING = "wh2_main_spell_ruin_warp_lightning"
GOLD, WARLOCK = "goldwiz", "warlock"
SPELL = {GOLD: SEARING_DOOM, WARLOCK: WARP_LIGHTNING}
GAP_M = 120              # caster to the target's front: inside the spells' 200 m (target_intercept_range)
FAR_M = 260              # ... and out of it (battle 3)
WINDS_MS = 1000
MEN_NEAR_M = 400         # the soldiers' places whenever caster and target are this close (always: no contact)
# battle -> the scripted winds at the start: [{side (1 Empire, 2 Skaven), current, reserve}] (amounts added)
WINDS_SET = {2: [{"side": 1, "current": 30, "reserve": 40}, {"side": 2, "current": 30, "reserve": -60}],
             4: [{"side": 1, "current": 30, "reserve": 0}, {"side": 2, "current": 30, "reserve": 0}]}
FRIEND_GAPS_M = (2, 5)   # battle 4: the friend's front to the target's back
FRIEND_T_W = 45          # battle 4: the targets' frontage (a shallow block: the friend within the strikes' reach)
WINDOWS = ((0, 2), (2, 4), (4, 6), (6, 8), (8, 12), (12, 20))   # s after the cast (its winds drop)
DROP = 2.5               # a fall of the side's current winds this big in one second is a cast
SPELL_COST = 6           # both spells' mana_cost (the passports): a drop of 12 in one second is two casts


def tries(first_s, from_s, to_s, every_s=1):
    """A caster's casts: one at first_s, then a try every every_s from from_s to to_s (the cooldown's end)."""
    out = [first_s] + list(range(from_s, to_s + 1, every_s))
    return sorted(set(out))


def caster(who, target, at, gap_m=GAP_M, max_s=160, kind="", target_mode="hold", abilities=True, **extra):
    """One lane: the caster `who` (GOLD / WARLOCK) and its target; at: the cast seconds after the go."""
    key = SPELL[who]
    if abilities:                         # (none: the game's defaults; the entry config takes no None)
        extra["a_abilities"] = [key]
    return cp.lane(who, target, "cast", target_mode, gap_m=gap_m, fight_s=max_s, max_s=max_s, answer=False,
                   casts=[{"at_s": float(s), "key": key, "tg": "t"} for s in at],
                   kind=kind or f"{who}>{target}", **extra)


def battles():
    """[[lane spec]] of the magic plan (tools/nn/charge_probe.battles hands these over)."""
    b1 = [caster(GOLD, "clanrat", tries(40, 41, 150), kind="b1_sd_clanrat"),
          caster(WARLOCK, "spear", tries(40, 41, 150), kind="b1_wl_spear"),
          caster(GOLD, "slave", [], abilities=False, kind="b1_default_gold"),
          caster(WARLOCK, "swords", [], abilities=False, kind="b1_default_warlock")]
    b2 = [caster(GOLD, "svsh", tries(10, 35, 80), max_s=130, kind="b2_sd_svsh"),
          caster(GOLD, "slave", tries(20, 45, 90), max_s=130, kind="b2_sd_slave"),
          caster(WARLOCK, "swords", tries(10, 35, 80), max_s=130, kind="b2_wl_swords"),
          caster(WARLOCK, "gs", tries(20, 45, 90), max_s=130, kind="b2_wl_gs")]
    b3 = [caster(GOLD, "clanrat", [5], max_s=100, target_mode="push", push_m=60, kind="b3_sd_walking"),
          caster(WARLOCK, "spear", [5], gap_m=FAR_M, max_s=100, kind="b3_wl_far"),
          caster(GOLD, "svsh", [20], max_s=100, kind="b3_sd_svsh"),
          caster(WARLOCK, "swords", [30], max_s=100, kind="b3_wl_swords")]
    b4 = [friendly(GOLD, "clanrat", "spear", FRIEND_GAPS_M[0]), friendly(GOLD, "slave", "swords", FRIEND_GAPS_M[1]),
          friendly(WARLOCK, "spear", "clanrat", FRIEND_GAPS_M[0]), friendly(WARLOCK, "swords", "slave", FRIEND_GAPS_M[1])]
    return [b1, b2, b3, b4]


def friendly(who, target, friend, gap_m, max_s=120):
    """A battle-4 lane: the caster's own unit `friend` right behind his target, its front gap_m from the target's back
    (the target FRIEND_T_W wide, the friend cp.WIDTH_M; both facing away from each other: the friend faces as the
    caster does), two casts (10 s, then a try every second from 50 to 90 s)."""
    t_depth = cp.depth(cp.UNITS[target][0], cp.UNITS[target][1], FRIEND_T_W)
    e_depth = cp.depth(cp.UNITS[friend][0], cp.UNITS[friend][1], cp.WIDTH_M)
    extra = {"short": friend, "dx": 0.0, "dz": -round(t_depth / 2 + gap_m + e_depth / 2, 1), "width": cp.WIDTH_M}
    return caster(who, target, tries(10, 50, 90), max_s=max_s, t_w=FRIEND_T_W, extras=[extra], friend_gap_m=gap_m,
                  kind=f"b4_{'sd' if who == GOLD else 'wl'}_{target}_ff{gap_m}")


def config_extra(index):
    """The entry config's magic fields of battle `index` (1-based)."""
    return {"winds_ms": WINDS_MS, "winds_set": WINDS_SET.get(index, []), "men_ms": 1000, "men_near_m": MEN_NEAR_M}


# ---------------------------------------------------------------- the recordings

def load_magic(run_dir):
    """The magic events of one run: {winds [(t s, {side: row})], api, set, mages, tries [probe_ability rows]}."""
    out = {"winds": [], "api": {}, "set": [], "mages": {}, "tries": []}
    for line in (Path(run_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if '"probe_' not in line:
            continue
        r = json.loads(line)
        ev = r["event"]
        if ev == "probe_winds":
            out["winds"].append((r["t"] / 1000, {1: r.get("s1") or {}, 2: r.get("s2") or {}}, r.get("m")))
        elif ev == "probe_winds_api":
            out["api"][r["side"]] = r
        elif ev == "probe_winds_set":
            out["set"].append(r)
        elif ev == "probe_mage":
            out["mages"][r["lane"]] = r
        elif ev == "probe_ability":
            out["tries"].append(r)
    return out


def side_of(lane):
    """1 (Empire, alliance 0) or 2 (Skaven) for the lane's caster."""
    return 1 if lane["spec"]["attacker"].startswith("own_") else 2


def casts_of(winds, side, drop=DROP):
    """[(t s, drop)] of a side: the 1-s falls of its current winds of at least `drop` (the casts)."""
    pts = [(t, w[side].get("cur")) for t, w, _ in winds if w[side].get("cur") is not None]
    out = []
    for (t0, a), (t1, b) in zip(pts, pts[1:]):
        if a - b >= drop:
            out.append((t1, round(a - b, 3)))
    return out


def winds_summary(winds, side, costs=()):
    """A side's winds: the first, highest and last current (and reserve), and the flow a second from the reserve into
    the current: (last - first + the casts' costs) / the time (the game reads whole numbers: a rise of 1 at a time, so
    the flow is taken over the whole recording, not second by second); the reserve's fall a second likewise."""
    pts = [(t, w[side].get("cur"), w[side].get("res"), w[side].get("rate")) for t, w, _ in winds
           if w[side].get("cur") is not None]
    if len(pts) < 2:
        return None
    t = np.array([p[0] for p in pts])
    cur = np.array([p[1] for p in pts], float)
    span = float(t[-1] - t[0])
    res = [p[2] for p in pts if p[2] is not None]
    rates = [p[3] for p in pts if p[3] is not None]
    return {"first": float(cur[0]), "max": float(cur.max()), "last": float(cur[-1]),
            "flow": round((cur[-1] - cur[0] + sum(costs)) / span, 4) if span > 0 else None,
            "res_first": res[0] if res else None, "res_last": res[-1] if res else None,
            "res_fall": round((res[0] - res[-1]) / span, 4) if res and span > 0 else None,
            "rate_first": rates[0] if rates else None, "rate_last": rates[-1] if rates else None}


def damage_after(samples, t_cast, windows=WINDOWS):
    """The target's health and men lost in each window after t_cast ({"0-2": (hp, men)}), and in total to 20 s."""
    t = np.array([s[0] for s in samples])
    hp = cp.series(samples, "tg", "hp")
    men = cp.series(samples, "tg", "men")
    out = {}
    for lo, hi in windows:
        a, b = cp.at(t, hp, t_cast + lo), cp.at(t, hp, t_cast + hi)
        ma, mb = cp.at(t, men, t_cast + lo), cp.at(t, men, t_cast + hi)
        out[f"{lo}-{hi}"] = (None if np.isnan(a - b) else round(a - b), None if np.isnan(ma - mb) else round(ma - mb))
    hp0, men0 = cp.at(t, hp, t_cast), cp.at(t, men, t_cast)
    return {"hp0": None if np.isnan(hp0) else round(hp0), "men0": None if np.isnan(men0) else round(men0),
            "windows": out}


STRIKE_S = (4.0, 11.0)   # s after the try: the window a strike's damage lies in (the first hit ~5.5-6 s, Searing Doom's
#                          last ~9.5 s: probe1.md)
BASE_S = 6.0             # s before the try: the melee's own rate (battle 4: the friend fought the target throughout)


def friend_after(lane, t_cast, strike=STRIKE_S, base_s=BASE_S):
    """The caster's own units near the target (battle 4's extras), each: {"hp", "men" - lost in the strike window
    (STRIKE_S after t_cast), "hp_net", "men_net" - the same less the loss at the rate of the base_s before the try (the
    melee's own: the friend and the target fight), "fight" - in melee in the window}; samples lane["extras"]
    [(t, [row])]; [] without extras."""
    rows = lane.get("extras") or []
    if not rows:
        return []
    t = np.array([r[0] for r in rows])
    out = []
    for j in range(len(rows[0][1])):
        rec = {"fight": any(bool((r[1][j] or {}).get("m")) for r in rows
                            if t_cast + strike[0] <= r[0] <= t_cast + strike[1])}
        for key in ("hp", "men"):
            v = np.array([np.nan if (r[1][j] or {}).get(key) is None else float(r[1][j][key]) for r in rows])
            lost = cp.at(t, v, t_cast + strike[0]) - cp.at(t, v, t_cast + strike[1])
            rate = (cp.at(t, v, t_cast - base_s) - cp.at(t, v, t_cast)) / base_s
            rec[key] = None if np.isnan(lost) else round(float(lost))
            net = lost - (rate if np.isfinite(rate) else 0.0) * (strike[1] - strike[0])
            rec[f"{key}_net"] = None if np.isnan(net) else round(float(net), 1)
        out.append(rec)
    return out


def measure(run_dir):
    """Rows of one run: per lane its casts (from the side's winds drops, matched to the lane's last try before them)
    with the damage after each; per side the winds."""
    lanes = cp.load_run(run_dir)
    magic = load_magic(run_dir)
    by_side = {}
    for ln in lanes:
        if ln["spec"].get("mode") == "cast":
            by_side.setdefault(side_of(ln), []).append(ln)
    rows, sides = [], {}
    for side in (1, 2):
        drops = casts_of(magic["winds"], side)
        sides[side] = {"api": (magic["api"].get(side) or {}).get("found"),
                       "set": [s for s in magic["set"] if s["side"] == side],
                       "casts": drops, "winds": winds_summary(magic["winds"], side, [d[1] for d in drops])}
        for t_drop, cost in drops:
            # the lane of this side whose try came last before the drop's sample (its given tries only; a try in the same
            # second as the sample is taken as after it: battle 1, the first try at 40 s showed at the 41 s sample); a
            # drop of several casts' cost (battle 4: two casters a side cast at the same second) goes to as many lanes
            # whose last try is that late
            last = []
            for ln in by_side.get(side, []):
                ts = [a["t"] / 1000 for a in ln["abilities"] if a.get("status") == "given" and a["t"] / 1000 < t_drop - 0.25]
                if ts:
                    last.append((max(ts), ln))
            if not last:
                continue
            last.sort(key=lambda x: -x[0])
            n = max(1, round(cost / SPELL_COST))
            for t_try, ln in [x for x in last if x[0] >= last[0][0] - 0.5][:n]:
                rows.append({"run": ln["run"], "battle": ln["battle"], "lane": ln["spec"]["name"],
                             "kind": ln["spec"].get("kind"), "side": side, "try_s": round(t_try, 1), "drop_s": t_drop,
                             "cost": round(cost / n, 3), **damage_after(ln["samples"], t_try),
                             "friends": friend_after(ln, t_try)})
    mages = {k: {f: v for f, v in m.items() if f in ("u", "active", "passive", "magic", "can", "cco")}
             for k, m in magic["mages"].items()}
    return {"run": Path(run_dir).name, "casts": rows, "sides": sides, "mages": mages}


# ---------------------------------------------------------------- the simulator's twin

def sim_params(params=None, blows=True):
    """The simulator's numbers with the probe's units (data/units/probe-units.json: the casters) beside the pool's (the
    pool's file wins) and the game's own randomness of hits (noise.blows) for the twin's copies."""
    import dataclasses
    from tools.nn.sim.params import load
    params = params or load()
    probe = json.loads(cp.PROBE_UNITS_JSON.read_text(encoding="utf-8"))["units"]
    params = dataclasses.replace(params, units={**probe, **params.units})
    return params.with_cal("noise", blows=bool(blows))


def first_winds(magic):
    """{side: (current, reserve)} of the run's first winds sample."""
    for _, w, _ in magic["winds"]:
        if w[1].get("cur") is not None and w[2].get("cur") is not None:
            return {s: (float(w[s]["cur"]), float(w[s].get("res") or 0.0)) for s in (1, 2)}
    return {}


def sim_lanes(lanes, winds, params, copies=4, device="cpu", seed=0):
    """The simulator on the recorded cast lanes: each lane its own battle - the caster and its target where the first
    sample has them, the target held (a 'push' target walks to where the game's ended), everyone fearless, the sides'
    winds as the game's first sample, the game's casts asked at their seconds (a try on cooldown or without winds is
    dropped, as the game drops it). Returns per lane and copy {"samples": [(t, hp, men)], "casts": [t of an accepted
    try], "caster": [(t, x, z)]} on the game's clock."""
    import torch
    from tools.nn.sim import battle, magic, orders as O, scenario as sim_scenario
    armies, meta = [], []
    for ln, w in zip(lanes, winds):
        spec, s0 = ln["spec"], ln["samples"][0]
        sides = {1: {"faction": cp.EMP, "ai": False, "units": []}, 2: {"faction": cp.SKV, "ai": False, "units": []}}
        for role, row, key, width in (("a", s0[1], spec["a_key"], spec["a_width"]),
                                      ("tg", s0[2], spec["t_key"], spec["t_width"])):
            fac = 1 if cp.FACTION[key] == cp.EMP else 2
            sides[fac]["units"].append({"key": key, "x": row["x"], "z": row["z"], "b": row["b"], "men": row["men"],
                                        "width": width, "general": False, "name": role})
        for _ in range(copies):
            armies.append({"attacker": 1, "sides": sides})
            meta.append((ln, w))
    st = sim_scenario.build(armies, params, device=device, per_side=1)
    B, N = st.B, st.N
    a_slot = torch.tensor([0 if cp.FACTION[m[0]["spec"]["a_key"]] == cp.EMP else 1 for m in meta], device=device)
    t_slot = 1 - a_slot
    b_idx = torch.arange(B, device=device)
    st.u["leadership"] = st.u["leadership"] + 1e4
    st.u["morale"] = st.u["morale"] + 1e4
    for b, (_, w) in enumerate(meta):
        for side, (cur, res) in w.items():
            mask = st.u["side"][b] == side
            st.u["wm_cur"][b, mask] = cur
            st.u["wm_res"][b, mask] = res
    t0 = [float(m[0]["samples"][0][0]) for m in meta]
    tries = [sorted(a["t"] / 1000 for a in m[0]["abilities"] if a.get("status") == "given") for m in meta]
    asked = [0] * B
    push = [m[0]["spec"].get("target_mode") == "push" for m in meta]
    end_xz = [(m[0]["samples"][-1][2]["x"], m[0]["samples"][-1][2]["z"]) for m in meta]
    out = [{"samples": [], "casts": [], "caster": [], "starts": []} for _ in range(B)]
    pending = {}
    before = {"k": [-1] * B, "w": st.u["wm_cur"][b_idx, a_slot].tolist()}

    def policy(st):
        o = O.hold(B, N, device)
        kind = torch.full_like(o.kind, O.KEEP)
        x, z = o.x.clone(), o.z.clone()
        slot = torch.full_like(o.kind, -1)
        target = torch.full_like(o.kind, -1)
        t = float(st.t.max())
        for b in range(B):
            A, T = int(a_slot[b]), int(t_slot[b])
            kind[b, T] = O.HOLD
            if push[b]:
                kind[b, T], x[b, T], z[b, T] = O.MOVE, end_xz[b][0], end_xz[b][1]
            while asked[b] < len(tries[b]) and tries[b][asked[b]] <= t0[b] + t + 1e-6:
                slot[b, A], target[b, A] = 0, T
                pending[b] = tries[b][asked[b]]
                asked[b] += 1
        magic.request(st.u, slot, target)
        before["k"] = st.u["cast_k"][b_idx, a_slot].tolist()
        return O.Orders(kind=kind, x=x, z=z, target=o.target, run=o.run, ability=o.ability)

    def record(st):
        t = float(st.t.max())
        hp = st.u["hp_abs"][b_idx, t_slot].tolist()
        men = st.u["men"][b_idx, t_slot].tolist()
        ax, az = st.u["x"][b_idx, a_slot].tolist(), st.u["z"][b_idx, a_slot].tolist()
        k = st.u["cast_k"][b_idx, a_slot].tolist()
        w = st.u["wm_cur"][b_idx, a_slot].tolist()
        for b in range(B):
            tg = t0[b] + t
            out[b]["samples"].append((tg, hp[b], men[b]))
            out[b]["caster"].append((tg, ax[b], az[b]))
            if b in pending:
                # a try taken: the caster was not casting before it and is now (pending or winding up)
                if before["k"][b] < 0 and k[b] >= 0:
                    out[b]["casts"].append(pending[b])
                pending.pop(b)
            if before["w"][b] - w[b] >= DROP:
                out[b]["starts"].append(round(tg, 2))         # the wind-up's start: the side's winds drop
            before["w"][b] = w[b]

    max_s = max(float(m[0]["spec"].get("max_s", 160)) for m in meta)
    battle.run(st, policy, params, until_s=max_s, record=record)
    return [(m[0], o) for m, o in zip(meta, out)]


def twin_damage(samples, t_cast, windows=WINDOWS):
    """damage_after of a twin's samples [(t, hp, men)]."""
    rows = [(t, None, {"hp": hp, "men": men}, None) for t, hp, men in samples]
    return damage_after(rows, t_cast, windows)


def first_hit(samples, t_cast, until=20.0):
    """Seconds from t_cast to the first health lost (samples [(t, hp)]), None: none."""
    prev = None
    for t, hp in samples:
        if t < t_cast:
            prev = hp
            continue
        if t > t_cast + until:
            break
        if prev is not None and hp < prev - 0.5:
            return round(t - t_cast, 2)
        prev = hp
    return None


def rms(v):
    """The root mean square of log(v) (0: every ratio 1), None for none."""
    v = [x for x in v if x]
    return float(np.sqrt(np.mean(np.log(np.maximum(v, 1e-3)) ** 2))) if v else None


def sim_report(run_dirs, copies=4, device="cpu", params=None, out=None):
    """The game's casts against the twin's: per cast HP and men lost (to 20 s after the try), their ratio twin / game
    (1: the same), the first hit's delay; the geometric means per spell."""
    params = sim_params(params)
    rows = []
    for run in run_dirs:
        lanes = [ln for ln in cp.load_run(run) if ln["spec"].get("mode") == "cast" and not ln["spec"].get("extras")
                 and any(a.get("status") == "given" for a in ln["abilities"])]
        if not lanes:
            continue
        w = first_winds(load_magic(run))
        game = {(c["lane"]): [] for c in measure(run)["casts"]}
        for c in measure(run)["casts"]:
            game[c["lane"]].append(c)
        twins = sim_lanes(lanes, [w] * len(lanes), params, copies=copies, device=device)
        by_lane = {}
        for ln, tw in twins:
            by_lane.setdefault(ln["spec"]["name"], (ln, []))[1].append(tw)
        for name, (ln, tws) in by_lane.items():
            g_casts = sorted(game.get(name, []), key=lambda c: c["try_s"])
            t_g = np.array([s[0] for s in ln["samples"]])
            hp_g = cp.series(ln["samples"], "tg", "hp")
            for i, gc in enumerate(g_casts):
                g_hp = sum(v[0] or 0 for v in gc["windows"].values())
                g_men = sum(v[1] or 0 for v in gc["windows"].values())
                s_hp, s_men, s_hit, s_try, s_start = [], [], [], [], []
                for tw in tws:
                    if i >= len(tw["casts"]):
                        continue
                    if i < len(tw["starts"]):
                        s_start.append(tw["starts"][i])
                    d = twin_damage(tw["samples"], tw["casts"][i])
                    s_hp.append(sum(v[0] or 0 for v in d["windows"].values()))
                    s_men.append(sum(v[1] or 0 for v in d["windows"].values()))
                    s_hit.append(first_hit([(t, hp) for t, hp, _ in tw["samples"]], tw["casts"][i]))
                    s_try.append(tw["casts"][i])
                g_hit = first_hit(list(zip(t_g, hp_g)), gc["try_s"])
                rows.append({"run": Path(run).name, "lane": name, "kind": gc["kind"], "n": i + 1,
                             "spell": SPELL[GOLD] if "_sd_" in gc["kind"] else SPELL[WARLOCK],
                             "try_game": gc["try_s"], "try_twin": float(np.mean(s_try)) if s_try else None,
                             "start_game": gc["drop_s"], "start_twin": float(np.mean(s_start)) if s_start else None,
                             "hp_game": g_hp, "men_game": g_men,
                             "hp_twin": float(np.mean(s_hp)) if s_hp else None,
                             "men_twin": float(np.mean(s_men)) if s_men else None,
                             "hit_game_s": g_hit, "hit_twin_s": float(np.mean([h for h in s_hit if h is not None]))
                             if any(h is not None for h in s_hit) else None, "copies": len(s_hp)})
    print(f"{'cast':<22}{'n':>2} {'try g/t':>13} {'HP game':>8} {'twin':>7} {'x':>5}  {'men g':>5} {'twin':>6} {'x':>5}"
          f"  {'1st hit g/t':>11}  {'cast g/t':>11}")
    for r in rows:
        hx = r["hp_twin"] / r["hp_game"] if r["hp_twin"] is not None and r["hp_game"] else None
        mx = r["men_twin"] / r["men_game"] if r["men_twin"] is not None and r["men_game"] else None
        r["hp_x"], r["men_x"] = hx, mx
        print(f"{r['kind']:<22}{r['n']:>2} {r['try_game']:>6.1f}/{(r['try_twin'] or 0):>6.1f} {r['hp_game']:>8.0f} "
              f"{(r['hp_twin'] or 0):>7.0f} {(hx or 0):>5.2f}  {r['men_game']:>5.0f} {(r['men_twin'] or 0):>6.1f} "
              f"{(mx or 0):>5.2f}  {str(r['hit_game_s']):>5}/{str(r['hit_twin_s']):>5}  "
              f"{str(r['start_game']):>5}/{str(r['start_twin']):>5}")
    summary = {}
    for spell in sorted({r["spell"] for r in rows}):
        sel = [r for r in rows if r["spell"] == spell and r["hp_x"]]
        first = [r for r in sel if r["n"] == 1]
        gm = lambda v: float(np.exp(np.mean(np.log(np.maximum(v, 1e-3))))) if len(v) else None
        summary[spell] = {"casts": len(sel), "hp_x": gm([r["hp_x"] for r in sel]),
                          "men_x": gm([r["men_x"] for r in sel if r["men_x"]]),
                          "hp_x_first": gm([r["hp_x"] for r in first]),
                          "men_x_first": gm([r["men_x"] for r in first if r["men_x"]]),
                          "hp_sum_x": sum(r["hp_twin"] for r in sel) / max(1e-9, sum(r["hp_game"] for r in sel)),
                          "men_sum_x": sum(r["men_twin"] for r in sel) / max(1e-9, sum(r["men_game"] for r in sel)),
                          # the spread of the casts' ratios about 1: the root mean square of their logarithm
                          "hp_rms": rms([r["hp_x"] for r in sel]), "men_rms": rms([r["men_x"] for r in sel if r["men_x"]])}
        s = summary[spell]
        print(f"{spell}: {s['casts']} casts, twin / game HP {s['hp_x']:.2f} (sum {s['hp_sum_x']:.2f}, first casts "
              f"{s['hp_x_first']:.2f}, rms log {s['hp_rms']:.2f}), men {s['men_x']:.2f} (sum {s['men_sum_x']:.2f}, "
              f"first {s['men_x_first']:.2f}, rms log {s['men_rms']:.2f})")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    return rows, summary


def _row_at(rows, t):
    """The sample row (rows [(t, row)]) nearest t."""
    return min(rows, key=lambda r: abs(r[0] - t))[1]


def friendly_cases(run_dir):
    """Battle 4's casts as the twin's starting points: [{kind, spell, try_s, caster, target, friend (rows at the try:
    x, z, b, men), widths, game: friend_after}] (the units where the game had them at the try)."""
    lanes = {ln["spec"]["name"]: ln for ln in cp.load_run(run_dir) if ln["spec"].get("extras")}
    out = []
    for c in measure(run_dir)["casts"]:
        ln = lanes.get(c["lane"])
        if ln is None or not ln.get("extras"):
            continue
        spec = ln["spec"]
        smp = [(s[0], s) for s in ln["samples"]]
        s = _row_at(smp, c["try_s"])
        e = _row_at(ln["extras"], c["try_s"])[0]
        out.append({"kind": c["kind"], "spell": SPELL[GOLD] if "_sd_" in c["kind"] else SPELL[WARLOCK],
                    "try_s": c["try_s"], "caster": (spec["a_key"], s[1], spec["a_width"]),
                    "target": (spec["t_key"], s[2], spec["t_width"]),
                    "friend": (cp.UNITS[spec["extras"][0]["short"]][0], e, spec["extras"][0].get("width") or cp.WIDTH_M),
                    "game": c["friends"][0], "target_game": c["windows"]})
    return out


def friendly_twin(cases, params, copies=8, device="cpu", strike=STRIKE_S):
    """The simulator on battle 4's casts: each case twice (with the cast at 0 s and without: a control), copies each;
    the caster, his target and his friend where the game had them at the try, held, fearless, 30 current winds. Per
    case {"hp", "men"}: what the cast took from the friend in the strike window (STRIKE_S after the order) - the cast's
    copies less the control's (the melee the boxes may start is the same in both)."""
    import torch
    from tools.nn.sim import battle, magic, orders as O, scenario as sim_scenario
    armies, meta = [], []
    for i, c in enumerate(cases):
        sides = {1: {"faction": cp.EMP, "ai": False, "units": []}, 2: {"faction": cp.SKV, "ai": False, "units": []}}
        for key, row, width in (c["caster"], c["friend"], c["target"]):
            fac = 1 if cp.FACTION[key] == cp.EMP else 2
            sides[fac]["units"].append({"key": key, "x": row["x"], "z": row["z"], "b": row["b"], "men": row["men"],
                                        "width": width, "general": False})
        for cast in (True, False):
            for _ in range(copies):
                armies.append({"attacker": 1, "sides": sides})
                meta.append((i, cast))
    st = sim_scenario.build(armies, params, device=device, per_side=2)
    B, N = st.B, st.N
    a_side = torch.tensor([0 if cp.FACTION[cases[i]["caster"][0]] == cp.EMP else 1 for i, _ in meta], device=device)
    a_slot, f_slot, t_slot = a_side * 2, a_side * 2 + 1, (1 - a_side) * 2
    b_idx = torch.arange(B, device=device)
    st.u["leadership"] = st.u["leadership"] + 1e4
    st.u["morale"] = st.u["morale"] + 1e4
    st.u["wm_cur"] = torch.where(st.u["side"] > 0, torch.full_like(st.u["wm_cur"], 30.0), st.u["wm_cur"])
    casting = torch.tensor([cast for _, cast in meta], device=device)
    asked = {"done": False}
    hp_f, men_f = [], []

    def policy(st):
        o = O.hold(B, N, device)
        slot = torch.full_like(o.kind, -1)
        target = torch.full_like(o.kind, -1)
        if not asked["done"]:
            slot[b_idx, a_slot] = torch.where(casting, torch.zeros_like(a_slot), torch.full_like(a_slot, -1))
            target[b_idx, a_slot] = t_slot
            asked["done"] = True
        magic.request(st.u, slot, target)
        return o

    def record(st):
        hp_f.append((float(st.t.max()), st.u["hp_abs"][b_idx, f_slot].tolist(), st.u["men"][b_idx, f_slot].tolist()))

    battle.run(st, policy, params, until_s=strike[1] + 1.0, record=record)
    t = np.array([r[0] for r in hp_f])
    hp = np.array([r[1] for r in hp_f])          # [T, B]
    men = np.array([r[2] for r in hp_f])
    i0, i1 = int(np.abs(t - strike[0]).argmin()), int(np.abs(t - strike[1]).argmin())
    lost_hp, lost_men = hp[i0] - hp[i1], men[i0] - men[i1]
    out = []
    for i in range(len(cases)):
        on = [b for b, (j, cast) in enumerate(meta) if j == i and cast]
        off = [b for b, (j, cast) in enumerate(meta) if j == i and not cast]
        out.append({"hp": float(lost_hp[on].mean() - lost_hp[off].mean()),
                    "men": float(lost_men[on].mean() - lost_men[off].mean()),
                    "hp_control": float(lost_hp[off].mean())})
    return out


def friendly_report(run_dirs, copies=8, device="cpu", params=None, out=None):
    """Battle 4: what each cast took from the caster's own unit behind its target, the game (less the melee's own rate)
    against the twin (less its control); per spell the sums."""
    params = sim_params(params, blows=False)
    cases = [c for r in run_dirs for c in friendly_cases(r)]
    if not cases:
        print("no battle-4 casts")
        return [], {}
    twin = friendly_twin(cases, params, copies=copies, device=device)
    print(f"{'cast':<22}{'try':>6} {'friend HP game':>14} {'(raw)':>7} {'twin':>7}  {'men game':>8} {'twin':>6}  fight")
    rows = []
    for c, tw in zip(cases, twin):
        g = c["game"]
        rows.append({"kind": c["kind"], "spell": c["spell"], "try_s": c["try_s"], "game_hp": g["hp_net"],
                     "game_hp_raw": g["hp"], "game_men": g["men_net"], "twin_hp": tw["hp"], "twin_men": tw["men"],
                     "twin_hp_control": tw["hp_control"], "fight": g["fight"]})
        print(f"{c['kind']:<22}{c['try_s']:>6.1f} {g['hp_net']:>14.0f} {g['hp']:>7.0f} {tw['hp']:>7.0f}  "
              f"{g['men_net']:>8.1f} {tw['men']:>6.1f}  {g['fight']}")
    summary = {}
    for spell in sorted({r["spell"] for r in rows}):
        sel = [r for r in rows if r["spell"] == spell]
        summary[spell] = {"casts": len(sel), "game_hp": sum(r["game_hp"] for r in sel),
                          "twin_hp": sum(r["twin_hp"] for r in sel), "game_men": sum(r["game_men"] for r in sel),
                          "twin_men": sum(r["twin_men"] for r in sel)}
        s = summary[spell]
        print(f"{spell}: {s['casts']} casts, friend HP game {s['game_hp']:.0f} twin {s['twin_hp']:.0f}, men game "
              f"{s['game_men']:.1f} twin {s['twin_men']:.1f}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    return rows, summary


def report(run_dirs, out=None):
    docs = [measure(r) for r in run_dirs]
    for d in docs:
        print(f"== {d['run']}")
        for side, s in d["sides"].items():
            print(f"  side {side}: readers {s['api']}, set {[(x['current'], x['reserve'], x['status']) for x in s['set']]}")
            print(f"    winds {s['winds']}")
            print(f"    drops {s['casts']}")
        for lane, m in d["mages"].items():
            print(f"  {lane} {m['u']}: active {m.get('active')} passive {m.get('passive')} magic {m.get('magic')} "
                  f"can {m.get('can')} cco {m.get('cco')}")
        for c in d["casts"]:
            w = " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in c["windows"].items())
            print(f"  {c['kind']:<20} try {c['try_s']:>6} drop {c['drop_s']:>6} cost {c['cost']:>5}  hp0 {c['hp0']} "
                  f"men0 {c['men0']}  lost hp/men {w}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    return docs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run folders (default: the charge probe's magic runs)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sim", action="store_true", help="the simulator's twin of the casts beside the game's")
    parser.add_argument("--friendly", action="store_true",
                        help="battle 4: the friendly hits, the game's against the twin's (friendly_report)")
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="the twin with sim.json magic.KEY = VALUE (JSON), e.g. --set wounds='\"even\"'")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    run_dirs = args.runs or [r for r in cp.runs() if json.loads((r / "manifest.json").read_text(encoding="utf-8"))
                             ["config"].get("plan") == "magic"]
    if args.sim or args.friendly:
        params = None
        if args.set:
            from tools.nn.sim.params import load
            params = load().with_cal("magic", **{k: json.loads(v) for k, v in (a.split("=", 1) for a in args.set)})
        if args.friendly:
            friendly_report(run_dirs, copies=args.copies, device=args.device, params=params, out=args.out)
        else:
            sim_report(run_dirs, copies=args.copies, device=args.device, params=params, out=args.out)
    else:
        report(run_dirs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
