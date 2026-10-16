"""Spells: the side's winds of magic, a cast at a unit and the bombardment it brings down (docs/en/training/simulator.md,
"Magic"). Every number of a spell is its passport (config/nn/abilities.json "spells", python -m tools.nn.abilities: the
game's database); config/nn/sim.json `magic` holds the rules the database has no column for, each with its source. The
magic probe (tools/nn/magic_probe.py, build/magic/probe1.md: 3 battles, 22 casts) measured what the database does not
say.

Winds (wm_cur, wm_res: the side's pool, the same on every unit of the side): a current pool the casts pay from and a
reserve that flows into it at magic.winds.flow_per_s (the database's _kv_winds_of_magic_params restored_points_base
0.1; the probe: +1 every 10 s, whatever the reserve, none once it is empty); a battle starts with
custom_battle_pool_initial_amount 80 x custom_battle_pool_initial_mana_available_ratio 0.15 = 12 current, 68 reserve
(the probe: 12 / 68 on both sides, 3 battles). No cap on the current pool was found (the probe: 42-43 and rising).

A cast (request(): a caster's spell slot at a unit): a spell ready (its cooldown over), its side's current winds at
least its cost and a target alive of the side the spell aims at (passport targets). Out of range (range_m, centre to
centre) the caster walks in at a run and casts once in range (the probe: ordered 260 m off, the Warlock Engineer ran
~65 m in ~16 s and cast ~205 m from the target's centre); in range he stands for wind_up_s (the cast's pose), the cost
leaving the current pool at its start. At the wind-up's end the spell's cooldown starts: active_s + recharge_s (the
probe: casts 40 s / 37 s apart = 1 + 8 + 30 / 1 + 5 + 30 and the sampling's second - the recharge runs from the end of
the effect). A cast that cannot start (no winds in range, the target gone, the caster routing) is dropped. While a
cast is pending or winding up the caster's own orders wait (the next cast request replaces a pending one).

A bombardment (passport bombardment): start_s after the wind-up the strike point is fixed at the target's centre then
(the probe: Searing Doom on clanrats walking 1.3 m/s - the dead lay where the unit's centre was 3 s after the order,
wind-up 1 + start 2); its `projectiles` land uniformly in time over window_s from flight_s later (their fall from
launch_height_m at muzzle_v, fixed_elevation down, under gravity: the first hit 5.96 / 5.17 s after the order, the probe
6.0 / 5.0-5.5 s), at a distance from the point uniform in [0, spread_m] at any angle (magic.landing "radius": the
landings gather at the centre - measured, the dead by their distance on the recorded men; "area": even over the
disc). Each strikes every unit whose men are there (friends too when the projectile can damage allies), by the density
of its formation (men / (front x depth), the simulator's own box, geometry.dims) - no fitted number:
    contact    the men within the projectile's collision_m + a man's radius (magic.contact_body) of where it lands,
               at most its penetration (projectile_penetration_junctions max_penetration): damage + ap each;
    explosion  the men within the detonation radius less magic.explosion_radius_cut_m (the CA bug report: "the actual
               effective value of the radius of ranged blast damage is 1 less than the value filled in") take the
               explosion's damage x (1 - (d / R)^p), p = magic.explosion_falloff_power (measured: no source names a
               falloff; a flat explosion gives 1.5x / 1.2x the game's HP on the probe's casts).
    knock      (magic.knock) a man knocked down takes nothing more of the bombardment until he is up: the projectile
               knocks the man it strikes when its mass x impact_v / his mass reaches the game's
               collision_speed_delta_threshold_knocked_down, the shockwave a man at d when force / his mass x (1 - d / R)
               reaches shockwave_speed_delta_threshold_knocked_down; he is down knockdown_getting_up_delay's mean.
               Each man's hits are cut to the share that finds him up (open_share: a dead-time counter over the
               bombardment's window); a hit that kills a whole man outright is not cut. The probe's battle 5: Warp
               Lightning strikes a lone man once a cast at most, Searing Doom a light Warlock Engineer 70, a Warlord 228.
By rings round the strike point (bombard(), profile()): a ring's men, their hits and explosion dose; the kills Poisson
a man with his own health, the wounds the bombardments left (bm_w) where the strike's dose falls (magic.wounds). Damage
per man: the armour roll on the base part (melee.per_hit, as missiles), the AP part whole; a magical projectile is cut
by spell resistance and ward only (resist_magic: the database's damage_mod_magic + damage_mod_all;
docs/en/game/mechanics/magic.md), shields do not block it (it falls from above). Its HP counts with the missiles' of
the step (battle.py: casualties, the fight's balance, under fire), its kills are its own.
The probe's twin: python -m tools.nn.magic_probe <runs> --sim.

Not modelled: overcasts and miscasts, spells other than bombardments, the cast in melee or on the move (unmeasured),
the game's AI casting (ai_like and the game-AI rule: next steps), a bombardment aimed at the ground.
Only json here and in keys() / slots_of(); the tensors need torch.
"""
import math

try:
    import torch
except ImportError:
    torch = None

SLOTS = 2            # spells a caster holds (sp0, sp1)
COLS = ("mana", "wind_up_s", "range_m", "active_s", "recharge_s", "initial_s", "t_enemies", "t_friends", "bomb", "n",
        "start_s", "window_s", "spread_m", "flight_s", "c_base", "c_ap", "c_radius_m", "c_pen", "e_base", "e_ap",
        "e_radius_m", "allies", "magical", "c_mom", "e_force")
COL = {c: i for i, c in enumerate(COLS)}


def keys(params):
    """Spell keys in table order (the passports, sorted)."""
    return sorted(params.spells or {})


def slots_of(params, unit_key):
    """Indices of the unit's spells (its passport's `spells`) in the table, SLOTS of them (-1: none)."""
    names = keys(params)
    own = [k for k in ((params.units.get(unit_key) or {}).get("spells") or ()) if k in names][:SLOTS]
    return [names.index(k) for k in own] + [-1] * (SLOTS - len(own))


def cal(params):
    return (params.sim.get("magic") or {}) if params.sim else {}


def on(params):
    """The spells are modelled (magic.on) and some unit of params casts (its passport's `spells`): without a caster
    among the units (the training pools today) the step skips magic altogether - nothing to cast, no cost."""
    if not (cal(params).get("on") and params.spells):
        return False
    return any((u or {}).get("spells") for u in (params.units or {}).values())


def row(params, key):
    """One spell's row of the table (COLS)."""
    p = params.spells[key]
    c = cal(params)
    b = p.get("bombardment") or {}
    pr = b.get("projectile") or {}
    ex = b.get("explosion") or {}
    t = p.get("targets") or {}
    r_e = max(0.0, float(ex.get("radius_m") or 0.0) - float(c.get("explosion_radius_cut_m", 0.0))) if ex else 0.0
    return [float(p["mana"]), float(p["wind_up_s"]), float(p["range_m"]), float(max(p["active_s"], 0.0)),
            float(max(p["recharge_s"], 0.0)), float(p.get("initial_s", 0.0)), float(bool(t.get("enemies"))),
            float(bool(t.get("friends"))), float(bool(b)), float(b.get("projectiles") or 0),
            float(b.get("start_s") or 0.0), float(b.get("window_s") or 0.0), float(b.get("spread_m") or 0.0),
            float(b.get("flight_s") or 0.0), float(pr.get("damage") or 0.0), float(pr.get("ap") or 0.0),
            float(pr.get("collision_m") or 0.0), float(pr.get("penetration") or 1), float(ex.get("damage") or 0.0),
            float(ex.get("ap") or 0.0), r_e, float(bool(pr.get("allies")) or bool(ex.get("allies"))),
            float(bool(pr.get("magical"))), float(pr.get("mass") or 0.0) * float(pr.get("impact_v") or 0.0),
            float(ex.get("force") or 0.0) if ex else 0.0]


def table(params, device):
    """[S + 1, COLS]: every spell; the last row (index -1) is no spell."""
    rows = [row(params, k) for k in keys(params)]
    return torch.tensor(rows + [[0.0] * len(COLS)], device=device)


def start(u, params):
    """At the battle's start (scenario.build): the sides' winds (magic.winds) and the spells' initial cooldowns. In place."""
    if not params.spells:
        return
    w = cal(params).get("winds") or {}
    present = u["side"] > 0
    u["wm_cur"] = torch.where(present, torch.full_like(u["wm_cur"], float(w.get("current", 0.0))), u["wm_cur"])
    u["wm_res"] = torch.where(present, torch.full_like(u["wm_res"], float(w.get("reserve", 0.0))), u["wm_res"])
    T = table(params, u["men"].device)
    for k in range(SLOTS):
        u[f"sp{k}_cd"] = T[u[f"sp{k}"], COL["initial_s"]].clone()


def leave_scripted(u, by_script):
    """The casters (a spell in a slot) of the sides a script plays leave the battle before it starts: no script casts
    (opponents.py), and a caster that never casts would be a dead unit of its army (the user, 16.10.2026: the scripts'
    armies get no caster; the network meets magic in self-play and past versions). by_script [B, 2] bool (side 1, side
    2; rollout.py by_rule). A caster that leaves is an empty slot (side 0, no men), as the slots no unit fills. Call it
    after every restart (the bank's rows bring the casters back). In place; a no-op without casters."""
    if "sp0" not in u:
        return
    caster = (u["sp0"] >= 0) | (u["sp1"] >= 0)
    if not bool(caster.any()):
        return
    side = u["side"]
    rule = by_script.to(torch.bool)
    gone = caster & (((side == 1) & rule[:, 0:1]) | ((side == 2) & rule[:, 1:2]))
    if not bool(gone.any()):
        return
    for name in ("side", "men", "men0", "hp", "hp_abs", "hp0"):
        if name in u:
            u[name] = torch.where(gone, torch.zeros_like(u[name]), u[name])
    for name in ("sp0", "sp1", "cast_k", "cq_k"):
        u[name] = torch.where(gone, torch.full_like(u[name], -1), u[name])
    u["lord"] = u["lord"] & ~gone


def request(u, slot, target):
    """Ask the units to cast: slot [B, N] - the spell slot (0..SLOTS-1, -1 none), target [B, N] - the unit to cast at
    (a slot of the state's layout). Taken by the next step (steer); in place."""
    ask = (slot >= 0) & (slot < SLOTS)
    u["cq_k"] = torch.where(ask, slot, u["cq_k"])
    u["cq_t"] = torch.where(ask, target, u["cq_t"])


def _side_sum(u, x):
    """Per unit the sum of x over its side's units ([B, N])."""
    side = u["side"]
    s1 = (x * (side == 1)).sum(1, keepdim=True)
    s2 = (x * (side == 2)).sum(1, keepdim=True)
    return torch.where(side == 1, s1, torch.where(side == 2, s2, torch.zeros_like(x)))


def _pick(u, name, k, suffix=""):
    """u[name{k}{suffix}] by the slot k [B, N] (0..SLOTS-1; the caller masks the units without one)."""
    out = u[f"{name}0{suffix}"].clone()
    for j in range(1, SLOTS):
        out = torch.where(k == j, u[f"{name}{j}{suffix}"], out)
    return out


def steer(u, params, dt, kind, tgt, run, standing, alive, same_side):
    """Before the movement (battle.py, after the orders): the winds flow, the cooldowns run, cast requests are taken,
    pending casts start in range, wind-ups end in the bombardment's launch. Returns (kind, tgt, run, move, mx, mz): the
    orders with the casters' own (MOVE at a run to the target while out of range - move, mx, mz: the point; HOLD while
    winding up)."""
    from tools.nn.sim import orders as O
    c = cal(params)
    w = c.get("winds") or {}
    T = table(params, u["men"].device)
    # the reserve flows into the current pool (the side's pool: the same on every unit of the side)
    flow = torch.minimum(torch.full_like(u["wm_res"], float(w.get("flow_per_s", 0.0)) * dt), u["wm_res"].clamp(min=0))
    u["wm_cur"] = u["wm_cur"] + flow
    u["wm_res"] = u["wm_res"] - flow
    if w.get("current_cap") is not None:
        u["wm_cur"] = u["wm_cur"].clamp(max=float(w["current_cap"]))
    for k in range(SLOTS):
        u[f"sp{k}_cd"] = (u[f"sp{k}_cd"] - dt).clamp(min=0)
        if f"sp{k}_on" in u:
            u[f"sp{k}_on"] = (u[f"sp{k}_on"] - dt).clamp(min=0)
    N = u["men"].shape[1]
    ar = torch.arange(N, device=u["men"].device)[None, :].expand_as(u["cq_k"])

    def target_ok(spell, t_slot):
        r = T[spell]
        ok_t = (t_slot >= 0) & alive.gather(1, t_slot.clamp(min=0))
        same = same_side.gather(2, t_slot.clamp(min=0)[:, :, None]).squeeze(2)
        return ok_t & torch.where(same, r[..., COL["t_friends"]] > 0, r[..., COL["t_enemies"]] > 0) & (t_slot != ar)

    # requests: a spell the caster holds, ready, its side's winds enough, a target of the right side alive
    req = u["cq_k"]
    sp = torch.where(req >= 0, _pick(u, "sp", req.clamp(min=0)), torch.full_like(req, -1))
    cd = _pick(u, "sp", req.clamp(min=0), "_cd")
    mana = T[sp, COL["mana"]]
    free = (u["cast_k"] < 0) | (u["cast_s"] < 0)            # not winding up (a pending cast is replaced)
    take = (req >= 0) & (sp >= 0) & standing & (cd <= 0) & free & (u["wm_cur"] >= mana) & target_ok(sp, u["cq_t"])
    u["cast_k"] = torch.where(take, req, u["cast_k"])
    u["cast_t"] = torch.where(take, u["cq_t"], u["cast_t"])
    u["cast_s"] = torch.where(take, torch.full_like(u["cast_s"], -1.0), u["cast_s"])
    u["cq_k"] = torch.full_like(u["cq_k"], -1)
    u["cq_t"] = torch.full_like(u["cq_t"], -1)
    # the casts going on
    k = u["cast_k"]
    casting = k >= 0
    sp = torch.where(casting, _pick(u, "sp", k.clamp(min=0)), torch.full_like(k, -1))
    r = T[sp]
    t_slot = u["cast_t"]
    keep = casting & standing & target_ok(sp, t_slot)
    tx, tz = u["x"].gather(1, t_slot.clamp(min=0)), u["z"].gather(1, t_slot.clamp(min=0))
    dist = torch.sqrt((tx - u["x"]) ** 2 + (tz - u["z"]) ** 2)
    pend = keep & (u["cast_s"] < 0)
    inside = dist <= r[..., COL["range_m"]]
    begin = pend & inside & (u["wm_cur"] >= r[..., COL["mana"]])
    drop = (casting & ~keep) | (pend & inside & ~begin)
    approach = pend & ~inside
    spent = torch.where(begin, r[..., COL["mana"]], torch.zeros_like(u["wm_cur"]))
    u["wm_cur"] = u["wm_cur"] - _side_sum(u, spent)
    u["cast_s"] = torch.where(begin, r[..., COL["wind_up_s"]], u["cast_s"])
    winding = keep & (u["cast_s"] >= 0)
    u["cast_s"] = torch.where(winding, u["cast_s"] - dt, u["cast_s"])
    done = winding & (u["cast_s"] <= 1e-6)
    for j in range(SLOTS):
        launch = done & (k == j)
        u[f"bm{j}_s"] = torch.where(launch & (r[..., COL["bomb"]] > 0), torch.zeros_like(u[f"bm{j}_s"]), u[f"bm{j}_s"])
        u[f"bm{j}_t"] = torch.where(launch, t_slot, u[f"bm{j}_t"])
        u[f"bm{j}_fix"] = torch.where(launch, torch.zeros_like(u[f"bm{j}_fix"]), u[f"bm{j}_fix"])
        u[f"sp{j}_cd"] = torch.where(launch, r[..., COL["active_s"]] + r[..., COL["recharge_s"]], u[f"sp{j}_cd"])
        if f"sp{j}_on" in u:
            u[f"sp{j}_on"] = torch.where(launch, r[..., COL["active_s"]], u[f"sp{j}_on"])
    end = drop | done
    u["cast_k"] = torch.where(end, torch.full_like(k, -1), u["cast_k"])
    u["cast_t"] = torch.where(end, torch.full_like(t_slot, -1), u["cast_t"])
    hold = winding
    kind = torch.where(hold, torch.full_like(kind, O.HOLD), torch.where(approach, torch.full_like(kind, O.MOVE), kind))
    tgt = torch.where(hold | approach, torch.full_like(tgt, -1), tgt)
    run = torch.where(approach, torch.ones_like(run), run)
    return kind, tgt, run, approach, tx, tz


def _overlap(c, half, l):
    """Length of [c - l, c + l] inside [-half, half]."""
    return (torch.minimum(c + l, half) - torch.maximum(c - l, -half)).clamp(min=0)


BAND_EDGES = (0.0, 0.06, 0.13, 0.21, 0.3, 0.42, 0.57, 0.76, 1.0)    # shares of the strike's reach (bands)
SMOOTH_M = 1.0       # the landings' density a man meets: averaged over a disc of this radius round him (a body's reach)
_PROFILES = {}


def profile(R, R_e, p, landing, device):
    """The strike's radial profile [K] by band (BAND_EDGES of R + max(R_e, SMOOTH_M)): (edges [K + 1], m, f - the
    landings a m^2 near a man of the band, per projectile; m1, m2 - the mean and the mean square of the explosion's
    falloff (1 - (d / R_e)^p) he gets, per projectile). Landing: "radius" - the radius uniform in [0, R] (the density
    1 / (2 pi R r), central; measured), "area" - uniform over the disc. Numerical (a polar grid of landings), cached."""
    key = (round(R, 4), round(R_e, 4), round(p, 4), landing, str(device))
    if key in _PROFILES:
        return _PROFILES[key]
    reach = R + max(R_e, SMOOTH_M)
    edges = torch.tensor(BAND_EDGES, dtype=torch.float64) * reach
    nr, nt = 240, 96
    q = (torch.arange(nr, dtype=torch.float64) + 0.5) / nr
    r = R * (q if landing == "radius" else torch.sqrt(q))
    th = (torch.arange(nt, dtype=torch.float64) + 0.5) / nt * 2 * math.pi
    lx = (r[:, None] * torch.cos(th)[None, :]).reshape(-1)
    lz = (r[:, None] * torch.sin(th)[None, :]).reshape(-1)
    f, m1, m2 = [], [], []
    for a, b in zip(edges[:-1].tolist(), edges[1:].tolist()):
        rho = torch.sqrt(a * a + (b * b - a * a) * (torch.arange(6, dtype=torch.float64) + 0.5) / 6)   # by area
        d = torch.sqrt((lx[None, :] - rho[:, None]) ** 2 + lz[None, :] ** 2)
        f.append(float((d < SMOOTH_M).double().mean()) / (math.pi * SMOOTH_M ** 2))
        k = ((1 - (d / R_e).clamp(max=1) ** p) * (d < R_e)) if R_e > 0 else torch.zeros_like(d)
        m1.append(float(k.mean()))
        m2.append(float((k * k).mean()))
    out = tuple(torch.tensor(v, dtype=torch.float32, device=device) for v in (edges.tolist(), f, m1, m2))
    _PROFILES[key] = out
    return out


def open_share(lam, window, t_down):
    """The share of a man's knocking hits (Poisson, lam of them over window s) that strike him when each knocks him
    down for t_down s (immune meanwhile): the registered count of a dead-time counter that starts open, E[N] =
    sum_k P(Poisson(lam / window x (window - (k - 1) t_down)) >= k) (the k-th hit no earlier than (k - 1) t_down after
    the first), over lam; 1 for lam -> 0; a window of 0: 1 - e^(-lam) of lam (one hit at most)."""
    lam = lam.clamp(min=0)
    if window <= 0:
        n = 1 - torch.exp(-lam)
    else:
        n = torch.zeros_like(lam)
        for k in range(1, int(window / max(t_down, 1e-6)) + 2):
            span = window - (k - 1) * t_down
            if span <= 0:
                break
            n = n + torch.special.gammainc(torch.full_like(lam, float(k)), lam * (span / window))
    return torch.where(lam > 1e-6, n / lam.clamp(min=1e-6), torch.ones_like(lam))


def _kills(H, A, hm, hpA, ph, rem, many):
    """(kills, HP) of H hits Poisson over the A men of a band with health hpA (rem a man): a hit of a man's health or more
    (K <= 1) kills the man it strikes, the hits on a man already struck lost - A (1 - e^(-H / A)) men, each his health
    left; smaller hits by melee.shot_kills, each at most the struck man's health left. A lone man (not many): the hits'
    mean, no kills here."""
    from tools.nn.sim import melee
    one = ph >= hm - 1e-6
    k_one = A * (1 - torch.exp(-H / A.clamp(min=1e-6)))
    k_many = torch.where(H > 0, melee.shot_kills(H, A, hm, hpA, ph), torch.zeros_like(H))
    k = torch.where(many, torch.where(one, k_one, k_many), torch.zeros_like(H))
    hp = torch.where(many, torch.where(one, k_one * rem, torch.minimum(H * torch.minimum(ph, rem), hpA)), H * ph)
    return k, hp


def strikes(u, params, dt):
    """The bombardments landing this step: [(k, rows, n)] - spell slot k, rows [M, 2] (battle, caster) with n [M]
    projectiles landing now; advances the bombardments' clocks and fixes their strike points. In place."""
    T = table(params, u["men"].device)
    out = []
    for k in range(SLOTS):
        s = u[f"bm{k}_s"]
        live = s >= 0
        if not bool(live.any()):
            # (new tensors as when it lands: the state's fields are replaced whatever the data, so a compiled step's
            # code after bombard() - battle.stepper - sees the same aliasing every step and is not recompiled)
            for f in ("s", "x", "z", "fix"):
                u[f"bm{k}_{f}"] = u[f"bm{k}_{f}"].clone()
            continue
        r = T[u[f"sp{k}"]]
        col = lambda name: r[..., COL[name]]
        # (a bombardment launched this step - steer set it to 0 at the wind-up's end, the end of this step - does not advance)
        s_new = torch.where(live & (s > 0), s + dt, torch.where(live, torch.full_like(s, 1e-6), s))
        # the strike point: the target's centre start_s after the wind-up (then fixed)
        t_slot = u[f"bm{k}_t"].clamp(min=0)
        fix = live & ~u[f"bm{k}_fix"] & (s_new >= col("start_s"))
        u[f"bm{k}_x"] = torch.where(fix, u["x"].gather(1, t_slot), u[f"bm{k}_x"])
        u[f"bm{k}_z"] = torch.where(fix, u["z"].gather(1, t_slot), u[f"bm{k}_z"])
        u[f"bm{k}_fix"] = u[f"bm{k}_fix"] | fix
        a0 = col("start_s") + col("flight_s")
        a1 = a0 + col("window_s")
        win = col("window_s")
        share = torch.where(win > 0, ((torch.minimum(s_new, a1) - torch.maximum(s, a0)).clamp(min=0)
                                      / win.clamp(min=1e-6)),
                            ((s < a0) & (s_new >= a0)).float())
        n_land = torch.where(live & u[f"bm{k}_fix"], col("n") * share, torch.zeros_like(share))     # [B, i]
        u[f"bm{k}_s"] = torch.where(live & (s_new > a1), torch.full_like(s, -1.0), s_new)
        rows = torch.nonzero(n_land > 0)
        if len(rows):
            out.append((k, rows, n_land[rows[:, 0], rows[:, 1]]))
    return out


def _eager(fn):
    """fn kept out of torch.compile (battle.stepper): its landings' shapes depend on the data (nonzero), and compiled it
    recompiled every step (1.3 s a step of 1024 battles instead of ~10 ms); eager it is a few small operations."""
    return torch.compiler.disable(fn) if torch is not None else fn


@_eager
def bombard(u, params, dt, front, depth, noise=False, extra=None):
    """The bombardments' damage this step: (hp, kills, hits) [B, i, j] - the HP caster i's strike takes from unit j, the
    men it kills, the hits it lands (contact and explosion); zeros when nothing lands. front, depth [B, N]: the
    formations' boxes (geometry.dims); noise: Poisson hits (the game's own randomness, noise.blows). In place (the
    bombardments' clocks).

    By bands round the strike point (BAND_EDGES; profile()): a band's men are the unit's density x its box's area in the
    band (a disc's area taken as a square's); each man of the band gets the band's contact hits - the landings near him
    x the men a projectile strikes (collision_m + his radius, at most its penetration) - and its explosion dose. An
    explosion's damage varies with the distance: Poisson hits of one size with the same mean and mean square (size
    e_ap x m2 / m1, m1^2 / m2 of them a landing), so the kills see its spread, not only its mean. Each man has his own
    health (CA): the band's men carry the unit's wounds by their share of its men (magic.wounds), Poisson a man
    (melee.shot_kills, the contact's hits first, then the explosion's); a hit takes at most the struck man's health
    left (overkill, CA): a bolt that kills a wounded man takes his health, not its damage."""
    from tools.nn.sim import melee
    B, N = u["men"].shape
    dev = u["men"].device
    hp = torch.zeros((B, N, N), device=dev)
    kills, hits = torch.zeros_like(hp), torch.zeros_like(hp)
    knocks = torch.zeros_like(hp)
    if extra is not None:
        extra["knock"] = knocks
    wounded = torch.zeros((B, N), device=dev)
    landing = strikes(u, params, dt)
    if not landing:
        if "bm_w" in u:
            u["bm_w"] = u["bm_w"].clone()          # (replaced every step: strikes())
        return hp, kills, hits
    c = cal(params)
    T = table(params, dev)
    p = float(c.get("explosion_falloff_power", 1.0))
    body = float(c.get("contact_body", 1.0))
    mode = str(c.get("landing", "radius"))
    knock = bool(c.get("knock"))
    if knock:
        rules = params.battle
        thr_c = float(rules["collision_speed_delta_threshold_knocked_down"])
        thr_s = float(rules["shockwave_speed_delta_threshold_knocked_down"])
        t_down = (float(rules["knockdown_getting_up_delay_min"]) + float(rules["knockdown_getting_up_delay_max"])) / 2
    present = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    men = u["men"].clamp(min=0)
    rho = torch.where(present, men / (front * depth).clamp(min=1e-3), torch.zeros_like(men))
    hm = u["hp_man"].clamp(min=1e-6)
    W = (men * hm - u["hp_abs"]).clamp(min=0)
    many = u["men0"] > 1
    rad = torch.deg2rad(u["b"])
    for k, rows, n in landing:
        b_, i_ = rows[:, 0], rows[:, 1]
        spell = u[f"sp{k}"][b_, i_]                                         # [M]
        for s_idx in torch.unique(spell).tolist():
            sel = spell == s_idx
            bb, ii, nn = b_[sel], i_[sel], n[sel]
            r = T[s_idx]
            col = lambda name: float(r[COL[name]])
            edges, f, m1, m2 = profile(col("spread_m"), col("e_radius_m"), p, mode, dev)
            # j's box seen from the strike point
            dx = u[f"bm{k}_x"][bb, ii][:, None] - u["x"][bb]               # [M, N]
            dz = u[f"bm{k}_z"][bb, ii][:, None] - u["z"][bb]
            sb, cb = torch.sin(rad[bb]), torch.cos(rad[bb])
            along, across = dx * sb + dz * cb, dx * cb - dz * sb
            hf, hd = front[bb] / 2, depth[bb] / 2
            half = math.sqrt(math.pi) / 2 * edges                          # [K + 1]
            inside = (_overlap(across[..., None], hf[..., None], half) * _overlap(along[..., None], hd[..., None], half))
            band_men = (rho[bb][..., None] * (inside[..., 1:] - inside[..., :-1])).clamp(min=0)     # [M, N, K]
            # whom it strikes: units alive; the caster's friends only when it can damage allies
            friend = u["side"][bb] == u["side"][bb, ii][:, None]
            whom = present[bb] & (~friend | (col("allies") > 0))
            band_men = band_men * whom[..., None]
            # contact: the landings near a man x the men a projectile strikes there, a man's share of them
            r_c = col("c_radius_m") + body * u["radius"][bb]                # [M, N]
            per_landing = rho[bb] * math.pi * r_c ** 2
            cap = torch.where(per_landing > col("c_pen"), col("c_pen") / per_landing.clamp(min=1e-9),
                              torch.ones_like(per_landing))
            lam_c = nn[:, None, None] * f * (math.pi * r_c ** 2 * cap)[..., None]               # a man, [M, N, K]
            lam_c = lam_c * float(col("c_base") + col("c_ap") > 0)
            # explosion: equal hits of the band's size
            ok = (m1 > 0) & (col("e_base") + col("e_ap") > 0)
            size = torch.where(ok, m2 / m1.clamp(min=1e-12), torch.zeros_like(m1))              # [K]
            lam_e = torch.where(ok, nn[:, None, None] * (m1 * m1 / m2.clamp(min=1e-12)), torch.zeros_like(m1))
            lam_e = lam_e.expand_as(lam_c)
            res = (u["resist_magic"] if col("magical") > 0
                   else (u["resist_missile"] + u["resist_physical"]).clamp(max=0.9))[bb][..., None]
            arm, hmj = u["armour"][bb][..., None], hm[bb][..., None]
            single = ~many[bb][..., None]
            ph_c = melee.per_hit(torch.full_like(hmj, col("c_base")), torch.full_like(hmj, col("c_ap")), arm, hmj, res,
                                 single=single)                              # [M, N, 1]
            ph_e = melee.per_hit(col("e_base") * size, col("e_ap") * size, arm, hmj, res, single=single)   # [M, N, K]
            if knock:
                # the knock (magic.knock): a man the projectile or its shockwave knocks down takes nothing more of the
                # bombardment until he is up - his share of its hits is his open time's (open_share); a hit that kills
                # a whole man outright is not cut (the knocked man is dead already)
                mass = u["mass"][bb].clamp(min=1.0)                                             # [M, N]
                k_c = (col("c_mom") / mass >= thr_c).float()
                r_k = (col("e_radius_m") * (1 - thr_s * mass / max(col("e_force"), 1e-6))).clamp(min=0)
                lam_k = col("n") * f * (math.pi * (r_c ** 2 * cap * k_c + r_k ** 2 * float(col("e_force") > 0)))[..., None]
                op = open_share(lam_k, col("window_s"), t_down)
                # (bookkeeping) the knock events this step: the landings now / all of them x the knocking hits that
                # find a man up, by the band's men
                knocks.index_put_((bb, ii), (lam_k * op * band_men * (nn / max(col("n"), 1e-6))[:, None, None]
                                             ).sum(-1), accumulate=True)
                lam_c = lam_c * torch.where(ph_c >= hmj - 1e-6, torch.ones_like(op), op)
                lam_e = lam_e * torch.where(ph_e >= hmj - 1e-6, torch.ones_like(op), op)
            Hc, He = lam_c * band_men, lam_e * band_men
            if noise:
                Hc, He = torch.poisson(Hc), torch.poisson(He)
            # the band's men and their health: the wounds bombardments left (bm_w) lie where this strike's dose falls
            # (a bombardment strikes its target's centre, as the last one did), the unit's other wounds by the band's
            # share of its men
            A = torch.minimum(band_men, men[bb][..., None])
            strike_w = "bm_w" in u and str(c.get("wounds", "strike")) == "strike"
            w_b = torch.minimum(u["bm_w"][bb], W[bb])[..., None] if strike_w else torch.zeros_like(A[..., :1])
            dose = A * (lam_c * ph_c + lam_e * ph_e)
            at = dose / dose.sum(-1, keepdim=True).clamp(min=1e-9)
            Wk = (W[bb][..., None] - w_b) * A / men[bb][..., None].clamp(min=1e-6) + w_b * at
            hpA = (A * hmj - Wk).clamp(min=0)
            m_ = many[bb][..., None]
            rem = hpA / A.clamp(min=1e-6)
            kc, hpc = _kills(Hc, A, hmj, hpA, ph_c.expand_as(Hc), rem, m_)
            A2, hpA2 = (A - kc).clamp(min=0), (hpA - hpc).clamp(min=0)
            rem2 = hpA2 / A2.clamp(min=1e-6)
            ke, hpe = _kills(He, A2, hmj, hpA2, ph_e, rem2, m_)
            hp.index_put_((bb, ii), (hpc + hpe).sum(-1), accumulate=True)
            kills.index_put_((bb, ii), (kc + ke).sum(-1), accumulate=True)
            hits.index_put_((bb, ii), (Hc + He).sum(-1), accumulate=True)
            # the wounds it leaves: its HP less the health the killed men had
            wounded.index_put_((bb,), ((hpc - kc * rem).clamp(min=0) + (hpe - ke * rem2).clamp(min=0)).sum(-1),
                               accumulate=True)
    if "bm_w" in u:
        u["bm_w"] = torch.minimum(u["bm_w"], W) + wounded
    return hp, kills, hits
