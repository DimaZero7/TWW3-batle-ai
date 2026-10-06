"""What happened in one battle, from its recording (docs/en/training/workflow.md "Game-vs-sim gap card").

The same numbers for a battle of the game (build/nn-arena/runs/<time>/, tools/nn/gamedata.py) and for
a simulated one recorded the same way (tools/nn/sim/check.py Recorder -> gamedata.Battle): the gap
card (tools/ops/gapcard.py) sets them side by side, the gate summary (tools/nn/gate.py --profile)
shows the game's. Side 1 is our network's army, side 2 the game's AI (or ai_like in the simulator).
Everything but the result and the battle length is measured up to `cut_s` (the gap card: the game's
end time, so a longer simulated battle is judged on the same window). Groups (metric profiles,
tools/nn/train/profiles.py GAME):

    mandatory  win (1 / 0), trade = (enemy gold lost - own gold lost) / budget (budget = the two armies'
               cost / 2), own / enemy gold lost as a share of that army's cost (gold as the gate counts
               it: dead, shattered or gone whole, routing half of what is left), own lord dead
               (hp or men 0), battle length s; own gold lost and the trade 30 s before the window's end
               (E30_S: the game's last seconds, a pursuit of a broken army, apart); own army destroyed by
               the window's end (the database's army destruction rule on the recording: own strength <= 0.22
               of its start and the enemy's >= 2.6 times ours, strength as the simulator's
               morale.army_collapse: config/nn/sim.json morale.collapse)
    routs      rout onsets (r false -> true) per unit, own / enemy
    lords      lord HP lost per second of that lord in melee: ours ("lost") and the enemy's ("dealt" by
               our army); enemy lord dead; our lord's ability uses (count, first use s)
    fatigue    the share of the units on the field (men > 0, not shattered) tired or worse (tired,
               very tired, exhausted) in the time bins 0-120 / 120-240 / 240-360 / 360+ s, own / enemy;
               exhausted over the whole window; the share of our units' time under a move / withdraw
               order far from the fight (before the battle's first melee and no standing enemy within
               FAR_M = 150 m of the unit, about a bow's range) that runs (the orders in force: the
               bridge's given nn_orders in the game, the simulator's order_kind / order_run)
    activity   the share of standing unit-time (alive, not routing) in melee / firing / moving / still
    shooters   missile units' (passport ammo > 0, not the lord) standing time in melee, own / enemy
    wrap       melee entries (a melee unit, not the lord nor a missile unit, going into melee) from the rear
               (the angle between the enemy's facing and the enemy -> unit line >= REAR_DEG; the enemy: its
               target when that one is in melee, else the nearest standing enemy), own / enemy; the share of the
               melee time of the side's units (not the lord) with an enemy behind (the rear flag bf), own / enemy;
               how far the enemy's melee line laps ours at the first melee, both ends summed, m (across the
               line between the two melee lines' centres)

Only numpy: the host runs it (gate.py).
"""
import json
from pathlib import Path

import numpy as np

from tools.nn.model import passport

ROUT_SHARE = 0.5          # = tools/nn/train/reward.py Weights.rout_share (gate.py ROUT_SHARE)
BINS = ((0, 120), (120, 240), (240, 360), (360, None))
TIRED = 3                 # gamedata.FATIGUE_LEVELS index of threshold_tired
EXHAUSTED = 5
MAX_DT = 2.0              # s: a longer gap between two samples counts as this (as gate.liveliness)
SIDES = (("own", 1), ("enemy", 2))
FAR_M = 150.0             # = tools/nn/train/behaviour.py FAR_M
POINT_KINDS = ("move", "withdraw")
REAR_DEG = 120.0          # = the simulator's rear sector (config/nn/sim.json threat), build/gangup's rear entry
E30_S = 30.0              # the own loss and the trade also this long before the window's end
CONFIG = Path(__file__).resolve().parents[2] / "config" / "nn"


def _bin_name(lo, hi):
    return f"{lo}-{hi}" if hi is not None else f"{lo}+"


# (key, group, title, format); the group is the metric profile that computes it.
ROWS = [("win", "mandatory", "win (1 / 0)", "{:.2f}"),
        ("trade", "mandatory", "trade (enemy - own gold lost) / budget", "{:+.3f}"),
        ("gold_own", "mandatory", "own gold lost / own army", "{:.3f}"),
        ("gold_enemy", "mandatory", "enemy gold lost / enemy army", "{:.3f}"),
        ("gold_own_e30", "mandatory", f"own gold lost / own army, {E30_S:g} s before the end", "{:.3f}"),
        ("trade_e30", "mandatory", f"trade, {E30_S:g} s before the end", "{:+.3f}"),
        ("army_beaten_own", "mandatory", "own army destroyed by the end (army destruction rule)", "{:.2f}"),
        ("lord_dead_own", "mandatory", "own lord dead", "{:.2f}"),
        ("length_s", "mandatory", "battle length, s", "{:.0f}"),
        ("routs_own", "routs", "rout onsets per unit, own", "{:.2f}"),
        ("routs_enemy", "routs", "rout onsets per unit, enemy", "{:.2f}"),
        ("lord_lost_own", "lords", "own lord HP lost / its melee s", "{:.4f}"),
        ("lord_lost_enemy", "lords", "enemy lord HP lost / its melee s (dealt)", "{:.4f}"),
        ("lord_dead_enemy", "lords", "enemy lord dead", "{:.2f}"),
        ("abilities_own", "lords", "own lord ability uses", "{:.2f}"),
        ("ability_first_s", "lords", "own lord first ability use, s", "{:.0f}")]
ROWS += [(f"tired_{s}_{_bin_name(lo, hi)}", "fatigue", f"tired or worse, {s}, {_bin_name(lo, hi)} s", "{:.3f}")
         for s, _ in SIDES for lo, hi in BINS]
ROWS += [(f"exhausted_{s}", "fatigue", f"exhausted, {s}, whole window", "{:.3f}") for s, _ in SIDES]
ROWS += [("run_far_own", "fatigue", f"own move orders running, far from the fight (no enemy within {FAR_M:g} m)",
          "{:.3f}")]
ROWS += [(f"{a}_{s}", "activity", f"standing time {a}, {s}", "{:.3f}")
         for s, _ in SIDES for a in ("melee", "fire", "move", "still")]
ROWS += [(f"missile_melee_{s}", "shooters", f"missile units' time in melee, {s}", "{:.3f}") for s, _ in SIDES]
ROWS += [(f"rear_entries_{s}", "wrap", f"melee entries from the rear, {s}", "{:.3f}") for s, _ in SIDES]
ROWS += [(f"behind_{s}", "wrap", f"melee time with an enemy behind, {s}", "{:.3f}") for s, _ in SIDES]
ROWS += [("lap_enemy", "wrap", "enemy line beyond ours at the first melee, m", "{:.0f}")]
GROUPS = tuple(dict.fromkeys(g for _, g, _, _ in ROWS))


def rows_for(chosen):
    """ROWS of the mandatory group and the chosen profiles."""
    return [r for r in ROWS if r[1] == "mandatory" or r[1] in chosen]


def abilities_used(events_path, unit=None):
    """[(t s, unit name)] of the bridge's nn_ability events with status "used" (our network's orders;
    the game's AI gives none), only `unit`'s when given."""
    out = []
    try:
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                if '"event":"nn_ability"' in line:
                    ev = json.loads(line)
                    if ev.get("status") == "used" and (unit is None or ev.get("u") == unit):
                        out.append((ev.get("t", 0) / 1000, ev.get("u")))
    except OSError:
        pass
    return out


def orders_in_force(events_path, names, t):
    """{"move": [T, N], "run": [T, N]} bools: our network's order in force at each sample time t (the
    bridge's nn_orders with status "given": a move / withdraw order, its run flag); None without events."""
    index = {n: i for i, n in enumerate(names)}
    given = []
    try:
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                if '"event":"nn_orders"' in line:
                    ev = json.loads(line)
                    for o in ev.get("orders") or []:
                        if o.get("status") == "given" and o.get("u") in index:
                            given.append((ev.get("t", 0) / 1000, index[o["u"]], o.get("k") in POINT_KINDS,
                                          bool(o.get("run"))))
    except OSError:
        return None
    T, N = len(t), len(names)
    move, run = np.zeros((T, N), dtype=bool), np.zeros((T, N), dtype=bool)
    cur_m, cur_r = np.zeros(N, dtype=bool), np.zeros(N, dtype=bool)
    j = 0
    given.sort(key=lambda g: g[0])
    for i, ti in enumerate(t):
        while j < len(given) and given[j][0] <= ti + 1e-6:
            _, u, mv, r = given[j]
            cur_m[u], cur_r[u] = mv, r
            j += 1
        move[i], run[i] = cur_m, cur_r
    return {"move": move, "run": run}


def far_from_fight(b, stand):
    """[T, N] units far from the fight: no melee in the battle yet and no standing enemy within FAR_M
    (tools/nn/train/behaviour.py far_from_fight, on the samples)."""
    x, z = np.nan_to_num(b.f["x"]), np.nan_to_num(b.f["z"])
    side = np.asarray(b.side)
    d2 = (x[:, :, None] - x[:, None, :]) ** 2 + (z[:, :, None] - z[:, None, :]) ** 2
    enemy = (side[:, None] != side[None, :])[None] & stand[:, None, :]
    near = (enemy & (d2 < FAR_M * FAR_M)).any(2)
    melee = (b.f["m"].astype(bool) & stand).any(1)
    before = np.cumsum(melee) == 0
    return ~near & before[:, None]


def _steps(t):
    """[T] seconds each sample stands for (to the next one, at most MAX_DT; the last 0)."""
    return np.clip(np.diff(t, append=t[-1] if len(t) else 0.0), 0.0, MAX_DT)


def _share(num, den):
    den = float(den)
    return float(num) / den if den > 0 else None


def gold_lost(f, i, cost):
    """[N] gold lost per unit at sample i: dead, shattered or gone (men 0 / missing) whole, else the
    health lost plus ROUT_SHARE of what is left when routing (gate.gold, tools/nn/train/reward.py)."""
    men, hp = f["men"][i], f["hp"][i]
    gone = ~(np.nan_to_num(men, nan=0.0) > 0) | f["s"][i].astype(bool)
    h = np.clip(np.nan_to_num(hp, nan=0.0), 0.0, 1.0)
    share = np.where(gone, 1.0, (1 - h) + h * ROUT_SHARE * f["r"][i].astype(bool))
    return share * cost


def collapse_rules(sim=None, rules=None):
    """(config/nn/sim.json morale.collapse, (alliance ratio 0.22, enemy ratio 2.6) from game_rules.json morale)."""
    sim = sim if sim is not None else json.loads((CONFIG / "sim.json").read_text(encoding="utf-8"))["morale"]["collapse"]
    rules = rules if rules is not None else json.loads((CONFIG / "game_rules.json").read_text(encoding="utf-8"))["morale"]
    return sim, (float(rules["army_destruction_alliance_strength_ratio"]),
                 float(rules["army_destruction_enemy_strength_ratio"]))


def beaten(b, passports=None, cal=None, ratios=None):
    """[T, 2] bools: side 1 / side 2 meets the database's army destruction rule at each sample (own strength
    <= ratios[0] of its start, the enemy's >= ratios[1] times its own); strength as the simulator's
    morale.army_collapse (tools/nn/sim/morale.py): strength "cp" - the database's combat potential (fixed + missile
    x the ammunition curve) x health x routing_weight when routing, units alive and not shattered; the old
    "strategic" - cost (the lord x lord_value_scale) x health x the missile units' ammunition weight."""
    if cal is None or ratios is None:
        c, r = collapse_rules()
        cal, ratios = (c if cal is None else cal), (r if ratios is None else ratios)
    passports = passports or passport.load()
    f = b.f
    keys = list(b.keys)
    side = np.asarray(b.side)
    lord = passport.lords(keys, passports)
    direct = np.array([bool(((passports[k].get("missile") or {}).get("direct"))) if k else False for k in keys])
    value = passport.cost(keys, passports).astype(float) * np.where(lord, float(cal.get("lord_value_scale", 1.0)), 1.0)
    men = np.nan_to_num(np.asarray(f["men"], dtype=float), nan=0.0)
    alive = (men > 0) & ~f["s"].astype(bool)
    hp = np.clip(np.nan_to_num(np.asarray(f["hp"], dtype=float), nan=0.0), 0.0, 1.0) ** float(cal.get("hp_power", 1.0))
    a = np.asarray(f["a"], dtype=float)
    first = np.where(np.isfinite(a).any(0), a[np.isfinite(a).argmax(0), np.arange(a.shape[1])], 0.0)
    a0 = np.nan_to_num(first, nan=0.0)
    weight = np.ones_like(hp)
    init_value = value
    if cal.get("strength") == "cp":
        fixed, missile = passport.combat_potential(keys, passports)
        sl, mid = float(cal["ammo_slope"]), float(cal["ammo_midpoint"])
        amm = np.clip(np.nan_to_num(a, nan=0.0) / np.maximum(a0, 1.0), 0.0, 1.0)
        value = fixed[None, :] + missile[None, :] / (1 + np.exp(-sl * (amm - mid)))
        init_value = fixed + missile / (1 + np.exp(-sl * (1 - mid)))
        weight = np.where(f["r"].astype(bool), float(cal["routing_weight"]), 1.0)
    elif cal.get("strength") == "strategic":
        amm = np.clip(np.nan_to_num(a, nan=0.0) / np.maximum(a0, 1.0), 0.0, 1.0)
        sl, mid = float(cal["ammo_slope"]), float(cal["ammo_midpoint"])
        lo, hi = 1 / (1 + np.exp(sl * mid)), 1 / (1 + np.exp(-sl * (1 - mid)))
        avail = (1 / (1 + np.exp(-sl * (amm - mid))) - lo) / (hi - lo)
        floor = np.where(direct, float(cal["ammo_empty_direct"]), float(cal["ammo_empty"]))
        weight = np.where((a0 > 0)[None, :], floor + (1 - floor) * avail, 1.0)
        weight = weight * np.where(f["r"].astype(bool), float(cal["routing_weight"]), 1.0)
    count = alive & ~f["r"].astype(bool) if cal.get("count") == "standing" else alive
    power = (value if value.ndim == 2 else value[None, :]) * hp * weight * count
    st = np.stack([power[:, side == 1].sum(1), power[:, side == 2].sum(1)], 1)
    init = np.array([init_value[side == 1].sum(), init_value[side == 2].sum()])
    other = st[:, ::-1]
    return (init[None, :] > 0) & (other > 0) & (other >= ratios[1] * st) & (st <= ratios[0] * init[None, :])


def measure(b, winner, cut_s=None, abilities=None, chosen=GROUPS, passports=None, orders=None):
    """{key: value} (ROWS of the mandatory group and `chosen`) of one battle b (gamedata.Battle; side 1
    ours). winner: 1 ours, 2 theirs, else no result (win None); cut_s: the window's end (default the
    battle's); abilities: [t s] of our lord's ability uses (abilities_used); orders: our orders in force
    per sample (orders_in_force) or None."""
    t = np.asarray(b.t, dtype=float)
    f = b.f
    side = np.asarray(b.side)
    keys = list(b.keys)
    cost = passport.cost(keys, passports).astype(float)
    lord = passport.lords(keys, passports)
    ammo = passport.ammo(keys, passports) > 0
    end = float(t[-1]) if len(t) else 0.0
    cut = end if cut_s is None else min(float(cut_s), end)
    w = t <= cut + 1e-6
    i_cut = int(np.nonzero(w)[0][-1]) if w.any() else 0
    dt = _steps(t) * w
    men = np.nan_to_num(f["men"], nan=0.0)
    field = (men > 0) & ~f["s"].astype(bool)                       # on the field (routing ones too)
    stand = field & ~f["r"].astype(bool)
    out = {}
    lost = gold_lost(f, i_cut, cost)
    start = {s: float(cost[side == n].sum()) for s, n in SIDES}
    budget = (start["own"] + start["enemy"]) / 2
    out["win"] = None if winner not in (1, 2) else float(winner == 1)
    out["trade"] = _share(lost[side == 2].sum() - lost[side == 1].sum(), budget)
    out["gold_own"] = _share(lost[side == 1].sum(), start["own"])
    out["gold_enemy"] = _share(lost[side == 2].sum(), start["enemy"])
    w30 = t <= cut - E30_S + 1e-6
    if w30.any():
        lost30 = gold_lost(f, int(np.nonzero(w30)[0][-1]), cost)
        out["gold_own_e30"] = _share(lost30[side == 1].sum(), start["own"])
        out["trade_e30"] = _share(lost30[side == 2].sum() - lost30[side == 1].sum(), budget)
    else:
        out["gold_own_e30"] = out["trade_e30"] = None
    out["army_beaten_own"] = float(beaten(b, passports)[:i_cut + 1, 0].any()) if len(t) else None
    dead = (np.nan_to_num(f["hp"], nan=1.0) <= 0) | ~(men > 0) | f["s"].astype(bool)     # as st.lord_dead_s
    for s, n in SIDES[:2 if "lords" in chosen else 1]:          # the enemy's: the lords group
        j = np.nonzero(lord & (side == n))[0]
        out[f"lord_dead_{s}"] = float(dead[:i_cut + 1, j].any()) if len(j) else None
    out["length_s"] = end
    if "routs" in chosen:
        onset = f["r"][1:].astype(bool) & ~f["r"][:-1].astype(bool) & w[1:, None]
        for s, n in SIDES:
            out[f"routs_{s}"] = _share(onset[:, side == n].sum(), (side == n).sum())
    if "lords" in chosen:
        for s, n in SIDES:
            j = np.nonzero(lord & (side == n))[0]
            if not len(j):
                out[f"lord_lost_{s}"] = None
                continue
            hp = f["hp"][:, j[0]]
            drop = np.nan_to_num(np.clip(hp[:-1] - hp[1:], 0.0, None), nan=0.0) * w[1:]
            melee_s = float((dt * (f["m"][:, j[0]].astype(bool) & stand[:, j[0]])).sum())
            out[f"lord_lost_{s}"] = _share(drop.sum(), melee_s)
        uses = sorted(x for x in (abilities or []) if x <= cut + 1e-6)
        out["abilities_own"] = float(len(uses))
        out["ability_first_s"] = float(uses[0]) if uses else None
    if "fatigue" in chosen:
        fat = f.get("fat")
        known = field & np.isfinite(fat) if fat is not None else np.zeros_like(field)
        level = np.nan_to_num(fat, nan=0.0) if fat is not None else np.zeros(field.shape)
        for s, n in SIDES:
            m = known[:, side == n]
            lv = level[:, side == n]
            for lo, hi in BINS:
                tb = ((t >= lo) & (t < hi if hi is not None else True))[:, None] & w[:, None]
                out[f"tired_{s}_{_bin_name(lo, hi)}"] = _share((m & tb & (lv >= TIRED)).sum(), (m & tb).sum())
            out[f"exhausted_{s}"] = _share((m & w[:, None] & (lv >= EXHAUSTED)).sum(), (m & w[:, None]).sum())
        out["run_far_own"] = None
        if orders is not None:
            far = far_from_fight(b, stand) & stand & (side == 1)[None] & orders["move"]
            wt = far * dt[:, None]
            out["run_far_own"] = _share((wt * orders["run"]).sum(), wt.sum())
    if "activity" in chosen:
        melee = f["m"].astype(bool)
        fire = f["fire"].astype(bool) & ~melee
        move = f["mv"].astype(bool) & ~melee & ~fire
        for s, n in SIDES:
            st = stand[:, side == n] * dt[:, None]
            den = st.sum()
            for name, x in (("melee", melee), ("fire", fire), ("move", move), ("still", ~(melee | fire | move))):
                out[f"{name}_{s}"] = _share((st * x[:, side == n]).sum(), den)
    if "shooters" in chosen:
        shooter = ammo & ~lord
        for s, n in SIDES:
            m = shooter & (side == n)
            st = stand[:, m] * dt[:, None]
            out[f"missile_melee_{s}"] = _share((st * f["m"][:, m]).sum(), st.sum())
    if "wrap" in chosen:
        out.update(wrap(b, w, stand, lord, ammo, dt))
    return out


def wrap(b, w, stand, lord, ammo, dt):
    """The wrap group of measure() (module doc): {rear_entries_own/enemy, behind_own/enemy, lap_enemy}."""
    f = b.f
    side = np.asarray(b.side)
    T, N = f["m"].shape
    x, z = np.nan_to_num(f["x"]), np.nan_to_num(f["z"])
    m = f["m"].astype(bool) & stand
    body = stand & ~lord[None] & ~ammo[None]
    out = {}
    tg = b.target if b.target is not None else np.full((T, N), -1)
    jj = np.clip(tg, 0, N - 1)
    take = lambda a: np.take_along_axis(a, jj, axis=1)
    foe = (side[:, None] != side[None, :])[None] & stand[:, None, :]
    d2 = (x[:, :, None] - x[:, None, :]) ** 2 + (z[:, :, None] - z[:, None, :]) ** 2
    ok = (tg >= 0) & take(m) & (side[jj] != side[None])
    ej = np.where(ok, jj, np.where(foe, d2, np.inf).argmin(2))
    ex, ez = np.take_along_axis(x, ej, axis=1), np.take_along_axis(z, ej, axis=1)
    eb = np.take_along_axis(np.nan_to_num(f["b"]), ej, axis=1)
    ang = np.abs((np.degrees(np.arctan2(x - ex, z - ez)) - eb + 180.0) % 360.0 - 180.0)
    entry = m & ~np.vstack([np.zeros((1, N), bool), m[:-1]]) & body & w[:, None] & foe.any(2)
    in_melee = m & ~lord[None]
    bf = f["bf"].astype(bool)
    for s, n in SIDES:
        e = entry[:, side == n]
        out[f"rear_entries_{s}"] = _share((e & (ang[:, side == n] >= REAR_DEG)).sum(), e.sum())
        wt = in_melee[:, side == n] * dt[:, None]
        out[f"behind_{s}"] = _share((wt * bf[:, side == n]).sum(), wt.sum())
    out["lap_enemy"] = None
    first = np.flatnonzero((m & w[:, None]).any(1))
    if len(first):
        i = first[0]
        k1, k2 = body[i] & (side == 1), body[i] & (side == 2)
        if k1.any() and k2.any():
            c1, c2 = np.array([x[i, k1].mean(), z[i, k1].mean()]), np.array([x[i, k2].mean(), z[i, k2].mean()])
            fwd = c1 - c2
            fwd = fwd / max(float(np.hypot(*fwd)), 1e-6)
            lat = (x[i] - c2[0]) * -fwd[1] + (z[i] - c2[1]) * fwd[0]
            out["lap_enemy"] = float(max(0.0, lat[k1].min() - lat[k2].min()) + max(0.0, lat[k2].max() - lat[k1].max()))
    return out


def game_battle(run_dir, winner, passports=None, chosen=GROUPS):
    """measure() of a game recording (build/nn-arena/runs/<time>/) with our lord's ability uses read
    from its events; None when the recording is missing or empty."""
    from tools.nn import gamedata
    run_dir = Path(run_dir)
    try:
        b = gamedata.load(run_dir)
    except (OSError, ValueError, KeyError):
        return None
    if not len(b.t):
        return None
    lords = [n for n, k, s in zip(b.names, b.keys, b.side) if s == 1 and k and passport.lords([k], passports)[0]]
    uses = [t for t, _ in abilities_used(run_dir / "events.jsonl", lords[0] if lords else None)]
    orders = orders_in_force(run_dir / "events.jsonl", b.names, b.t) if "fatigue" in chosen else None
    return measure(b, winner, None, uses, chosen, passports, orders)


def pooled(values):
    """(mean, count) of the numbers in values that are not None."""
    v = [x for x in values if x is not None]
    return (float(np.mean(v)) if v else None), len(v)
