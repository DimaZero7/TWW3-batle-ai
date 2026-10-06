"""Step 2 of the battle simulator: numbers measured in the game, from recorded arena runs.

Reads the runs of the measurement arenas (config/nn/arenas.json; recorded with
python -m tools.build nn-arena --arena <name>, then tools/launcher/launch.ps1) and
writes what the simulator must reproduce to build/nn-measure/targets.json (not in
Git: research output). Three kinds, by the arena's name:

    pair_*    melee, one unit against one: losses per second in contact, time to
              wavering and rout, morale, implied men fighting at once;
    missile_* one missile unit shoots a unit that holds: shots, time to the first
              shot, shots per man per second, damage per shot by distance;
    whole_*   whole battles: winner, duration, losses over time, routs and rallies.

    python -m tools.nn.measure              # every measurement arena; prints a summary
    python -m tools.nn.measure --out x.json

Only fair runs (Normal difficulty, tools/nn/gamedata.runs) count. Only numpy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools import config as project
from tools.nn import gamedata

OUT = project.BUILD / "nn-measure" / "targets.json"
PASSPORTS = project.CONFIG_DIR / "nn" / "units.json"
KINDS = ("pair", "missile", "whole")
FLIGHT_S = 2          # damage is counted this many seconds after the shots (arrow flight, 1 s samples)
DISTANCE_BINS = (0, 60, 90, 110, 140)
MISS_S = 0.5          # a missed blow costs this long (config/nn/sim.json melee.miss_s, build/hitchance)
CHARGE_S = 15         # the first seconds of contact are the charge; the rest is the steady fight
CURVE_EVERY_S = 30


def kind(arena):
    head = arena.split("_", 1)[0]
    return head if head in KINDS else None


def passports(path=PASSPORTS):
    return json.loads(Path(path).read_text(encoding="utf-8"))["units"]


def stats(values):
    """n, mean, min and max of the numbers that are there (None when there are none)."""
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not v:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": len(v), "mean": round(float(np.mean(v)), 4), "min": round(min(v), 4), "max": round(max(v), 4)}


def first(mask, start=0):
    """First index from start where mask is true, or None."""
    hits = np.nonzero(mask[start:])[0]
    return int(hits[0]) + start if hits.size else None


def hp_abs(b, i, p):
    return b.f["hp"][:, i] * p[b.keys[i]]["hp_total"]


def distance(b, i, j):
    return np.hypot(b.f["x"][:, i] - b.f["x"][:, j], b.f["z"][:, i] - b.f["z"][:, j])


def speed(b, i):
    """m/s between samples ([T], first is 0)."""
    d = np.hypot(np.diff(b.f["x"][:, i]), np.diff(b.f["z"][:, i])) / np.maximum(np.diff(b.t), 1e-9)
    return np.concatenate([[0.0], d])


def hit_chance(attacker, target):
    """The database rule (docs/*/game/units/melee.md): 35 + attack - defence, within 8-90 %."""
    return min(90, max(8, 35 + attacker["melee"]["attack"] - target["melee"]["defence"])) / 100


def per_hit(damage, ap_damage, armour, hp_per_man, single=False):
    """HP a hit takes (the simulator's rule, tools/nn/sim/melee.py per_hit, in plain Python): the armour-piercing
    part whole, the base part less the armour roll (U(0.5, 1) x armour %, at most 100 %); damage beyond the struck
    man's health is lost (overkill, smoothed); a lone man (single) loses the mean."""
    a = max(0.0, armour / 100)
    cut = 0.75 * a if a <= 1 else (2 - 1 / a - a / 4 if a < 2 else 1.0)
    lo = max(1.0, ap_damage + damage * max(0.0, 1 - a))
    hi = max(1.0, ap_damage + damage * max(0.0, 1 - a / 2))
    mu = max(1.0, ap_damage + damage * (1 - cut))
    if single:
        return mu
    p1 = min(1.0, max(0.0, (hp_per_man - lo) / (hi - lo))) if hi - lo > 1e-6 else float(lo < hp_per_man)
    tail = max(1.0, (hp_per_man - (lo + min(hi, hp_per_man)) / 2) / mu + 0.5)
    return hp_per_man / (1 + p1 * tail)


def melee_per_fighter(attacker, target):
    """HP per second one fighting man of attacker takes from target, by the database rule: p / (p x attack
    interval + MISS_S) blows a second (the simulator's rule, config/nn/sim.json melee.miss_s)."""
    m = attacker["melee"]
    p = hit_chance(attacker, target)
    return p * per_hit(m["damage"], m["ap_damage"], target["armour"], target["hp_per_man"],
                       single=target["men"] <= 1) / (p * m["attack_interval_s"] + MISS_S)


def rallied(b, i, start=0):
    """True if unit i routed and later stood again (not routing, not shattered, men left)."""
    r, s, men = b.f["r"][start:, i], b.f["s"][start:, i], b.f["men"][start:, i]
    routed = first(r)
    if routed is None:
        return False
    back = (~r[routed:]) & (~s[routed:]) & (men[routed:] > 0)
    return bool(back.any())


def melee_pair(b, p):
    """One unit against one: the fight from first contact to the first rout (or the end). Times count from the
    first record with the melee flag (c); the charge's HP from the record before it (c0): the blows of the contact
    second land before the flag is first recorded (the first strike), and a sample of the flag alone would miss
    them (charge_hp_lost: c0 to CHARGE_S after c; contact_s_hp_lost: c0 to c)."""
    c = first(b.f["m"].any(axis=1))
    if c is None:
        return {"run": b.run, "contact": False}
    c0 = max(c - 1, 0)
    routs = b.f["r"] | b.f["s"]
    e = first(routs.any(axis=1), c)
    e = len(b.t) - 1 if e is None else e
    out = {"run": b.run, "contact": True, "contact_t_s": float(b.t[c]), "fight_s": float(b.t[e] - b.t[c]),
           "status": b.result.get("status"), "winner_side": b.winner, "units": []}
    for i in range(2):
        j = 1 - i
        me, foe = p[b.keys[i]], p[b.keys[j]]
        hp = hp_abs(b, i, p)
        steady = first(b.t >= b.t[c] + CHARGE_S, c)
        row = {"slot": b.names[i], "key": b.keys[i], "side": int(b.side[i]),
               "men_start": float(b.f["men"][c, i]), "men_end": float(b.f["men"][e, i]),
               "hp_start": round(float(hp[c]), 1), "hp_end": round(float(hp[e]), 1)}
        dur = max(b.t[e] - b.t[c], 1e-9)
        row["men_per_s"] = round((row["men_start"] - row["men_end"]) / dur, 4)
        row["hp_per_s"] = round((hp[c] - hp[e]) / dur, 2)
        row["contact_s_hp_lost"] = round(float(hp[c0] - hp[c]), 1)
        if steady is not None and steady < e:
            row["charge_hp_lost"] = round(float(hp[c0] - hp[steady]), 1)
            row["steady_hp_per_s"] = round(float((hp[steady] - hp[e]) / (b.t[e] - b.t[steady])), 2)
            row["steady_men_per_s"] = round(float((b.f["men"][steady, i] - b.f["men"][e, i]) /
                                                  (b.t[e] - b.t[steady])), 4)
            # How many of the other side's men would have to hit at once to take this much.
            row["implied_foes_fighting"] = round(row["steady_hp_per_s"] / melee_per_fighter(foe, me), 2)
        waver, rout = first(b.f["w"][:, i], c), first(routs[:, i], c)
        row["waver_after_s"] = None if waver is None else float(b.t[waver] - b.t[c])
        row["rout_after_s"] = None if rout is None else float(b.t[rout] - b.t[c])
        row["rallied"] = rallied(b, i, c)
        step = max(1, int(round(10 / max(np.median(np.diff(b.t)), 1e-9)))) if len(b.t) > 1 else 1
        row["morale_every_10s"] = [None if not np.isfinite(v) else round(float(v), 3)
                                   for v in b.f["mp"][c:e + 1:step, i]]
        row["morale_state_every_10s"] = [None if not np.isfinite(v) else int(v)
                                         for v in b.f["ms"][c:e + 1:step, i]]
        out["units"].append(row)
    return out


def approach(b, i, p, contact):
    """A unit's run before contact: top speed, speed in the last 2 s, seconds from starting to move
    to 90 % of the top (1 s samples). None if it moved less than 20 m."""
    if contact is None or contact < 2:
        return None
    moved = np.hypot(b.f["x"][contact, i] - b.f["x"][0, i], b.f["z"][contact, i] - b.f["z"][0, i])
    if not np.isfinite(moved) or moved < 20:
        return None
    v = speed(b, i)[1:contact + 1]
    top = float(np.percentile(v, 90))
    go = first(v > 0.3)
    reach = first(v >= 0.9 * top, go or 0)
    return {"slot": b.names[i], "key": b.keys[i], "moved_m": round(float(moved), 1), "top_m_s": round(top, 2),
            "last_2s_m_s": round(float(v[-2:].mean()), 2),
            "passport_run_m_s": p[b.keys[i]]["speed"]["run"], "passport_charge_m_s": p[b.keys[i]]["speed"]["charge"],
            "starts_moving_s": None if go is None else float(b.t[go + 1]),
            "to_90pct_after_start_s": None if go is None or reach is None else float(reach - go + 1)}


def missile_run(b, p):
    """The game's AI (side 2) shoots our unit that holds (side 1)."""
    s = int(np.nonzero(b.side == 2)[0][0])
    g = int(np.nonzero(b.side == 1)[0][0])
    shooter, target = p[b.keys[s]], p[b.keys[g]]
    ammo = b.f["a"][:, s]
    fired = np.concatenate([[0.0], np.maximum(0.0, -np.diff(ammo))])
    fired[~np.isfinite(fired)] = 0
    dist = distance(b, s, g)
    hp = hp_abs(b, g, p)
    lost = np.concatenate([[0.0], np.maximum(0.0, -np.diff(hp))])
    men_lost = np.concatenate([[0.0], np.maximum(0.0, -np.diff(b.f["men"][:, g]))])
    melee = b.f["m"][:, g] | b.f["m"][:, s]
    shot = first(fired > 0)
    out = {"run": b.run, "shooter": b.keys[s], "target": b.keys[g], "status": b.result.get("status"),
           "ammo_start": float(ammo[0]), "ammo_end": float(ammo[-1]), "shots": float(fired.sum())}
    if shot is None:
        return dict(out, shooting=False)
    rng = shooter["missile"]["range_m"]
    stop = shot                     # the start of the halt in which the first shot came
    while stop > 0 and not b.f["mv"][stop - 1, s]:
        stop -= 1
    last = int(np.nonzero(fired > 0)[0][-1])
    firing = np.zeros(len(b.t), dtype=bool)
    # Distances are between unit centres: the AI halts a little beyond the range to the front rank.
    firing[shot:last + 1] = (~b.f["mv"][shot:last + 1, s]) & (dist[shot:last + 1] <= rng + 20)
    man_s = float(np.nansum(b.f["men"][firing, s]))
    rate = float(fired[firing].sum() / man_s) if man_s else None
    out.update(shooting=True, first_shot_t_s=float(b.t[shot]), first_shot_distance_m=round(float(dist[shot]), 1),
               first_shot_after_stop_s=None if stop == 0 else float(b.t[shot] - b.t[stop]),
               last_shot_t_s=float(b.t[last]), shots_per_man_per_s=None if rate is None else round(rate, 4),
               implied_reload_s=None if not rate else round(1 / rate, 2),
               passport_reload_s=shooter["missile"]["reload_s"],
               ammo_used_share=round(float(fired.sum() / ammo[0]), 3) if ammo[0] else None)
    # Damage per shot by distance, summed over seconds: the ammo counter drops in bursts, so a shot's
    # own second is not its damage's; damage at t counts for the distance at t - FLIGHT_S. No melee.
    # A second's shots count only while their damage can still be seen.
    at = np.clip(np.arange(len(b.t)) - FLIGHT_S, 0, None)
    counted = ~melee & (np.arange(len(b.t)) >= shot)
    seen = np.arange(len(b.t)) + FLIGHT_S < len(b.t)
    bins = []
    for lo, hi in zip(DISTANCE_BINS, DISTANCE_BINS[1:]):
        in_bin = (dist >= lo) & (dist < hi)
        shots = float(fired[counted & seen & in_bin].sum())
        hp_lost = float(lost[counted & in_bin[at]].sum())
        men = float(men_lost[counted & in_bin[at]].sum())
        if shots:
            expected = per_hit(shooter["missile"]["damage"], shooter["missile"]["ap_damage"], target["armour"],
                               target["hp_per_man"], single=target["men"] <= 1)
            bins.append({"distance_m": [lo, hi], "shots": shots, "hp_per_shot": round(hp_lost / shots, 3),
                         "men_per_100_shots": round(100 * men / shots, 2),
                         "implied_hit_rate": round(hp_lost / shots / expected, 3)})
    out["by_distance"] = bins
    window = (b.t >= b.t[shot]) & (b.t <= b.t[last]) & ~melee
    out["hp_per_s_while_shooting"] = round(float(lost[window].sum() / max(window.sum(), 1)), 2)
    waver, rout = first(b.f["w"][:, g], shot), first(b.f["r"][:, g] | b.f["s"][:, g], shot)
    out["target_waver_after_first_shot_s"] = None if waver is None else float(b.t[waver] - b.t[shot])
    out["target_rout_after_first_shot_s"] = None if rout is None else float(b.t[rout] - b.t[shot])
    out["melee_after_shooting"] = bool(melee[shot:].any())
    return out


def whole_run(b, p, factions):
    """A whole battle: who won, how long, losses over time, routs and rallies per side."""
    attacker_side = 1 if b.own_ai == "attack" else 2
    names = {1: factions.get("own", "?"), 2: factions.get("enemy", "?")}
    out = {"run": b.run, "own_ai": b.own_ai, "status": b.result.get("status"), "duration_s": float(b.t[-1]),
           "attacker": names[attacker_side], "winner": names.get(b.winner, None),
           "first_contact_s": None, "sides": {}}
    c = first(b.f["m"].any(axis=1))
    out["first_contact_s"] = None if c is None else float(b.t[c])
    step = first(b.t >= CURVE_EVERY_S) or 1
    for side in (1, 2):
        idx = np.nonzero(b.side == side)[0]
        men = np.nansum(b.f["men"][:, idx], axis=1)
        hp = sum(hp_abs(b, i, p) for i in idx)
        routed = [b.names[i] for i in idx if b.f["r"][:, i].any() or b.f["s"][:, i].any()]
        out["sides"][names[side]] = {
            "units": len(idx), "men_start": float(men[0]), "men_end": float(men[-1]),
            "hp_lost_share": round(float(1 - hp[-1] / hp[0]), 3),
            "routed_units": len(routed), "rallied_units": sum(rallied(b, i) for i in idx),
            "shattered_units": int(sum(b.f["s"][-1, i] for i in idx)),
            "men_every_30s": [float(v) for v in men[::step]],
            "hp_share_every_30s": [round(float(v / hp[0]), 3) for v in hp[::step]]}
    return out


def summary(kind_name, rows):
    """Means and spread over the runs of one arena."""
    if kind_name == "pair":
        fought = [r for r in rows if r.get("contact")]
        out = {"runs": len(rows), "fought": len(fought), "fight_s": stats(r["fight_s"] for r in fought)}
        for i in range(2):
            us = [r["units"][i] for r in fought]
            if not us:
                continue
            out[us[0]["slot"]] = {k: stats(u.get(k) for u in us) for k in (
                "men_per_s", "hp_per_s", "charge_hp_lost", "steady_hp_per_s", "steady_men_per_s",
                "implied_foes_fighting", "waver_after_s", "rout_after_s")}
            out[us[0]["slot"]]["routed_runs"] = sum(u["rout_after_s"] is not None for u in us)
        return out
    if kind_name == "missile":
        shot = [r for r in rows if r.get("shooting")]
        out = {"runs": len(rows), "shooting": len(shot)}
        for k in ("first_shot_t_s", "first_shot_distance_m", "first_shot_after_stop_s", "shots_per_man_per_s",
                  "implied_reload_s", "ammo_used_share", "hp_per_s_while_shooting",
                  "target_waver_after_first_shot_s", "target_rout_after_first_shot_s"):
            out[k] = stats(r.get(k) for r in shot)
        pooled = {}
        for r in shot:
            for x in r["by_distance"]:
                key = f"{x['distance_m'][0]}-{x['distance_m'][1]}"
                pooled.setdefault(key, []).append(x)
        out["by_distance"] = {k: {"runs": len(v), "shots": sum(x["shots"] for x in v),
                                  "hp_per_shot": stats(x["hp_per_shot"] for x in v),
                                  "implied_hit_rate": stats(x["implied_hit_rate"] for x in v)}
                              for k, v in sorted(pooled.items(), key=lambda kv: int(kv[0].split("-")[0]))}
        return out
    wins = {}
    for r in rows:
        wins[str(r["winner"])] = wins.get(str(r["winner"]), 0) + 1
    return {"runs": len(rows), "wins": wins, "duration_s": stats(r["duration_s"] for r in rows)}


def targets(runs_dir=None, passports_path=PASSPORTS):
    """{arena: {kind, runs: [...], summary}} for every measurement arena with runs."""
    p = passports(passports_path)
    by_arena = {}
    for d in gamedata.runs(root=runs_dir):
        b = gamedata.load(d)
        k = kind(b.arena)
        if k is None:
            continue
        factions = json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))["config"].get("factions", {})
        row = {"pair": lambda: melee_pair(b, p), "missile": lambda: missile_run(b, p),
               "whole": lambda: whole_run(b, p, factions)}[k]()
        if k == "pair":
            c = first(b.f["m"].any(axis=1))
            row["approach"] = [x for x in (approach(b, i, p, c) for i in range(2)) if x]
        by_arena.setdefault(b.arena, {"kind": k, "runs": []})["runs"].append(row)
    for name, a in by_arena.items():
        a["summary"] = summary(a["kind"], a["runs"])
    return by_arena


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--runs", type=Path, help="runs folder instead of build/nn-arena/runs")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    data = targets(args.runs)
    doc = {"_description": "Numbers measured in the game for the battle simulator (step 2; "
                           "docs/en/training/measurements.md). Written by tools/nn/measure.py from build/nn-arena/runs.",
           "arenas": data}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, a in sorted(data.items()):
        print(f"== {name} ({a['kind']}): {', '.join(r['run'] for r in a['runs'])}")
        print(json.dumps(a["summary"], ensure_ascii=False))
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
