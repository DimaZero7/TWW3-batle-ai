"""The game check crest8 of drill S0-1 'the line on the crest' (tmp/drill_s1/design_merged.md section 6 (b)): does the
game's AI, attacking, walk up to our line standing on a crest? The starts and the report.

    python -m game.battle.crest_read --make      build/crest8/c01..c08.json (--arena-json), c01..c08.place.json (the
                                                 place switch's line), battles.json
    PLACE=build/crest8/c01.place.json bash game/battle/chain.sh tactic c01                  (ksl_plains_01)
    PLACE=build/crest8/c05.place.json ZONES=east bash game/battle/chain.sh tactic c05       (kho_wastes_01)
    python -m game.battle.crest_read <runs...> [--out <json>]   the report and the verdict so far

Starts: 8 = 2 sites x 2 armies x 2 mirrors; holder arm R's form (game/battle/holder_read.py rNN): the game's AI attacks
on Normal, ours hold the west zone with the tactic chain alone (chain.sh tactic); 3 infantry + lord a side from the
stage-0 roster, gold matched (net/stage0/specs._sides); mirror 1 - the two armies and factions swap sides. Our 3 are
teleported in the deployment onto the teacher's crest line of net/strategist/crest.py on the Kislev west zone (the
game's east.xml zone, 416 x 176 m at x -184; on kho_wastes_01 laid by bridge.build --map-zones east), the lord 150 m
behind (bridge/companion/deploy.LinePlace). c = 1 + site x 4 + army x 2 + mirror; site 0 ksl_plains_01, 1 kho_wastes_01.

Per battle (the run's events.jsonl nn_sample a second): THE FIRST OPPONENT by exam_s1's rule (Astra s1_step2_review
2): a pair (our infantry n, an enemy e) touches at the first sample where n is in melee with its target e, or e is in
melee with its target n (either direction; the sample's "m" and "t"); several enemies -> the nearest centre; lords out
on both sides (of the contact line and the height line); our first contact (any of our infantry) and whether by 300 s;
each unit's (slot, opponent, second, dh_c = its centre's height less the opponent's, heights on the captured grid
sim/maps/<map>/heights.npz), a unit that never touched kept as 'no contact' (in the denominator of the 2/3); the deploy
check (the companion's record: errors).
LINES (tmp/astra/KEY.md s1_step1; written before any run): YES - contact by 300 s in >= 7 of 8 and >= 2/3 of the
first-melee units (pooled over the battles) at dh_c >= +0.5 m; NO (freeze the drill) - contact in <= 5; 6 of 8 - two
battles by eye; trade not judged. Stop as soon as the verdict cannot change (printed after each battle read).
"""
import argparse
import json
import math
import random
from pathlib import Path

import game

ROOT = game.ROOT
OUT = game.BUILD / "crest8"
SITES = ("ksl_plains_01", "kho_wastes_01")
N_INF = 3
SEED0 = 801
CONTACT_S = 300.0
DH_M = 0.5
N_BATTLES = 8


def starts():
    out = []
    for si, mp in enumerate(SITES):
        for k in range(2):
            for mir in (0, 1):
                out.append((f"c{1 + si * 4 + k * 2 + mir:02d}", mp, k, mir))
    return out


def line_of(mp, zone):
    """The teacher's crest line (net/strategist/crest.py) of N_INF units in the zone of the map; asserts it fires."""
    from net.strategist import crest as CR
    fd = CR.find(CR.ground(mp), zone, N_INF)
    k = CR.teach(fd, attack=False)
    assert k > 0, f"{mp}: the teacher keeps the centre in {zone}"
    c = fd["cands"][k - 1]
    return {k_: round(float(c[k_]), 3) for k_ in ("x", "z", "b", "front", "band", "share05", "s", "f", "turn")}, \
        round(float(fd["centre"]["band"]), 3)


def make():
    from net.stage0 import specs as S
    from game.battle import aivai as AV
    OUT.mkdir(parents=True, exist_ok=True)
    allb = {}
    mz, ez = S.zone(0), S.zone(1)
    for name, mp, k, mir in starts():
        seed = SEED0 + k
        rnd = random.Random(seed)
        fac = ("emp", "skv") if seed % 2 else ("skv", "emp")
        f1, f2, mine, theirs = S._sides(rnd, N_INF, fac=fac)
        if mir:
            f1, f2, mine, theirs = f2, f1, theirs, mine
        sp = {"fac": {1: f1, 2: f2}, "mine": mine, "theirs": theirs, "map": mp, "mine_zone": mz, "enemy_zone": ez,
              "places": {1: S.rows(mz, len(mine)), 2: S.rows(ez, len(theirs))}, "gold": (S.gold(mine), S.gold(theirs))}
        units = AV.world_units(sp)
        arena = AV.arena_of(name, sp, units)
        arena["name"], arena["note"] = f"crest8_{name}", "crest8: stage 0, 3 + lord, the game's AI attacks, ours on the crest"
        (OUT / f"{name}.json").write_text(json.dumps(arena, indent=1), encoding="utf-8")
        line, centre_band = line_of(mp, mz)
        place = {"line": line, "n": N_INF, "map": mp, "zone": mz, "centre_band": centre_band,
                 "site": f"{mp} W band {line['band']}",
                 "why": "net/strategist/crest.py find + teach on the Kislev west zone (net/stage0/specs.zone(0))"}
        (OUT / f"{name}.place.json").write_text(json.dumps(place, indent=1), encoding="utf-8")
        allb[name] = {"seed": seed, "army": k, "mirror": mir, "map": mp, "role": "attack", "fac": sp["fac"],
                      "gold": sp["gold"], "line": line, "units": {str(s): v for s, v in units.items()}}
        print(name, mp, "army", k, "mirror", mir, sp["fac"], [u.split("_inf_")[-1] for u in mine], "vs",
              [u.split("_inf_")[-1] for u in theirs], [round(g) for g in sp["gold"]], "line", line)
    (OUT / "battles.json").write_text(json.dumps(allb, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ the report

def _jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def battle(run):
    from net.strategist import crest as CR
    run = Path(run)
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    start = Path(str(man["config"]["army"]["from_file"]).replace("\\", "/")).stem
    b = json.loads((OUT / "battles.json").read_text(encoding="utf-8"))[start]
    G = CR.ground(b["map"])
    ev = _jsonl(run / "events.jsonl")
    fr = sorted([(d["t"] / 1000.0, {u["n"]: u for u in d["units"]}) for d in ev if d.get("event") == "nn_sample"],
                key=lambda x: x[0])
    res = next((d for d in ev if d.get("event") == "result"), {})
    names = list(fr[0][1]) if fr else []
    own = [n for n in names if n.startswith("own_")]
    slot_no = lambda n: int("".join(c for c in n if c.isdigit()) or 10 ** 6)       # noqa: E731
    inf = sorted([n for n in own if not n.endswith("_lord")], key=slot_no)
    foes = [n for n in names if n.startswith("enemy_") and not n.endswith("_lord")]
    h = lambda u: float(CR.height(G, u["x"], u["z"]))          # noqa: E731
    d = lambda a, b: math.hypot(a["x"] - b["x"], a["z"] - b["z"])      # noqa: E731
    first = {}
    for t, us in fr:
        for n in inf:
            u = us.get(n)
            if n in first or not u or u.get("x") is None:
                continue
            touch = [us[e] for e in foes if us.get(e) and us[e].get("x") is not None
                     and ((u.get("m") and u.get("t") == e) or (us[e].get("m") and us[e].get("t") == n))]
            if touch:
                e = min(touch, key=lambda v: d(v, u))
                first[n] = {"unit": n, "slot": inf.index(n), "first_melee_s": t, "opponent": e["n"],
                            "dh_c": round(h(u) - h(e), 2), "gap_m": round(d(e, u), 1)}
    units = [first.get(n, {"unit": n, "slot": inf.index(n), "first_melee_s": None, "dh_c": None, "note": "no contact"})
             for n in inf]
    secs = [x["first_melee_s"] for x in units if x["first_melee_s"] is not None]
    contact = min(secs) if secs else None
    comp = run / "companion.jsonl"
    dep = next((r for r in (_jsonl(comp) if comp.exists() else []) if r.get("event") == "deploy"), {})
    start_h = {n: round(h(fr[0][1][n]), 1) for n in inf if fr and n in fr[0][1]}
    return {"run": run.name, "start": start, "map": b["map"], "line_band": b["line"]["band"],
            "deploy_who": dep.get("who"), "deploy_errors": dep.get("errors"), "start_heights": start_h,
            "contact_s": contact, "contact_300": contact is not None and contact <= CONTACT_S,
            "status": res.get("status"), "winner": res.get("winner"), "units": units}


def verdict(rows, total=N_BATTLES):
    """(verdict, final?) by the lines (module doc) over the battles read so far."""
    k = len(rows)
    c = sum(r["contact_300"] for r in rows)
    miss = k - c
    us = [u for r in rows if r["contact_300"] for u in r["units"]]      # (non-contacts in the denominator: exam_s1)
    good = sum(u["dh_c"] is not None and u["dh_c"] >= DH_M for u in us)
    left = total - k
    share = good / len(us) if us else float("nan")
    if miss >= total - 5:                                  # (contact can no longer exceed 5)
        return f"NO: contact {c} of {k}, at most {c + left} of {total} - freeze the drill", True
    yes_possible = miss <= total - 7
    # YES for sure: contact already >= 7 and the share holds even if every unit still to come fails
    if c >= 7 and us and good / (len(us) + N_INF * left) >= 2 / 3:
        return f"YES: contact {c} of {k}, first-melee units at dh_c >= +{DH_M} m {good} of {len(us)}", True
    if not yes_possible and left:
        return f"not YES (contact {c} of {k}, at most {c + left}): one more battle without contact - NO", False
    if left == 0:
        return (f"{'YES' if c >= 7 and share >= 2 / 3 else 'NOT YES'}: contact {c} of {total}, units dh_c >= +{DH_M} m "
                f"{good} of {len(us)}" + ("" if c >= 7 and share >= 2 / 3 else " - two battles by eye"), True)
    return f"open: contact {c} of {k}, units dh_c >= +{DH_M} m {good} of {len(us)} ({left} battles left)", False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make", action="store_true")
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.make:
        return make()
    rows = []
    for r in a.runs:
        row = battle(r)
        rows.append(row)
        fm = [u for u in row["units"] if u["first_melee_s"] is not None]
        print(f"{row['start']} {row['run']} {row['map']} band {row['line_band']} deploy {row['deploy_who']} "
              f"errors {row['deploy_errors']} | contact by 300 s: {'yes' if row['contact_300'] else 'no'} "
              f"({row['contact_s']}) | first-melee units dh_c {[u['dh_c'] for u in fm]} | {row['status']} {row['winner']}")
        print("   verdict so far:", verdict(rows)[0])
    v, final = verdict(rows) if rows else ("no runs", False)
    print("VERDICT:", v, "| final - stop" if final else "| go on")
    if a.out:
        Path(a.out).write_text(json.dumps({"rows": rows, "verdict": v, "final": final}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
