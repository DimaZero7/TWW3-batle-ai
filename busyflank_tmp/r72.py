"""TEMPORARY: the 72 network-vs-game-AI battles of build/lordrout/replay72_*.json replayed (check.measure_battle rows):
same winner, routs per unit, routed shares, coverage.
DOCK_BUILD=<main>/build bash tools/nn/dock.sh walk3_tmp.r72 OUT.json [COPIES [WORKERS]] [--old]"""
import json
import sys
from pathlib import Path

import numpy as np

from tools.nn.sim import check, replay
from tools.nn.sim.params import load as load_params

# --old: the replay before WALK_IN_MELEE (module level: spawned workers re-import this module with the same argv)
replay.WALK_IN_MELEE = "--old" not in sys.argv


def jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def main():
    sets = [x[6:] for x in sys.argv[1:] if x.startswith("--set=")]
    args = [x for x in sys.argv[1:] if x != "--old" and not x.startswith("--set=")]
    out = args[0]
    copies = int(args[1]) if len(args) > 1 else 4
    workers = int(args[2]) if len(args) > 2 else 4
    print("WALK_IN_MELEE", replay.WALK_IN_MELEE, flush=True)
    params = load_params()
    for kv in sets:
        k, v = kv.split("=", 1)
        sec, key = k.split(".", 1)
        params.sim[sec][key] = json.loads(v)
        print("set", sec, key, v, flush=True)
    names = [r["run"] for r in json.load(open("build/lordrout/replay72_mff1.json"))["rows"]]
    runs = [Path("build/nn-arena/runs") / n for n in names]
    rows = check.simulate(runs, params, "cpu", copies, check.JITTER_M, end_at_recording=True,
                          measure_fn=check.measure_battle, workers=workers)
    cases = [c for r in rows for c in r.pop("cases")]
    fc = check.forecast(cases)
    m = [r["match"] for r in rows if r["match"] is not None]
    print(f"battles {len(rows)}, same winner {sum(m):g} of {len(m)}, coverage {fc['coverage'][0]:.3f}")
    for k in ("routed_1", "routed_2", "hp_lost_1", "hp_lost_2"):
        print(f"  {k:10s} game {np.mean([r['game_' + k] for r in rows]):.3f} sim {np.mean([r['sim_' + k] for r in rows]):.3f}")
    for q in fc["per_q"]:
        if q["q"] in ("routs_per_unit", "own.routed_share", "enemy.routed_share", "own_won", "own.hp_lost_end",
                      "enemy.hp_lost_end", "rallies_per_unit"):
            print(f"  {q['q']:20s} game {q['game']:.3f} sim {q['sim']:.3f} coverage {q['coverage']:.2f}")
    sys.stdout.flush()
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"forecast": fc, "rows": rows}, f, default=jsonable)


if __name__ == "__main__":
    main()
