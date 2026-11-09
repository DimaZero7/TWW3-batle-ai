"""Per run: own snipers (shots, first melee, first rout), their guard pairs (the two halberds / spears / line nearest
each sniper at t = 0: first rout), the enemy's snipers (men start -> end, HP lost share). python -m game.battle.snip <run>..."""
import json
import math
import sys

from game.battle import run_dir, units

SN = ("long_rifles", "jezzails")
GUARD = ("halberd", "spear", "greatsword", "sword", "clanrat", "stormvermin", "slave")


def snipers(run):
    R = run_dir(run)
    us = units.per_unit(R)
    first = last = None
    for line in (R / "events.jsonl").open(encoding="utf-8"):
        if '"nn_sample"' in line:
            d = json.loads(line)
            if first is None:
                first = {u["n"]: u for u in d["units"]}
            last = {u["n"]: u for u in d["units"]}
    out = {"run": R.name, "own": {}, "enemy": {}}
    for n, u in first.items():
        short = n.removeprefix("own_")
        if n.startswith("own_") and any(k in n for k in SN):
            g = sorted((math.hypot(v["x"] - u["x"], v["z"] - u["z"]), m) for m, v in first.items()
                       if m.startswith("own_") and any(k in m for k in GUARD))[:2]
            gr = [us.get(m.removeprefix("own_"), {}).get("rout") for _, m in g]
            su = us.get(short, {})
            out["own"][short] = {"shots": su["ammo"][0] - su["ammo"][1] if su else None, "melee": su.get("melee"),
                                 "rout": su.get("rout"), "guards": [m.removeprefix("own_") for _, m in g],
                                 "guards_d": [round(d) for d, _ in g], "guards_rout": gr}
        if n.startswith("enemy_") and any(k in n for k in SN):
            e = last.get(n, {})
            out["enemy"][n.removeprefix("enemy_")] = {"men": [u.get("men"), e.get("men")],
                                                      "hp_lost": round(1 - float(e.get("hp", 0) or 0), 2)}
    return out


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(json.dumps(snipers(run)))
