"""Per own unit of a battle: when it first met the enemy in melee, when it routed, seconds firing, ammunition start -> end,
seconds standing still while the battle was on, the distance walked. python -m game.battle.units <run>"""
import json
import math
import sys

from game.battle import run_dir


def per_unit(run):
    """{unit name without own_: {melee, rout, fire_s, ammo [start, end], still_s, walk_m, dead}}."""
    run = run_dir(run)
    s = [json.loads(line) for line in (run / "events.jsonl").open(encoding="utf-8") if '"nn_sample"' in line]
    rows = {}
    for d in s:
        t = round(d["t"] / 1000)
        for u in d["units"]:
            if int(u.get("side") or 0) != 1:
                continue
            r = rows.setdefault(u["n"], {"melee": None, "rout": None, "fire_s": 0, "ammo": [u.get("a"), None], "still_s": 0,
                                         "walk_m": 0.0, "last": (u["x"], u["z"]), "dead": None})
            if u.get("m") and r["melee"] is None:
                r["melee"] = t
            if u.get("r") and r["rout"] is None:
                r["rout"] = t
            if (u.get("men") or 0) == 0 and r["dead"] is None:
                r["dead"] = t
            if u.get("fire"):
                r["fire_s"] += 1
            r["ammo"][1] = u.get("a")
            if not u.get("mv") and not u.get("m") and not u.get("r"):
                r["still_s"] += 1
            r["walk_m"] += math.hypot(u["x"] - r["last"][0], u["z"] - r["last"][1])
            r["last"] = (u["x"], u["z"])
    for r in rows.values():
        r.pop("last")
        r["walk_m"] = round(r["walk_m"])
    return {n.replace("own_", ""): r for n, r in rows.items()}


if __name__ == "__main__":
    for name, row in per_unit(sys.argv[1]).items():
        print(name, json.dumps(row))
