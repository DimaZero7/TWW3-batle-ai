"""Per shooter / gun: orders the bridge gave the game (nn_orders status given / kept), the bridge's own re-gives
(nn_reaim, nn_duty resume / release), from a run's events. python -m game.battle.churn <run>..."""
import json
import sys

from game.battle import run_dir

UNITS = ("long_rifles", "jezzails", "mortar", "warp_lightning", "plagueclaw", "huntsmen", "ratling", "handgun")


def orders(run):
    """{unit name without own_: (orders given, the bridge's own re-gives)}."""
    given, own = {}, {}
    for line in (run_dir(run) / "events.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        if d.get("event") == "nn_orders":
            for o in d.get("orders") or []:
                if any(k in o["u"] for k in UNITS) and o.get("status") != "kept":
                    given[o["u"]] = given.get(o["u"], 0) + 1
        elif d.get("event") in ("nn_reaim", "nn_duty") and any(k in (d.get("u") or "") for k in UNITS):
            a = d.get("action") or "reaim"
            if a in ("reaim", "resume", "release"):
                own[d["u"]] = own.get(d["u"], 0) + 1
    return {u.removeprefix("own_"): (given.get(u, 0), own.get(u, 0)) for u in sorted(set(given) | set(own))}


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(run, orders(run))
