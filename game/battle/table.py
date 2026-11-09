"""Per run, one json row: winner, time, contact, gold own / enemy (compare), inside share, regroups (analyze), lord
melee, snipers' and guns' shots / first melee / orders given (churn). python -m game.battle.table <run>...
(the after-N tables of the chain's battles: build/nn-arena/runs or data/nn-arena/runs)"""
import json
import sys

from game.battle import analyze, churn, compare, units

SN = ("long_rifles", "jezzails")
GUN = ("mortar", "warp_lightning", "plagueclaw")


def row(run):
    a = analyze.main(run)
    c = compare.one(run)
    us = units.per_unit(run)
    given = churn.orders(run)
    shoot = {n: (u["ammo"][0] - u["ammo"][1], u["melee"], given.get(n, (0, 0))[0]) for n, u in us.items()
             if any(k in n for k in SN + GUN)}
    return {"run": c["run"], "winner": c["winner"], "t": c["time_s"], "contact": c["first_contact_s"],
            "gold": [c["side1"]["gold_lost"], c["side2"]["gold_lost"]], "inside": a["areas"]["inside_share"],
            "reforms": a["operative"]["reforms"], "lord_melee": us.get("lord", {}).get("melee"),
            "missed": a["game_wait"]["missed"], "strat_err": (a.get("strategist") or {}).get("errors"), "shoot": shoot}


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(json.dumps(row(run)))
