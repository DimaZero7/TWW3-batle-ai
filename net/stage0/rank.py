"""The rank check of a stage 0 drill and the 'before' reading (tmp/old-build/diag_stage1/plan.md §4 step 2).

    bash ops/dock.sh net.stage0.rank rank --drill c01 --out tmp/old-build/diag_stage0/rank_c01.jsonl
    bash ops/dock.sh net.stage0.rank before --out tmp/old-build/diag_stage0/before.json

rank: each scripted behaviour of the drill (behave.SETS) on the same n drill battles and the same n_gen general battles
of the stage (c01: the Kislev ones - the flat field has no height; o03: the armies of 4 and more - a smaller one has no
reserve); a row per behaviour (the drill's and the general battles' summaries), then the rank row - Spearman across the
behaviours of
    measure   the drill's measure against the same number in the general battles (the transfer)
    trade     the drill's trade against the general battles' trade
    score     the drill's score (its measure + its trade: what a training on it would raise) against the general trade
before: the chain as it is (every level the formula) in n general battles: every measure, by size, enemy and map.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch


from net.stage0 import battle as BA
from net.stage0 import behave as BH
from net.stage0 import specs as SP
from net.stage0 import stat as STT
from net.stage0 import watch as W

DRILL = {
    "c01": {"measure": "dh", "transfer": "dh", "gen": {"mp": "ksl_plains_01"}, "sign": 1.0},
    "t02": {"measure": "pin_flank", "transfer": "flank_dmg", "gen": {}, "sign": 1.0, "trade": "trade_after"},
    "o03": {"measure": "reserve_s", "transfer": "reserve_s", "gen": {"min_n": 4}, "sign": -1.0},
}


def score(kind, s):
    """The drill's score from its summary (module doc): c01 trade + dh / 10 m; t02 pin_flank + trade_after; o03
    trade - reserve_s / reserve_cap_s - reserve_early (a battle without a breach: no term)."""
    v = lambda k: 0.0 if s[k][0] != s[k][0] else s[k][0]                     # noqa: E731
    if kind == "c01":
        return v("trade") + v("dh") / 10.0
    if kind == "t02":
        return v("pin_flank") + v("trade_after")
    from net.stage0 import cfg
    return v("trade") - v("reserve_s") / float(cfg()["general"]["reserve_cap_s"]) - v("reserve_early")


def generals(n, rnd, mp=None, min_n=None):
    out = []
    while len(out) < n:
        s = SP.general(rnd, mp=mp)
        if min_n and len(s["mine"]) < min_n:
            continue
        out.append(s)
    return out


def _done(path):
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "who" in r:
                rows[r["who"]] = r
    return rows


def rank(a, unit_net, P):
    kind = a.drill
    d = DRILL[kind]
    whos = a.who.split(",") if a.who else list(BH.SETS[kind])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out)
    dsp = SP.drills(kind, a.n, SP.seeded(a.seed))
    g = dict(d["gen"])
    gsp = generals(a.n_gen, SP.seeded(a.seed + 1000), mp=g.get("mp"), min_n=g.get("min_n"))
    for who in whos:
        if who in done:
            continue
        t0 = time.time()
        md, sd = BA.play(dsp, BH.by_name(who), unit_net, P, a.device)
        mg, sg = BA.play(gsp, BH.by_name(who), unit_net, P, a.device)
        row = {"who": who, "drill": W.summary(md), "general": W.summary(mg), "drill_stats": sd, "general_stats": sg,
               "wall_s": round(time.time() - t0, 1), "n": a.n, "n_gen": a.n_gen, "seed": a.seed}
        row["score"] = score(kind, row["drill"])
        print(json.dumps({"who": who, "wall_s": row["wall_s"], d["measure"]: row["drill"][d["measure"]],
                          "transfer": row["general"][d["transfer"]], "trade": row["drill"]["trade"],
                          "gen_trade": row["general"]["trade"], "score": round(row["score"], 4)}), flush=True)
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        done[who] = row
    rows = [done[w] for w in whos]
    tk = d.get("trade", "trade")
    col = lambda part, k: [r[part][k][0] for r in rows]                       # noqa: E731
    res = {"rank": kind, "whos": whos,
           "measure": round(STT.spearman(col("drill", d["measure"]), col("general", d["transfer"])), 3),
           "trade": round(STT.spearman(col("drill", tk), col("general", "trade")), 3),
           "score": round(STT.spearman([r["score"] for r in rows], col("general", "trade")), 3),
           "measure_vs_gen_trade": round(STT.spearman([d["sign"] * x for x in col("drill", d["measure"])],
                                                      col("general", "trade")), 3),
           "table": {r["who"]: {"drill": r["drill"][d["measure"]][0], "drill_trade": r["drill"][tk][0],
                                "score": round(r["score"], 4), "transfer": r["general"][d["transfer"]][0],
                                "gen_trade": r["general"]["trade"][0]} for r in rows}}
    print(json.dumps(res), flush=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")
    return res


def before(a, unit_net, P):
    sp = SP.generals(a.n, SP.seeded(a.seed + 2000))
    t0 = time.time()
    m, st = BA.play(sp, BH.Behaviour(), unit_net, P, a.device)
    res = {"all": W.summary(m), "stats": st, "wall_s": round(time.time() - t0, 1), "n": a.n}
    sel = lambda f: torch.tensor([bool(f(s)) for s in sp])                     # noqa: E731
    res["by_size"] = {"2-4": W.summary(m, sel(lambda s: len(s["mine"]) <= 4)),
                      "5-8": W.summary(m, sel(lambda s: len(s["mine"]) >= 5))}
    res["by_enemy"] = {e: W.summary(m, sel(lambda s, e=e: s["enemy"] == e)) for e in dict.fromkeys(s["enemy"] for s in sp)}
    res["by_map"] = {mp: W.summary(m, sel(lambda s, mp=mp: s["map"] == mp)) for mp in ("flat", "ksl_plains_01")}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res["all"].items() if not k.endswith("_n")}), flush=True)
    print(json.dumps(st), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("rank", "before"))
    ap.add_argument("--drill", choices=tuple(DRILL))
    ap.add_argument("--who", default="")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--n-gen", dest="n_gen", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from sim.params import load
    from net.tactic import train as TT
    P = load()
    unit_net = TT.unit_network(a.device)
    if a.what == "rank":
        rank(a, unit_net, P)
    else:
        before(a, unit_net, P)


if __name__ == "__main__":
    main()
