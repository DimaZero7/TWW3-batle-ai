"""Samples and statistics of random battles; the game's battle file and a simulator batch of them.

    python -m tools.nn.armies                          # 5 samples and statistics over 10000 battles
    python -m tools.nn.armies --samples 3 --xml build/nn-armies     # + scenarios/<name>.xml per sample
    bash tools/nn/dock.sh tools.nn.armies --samples 8 --sim         # + a simulator batch (torch)
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from tools.nn.armies import export
from tools.nn.armies import generate as G


def describe(arena):
    parts = []
    for side in ("own", "enemy"):
        s = arena["sides"][side]
        kinds = Counter(u["slot"].rsplit("_", 1)[0] for u in s["units"] if not u.get("general"))
        body = " + ".join(f"{n} {k}" for k, n in sorted(kinds.items())) or "no units"
        parts.append(f"{s['faction']} [{s['army']}] {s['cost']}: lord + {body}")
    return f"{arena['name']} budget {arena['budget']} | " + " | ".join(parts)


def stats(arenas, pools):
    """Summary lines over many battles."""
    n_units, missile, gaps, budgets, ratio, modes, mix = [], [], [], [], [], Counter(), Counter()
    realized, pairs = {}, {}
    for a in arenas:
        budgets.append(a["budget"])
        sides = [a["sides"][s] for s in ("own", "enemy")]
        if sides[0]["budget"] == sides[1]["budget"]:
            gaps.append(abs(sides[0]["cost"] - sides[1]["cost"]) / max(s["cost"] for s in sides))
        lo, hi = sorted(sides, key=lambda s: (pools[s["faction"]].budget_factor, s["faction"]))
        pairs.setdefault((lo["faction"], hi["faction"]), []).append(lo["cost"] / hi["cost"])
        counts = []
        for s in sides:
            units = [u for u in s["units"] if not u.get("general")]
            counts.append(len(units))
            n_units.append(len(units))
            miss = {u.key for u in pools[s["faction"]].units if u.missile}
            missile.append(sum(u["key"] in miss for u in units) / max(1, len(units)))
            modes["random" if s["army"] == "random" else "template"] += 1
            if s["army"] != "random":
                r = realized.setdefault(s["faction"], Counter())
                r.update(u["key"] for u in units)
        mix["mirror" if sides[0]["faction"] == sides[1]["faction"] else "cross"] += 1
        ratio.append(max(counts) / max(1, min(counts)))
    n_units, missile, gaps, ratio = map(np.asarray, (n_units, missile, gaps, ratio))
    N = len(arenas)
    hist = np.histogram(n_units, bins=[0, 1, 5, 10, 15, 19, 20])[0]
    out = [f"battles {N}: {mix['mirror'] / N:.0%} mirror; armies {modes['template'] / (2 * N):.0%} template, "
           f"{modes['random'] / (2 * N):.0%} random",
           f"budget: min {min(budgets)}, median {int(np.median(budgets))}, max {max(budgets)}",
           f"units per side (lord not counted): mean {n_units.mean():.1f}; 0: {hist[0]}, 1-4: {hist[1]}, "
           f"5-9: {hist[2]}, 10-14: {hist[3]}, 15-18: {hist[4]}, 19: {hist[5]}",
           f"cost difference between sides of equal budgets: mean {gaps.mean():.1%}, max {gaps.max():.1%}",
           f"cheap against elite (one side has x times the other's units): x>=1.5 {np.mean(ratio >= 1.5):.1%}, "
           f"x>=2 {np.mean(ratio >= 2):.1%}, x>=3 {np.mean(ratio >= 3):.1%}",
           f"missile share of a side's units: mean {missile.mean():.0%}; none {np.mean(missile == 0):.0%}, "
           f">=50% {np.mean(missile >= 0.5):.0%}, all {np.mean(missile == 1):.1%}"]
    for (f1, f2), r in sorted(pairs.items()):
        r = np.asarray(r)
        out.append(f"cost {f1} / {f2}: mean {r.mean():.3f}, min {r.min():.3f}, max {r.max():.3f} ({len(r)} battles)")
    for faction, r in sorted(realized.items()):
        total = sum(r.values())
        out.append(f"template armies of {faction}: " + ", ".join(
            f"{u.slot} {r[u.key] / total:.0%}" for u in pools[faction].units))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--stats", type=int, default=10000, help="battles for the statistics (0: none)")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--start", type=int, default=0, help="first seed, counted from the split's start")
    parser.add_argument("--xml", type=Path, help="write each sample's game battle file into this folder")
    parser.add_argument("--arenas", type=Path, help="write the samples as an arenas.json file")
    parser.add_argument("--sim", action="store_true", help="build a simulator batch of the samples and fight it out (torch)")
    parser.add_argument("--sim-s", type=float, default=900.0, help="--sim: battle time limit, s")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    first = (G.TRAIN_SEEDS if args.split == "train" else G.EVAL_SEEDS).start + args.start
    samples = [G.battle(first + i) for i in range(args.samples)]
    for a in samples:
        print(describe(a))
    if args.xml:
        args.xml.mkdir(parents=True, exist_ok=True)
        for a in samples:
            path = args.xml / f"{a['name']}.xml"
            path.write_text(export.scenario_xml(a), encoding="utf-8", newline="\n")
        print("battle files in", args.xml)
    if args.arenas:
        print("arenas in", export.write_arenas(samples, args.arenas))
    if args.sim:
        from tools.nn.sim import scenario as sim_scenario
        armies = export.to_sim(samples, "attack")
        st = sim_scenario.build(armies, per_side=G.MAX_UNITS + 1)
        side = st.u["side"]
        print(f"simulator batch: {tuple(side.shape)}, units per side "
              f"{(side == 1).sum(1).tolist()} / {(side == 2).sum(1).tolist()}, bounds {st.bounds}")
        from tools.nn.sim import battle, replay
        battle.run(st, replay.nearest_attack, until_s=args.sim_s, every_s=1.0)
        print(f"every unit attacks the nearest enemy, {args.sim_s} s at most: winners {st.winner.tolist()}, "
              f"over {st.done.tolist()}, time {[round(float(t)) for t in st.t]} s")
    if args.stats:
        arenas = [G.battle(first + args.samples + i) for i in range(args.stats)]
        for line in stats(arenas, G.default().pools):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
