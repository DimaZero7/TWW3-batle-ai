"""Battle viewer: records of battles and the page that plays them.

    python -m tools.viewer game <run dir> [--out DIR] [--preset approach|fire|debug]
    python -m tools.viewer sim <army> [--out DIR] [--preset ...]
    python -m tools.viewer compare <run dir> [--army NAME] [--out DIR] [--preset ...]
    python -m tools.viewer logistics <army> [--out DIR]
    python -m tools.viewer page <record.json> [<record.json> ...] --out <file.html> [--preset ...]

'game' turns a game run (build/<target>/runs/<run>) into DIR/record.json and
DIR/viewer.html (default research/analysis/viewer/<run>/); 'sim' does the same
for a simulated army (config/armies/<army>, default .../viewer/sim-<army>/);
'compare' puts the game run and the simulation of its army side by side, on one
clock. 'logistics' walks every step of the army round an obstacle in queues
(tools/sim/logistics.py): with the queue and all at once side by side, one
page per step (default .../viewer/logistics-<army>/). 'page' puts any records
side by side. Presets: approach, modules (what the AI's modules see), fire,
logistics, debug; --lang ru|en is the page's first language (it has a switch).
The page opens as a file, without a server.
"""
import argparse
import json
import sys
from pathlib import Path

from tools import config as project
from tools.viewer import game, logistics, record, sim

OUT = project.ROOT / "research" / "analysis" / "viewer"
PRESETS = ("approach", "modules", "fire", "logistics", "debug")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("game")
    g.add_argument("run", type=Path)
    g.add_argument("--out", type=Path)
    g.add_argument("--idle-s", type=float, default=5, help="cut still stretches longer than this (0: keep all)")
    s = sub.add_parser("sim")
    s.add_argument("army")
    s.add_argument("--out", type=Path)
    c = sub.add_parser("compare")
    c.add_argument("run", type=Path)
    c.add_argument("--army", help="default: the army of the run")
    c.add_argument("--out", type=Path)
    c.add_argument("--idle-s", type=float, default=5)
    lg = sub.add_parser("logistics")
    lg.add_argument("army")
    lg.add_argument("--out", type=Path)
    p = sub.add_parser("page")
    p.add_argument("records", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    for x in (g, s, c, lg, p):
        x.add_argument("--preset", choices=PRESETS, default="logistics" if x is lg else "approach")
        x.add_argument("--lang", choices=("ru", "en"), default="ru")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "page":
        recs = [json.loads(path.read_text(encoding="utf-8")) for path in args.records]
        print("written", record.page(recs, args.out, args.preset, args.lang))
        return 0
    if args.command == "logistics":
        out = args.out or OUT / f"logistics-{args.army}"
        steps = logistics.convert(args.army)
        if not steps:
            print(f"{args.army}: no step goes round an obstacle")
            return 1
        for n, pair in enumerate(steps, start=1):
            for r in pair:
                record.write(r, out / f"record-{r['meta']['name']}.json")
            print("written", record.page(list(pair), out / f"step-{n}.html", args.preset, args.lang))
        return 0
    recs = []
    if args.command in ("game", "compare"):
        recs.append(game.convert(args.run, idle_ms=int(args.idle_s * 1000)))
    if args.command in ("sim", "compare"):
        recs.append(sim.convert(args.army or recs[0]["meta"]["army"]))
    name = {"game": lambda: args.run.name, "sim": lambda: f"sim-{args.army}",
            "compare": lambda: f"{args.run.name}-vs-sim"}[args.command]()
    out = args.out or OUT / name
    for r in recs:
        record.write(r, out / f"record-{r['meta']['source']}.json")
        m = r["meta"]
        print(f"{m['title']['en']}: {len(r['frames'])} frames, {len(r['events'])} events, {m['duration_ms'] / 1000:.0f} s"
              + (f" (cut {sum(x['cut_ms'] for x in m['skipped']) / 1000:.0f} s still)" if m["skipped"] else ""))
    print("written", record.page(recs, out / "viewer.html", args.preset, args.lang))
    return 0


if __name__ == "__main__":
    sys.exit(main())
