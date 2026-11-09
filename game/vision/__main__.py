"""python -m game.vision <run>... [--out build/vision] [--frames 6] [--size 360] [--cols 3] [--png-only | --txt-only]

A run: a folder or a run name (game.battle.run_dir: build/nn-arena/runs, data/nn-arena/runs, data/human/runs)."""
import argparse
import json
import sys

import game
from game.vision import record, render


def main(argv=None):
    a = argparse.ArgumentParser(prog="python -m game.vision")
    a.add_argument("runs", nargs="+")
    a.add_argument("--out", default=str(game.BUILD / "vision"))
    a.add_argument("--frames", type=int, default=6)
    a.add_argument("--size", type=int, default=360, help="a frame's longer side, px")
    a.add_argument("--cols", type=int, default=3)
    g = a.add_mutually_exclusive_group()
    g.add_argument("--png-only", action="store_true")
    g.add_argument("--txt-only", action="store_true")
    o = a.parse_args(argv)
    for run in o.runs:
        res = render(record.load(run), o.out, frames=o.frames, size=o.size, cols=o.cols, png=not o.txt_only,
                     txt=not o.png_only)
        print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
