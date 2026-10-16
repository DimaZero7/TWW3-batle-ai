"""The step replay (tools/nn/sim/stepreplay.py) with the lab's men: every 10 s the simulator is put back into the
recorded state of a game battle and plays 9 s on the recorded orders - here with the men inside (hook.py). The men
carry over from the lab's own last window: their places as offsets from the unit's centre, the spread of their wounds
(state.conform puts them to the recording's men and health at each window start: the most wounded die first, the
missing health comes off the living by the wounds they have).

  python -m tools.nn.fighters.replay --runs-from build/contactgap/sr_free_rect --out build/fighters/sr_fighters
  python -m tools.nn.fighters.replay --runs-from ... --out build/fighters/sr_rect --rect    # the same, no men

Output: stepreplay's npz per battle (tools/nn/fighters/card.py compares them). Needs torch."""
import argparse
from pathlib import Path

from tools import config as project


def run(runs, out, rect=False, copies=4, device="cpu", chunk=12, seed=0, log=print, sets=()):
    from tools.nn.sim import stepreplay
    if rect:
        stepreplay.run(runs, out, H=10, copies=copies, device=device, chunk=chunk, workers=1, log=log, sets=sets)
        return
    from tools.nn.fighters import hook
    orig = stepreplay._params

    def params(sets=()):
        # the men roll their own blows and hits: no noise on the units' HP (noise.blows)
        return orig(sets).with_cal("noise", blows=False)
    stepreplay._params = params
    try:
        with hook.installed(seed=seed) as h:
            stepreplay.run(runs, out, H=10, copies=copies, device=device, chunk=chunk, workers=1, log=log, sets=sets)
            log(f"projectiles traced {h.stats['traced']:.0f} of {h.stats['wanted']:.0f}")
    finally:
        stepreplay._params = orig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--runs-from", default="", help="a folder of step-replay npz: its battles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rect", action="store_true", help="the simulator alone (the rectangle)")
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], help="section.key=json over config/nn/sim.json (stepreplay)")
    a = ap.parse_args(argv)
    names = list(a.runs)
    if a.runs_from:
        names += sorted(p.stem for p in Path(a.runs_from).glob("*.npz"))
    if a.limit:
        names = names[:a.limit]
    runs = [project.BUILD / "nn-arena" / "runs" / n for n in names]
    run(runs, a.out, rect=a.rect, copies=a.copies, device=a.device, chunk=a.chunk, sets=a.set)


if __name__ == "__main__":
    main()
