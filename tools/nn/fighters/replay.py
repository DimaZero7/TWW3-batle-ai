"""The step replay (tools/nn/sim/stepreplay.py) with the lab's men: every 10 s the simulator is put back into the
recorded state of a game battle and plays 9 s on the recorded orders - here with the men inside (hook.py). The men
carry over from the lab's own last window: their places as offsets from the unit's centre, the spread of their wounds
(state.conform puts them to the recording's men and health at each window start: the most wounded die first, the
missing health comes off the living by the wounds they have).

  python -m tools.nn.fighters.replay --runs-from build/contactgap/sr_free_rect --out build/fighters/sr_fighters
  python -m tools.nn.fighters.replay --runs-from ... --out build/fighters/sr_rect --rect    # the same, no men
  python -m tools.nn.fighters.replay --runs <run> --places --geometry --out ...  # each window from the game's own
      soldiers' places (a battle recorded with --soldiers-every 1, places.py), the men-in-reach geometry (soldiersdata)

Output: stepreplay's npz per battle (tools/nn/fighters/card.py compares them). Needs torch."""
import argparse
import json
from pathlib import Path

from tools import config as project


def run(runs, out, rect=False, copies=4, device="cpu", chunk=12, seed=0, log=print, sets=(), places=False,
        geometry=False):
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
    orig_batch = stepreplay._batch
    geo = {"near": 0, "inside": 0, "steps": 0, "near_by_s": [0] * 10, "samples_by_s": [0] * 10}
    try:
        with hook.installed(seed=seed) as h:
            def batch(job):
                if places:
                    from tools.nn.fighters.places import FromRecording
                    h.on_reset = FromRecording(job[0], job[4])
                return orig_batch(job)
            stepreplay._batch = batch
            if geometry:
                h.listeners.append(lambda st, F: _geometry(st, F, copies, geo))
            if places:
                from tools.nn.fighters.places import installed as places_installed
                with places_installed():
                    stepreplay.run(runs, out, H=10, copies=copies, device=device, chunk=chunk, workers=1, log=log,
                                   sets=sets)
            else:
                stepreplay.run(runs, out, H=10, copies=copies, device=device, chunk=chunk, workers=1, log=log,
                               sets=sets)
            log(f"projectiles traced {h.stats['traced']:.0f} of {h.stats['wanted']:.0f}")
            if places and h.on_reset is not None:
                log(f"places from the recording (last batch): {h.on_reset.stats}")
    finally:
        stepreplay._params = orig
        stepreplay._batch = orig_batch
    if geometry:
        share = geo["inside"] / max(geo["near"], 1)
        log(f"lab geometry: men within reach of an enemy {geo['near']}, nearest enemy inside their own formation "
            f"{share:.3f}; men within reach a battle by the window's second: "
            + " ".join(f"{a / max(n, 1):.0f}" for a, n in zip(geo["near_by_s"], geo["samples_by_s"])))
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "geometry.json").write_text(json.dumps(dict(geo, inside_share=share)), encoding="utf-8")


def _geometry(st, F, copies, geo, every_steps=4):
    """soldiersdata.geometry on the lab's men of each battle's first copy, every every_steps steps."""
    import numpy as np
    from tools.nn.fighters import soldiersdata, state as S
    geo["steps"] += 1
    if geo["steps"] % every_steps:
        return
    u = st.u
    px, pz = S.positions(F, u)
    live = S.alive(F)
    px, pz, live, uid = px.cpu().numpy(), pz.cpu().numpy(), live.cpu().numpy(), F.uid.cpu().numpy()
    ux, uz, ub, side = (u[k].cpu().numpy() for k in ("x", "z", "b", "side"))
    for b in range(0, F.B, copies):
        if bool(st.done[b]):
            continue
        pos, sides, cen, bea = {}, {}, {}, {}
        for n in range(u["men"].shape[1]):
            if side[b, n] <= 0:
                continue
            m = live[b] & (uid[b] == n)
            if not m.any():
                continue
            pos[n] = np.stack([px[b][m], pz[b][m]], 1)
            sides[n], cen[n], bea[n] = int(side[b, n]), (ux[b, n], uz[b, n]), float(ub[b, n])
        a, i = soldiersdata.geometry(pos, sides, cen, bea)
        geo["near"] += a
        geo["inside"] += i
        k = int(float(st.t[b]) % 10)          # the second of the step replay's window (0: its start)
        geo["near_by_s"][k] += a
        geo["samples_by_s"][k] += 1


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
    ap.add_argument("--places", action="store_true", help="each window from the game's recorded soldiers' places")
    ap.add_argument("--geometry", action="store_true", help="the lab's men-in-reach geometry (soldiersdata.geometry)")
    a = ap.parse_args(argv)
    names = list(a.runs)
    if a.runs_from:
        names += sorted(p.stem for p in Path(a.runs_from).glob("*.npz"))
    if a.limit:
        names = names[:a.limit]
    runs = [project.BUILD / "nn-arena" / "runs" / n for n in names]
    run(runs, a.out, rect=a.rect, copies=a.copies, device=a.device, chunk=a.chunk, sets=a.set, places=a.places,
        geometry=a.geometry)


if __name__ == "__main__":
    main()
