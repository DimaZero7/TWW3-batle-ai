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
        geometry=False, lab=(), shape_log=False):
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
    geo = {"near": 0, "inside": 0, "steps": 0, "near_by_s": [0] * 10, "samples_by_s": [0] * 10,
           "blows_front_flank_rear": [0, 0, 0]}
    lab_d = json.loads(json.dumps(hook.LAB))
    for kv in lab:
        # section.key=json over the lab's numbers (lab.json), as --set over the simulator's
        k, v = kv.split("=", 1)
        *path, last = k.split(".")
        d = lab_d
        for p in path:
            d = d.setdefault(p, {})
        d[last] = json.loads(v)
    try:
        with hook.installed(lab=lab_d, seed=seed) as h:
            def batch(job):
                if places:
                    from tools.nn.fighters.places import FromRecording
                    h.on_reset = FromRecording(job[0], job[4])
                return orig_batch(job)
            stepreplay._batch = batch
            if geometry:
                h.listeners.append(lambda st, F: _geometry(st, F, copies, geo))
            shapes = {}
            if shape_log:
                h.listeners.append(lambda st, F: _shape(st, F, copies, h, shapes))
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
    if shape_log:
        import numpy as np
        Path(out).mkdir(parents=True, exist_ok=True)
        for run, d in shapes.items():
            rows = sorted(d["f"])
            np.savez(Path(out) / f"{run}_shape.npz", rows=np.array(rows), feats=np.stack([d["f"][r] for r in rows]),
                     names=np.array(d["names"]))
        log(f"shape features at the windows' 5th and 10th second: {Path(out)}/<run>_shape.npz")
    if geometry:
        share = geo["inside"] / max(geo["near"], 1)
        log(f"lab geometry: men within reach of an enemy {geo['near']}, nearest enemy inside their own formation "
            f"{share:.3f}; men within reach a battle by the window's second: "
            + " ".join(f"{a / max(n, 1):.0f}" for a, n in zip(geo["near_by_s"], geo["samples_by_s"]))
            + f"; swings front / flank / rear {geo['blows_front_flank_rear']}")
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "geometry.json").write_text(json.dumps(dict(geo, inside_share=share)), encoding="utf-8")


def _shape(st, F, copies, h, shapes):
    """shape.unit_features of the lab's men (copy mean) at the 5th and the 10th second of each window (the replay from
    the places: h.on_reset knows the batch's runs and their slots)."""
    import numpy as np
    from tools.nn.fighters import shape, state as S
    t = float(st.t.min())
    if abs(t - round(t)) > 1e-6 or round(t) % 5 or h.on_reset is None:
        return
    row = int(round(t))
    u = st.u
    px, pz = S.positions(F, u)
    live = S.alive(F)
    feats = shape.unit_features(px, pz, F.uid, live, S.ugather(u, F, "side").long(), u["b"], u["men0"], u["hp"],
                                u["men"].shape[1]).cpu().numpy()
    for k, r in enumerate(h.on_reset.runs):
        bs = slice(k * copies, (k + 1) * copies)
        if bool(st.done[bs].all()):
            continue
        d = shapes.setdefault(r["name"], {"f": {}, "names": [None] * feats.shape[1]})
        for nm, n in r["slot"].items():
            d["names"][n] = nm
        d["f"][row] = feats[bs].mean(0)


def _geometry(st, F, copies, geo, every_steps=4):
    """soldiersdata.geometry on the lab's men of each battle's first copy, every every_steps steps."""
    import numpy as np
    from tools.nn.fighters import soldiersdata, state as S
    geo["steps"] += 1
    k_ = getattr(F, "dir_k", None)
    if k_ is not None:
        for j in range(3):
            geo["blows_front_flank_rear"][j] += int((k_ == j).sum())
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
    ap.add_argument("--lab", action="append", default=[], help="section.key=json over tools/nn/fighters/lab.json")
    ap.add_argument("--shape-log", action="store_true",
                    help="with --places: the lab's shape.unit_features at each window's 5th and 10th second")
    a = ap.parse_args(argv)
    names = list(a.runs)
    if a.runs_from:
        names += sorted(p.stem for p in Path(a.runs_from).glob("*.npz"))
    if a.limit:
        names = names[:a.limit]
    runs = [project.BUILD / "nn-arena" / "runs" / n for n in names]
    run(runs, a.out, rect=a.rect, copies=a.copies, device=a.device, chunk=a.chunk, sets=a.set, places=a.places,
        geometry=a.geometry, lab=a.lab, shape_log=a.shape_log)


if __name__ == "__main__":
    main()
