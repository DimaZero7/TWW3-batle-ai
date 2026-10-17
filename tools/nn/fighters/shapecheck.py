"""The shape features (shape.unit_features) of the lab against the game's on the same battles: the lab replayed from the
game's soldiers' places (replay --places --shape-log) 5 and 10 s into each window, the game's own at those seconds, and
the game's at the window's start (how much the shape changes by itself in those seconds - what the lab is up against).

  python -m tools.nn.fighters.shapecheck build/fighters2/shape --runs <run> [...]
  python -m tools.nn.fighters.shapecheck --time --runs <run> [--threads 4]   # the companion's cost a reading (CPU)

Per feature: mean |lab - game| and mean |game at the start - game|, the correlation of lab and game over units, by
horizon (5 / 10 s). Units: not lords, 10+ men in the game and the lab. Needs torch."""
import argparse
from pathlib import Path

import numpy as np

from tools import config as project
from tools.nn.fighters import shape, soldiersdata


def game_features(g, places, row):
    """[N, F] the game's shape.unit_features at recording row `row` (None without a record near it)."""
    rec = places.at(float(g.t[row]))
    if rec is None:
        return None
    men0 = np.nan_to_num(g.f["men"][0])
    return shape.features_from_rows(rec, list(g.names), np.asarray(g.side), g.f["b"][row], men0,
                                    g.f["hp"][row]).numpy()


def check(out_dir, runs):
    from tools.nn import gamedata
    names = shape.FEATURES + tuple(f"grid{i}" for i in range(shape.GRID * shape.GRID))
    acc = {hz: {"lab": [], "game": [], "start": []} for hz in (5, 10)}
    for run in runs:
        d = project.BUILD / "nn-arena" / "runs" / run
        z = np.load(Path(out_dir) / f"{run}_shape.npz", allow_pickle=True)
        g = gamedata.load(d)
        pl = soldiersdata.load(d)
        slot = {nm: k for k, nm in enumerate(z["names"]) if nm is not None}
        idx = [slot[nm] for nm in g.names]
        for row, lab in zip(z["rows"], z["feats"]):
            hz = 5 if row % 10 == 5 else 10
            row0 = row - hz
            if row >= len(g.t) or row0 < 0:
                continue
            gm, g0 = game_features(g, pl, row), game_features(g, pl, row0)
            if gm is None or g0 is None:
                continue
            lab_u = lab[idx]
            men = g.f["men"][row]
            ok = (np.nan_to_num(men) >= 10) & (g.f["men"][0] > 1) & (lab_u[:, 1] > 0) & (gm[:, 1] > 0)
            acc[hz]["lab"].append(lab_u[ok])
            acc[hz]["game"].append(gm[ok])
            acc[hz]["start"].append(g0[ok])
    print("feature | 5 s: |lab-game| |start-game| corr(lab,game) | 10 s: the same | game mean / lab mean (10 s)")
    for j, nm in enumerate(names):
        cells = []
        for hz in (5, 10):
            L, G, S0 = (np.concatenate(acc[hz][k])[:, j] for k in ("lab", "game", "start"))
            c = np.corrcoef(L, G)[0, 1] if L.std() > 0 and G.std() > 0 else np.nan
            cells.append(f"{np.abs(L - G).mean():7.3f} {np.abs(S0 - G).mean():7.3f} {c:5.2f}")
        G = np.concatenate(acc[10]["game"])[:, j]
        L = np.concatenate(acc[10]["lab"])[:, j]
        print(f"  {nm:12} | " + " | ".join(cells) + f" | {G.mean():7.3f} / {L.mean():7.3f}")
    print("unit-seconds:", {hz: int(sum(len(a) for a in acc[hz]["lab"])) for hz in (5, 10)})


def timing(runs, threads=4, every=20):
    """The companion's cost: shape.features_from_rows on recorded readings (as the bridge sends them), CPU, ms; and a
    reading's size in the state document (JSON bytes)."""
    import time
    import torch
    from tools.nn import gamedata
    torch.set_num_threads(threads)
    ms, men, size = [], [], []
    for run in runs:
        d = project.BUILD / "nn-arena" / "runs" / run
        g = gamedata.load(d)
        pl = soldiersdata.load(d)
        with open(d / "events.jsonl", encoding="utf-8") as f:
            size += [len(line) for line in f if '"nn_soldiers"' in line][::every]
        for t, rec in list(zip(pl.times, pl.rows))[::every]:
            i = int(np.argmin(np.abs(g.t - t)))
            t0 = time.perf_counter()
            shape.features_from_rows(rec, list(g.names), np.asarray(g.side), g.f["b"][i], g.f["men"][0], g.f["hp"][i])
            ms.append((time.perf_counter() - t0) * 1000)
            men.append(sum(len(p) for p in rec.values()))
    ms, men = np.array(ms[1:]), np.array(men[1:])
    print(f"{len(ms)} readings, {men.mean():.0f} men (max {men.max()}); features {np.median(ms):.1f} ms median, "
          f"{np.percentile(ms, 95):.1f} p95 ({threads} threads); a reading in JSON {np.mean(size) / 1024:.0f} KB")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", nargs="?", help="the replay's out folder (replay --places --shape-log)")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--time", action="store_true", help="the companion's cost instead")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    if a.time:
        timing(a.runs, a.threads)
    else:
        check(a.out, a.runs)


if __name__ == "__main__":
    main()
