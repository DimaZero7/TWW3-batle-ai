"""The soldiers' places a recorded battle carries (entries.nn_arena with config.soldiers_every: nn_soldiers, every
soldier's x, z of every unit; tools.build nn-arena --soldiers-every 1) and the geometry the fighters lab is checked by
(docs/en/training/fighters_lab.md, "Replay from the game's places").

  load(run_dir) -> Places: times (s, the recording's clock), per record {unit name: [k, 2] m}
  Places.at(t) -> the record nearest to t (s) within max_gap_s, else None
  geometry(pos, sides, centres, bearings) -> the men within reach of an enemy and how many of them have their nearest
      enemy inside their own formation (the box of their own unit's men in its frame, 1 m in from each edge)

  python -m tools.nn.fighters.soldiersdata build/nn-arena/runs/<run> [...]   # the game's geometry per battle

numpy only (no torch)."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

REACH_M = 3.15          # the formed combat distance 2.5 m + a man's radius 0.65 (the lab's reach, melee.py)
BOX_IN_M = 1.0


class Places:
    def __init__(self, times, rows):
        self.times = np.asarray(times, float)
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def at(self, t, max_gap_s=0.6):
        if not len(self.rows):
            return None
        i = int(np.argmin(np.abs(self.times - t)))
        return self.rows[i] if abs(self.times[i] - t) <= max_gap_s else None


def load(run_dir):
    """Places of a run (an empty Places when it recorded none)."""
    times, rows = [], []
    with open(Path(run_dir) / "events.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"nn_soldiers"' not in line:
                continue
            r = json.loads(line)
            if r.get("event") != "nn_soldiers":
                continue
            row = {}
            for u in r.get("units") or []:
                xz = u.get("xz_dm")
                if xz:
                    row[u["n"]] = np.asarray(xz, float).reshape(-1, 2) / 10.0
            times.append(r["t"] / 1000.0)
            rows.append(row)
    return Places(times, rows)


def geometry(pos, sides, centres, bearings, reach=REACH_M, box_in=BOX_IN_M):
    """pos {unit: [k, 2]}, sides {unit: 1 | 2}, centres {unit: (x, z)}, bearings {unit: deg}: (men within reach of an
    enemy man, of them those whose nearest enemy stands inside their own unit's box) - the box: the extent of the
    unit's own men along and across its facing from its centre, box_in metres in from each edge. Units of one man
    (lords) are left out as strikers."""
    names = [n for n in pos if len(pos[n])]
    if not names:
        return 0, 0
    P = np.concatenate([pos[n] for n in names])
    owner = np.concatenate([np.full(len(pos[n]), k) for k, n in enumerate(names)])
    side = np.array([sides[n] for n in names])[owner]
    near_n = inside_n = 0
    for k, n in enumerate(names):
        mine = pos[n]
        if len(mine) < 2:
            continue
        foe = side != sides[n]
        if not foe.any():
            continue
        F = P[foe]
        d = np.sqrt(((mine[:, None, :] - F[None, :, :]) ** 2).sum(-1))
        j = d.argmin(1)
        ok = d[np.arange(len(mine)), j] <= reach
        if not ok.any():
            continue
        b = math.radians(bearings[n])
        fx, fz = math.sin(b), math.cos(b)
        c = np.asarray(centres[n], float)
        rel = mine - c
        al, ac = rel[:, 0] * fx + rel[:, 1] * fz, rel[:, 0] * fz - rel[:, 1] * fx
        tr = F[j[ok]] - c
        tal, tac = tr[:, 0] * fx + tr[:, 1] * fz, tr[:, 0] * fz - tr[:, 1] * fx
        ins = ((tal < al.max() - box_in) & (tal > al.min() + box_in) & (tac < ac.max() - box_in)
               & (tac > ac.min() + box_in))
        near_n += int(ok.sum())
        inside_n += int(ins.sum())
    return near_n, inside_n


def battle_geometry(run_dir, every_s=1.0, by_row=None):
    """The game's own geometry over a recorded battle: (men within reach summed over records, of them inside);
    by_row: a dict filled {row % 10: [men within reach, records]} (the step replay's second of its window)."""
    from tools.nn import gamedata
    g = gamedata.load(run_dir)
    pl = load(run_dir)
    side = {n: int(s) for n, s in zip(g.names, g.side)}
    near = ins = 0
    last = -1e9
    for t, row in zip(pl.times, pl.rows):
        if t - last < every_s - 1e-6:
            continue
        last = t
        i = int(np.argmin(np.abs(g.t - t)))
        cen = {n: (g.f["x"][i, k], g.f["z"][i, k]) for k, n in enumerate(g.names)}
        bea = {n: g.f["b"][i, k] for k, n in enumerate(g.names)}
        live = {n: p for n, p in row.items() if n in side and np.isfinite(bea[n]) and np.isfinite(cen[n][0])}
        a, b = geometry(live, side, cen, bea)
        near, ins = near + a, ins + b
        if by_row is not None:
            acc = by_row.setdefault(i % 10, [0, 0])
            acc[0] += a
            acc[1] += 1
    return near, ins


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    for r in a.runs:
        pl = load(r)
        rows = {}
        near, ins = battle_geometry(r, by_row=rows)
        print(f"{Path(r).name}: {len(pl)} records; men within reach of an enemy {near}, nearest enemy inside their own "
              f"formation {ins / max(near, 1):.3f}; men within reach a record by row % 10: "
              + " ".join(f"{rows[k][0] / max(rows[k][1], 1):.0f}" for k in sorted(rows)))


if __name__ == "__main__":
    main()
