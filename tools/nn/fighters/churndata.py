"""Does a man in reach of the enemy stay in reach (docs/en/training/fighters_lab.md, "Killed per health: the window
reset and the men's turnover")? The game against the lab, the same seconds, the same way (numpy, no torch).

A man in reach: within 3.15 m (centre to centre) of an enemy man, at a step-replay window's second 1..7; 3 s later -
still in reach (among the survivors). The game's men are followed by their ManList slot (the soldiers' places once a
second, soldiersdata.py): a dead man's slot takes the list's last man, so a slot is trusted ('sure') only while it never
steps more than JUMP m in a second and stays inside the list. The lab (replay.py --places --churn: churn.npz) knows its
men; the same JUMP filter is put on its own 1 s steps. Both binned by the GAME's recorded unit centre shift over the 3 s;
routing units apart.

JUMP matters: 2.5 m (the first count) drops half the game's men in reach of units whose centre moves > 6 m in 3 s - the
men who moved with it - and leaves those who stood: 'still in reach' 0.89 there against 0.79 at 5 m.

  python -m tools.nn.fighters.churndata build/nn-arena/runs/<run> ... --lab build/night5/base [--jump 5]
"""
import argparse
import json
from pathlib import Path

import numpy as np

REACH_M = 3.15
LAG = 3
BINS = (0.5, 1.5, 3.0, 6.0, 1e9)


def in_reach(rec, side):
    """{unit: [k] bool} each man of a unit of 2+ within REACH_M of an enemy man (rec: unit -> [k, 2] places)."""
    names = [n for n in rec if n in side and len(rec[n]) >= 2]
    out = {}
    if not names:
        return out
    P = np.concatenate([rec[n] for n in names])
    S = np.concatenate([np.full(len(rec[n]), side[n]) for n in names])
    for n in names:
        foe = P[S != side[n]]
        if not len(foe):
            continue
        d = np.sqrt(((rec[n][:, None, :] - foe[None, :, :]) ** 2).sum(-1)).min(1)
        out[n] = d <= REACH_M
    return out


def sure_slots(seq, jump):
    """[k] bool: the slots of a unit's list (seq: its [k_i, 2] places at t, t+1 ... ; None: missing) that hold the
    same man all through - no step over `jump` m in a second, never past the list's end."""
    ok = np.ones(len(seq[0]), bool)
    prev = seq[0]
    for cur in seq[1:]:
        if cur is None:
            return np.zeros(len(ok), bool)
        m = min(len(prev), len(cur))
        step = np.full(len(ok), np.inf)
        step[:m] = np.sqrt(((prev[:m] - cur[:m]) ** 2).sum(-1))
        ok &= step <= jump
        prev = cur
    return ok


def centre_shift(g):
    """[T, N] the recorded unit centre's shift over LAG s from each row (NaN past the end)."""
    cs = np.full((len(g.t), len(g.names)), np.nan)
    cs[:-LAG] = np.hypot(g.f["x"][LAG:] - g.f["x"][:-LAG], g.f["z"][LAG:] - g.f["z"][:-LAG])
    return cs


def _bin(d):
    return next(j for j, hi in enumerate(BINS) if d < hi)


def game(run_dirs, jump, acc):
    """acc[(routing, bin)] += [men in reach, sure, still] over the window seconds 1..7 of each battle."""
    from tools.nn import gamedata
    from tools.nn.fighters import soldiersdata as SD
    for d in run_dirs:
        g = gamedata.load(d)
        pl = SD.load(d)
        cs = centre_shift(g)
        side = {n: g.side[i] for i, n in enumerate(g.names)}
        T = len(g.t)
        recs = {i: pl.at(float(g.t[i])) for i in range(T)}
        for s0 in range(0, T - 10, 10):
            for i in range(s0 + 1, s0 + 8):
                seq = [recs.get(i + s) for s in range(LAG + 1)]
                if any(r is None for r in seq):
                    continue
                r0, r1 = in_reach(seq[0], side), in_reach(seq[-1], side)
                for n, inr in r0.items():
                    k = g.names.index(n)
                    if n not in r1 or not np.isfinite(cs[i, k]):
                        continue
                    ok = inr & sure_slots([r.get(n) for r in seq], jump)
                    still = np.zeros(len(inr), bool)
                    m = min(len(inr), len(r1[n]))
                    still[:m] = r1[n][:m]
                    a = acc.setdefault((bool(g.f["r"][i, k]), _bin(cs[i, k])), np.zeros(3))
                    a += [inr.sum(), ok.sum(), (ok & still).sum()]
    return acc


def lab(lab_dir, run_dirs, jump, acc, copies=4):
    """The same from a replay's churn.npz (replay.py --churn): its rows (batch, battle row, second, unit, still, shift,
    jump, centre shift, routing, ...) binned by the game's centre shift of the unit at that second."""
    from tools.nn import gamedata
    z = np.load(Path(lab_dir) / "churn.npz")
    meta = json.loads(str(z["meta"]))
    r = z["rows"]
    games = {Path(d).name: gamedata.load(d) for d in run_dirs}
    shifts = {k: centre_shift(g) for k, g in games.items()}
    for bt, runs in enumerate(meta):
        rb = r[r[:, 0] == bt]
        for k, rr in enumerate(runs):
            if rr["name"] not in games:
                continue
            q = rb[(rb[:, 1] // copies) == k]
            names = list(games[rr["name"]].names)
            inv = {v: names.index(nm) for nm, v in rr["slot"].items()}
            gi = np.array([inv.get(int(x), -1) for x in q[:, 3]])
            cs = shifts[rr["name"]]
            d = np.where(gi >= 0, cs[np.minimum(q[:, 2].astype(int), len(cs) - 1), gi.clip(0)], np.nan)
            for j, (lo, hi) in enumerate(zip((-1.0,) + BINS[:-1], BINS)):
                for rout in (False, True):
                    s = np.isfinite(d) & (d >= lo) & (d < hi) & ((q[:, 8] > 0) == rout)
                    ok = s & (q[:, 6] <= jump)
                    a = acc.setdefault((rout, j), np.zeros(3))
                    a += [s.sum(), ok.sum(), (ok & (q[:, 4] > 0)).sum()]
    return acc


def table(acc, rout=False):
    """(still in reach over all bins, [per bin (still, sure share)]) for routing or not."""
    rows = [acc.get((rout, j), np.zeros(3)) for j in range(len(BINS))]
    tot = sum(rows)
    return tot[2] / max(tot[1], 1), [(a[2] / max(a[1], 1), a[1] / max(a[0], 1)) for a in rows]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="recorded battles with the soldiers' places")
    ap.add_argument("--lab", action="append", default=[], help="a replay folder with churn.npz (replay.py --churn)")
    ap.add_argument("--jump", type=float, default=5.0)
    a = ap.parse_args(argv)
    cols = [("game", game(a.runs, a.jump, {}))] + [(Path(d).name, lab(d, a.runs, a.jump, {})) for d in a.lab]
    print(f"still in reach {LAG} s later (sure share), by the game's unit centre shift < "
          + " / ".join(f"{b:g}" for b in BINS[:-1]) + " / more m; jump filter " + f"{a.jump:g} m")
    for rout in (False, True):
        for name, acc in cols:
            allv, per = table(acc, rout)
            print(f"  {'routing' if rout else 'standing':8} {name:12} all {allv:.3f} | "
                  + " ".join(f"{s:.3f} ({k:.2f})" for s, k in per))


if __name__ == "__main__":
    main()
