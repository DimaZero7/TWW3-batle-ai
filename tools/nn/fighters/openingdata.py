"""The opening of a fight man by man: when does a man reaching an enemy strike first (the swing probe's battle 2,
units of 25; numpy, no torch).

  python -m tools.nn.fighters.openingdata build/charge-probe/runs/<swing battle 2 run>

Per side and 0.5 s after the contact: the game's hits (the drops of the other's health, swingdata) against three
clocks for a man from the moment he first stands within reach (the running maximum of the men in reach counts the men
coming in):
  ready      he swings at once, then every interval (the lab's 'ready at once')
  random     his first swing at a random point of his interval (a steady stream p / interval from the start)
  delay D    he swings D s after he came within reach, then every interval
expected hits = p x swings. Cumulative to 1 / 2 / 3 / 5 / 10 s, game / model."""
import argparse
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp
from tools.nn.fighters import shape, swingdata

ENDS = (1.0, 2.0, 3.0, 5.0, 10.0)
DELAYS = (0.5, 1.0, 1.5, 2.0, 3.0)


def side(lane, who, oth, sk, tk):
    smp = lane["samples"]
    c = (lane.get("contacts") or {}).get(1)
    if c is None or not lane.get("men"):
        return None
    t = np.array([x[0] for x in smp])
    col = {"a": 1, "tg": 2}
    hp_o = np.array([float((x[col[oth]] or {}).get("hp") or np.nan) for x in smp])
    hit, _ = swingdata.mean_hit(sk, tk)
    d = np.nan_to_num(-np.diff(hp_o))
    n_hits = np.where(d > 0.5, np.maximum(1, np.round(d / hit)), 0)
    th = t[1:] - c
    # men in reach over time and the men coming in (running max)
    tm, nr = [], []
    for tt, fa, ft in lane["men"]:
        P = {"a": shape.xy(fa), "tg": shape.xy(ft)}
        if len(P[who]) and len(P[oth]):
            tm.append(tt - c)
            nr.append(int((shape.nearest(P[who], P[oth]) <= shape.REACH_M).sum()))
    tm, nr = np.array(tm), np.array(nr)
    keep = tm >= -1.0
    tm, nr = tm[keep], nr[keep]
    came = np.diff(np.concatenate([[0], np.maximum.accumulate(nr)]))       # men coming in at each sample
    p, interval = swingdata.p_hit(sk, tk)
    out = {"game": [], "ready": [], "random": [], **{f"delay{dl}": [] for dl in DELAYS}}
    for end in ENDS:
        out["game"].append(float(n_hits[(th >= -0.2) & (th < end)].sum()))
        t0 = tm[came > 0]
        k = came[came > 0]
        def swings(lag):
            s = np.where(end > t0 + lag, np.floor((end - t0 - lag) / interval) + 1, 0)
            return float(p * (k * s).sum())
        out["ready"].append(swings(0.0))
        out["random"].append(float(p * (k * np.clip(end - t0, 0, None)).sum() / interval))
        for dl in DELAYS:
            out[f"delay{dl}"].append(swings(dl))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    tot = {}
    for d in a.runs:
        for ln in cp.load_run(Path(d)):
            sp = ln["spec"]
            for who, oth, sk, tk in (("a", "tg", sp["a_key"], sp["t_key"]), ("tg", "a", sp["t_key"], sp["a_key"])):
                r = side(ln, who, oth, sk, tk)
                if not r:
                    continue
                print(sp.get("kind"), cp.SHORT[sk], ">", cp.SHORT[tk], "game", r["game"], "ready",
                      [round(x, 1) for x in r["ready"]])
                for k, v in r.items():
                    tot[k] = np.array(v) + tot.get(k, 0)
    print("cumulative hits to " + " / ".join(f"{e:g}" for e in ENDS) + " s after contact: game " +
          " / ".join(f"{x:.0f}" for x in tot["game"]))
    for k in tot:
        if k != "game":
            print(f"  {k:9} game / model " + " / ".join(f"{g / m:.2f}" if m else "-" for g, m in zip(tot["game"], tot[k])))


if __name__ == "__main__":
    main()
