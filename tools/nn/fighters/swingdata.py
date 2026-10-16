"""The blow rate man by man in the game: the swing probe (tools/nn/charge_probe.py plan 'swing': small units cut down
to 5 / 1 men, both attacking from 3 m; health, men and the soldiers' places every 0.1 s) read (numpy, no torch).

  python -m tools.nn.fighters.swingdata build/charge-probe/runs/<swing run>

For each side striking the other: its hits are the drops of the other's health between 0.1 s samples (a drop of k
mean hits counts k), the man-seconds its men stand within reach (3.15 m: the formed combat distance 2.5 + a man's
radius) of an enemy man; hits a second per man in reach against the two clocks:
  per swing    p / interval - every swing, hit or miss, an interval after the last (CA's schema: the interval is 'the
               smallest gap between the starts of attack animations')
  miss 0.5 s   p / (p x interval + 0.5) - the simulator's melee.miss_s
p = (35 + attack - defence) / 100 within 8-90 % (the passports; the skaven's Strength in Numbers +8 defence is off below
half health - these units are at 3-4 %)."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp
from tools.nn.fighters import shape


def mean_hit(sk, tk):
    s, t = cp.passport(sk), cp.passport(tk)
    a = t["armour"] / 100
    cut = 0.75 * a if a <= 1 else (2 - 1 / a - a / 4 if a < 2 else 1)
    return s["melee"]["ap_damage"] + s["melee"]["damage"] * (1 - cut), t["hp_per_man"]


def p_hit(sk, tk):
    s, t = cp.passport(sk)["melee"], cp.passport(tk)["melee"]
    return min(90, max(8, 35 + s["attack"] - t["defence"])) / 100, s["attack_interval_s"]


def side(lane, who, oth, key_s, key_t):
    smp = lane["samples"]
    t = np.array([x[0] for x in smp])
    col = {"a": 1, "tg": 2}
    hp_o = np.array([float((x[col[oth]] or {}).get("hp") or np.nan) for x in smp])
    c = (lane.get("contacts") or {}).get(1)
    if c is None:
        return None
    hit, hp_man = mean_hit(key_s, key_t)
    drops = -np.diff(hp_o)
    ok = np.isfinite(drops) & (t[1:] >= c - 0.2)
    d = np.where(ok, drops, 0.0)
    # a drop is whole hits: at least one; the hits of a killing blow are cut by the man's health (overkill)
    n_hits = np.where(d > 0.5, np.maximum(1, np.round(d / hit)), 0)
    events = int((d > 0.5).sum())
    # man-seconds within reach, from the places
    ms = 0.0
    ts = np.array([m[0] for m in lane["men"]])
    dt = float(np.median(np.diff(ts))) if len(ts) > 2 else 0.1
    for tm, fa, ft in lane["men"]:
        if tm < c - 0.2:
            continue
        P = {"a": shape.xy(fa), "tg": shape.xy(ft)}
        if len(P[who]) and len(P[oth]):
            ms += float((shape.nearest(P[who], P[oth]) <= shape.REACH_M).sum()) * dt
    p, interval = p_hit(key_s, key_t)
    return {"hits": float(n_hits.sum()), "events": events, "hp": float(d.sum()), "man_s": ms,
            "rate": float(n_hits.sum()) / ms if ms > 0 else None, "p": p, "interval": interval,
            "per_swing": p / interval, "miss05": p / (p * interval + 0.5), "mean_hit": hit,
            "gaps": [], "t0": c}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rows = []
    for d in a.runs:
        for ln in cp.load_run(Path(d)):
            sp = ln["spec"]
            for who, oth, sk, tk in (("a", "tg", sp["a_key"], sp["t_key"]), ("tg", "a", sp["t_key"], sp["a_key"])):
                r = side(ln, who, oth, sk, tk)
                if r:
                    r.update(kind=sp.get("kind"), striker=cp.SHORT[sk], target=cp.SHORT[tk])
                    rows.append(r)
    print("lane | striker > target | hits (drop events) | man-s in reach | hits/s a man | p / I | p / (pI + 0.5) | "
          "game / per swing | game / miss 0.5")
    for r in rows:
        if r["rate"] is None:
            continue
        print(f"{r['kind']:10} {r['striker']}>{r['target']:8} {r['hits']:5.0f} ({r['events']}) {r['man_s']:7.1f} "
              f"{r['rate']:.3f} {r['per_swing']:.3f} {r['miss05']:.3f} {r['rate'] / r['per_swing']:.2f} "
              f"{r['rate'] / r['miss05']:.2f}")
    H = sum(r["hits"] for r in rows if r["rate"])
    exp_swing = sum(r["per_swing"] * r["man_s"] for r in rows if r["rate"])
    exp_miss = sum(r["miss05"] * r["man_s"] for r in rows if r["rate"])
    print(f"all: hits {H:.0f}, per swing expects {exp_swing:.0f} (game / it {H / exp_swing:.2f}), miss 0.5 s expects "
          f"{exp_miss:.0f} ({H / exp_miss:.2f}); Poisson 95 % on the hits +-{1.96 * H ** 0.5 / H:.0%}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
