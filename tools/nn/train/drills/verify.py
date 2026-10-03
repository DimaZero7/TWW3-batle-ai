"""Check a drill in the simulator with our two scripts before it trains anything
(docs/en/training/training.md "Drills"): the naive script must lose clearly, the skilled one
win clearly, over generated battles of the drill (our side alternating 1 and 2), with the
training's randomised numbers (randomise.Spread()).

    bash tools/nn/dock.sh tools.nn.train.drills.verify --drill pincer --battles 256
    python -m tools.nn.train.drills.verify --drill pincer --battles 64 --device cpu --show 3
    bash tools/nn/dock.sh tools.nn.train.drills.verify --drill kiting --broad 1     # the broad frame only

--broad: the share of the battles from the drill's broad frame (default drills.BROAD; 0 the clean frame,
1 the broad one only); with both in the mix the summary also splits them ("clean", "broad").
Prints per script: win rate, gold trade ((enemy gold destroyed - own lost) / budget, reward.gold_sides),
mean battle seconds, timeouts; --show K: the first K battles' rosters and results.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

from tools.nn.sim import battle, scenario
from tools.nn.train import randomise, reward
from tools.nn.train import drills as D

PASS_NAIVE = 0.25        # the naive script wins at most this share ...
PASS_SKILLED = 0.75      # ... the skilled one at least this


def play(drill, script, n=256, seed=0, device="cpu", spread=randomise.Spread(), seeds=None, extra=None, ours_box=None,
         broad=None):
    """Battles of a drill with `script` on our side and the drill's enemy -> per-battle results:
    {"won", "trade", "seconds", "timeout", "ours", "broad"} (numpy) and the descriptions. broad: the share
    of the broad frame (drills.battles; default drills.BROAD)."""
    seeds = list(seeds if seeds is not None else range(seed, seed + n))
    pairs = D.battles(drill, seeds, broad=broad)
    descs = [p[0] for p in pairs]
    ours = torch.tensor([p[1] for p in pairs], device=device)
    if ours_box is not None:
        ours_box["ours"] = ours                  # (for an `extra` that needs our side per battle)
    st = scenario.build(descs, device=device)
    gen = torch.Generator(device=device).manual_seed(seed + 7)
    randomise.apply(st, torch.ones(st.B, dtype=torch.bool, device=device), spread, gen)
    from tools.nn.sim.params import load
    params = load()
    while not bool(st.done.all()):
        orders = D.merged(st, ours, script(st), drill.enemy(st))
        battle.step(st, orders, params, params.dt)
        st.u["lost_worst"] = reward.track(st.u)          # a loss counts once (reward.gold_lost)
        if extra is not None:
            extra(st)
    gold = reward.gold_sides(st.u).cpu().numpy()
    bud = reward.budget(st.u).cpu().numpy()
    o = ours.cpu().numpy()
    b = np.arange(st.B)
    own, en = gold[b, o - 1], gold[b, 2 - o]
    won = st.winner.cpu().numpy() == o
    t = st.t.cpu().numpy()
    return {"won": won, "trade": (en - own) / np.maximum(bud, 1e-9), "seconds": t, "timeout": t >= params.limit_s - 1e-6,
            "ours": o, "broad": np.array([bool(d.get("broad")) for d in descs])}, descs


def summary(res, sel=None):
    """Win rate, gold trade, seconds, timeouts over the battles sel (default all); with both frames in the
    battles also "clean" and "broad" (the same over each)."""
    s = np.ones(len(res["won"]), dtype=bool) if sel is None else sel
    out = {"battles": int(s.sum()), "win_rate": round(float(res["won"][s].mean()), 3),
           "gold_trade": round(float(res["trade"][s].mean()), 3), "seconds": round(float(res["seconds"][s].mean()), 1),
           "timeouts": round(float(res["timeout"][s].mean()), 3)}
    b = res.get("broad")
    if sel is None and b is not None and b.any() and not b.all():
        out["clean"], out["broad"] = summary(res, ~b), summary(res, b)
    return out


def roster(desc, side):
    keys = [u["key"].split("_", 2)[-1] for u in desc["sides"][side]["units"]]
    return ",".join(keys)


def check(name, n=256, seed=0, device="cpu", show=0, broad=None):
    """{"naive": summary, "skilled": summary, "pass": bool} of a drill (broad: the broad frame's share)."""
    drill = D.load([name])[name]
    out = {}
    for which in ("naive", "skilled"):
        t0 = time.time()
        res, descs = play(drill, getattr(drill, which), n, seed, device, broad=broad)
        out[which] = dict(summary(res), wall_s=round(time.time() - t0, 1))
        print(f"{name} {which:8s}: " + json.dumps(out[which]), flush=True)
        for i in range(min(show, len(descs))):
            o = int(res["ours"][i])
            print(f"   #{i} ours(side {o}) [{roster(descs[i], o)}] v [{roster(descs[i], 3 - o)}] attacker "
                  f"{descs[i]['attacker']}: won {bool(res['won'][i])} trade {res['trade'][i]:+.2f} "
                  f"{res['seconds'][i]:.0f} s", flush=True)
    out["pass"] = out["naive"]["win_rate"] <= PASS_NAIVE and out["skilled"]["win_rate"] >= PASS_SKILLED
    print(f"{name}: naive {out['naive']['win_rate']:.3f} / skilled {out['skilled']['win_rate']:.3f} -> "
          f"{'PASS' if out['pass'] else 'FAIL'}", flush=True)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drill", default=",".join(D.NAMES), help="comma-separated drills")
    ap.add_argument("--battles", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--broad", type=float, default=None, help="share of the broad frame (default drills.BROAD)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    have = D.load()
    for name in args.drill.split(","):
        if name in have:
            check(name, args.battles, args.seed, args.device, args.show, args.broad)
        else:
            print(f"{name}: no module yet", flush=True)


if __name__ == "__main__":
    main()
