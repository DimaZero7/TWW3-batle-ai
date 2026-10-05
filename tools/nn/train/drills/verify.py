"""Check a drill in the simulator with our two scripts before it trains anything
(docs/en/training/training.md "Drills"): the naive script must lose clearly, the skilled one
win clearly, over generated battles of the drill (our side alternating 1 and 2), with the
training's randomised numbers (randomise.Spread()).

    bash tools/nn/dock.sh tools.nn.train.drills.verify --drill pincer --battles 256
    python -m tools.nn.train.drills.verify --drill pincer --battles 64 --device cpu --show 3
    bash tools/nn/dock.sh tools.nn.train.drills.verify --drill kiting --broad 1 --embed 0    # the broad frame only
    bash tools/nn/dock.sh tools.nn.train.drills.verify --drill kiting,hold_fire --embed 1   # the embedded frame

--embed: the share of the battles from the drill's embedded frame (default drills.EMBED, 1: the situation
inside a normal battle; 0 the clean / broad mix); --broad: of the rest, the share from the broad frame
(default drills.BROAD; 0 the clean frame, 1 the broad one only); with more than one frame in the mix the
summary also splits them ("clean", "broad", "embedded").
Prints per script: win rate, gold trade ((enemy gold destroyed - own lost) / budget, reward.gold_sides),
mean battle seconds, timeouts; then skilled - naive on the same battles (paired: the same seeds and
randomised numbers): the gold trade's and the win rate's mean difference with a 95 % interval, and the share
of the battles where skilled traded better / worse. An embedded frame passes on that paired gold trade
(EMBED_PASS_TRADE: the rest of a normal battle makes the win rate a weak measure), the others on the win
rates. --show K: the first K battles' rosters and results.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

from tools.nn.sim import battle, scenario
from tools.nn.sim import state as S
from tools.nn.train import randomise, reward
from tools.nn.train import drills as D

PASS_NAIVE = 0.25        # the naive script wins at most this share ...
PASS_SKILLED = 0.75      # ... the skilled one at least this
EMBED_PASS_TRADE = 0.05  # an embedded frame: skilled - naive gold trade (paired) at least this, its 95 % interval above 0


def narrowed(st, keep):
    """The battles keep (indices, ascending) of the State st as a State of their own (the same tensors'
    rows; the map's bounds and the unit keys go along)."""
    kl = keep.tolist()
    return S.State({k: v[keep] for k, v in st.u.items()}, st.t[keep], st.attacker[keep], st.done[keep],
                   st.winner[keep], st.lord_dead_s[keep], st.bounds, [st.keys[i] for i in kl] if st.keys else [])


def put_back(full, rows, cur):
    """The running battles' State cur back into the whole batch's full at the places rows (in place)."""
    for k, v in cur.u.items():
        full.u[k][rows] = v
    for k in ("t", "attacker", "done", "winner", "lord_dead_s"):
        getattr(full, k)[rows] = getattr(cur, k)


def run(st, ours, script, enemy, params, advance=None, extra=None, compact=True, check_every=8):
    """Every battle of st to its end (in place): our side's `script` and the drill's `enemy` order every
    simulator step. compact: once some battles have ended, only the running ones step (checked every
    check_every steps; a battle is per battle in the simulator, so it ends the same in a smaller batch:
    the uncompiled CPU step costs in proportion to the battles it steps). extra(st, rows), if given,
    after every step: st the battles that stepped, rows their places in the whole batch (None: all)."""
    advance = advance or battle.step
    cur, rows, i = st, None, 0
    while True:
        if i % check_every == 0:
            live = ~cur.done
            n = int(live.sum())
            if n == 0:
                break
            if compact and n < cur.B:
                keep = live.nonzero().squeeze(1)
                if rows is not None:
                    put_back(st, rows, cur)
                rows = keep if rows is None else rows[keep]
                cur = narrowed(st, rows)
        o = ours if rows is None else ours[rows]
        advance(cur, D.merged(cur, o, script(cur), enemy(cur)), params, params.dt)
        cur.u["lost_worst"] = reward.track(cur.u)          # a loss counts once (reward.gold_lost)
        if extra is not None:
            extra(cur, rows)
        i += 1
    if rows is not None:
        put_back(st, rows, cur)
    return st


def play(drill, script, n=256, seed=0, device="cpu", spread=randomise.Spread(), seeds=None, extra=None, ours_box=None,
         broad=None, embed=None, compile=False, compact=None):
    """Battles of a drill with `script` on our side and the drill's enemy -> per-battle results:
    {"won", "trade", "seconds", "timeout", "ours", "broad", "frame"} (numpy) and the descriptions. broad, embed:
    the shares of the broad and embedded frames (drills.battles; default drills.BROAD, drills.EMBED).
    compile: the simulator's step by torch.compile on CUDA (battle.stepper; default off: plain, as before);
    compact (default: when not compiled, as a new batch size would compile again): only the running battles
    step (run()).
    extra(st, rows): see run()."""
    seeds = list(seeds if seeds is not None else range(seed, seed + n))
    pairs = D.battles(drill, seeds, broad=broad, embed=embed)
    descs = [p[0] for p in pairs]
    ours = torch.tensor([p[1] for p in pairs], device=device)
    if ours_box is not None:
        ours_box["ours"] = ours                  # (for an `extra` that needs our side per battle; "st": the end)
    st = scenario.build(descs, device=device)
    gen = torch.Generator(device=device).manual_seed(seed + 7)
    randomise.apply(st, torch.ones(st.B, dtype=torch.bool, device=device), spread, gen)
    from tools.nn.sim.params import load
    params = load()
    advance = battle.stepper(device, compile)
    run(st, ours, script, drill.enemy, params, advance, extra, compact=(advance is battle.step) if compact is None else compact)
    if ours_box is not None:
        ours_box["st"] = st                      # the whole batch at the end
    gold = reward.gold_sides(st.u).cpu().numpy()
    bud = reward.budget(st.u).cpu().numpy()
    o = ours.cpu().numpy()
    b = np.arange(st.B)
    own, en = gold[b, o - 1], gold[b, 2 - o]
    won = st.winner.cpu().numpy() == o
    t = st.t.cpu().numpy()
    return {"won": won, "trade": (en - own) / np.maximum(bud, 1e-9), "seconds": t, "timeout": t >= params.limit_s - 1e-6,
            "ours": o, "broad": np.array([bool(d.get("broad")) for d in descs]),
            "frame": np.array([d.get("frame", "broad" if d.get("broad") else "clean") for d in descs], dtype=object)}, descs


def summary(res, sel=None):
    """Win rate, gold trade, seconds, timeouts over the battles sel (default all); with more than one frame in
    the battles also per frame ("clean", "broad", "embedded": the same over each)."""
    s = np.ones(len(res["won"]), dtype=bool) if sel is None else sel
    out = {"battles": int(s.sum()), "win_rate": round(float(res["won"][s].mean()), 3),
           "gold_trade": round(float(res["trade"][s].mean()), 3), "seconds": round(float(res["seconds"][s].mean()), 1),
           "timeouts": round(float(res["timeout"][s].mean()), 3)}
    kinds = res.get("frame")
    if sel is None and kinds is not None:
        present = [k for k in D.FRAMES if (kinds == k).any()]
        if len(present) > 1:
            for k in present:
                out[k] = summary(res, kinds == k)
    return out


def paired(a, b, sel=None):
    """b - a on the same battles (sel, default all): {"trade", "trade_ci95", "win", "win_ci95", "better",
    "worse" (shares of the battles where b's trade is above / below a's by more than 0.01)}."""
    s = np.ones(len(a["won"]), dtype=bool) if sel is None else sel
    dt = b["trade"][s] - a["trade"][s]
    dw = b["won"][s].astype(float) - a["won"][s].astype(float)
    ci = (lambda x: float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"))
    return {"battles": int(s.sum()), "trade": round(float(dt.mean()), 4), "trade_ci95": round(ci(dt), 4),
            "win": round(float(dw.mean()), 4), "win_ci95": round(ci(dw), 4),
            "better": round(float((dt > 0.01).mean()), 3), "worse": round(float((dt < -0.01).mean()), 3)}


def roster(desc, side):
    keys = [u["key"].split("_", 2)[-1] for u in desc["sides"][side]["units"]]
    return ",".join(keys)


def check(name, n=256, seed=0, device="cpu", show=0, broad=None, embed=None):
    """{"naive": summary, "skilled": summary, "paired": skilled - naive, "pass": bool} of a drill (broad, embed:
    the broad and embedded frames' shares). An embedded-only check passes on the paired gold trade
    (EMBED_PASS_TRADE), the others on the win rates (PASS_NAIVE, PASS_SKILLED)."""
    drill = D.load([name])[name]
    out, raw = {}, {}
    for which in ("naive", "skilled"):
        t0 = time.time()
        res, descs = play(drill, getattr(drill, which), n, seed, device, broad=broad, embed=embed)
        raw[which] = res
        out[which] = dict(summary(res), wall_s=round(time.time() - t0, 1))
        print(f"{name} {which:8s}: " + json.dumps(out[which]), flush=True)
        for i in range(min(show, len(descs))):
            o = int(res["ours"][i])
            print(f"   #{i} ours(side {o}) [{roster(descs[i], o)}] v [{roster(descs[i], 3 - o)}] attacker "
                  f"{descs[i]['attacker']}: won {bool(res['won'][i])} trade {res['trade'][i]:+.2f} "
                  f"{res['seconds'][i]:.0f} s", flush=True)
    out["paired"] = paired(raw["naive"], raw["skilled"])
    kinds = raw["naive"]["frame"]
    for k in D.FRAMES:
        if (kinds == k).any() and not (kinds == k).all():
            out["paired"][k] = paired(raw["naive"], raw["skilled"], kinds == k)
    p = out["paired"]
    print(f"{name} skilled - naive (paired, {p['battles']} battles): gold trade {p['trade']:+.4f} ± {p['trade_ci95']:.4f}, "
          f"win {p['win']:+.4f} ± {p['win_ci95']:.4f}; skilled traded better in {p['better']:.3f}, worse in "
          f"{p['worse']:.3f}" + "".join(f"; {k} {v['trade']:+.4f} ± {v['trade_ci95']:.4f}" for k, v in p.items()
                                        if isinstance(v, dict)), flush=True)
    if (kinds == "embedded").all():
        out["pass"] = p["trade"] >= EMBED_PASS_TRADE and p["trade"] - p["trade_ci95"] > 0
        rule = f"paired gold trade >= {EMBED_PASS_TRADE} and above 0 at 95 %"
    else:
        out["pass"] = out["naive"]["win_rate"] <= PASS_NAIVE and out["skilled"]["win_rate"] >= PASS_SKILLED
        rule = f"win rates naive <= {PASS_NAIVE}, skilled >= {PASS_SKILLED}"
    print(f"{name}: naive {out['naive']['win_rate']:.3f} / {out['naive']['gold_trade']:+.3f} | skilled "
          f"{out['skilled']['win_rate']:.3f} / {out['skilled']['gold_trade']:+.3f} -> {'PASS' if out['pass'] else 'FAIL'} "
          f"({rule})", flush=True)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drill", default=",".join(D.NAMES), help="comma-separated drills")
    ap.add_argument("--battles", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--broad", type=float, default=None, help="share of the broad frame (default drills.BROAD)")
    ap.add_argument("--embed", type=float, default=None, help="share of the embedded frame (default drills.EMBED)")
    ap.add_argument("--out", help="write the results as json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    have = D.load()
    res = {}
    for name in args.drill.split(","):
        if name in have:
            res[name] = check(name, args.battles, args.seed, args.device, args.show, args.broad, args.embed)
        else:
            print(f"{name}: no module yet", flush=True)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
