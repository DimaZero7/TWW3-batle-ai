"""Does a drill's skill carry over to ordinary battles? (docs/en/training/training.md "Drills", "Transfer")

A drill teaches a skill in its own battles; the network may master them (kiting 1.00) and still never
use the skill in a normal battle. Every drill with a `transfer` detector (drills.Drill.transfer: State ->
(situation, applied, mistake) [B, N]) says, per unit and simulator step of ANY battle, whether the unit is
in the drill's situation, whether it applies the skill there, and whether it makes the drill's mistake.
The Tracker sums, per battle, the unit-seconds of each, for the two sides apart: the evaluated network's
units ("mine") and the opponent script's (in the battles against ai_like: the reference of what the
game-like script does in the same situations).

    transfer share = applied unit-s / situation unit-s      (the skill used where it applies)
    mistake share  = mistake unit-s / situation unit-s

evaluate.play (the normal evaluation battles: test5's ai_like / nearest / hold_shoot) runs it and returns
res["transfer"] = {drill: {"network": {...}, "by_opponent": {name: {"network": {...}, "script": {...}}},
"ai_like": {...}}} (the network over all its battles; per opponent both sides; "ai_like" = the opponent's
units in the battles against ai_like), each {"share", "mistake", "unit_s" (situation unit-seconds a
battle), "battles" (share of the battles where the situation came up)}.

    bash tools/nn/dock.sh tools.nn.train.drills.transfer --checkpoint build/nn-train/test5/w5_kiting_teach/m25.pt
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

KEYS = ("sit_s", "applied_s", "mistake_s")


def detectors(names=None):
    """{drill name: its transfer detector} of the drills (default: drills.READY) that have one."""
    from tools.nn.train import drills as D
    loaded = D.load([n for n in D.NAMES if n in (names if names is not None else D.READY)])
    return {n: d.transfer for n, d in loaded.items() if d.transfer is not None}


def _count(st, dt, mine, live, sums, fns):
    """One step's unit-seconds into sums (in place): per drill and side (mine, the other side)."""
    u = st.u
    run = live[:, None]
    other = (u["side"] > 0) & ~mine
    for name, fn in fns:
        sit, applied, mistake = fn(st)
        for tag, side in (("mine", mine), ("other", other)):
            m = side & run
            sums[f"{name}/{tag}/sit_s"] += (sit & m).float().sum(1) * dt
            sums[f"{name}/{tag}/applied_s"] += (sit & applied & m).float().sum(1) * dt
            sums[f"{name}/{tag}/mistake_s"] += (sit & mistake & m).float().sum(1) * dt


class Tracker:
    """Per battle [B] sums of the drills' situations (unit-seconds) for the units of mine [B, N] and the
    other side's, while the battle runs. wrap: e.g. tools/nn/train/rollout.py fast (compiled on CUDA)."""

    def __init__(self, st, params, mine, names=None, wrap=None):
        self.dt = float(params.dt)
        self.mine = mine
        self.fns = tuple(detectors(names).items())
        self.names = tuple(n for n, _ in self.fns)
        self.sums = {f"{n}/{t}/{k}": torch.zeros(st.B, device=st.device)
                     for n in self.names for t in ("mine", "other") for k in KEYS}
        self._count = wrap(_count) if wrap else _count

    def update(self, st, live):
        """After a simulator step; live [B]: the battle was running before it."""
        if self.fns:
            self._count(st, self.dt, self.mine, live, self.sums, self.fns)

    def summary(self, sel, tag="mine"):
        """{drill: {share, mistake, unit_s, battles}} over the battles sel [B] (numpy bool) for one side."""
        out = {}
        sel = np.asarray(sel, dtype=bool)
        n = max(1, int(sel.sum()))
        for name in self.names:
            s = {k: self.sums[f"{name}/{tag}/{k}"].cpu().numpy()[sel] for k in KEYS}
            tot = float(s["sit_s"].sum())
            out[name] = {"share": round(float(s["applied_s"].sum()) / tot, 4) if tot > 0 else None,
                         "mistake": round(float(s["mistake_s"].sum()) / tot, 4) if tot > 0 else None,
                         "unit_s": round(tot / n, 2), "battles": round(float((s["sit_s"] > 0).mean()), 4) if sel.any() else None}
        return out


def report(tracker, this_by_opponent):
    """res["transfer"] of an evaluation: this_by_opponent {opponent name: sel [B] numpy bool}."""
    if not tracker.names:
        return {}
    every = np.zeros_like(next(iter(this_by_opponent.values())), dtype=bool)
    for sel in this_by_opponent.values():
        every = every | sel
    net = tracker.summary(every, "mine")
    per = {o: {"network": tracker.summary(sel, "mine"), "script": tracker.summary(sel, "other")}
           for o, sel in this_by_opponent.items()}
    out = {}
    for name in tracker.names:
        out[name] = {"network": net[name], "by_opponent": {o: {k: v[name] for k, v in p.items()} for o, p in per.items()}}
        if "ai_like" in per:
            out[name]["ai_like"] = per["ai_like"]["script"][name]
    return out


def text(tr):
    """Lines: per drill the network's transfer share against the ai_like reference."""
    f = (lambda v: "-" if v is None else f"{v:.3f}")
    lines = []
    for name, x in (tr or {}).items():
        net, ref = x.get("network") or {}, x.get("ai_like") or {}
        lines.append(f"{name}: network {f(net.get('share'))} (mistake {f(net.get('mistake'))}, {net.get('unit_s', 0):.1f} "
                     f"unit-s/battle, in {f(net.get('battles'))} of battles) | ai_like {f(ref.get('share'))} "
                     f"(mistake {f(ref.get('mistake'))}, {ref.get('unit_s', 0):.1f} unit-s/battle)")
        for o, p in (x.get("by_opponent") or {}).items():
            lines.append(f"    vs {o}: network {f(p['network']['share'])} / {f(p['network']['mistake'])} "
                         f"({p['network']['unit_s']:.1f} unit-s); {o} itself {f(p['script']['share'])} / "
                         f"{f(p['script']['mistake'])} ({p['script']['unit_s']:.1f} unit-s)")
    return lines


def main():
    """The transfer metrics of a checkpoint on test5's normal evaluation battles (no training)."""
    sys.stdout.reconfigure(encoding="utf-8")
    from tools.nn.train import cadence as cad
    from tools.nn.train import checkpoint, evaluate, test5
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval", type=int, default=512, help="battles per opponent (test5's --eval)")
    ap.add_argument("--out", help="write the whole evaluation as json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cad.add_args(ap)
    args = ap.parse_args()
    actor = checkpoint.load_policy(args.checkpoint, args.device)
    t0 = time.time()
    res = evaluate.play(actor, opponents=test5.OPPONENTS, device=args.device, generated=args.eval, max_units=19, seed=1,
                        together=True, paired=True, baseline=False, cadence=cad.of_args(args))
    print(f"{args.checkpoint}: {args.eval} battles per opponent, {time.time() - t0:.0f} s", flush=True)
    print("\n".join(text(res.get("transfer"))), flush=True)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "transfer": res.get("transfer"),
                                              "skill": res.get("skill")}, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
