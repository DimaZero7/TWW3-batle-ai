"""Teach the student (tools/nn/student, docs/en/training/student.md): imitation of the game AI on its recorded battles.

    python -m tools.nn.student.train --name s12 --init build/v2/itS12.pt --epochs 4        # from S12's weights
    python -m tools.nn.student.train --name scratch --scratch build/v2/itS12.pt --epochs 4  # S12's build, fresh weights

Every epoch: one pass over the training starts (dataset.Streams: B streams of battle sides, chunks of T seconds, the
memory carried), AdamW, the gradient's norm clipped; then the held-out starts (val; and --exam-check sides of the exam)
and a checkpoint: <out>/latest.pt, <out>/best.pt (the lowest val nll), an ordinary checkpoint (format 2, the actor; no
critic) the self-play pool takes as it is (run.py --pool-extra <out>/best.pt). The log: <out>/log.jsonl.
--minutes: stop after that many minutes (the epoch in progress ends its validation and checkpoint first).
--max-sides: only that many training sides (a short check).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tools import config as project
from tools.nn.model import policy as model_policy
from tools.nn.student import dataset as ds
from tools.nn.student import objective as obj
from tools.nn.train import checkpoint

OUT = project.BUILD / "student"


def actor_from(args, device):
    """(actor, its source note): --init's weights, or --scratch's build with fresh weights (seed)."""
    if args.init:
        return checkpoint.load_policy(args.init, device).train(), f"init {args.init}"
    cfg = checkpoint.config_of(checkpoint.read(args.scratch))
    torch.manual_seed(args.seed)
    return model_policy.Actor(cfg).to(device).train(), f"scratch (the build of {args.scratch}, seed {args.seed})"


@torch.no_grad()
def check(actor, sides, args, device, N):
    """The accuracies' sums over these sides (one pass, the memory from each side's start)."""
    if not sides:
        return {}
    was = actor.training
    actor.eval()
    st = ds.Streams(sides, B=args.B, T=args.T, N=N, shuffle=False)
    acc, h = {}, None
    for steps in st:
        b = ds.to_batch(steps, device)
        if h is None:
            h = actor.initial({k: v[0] for k, v in b["obs"].items()})
        _, s, h, _ = obj.run(actor, b, h, args.new_weight)
        obj.add(acc, s)
    actor.train(was)
    return acc


def pick(sides, n, seed):
    if not n or len(sides) <= n:
        return list(sides)
    rng = np.random.default_rng(seed)
    return [sides[i] for i in sorted(rng.choice(len(sides), n, replace=False))]


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--init", help="start from this checkpoint's actor")
    g.add_argument("--scratch", help="fresh weights in this checkpoint's build (its ModelConfig)")
    ap.add_argument("--data", nargs="*", default=[str(d) for d in ds.DIRS], help="folders of converted battles")
    ap.add_argument("--out", help=f"default {OUT}/<name>")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--B", type=int, default=16, help="streams (battle sides at once)")
    ap.add_argument("--T", type=int, default=32, help="seconds a chunk (the memory's gradient reaches this far)")
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 with --init, 3e-4 from scratch")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--new-weight", type=float, default=1.0, help="a new order's weight in the loss (keep: 1)")
    ap.add_argument("--val-share", type=float, default=0.15)
    ap.add_argument("--exam-train", default="", help="exam army seeds (comma-separated) to train on")
    ap.add_argument("--exam-check", type=int, default=30, help="exam sides in the check after an epoch (0: none)")
    ap.add_argument("--max-sides", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0, help="at most this many val sides in the check (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    device = torch.device(a.device)
    out = Path(a.out) if a.out else OUT / a.name
    out.mkdir(parents=True, exist_ok=True)
    actor, source = actor_from(a, device)
    lr = a.lr or (1e-4 if a.init else 3e-4)
    sides = ds.discover([Path(d) for d in a.data], actor.cfg)
    parts = ds.split(sides, a.val_share, [int(s) for s in a.exam_train.split(",") if s])
    train = pick(parts["train"], a.max_sides, a.seed)
    val = pick(parts["val"], a.max_val, a.seed)
    exam = pick(parts["exam"], a.exam_check, a.seed)
    N = max(s.N for s in sides)
    groups = lambda xs: len({s.group for s in xs})
    print(f"student {a.name}: {source}; lr {lr:g}, B {a.B}, T {a.T}, new weight {a.new_weight:g}; train {len(train)} "
          f"sides ({groups(train)} starts, {sum(s.T for s in train)} s), val {len(val)} ({groups(val)} starts, "
          f"{sum(s.T for s in val)} s), exam check {len(exam)} ({groups(exam)} starts); units padded to {N}",
          flush=True)
    (out / "split.json").write_text(json.dumps({k: [s.name for s in v] for k, v in
                                                (("train", train), ("val", val), ("exam", exam))}, indent=0),
                                    encoding="utf-8", newline="\n")
    opt = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=a.wd)
    streams = ds.Streams(train, B=a.B, T=a.T, N=N, seed=a.seed)
    t0, best, step = time.time(), float("inf"), 0
    meta = {"student": a.name, "source": source, "val_share": a.val_share, "exam_train": a.exam_train, "lr": lr}
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    stop = False
    for epoch in range(1, a.epochs + 1):
        acc, h, te = {}, None, time.time()
        for steps in streams:
            b = ds.to_batch(steps, device)
            if h is None:
                h = actor.initial({k: v[0] for k, v in b["obs"].items()})
            loss, st, h, _ = obj.run(actor, b, h, a.new_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(actor.parameters(), a.clip)
            opt.step()
            obj.add(acc, st)
            step += 1
            if step % 50 == 0:
                sm = obj.summary(acc)
                print(f"  epoch {epoch} step {step}: {obj.text(sm)}; grad {float(gn):.2f}; "
                      f"{acc['labels'] / max(1e-9, time.time() - te):.0f} labels/s, {time.time() - t0:.0f} s", flush=True)
            if a.minutes and time.time() - t0 > a.minutes * 60:
                stop = True
                break
        train_s = time.time() - te
        sm = obj.summary(acc)
        tv = time.time()
        vs = obj.summary(check(actor, val, a, device, N))
        es = obj.summary(check(actor, exam, a, device, N)) if exam else None
        rec = {"epoch": epoch, "step": step, "train": sm, "val": vs, "exam": es, "train_s": train_s,
               "check_s": time.time() - tv, "counts": streams.counts(), "seconds": time.time() - t0}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"epoch {epoch} ({train_s:.0f} s, check {time.time() - tv:.0f} s; {streams.counts_text()}):\n"
              f"  train {obj.text(sm)}\n  val   {obj.text(vs)}"
              + (f"\n  exam  {obj.text(es)}" if es else ""), flush=True)
        m = dict(meta, epoch=epoch, step=step, val=vs, exam=es)
        checkpoint.save(out / "latest.pt", actor, None, "v2", m)
        if vs["nll"] < best:
            best = vs["nll"]
            checkpoint.save(out / "best.pt", actor, None, "v2", m)
            print(f"  best (val nll {best:.4f}) -> {out / 'best.pt'}", flush=True)
        if stop:
            break
    log.close()
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
