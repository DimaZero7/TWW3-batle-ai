"""The learned group collection put into another general (docs/ru/training/general.md "Сбор групп: турнир"): the
tourney branch trains only general.assign. and general.k_head. (tourney_train.py) on a frozen network; the main line
goes on meanwhile. graft copies the collection's weights into a checkpoint of that line (a general of the same card
and places; one without the collection - an older one - takes them as fresh parts, executor.FRESH) and switches its
groups to "learned" with the branch's capacity bounds; the rest of the checkpoint (the executor, the general's other
parts, the critic) stays as it was.

    python -m tools.nn.general.graft --from build/nn-train/groups/t1/latest.pt \\
        --into build/nn-train/test5/g38_general/m40.pt --out build/learned_groups/g38_learned.pt
    python -m tools.nn.train.exam_sim build/learned_groups/g38_learned.pt --R 2 --plain --device cpu

The target's training state ("train": Adam, the trainer's counters, the ability critic, the kind Q's buffer) is
dropped: run.py --init from the grafted file starts them afresh. --keep-train keeps it (the line g41: the step's
latest.pt goes on as a chain's step does; a target without the collection has an Adam of fewer parameters - run.py
then starts Adam afresh and keeps the rest; one with it keeps Adam too: the collection's weights have no moments, PPO
does not train them).
"""
import argparse
import sys

import torch

from tools.nn.general.assign_learn import PREFIXES

KEYS = ("assign_lo", "assign_hi", "assign_stick")


def graft(src, dst, keep_train=False):
    """(the grafted checkpoint dict, tensors copied) of the source and target checkpoint dicts (checkpoint.read);
    keep_train: the target's "train" stays (module doc)."""
    if not src.get("general") or not dst.get("general"):
        raise ValueError("graft: both checkpoints need a general")
    for f in ("card", "places"):
        if src["general"].get(f) != dst["general"].get(f):
            raise ValueError(f"graft: the generals' {f} differ ({src['general'].get(f)} vs {dst['general'].get(f)})")
    out = dict(dst)
    actor = dict(dst["actor"])
    n = 0
    for k, v in src["actor"].items():
        if k.startswith(PREFIXES):
            actor[k] = v.clone()
            n += 1
    if not n:
        raise ValueError("graft: the source has no collection (general.assign., general.k_head.)")
    out["actor"] = actor
    gen = dict(dst["general"], groups="learned")
    for k in KEYS:
        if k in src["general"]:
            gen[k] = src["general"][k]
    out["general"] = gen
    if not keep_train:
        out.pop("train", None)
    out["meta"] = dict(dst.get("meta") or {}, graft={"collection": src.get("meta", {}).get("tourney"),
                                                     "tensors": n})
    return out, n


def main(argv=None):
    from tools.nn.train import checkpoint
    from tools.nn.train.exam_sim import resolve
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, help="the branch's checkpoint (its collection)")
    ap.add_argument("--into", required=True, help="the general's checkpoint to take it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-train", action="store_true", help="keep the target's training state (module doc)")
    args = ap.parse_args(argv)
    data, n = graft(checkpoint.read(resolve(args.src)), checkpoint.read(resolve(args.into)), args.keep_train)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(out)
    actor = checkpoint.load_policy(out)                  # (it loads: the shapes fit)
    print(f"written: {out} ({n} tensors of the collection; groups {actor.gcfg.groups}, sticky "
          f"{getattr(actor.gcfg, 'groups_sticky', False)}, bounds {actor.gcfg.assign_lo}/{actor.gcfg.assign_hi}; "
          f"training state {'kept' if 'train' in data else 'dropped'})", flush=True)


if __name__ == "__main__":
    main()
