"""The thin executor (line g48, docs/ru/training/general.md "Тонкий исполнитель"): a new executor of other sizes with
random weights under the general of a checkpoint.

    python -m tools.nn.general.thin --general-from build/nn-train/line_g47/g47_3_grafted.pt \\
        --out build/nn-train/line_g48/g48_start.pt [--d 128 --layers 3 --heads 4 --ff 2 --pointer 64 --memory 1 --eyes 1]

What the new checkpoint holds:
- the executor (the unit network, its task input) of the sizes given, every weight as a fresh network's (seeded): no
  imitation, no distillation of the source's executor; its inputs and outputs (heads) are the source's - the
  observation, the sectors, the chained heads, the abilities, the task: only the widths change (ModelConfig d, layers,
  heads, ff, pointer, memory, eyes), so the bridge, the game's companion and the cards read it as any checkpoint (they
  rebuild the network from its "config");
- the general (every general.* parameter) as the source saved it - but its executor's own kind table
  (general.exec_kinds, GeneralConfig.split_kinds: the old executor's input) restarts as a fresh general's, a copy of the
  general's kind table; the GeneralConfig the source's;
- the source's critic (the full view of the battle: it does not depend on the executor's size; its width is kept by
  ModelConfig.critic_ff, the executor's ff may differ) and the ability critic's state (ABILITY_STATE: prices of the
  battle's moments from the observation, not of the executor) - so the start's values are not noise; no Adam state, no
  trainer's counters (the run starts them afresh, --critic-warmup applies).
Train it with run.py --general-freeze 1 (only the executor and the critic learn).
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import torch

from tools.nn.general.executor import GeneralActor
from tools.nn.train import checkpoint

# the executor's sizes of line g48 (build/diag_exec/exec_map.md: 0.84 M weights without the eyes, x 4.6 fewer)
THIN = {"d": 128, "layers": 3, "heads": 4, "ff": 2, "pointer": 64, "memory": True, "eyes": True}
ABILITY_STATE = ("ability_critic", "ability_critic_openloop", "ability_critic_kind", "ability_norm")


def executor_weights(actor):
    """The executor's weights (every parameter but the general's, its own kind table counted as the executor's)."""
    return sum(p.numel() for n, p in actor.named_parameters()
               if not n.startswith("general.") or n == "general.exec_kinds")


def build(src, sizes=None, seed=0):
    """(actor, critic state or None, train dict or None, source config) of a thin executor under src's general (module
    doc). sizes: ModelConfig fields over the source's (default THIN)."""
    data = checkpoint.read(src)
    if not data.get("general"):
        raise ValueError(f"{src}: no general in the checkpoint")
    old = checkpoint.config_of(data)
    sizes = dict(THIN if sizes is None else sizes)
    cfg = dataclasses.replace(old, critic_ff=old.critic_ff or old.ff, **sizes)
    torch.manual_seed(seed)
    actor = GeneralActor(cfg, checkpoint.general_of(data))
    gen = {k: v for k, v in data["actor"].items() if k.startswith("general.")}
    gen["general.exec_kinds"] = gen["general.kinds"].clone()           # (a fresh general's: a copy of its kinds)
    missing, unexpected = actor.load_state_dict(gen, strict=False)
    left = [k for k in missing if k.startswith("general.")]
    if left or unexpected:
        raise ValueError(f"the general does not fit: missing {left[:5]}, unexpected {list(unexpected)[:5]}")
    ts = data.get("train") or {}
    train = {k: v for k, v in ts.items() if k in ABILITY_STATE} or None
    return actor.eval(), data.get("critic"), train, old


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--general-from", required=True, help="the checkpoint whose general (and critic) the start takes")
    ap.add_argument("--out", required=True)
    for k, v in THIN.items():
        ap.add_argument(f"--{k}", type=int, default=int(v))
    ap.add_argument("--seed", type=int, default=48)
    args = ap.parse_args(argv)
    sizes = {k: (bool(getattr(args, k)) if isinstance(v, bool) else getattr(args, k)) for k, v in THIN.items()}
    actor, critic_state, train, old = build(args.general_from, sizes, args.seed)
    from tools.nn.model import critic as mcritic
    critic = mcritic.Critic(actor.cfg)
    if critic_state is not None:
        critic.load(critic_state)
    old_actor = checkpoint.load_policy(args.general_from)
    meta = {"update": 0, "battles": 0, "seed": args.seed,
            "note": f"thin executor {sizes} (random weights) under the general of {args.general_from} (frozen in "
                    "training: run.py --general-freeze 1)"}
    path = checkpoint.save(Path(args.out), actor, critic, "v2", meta, train)
    gsum = float(sum(p.detach().double().sum() for n, p in actor.named_parameters()
                     if n.startswith("general.") and n != "general.exec_kinds"))
    print(f"{path}: executor {executor_weights(actor) / 1e6:.3f} M weights (source {executor_weights(old_actor) / 1e6:.3f}"
          f" M), general {sum(p.numel() for n, p in actor.named_parameters() if n.startswith('general.')) / 1e6:.3f} M "
          f"(sum of its weights {gsum:.6f}), critic {sum(p.numel() for p in critic.parameters()) / 1e6:.3f} M "
          f"{'from the source' if critic_state is not None else 'fresh'}, ability state "
          f"{sorted(train) if train else 'none'}; config {dataclasses.asdict(actor.cfg)}", flush=True)
    return path


if __name__ == "__main__":
    main()
