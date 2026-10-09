"""Checkpoints of the network: one file per version (docs/en/training/training.md).

The contract with the game's companion (tools/nn/companion) - keep it stable:

    build/nn-train/random.pt   an untrained network (random weights), written first
    build/nn-train/latest.pt   the last trained network
    build/nn-train/pool/*.pt   past versions (the opponents of self-play)

A file is a plain dict saved by torch.save (loads with weights_only=True):

    format   2 (1 had no KEEP order: 4 order kinds)
    kinds    the order kinds in code order: hold 0, move 1, attack 2, withdraw 3, keep 4
             (tools/nn/sim/orders.py KINDS)
    preset   "small" or "target" (tools/nn/model/config.py), for the record
    config   the ModelConfig fields as a dict: rebuilds the actor
    actor    the actor's state_dict (all the game needs)
    critic   the critic's state_dict (training only; may be missing)
    meta     dict: update, battles, steps, seconds, when written, notes
    train    (training only; may be missing) what a run needs to go on seamlessly from this file (run.py --init):
             "optim" the optimizer's state_dict (Adam's moments over the actor's then the critic's parameters),
             "state" the trainer's counters (the entropy floor's weight, the entropy decay's seconds, updates and
             seconds over the chain)

    policy = load_policy("build/nn-train/latest.pt")      # an Actor in eval mode
"""
import dataclasses
import os
import time
from pathlib import Path

import torch

from tools import config as project
from tools.nn.model import config as model_config
from tools.nn.model import critic as model_critic
from tools.nn.model import policy as model_policy
from tools.nn.sim.orders import KINDS

FORMAT = 2
DIR = project.BUILD / "nn-train"
RANDOM = DIR / "random.pt"
RANDOM_V2 = DIR / "random_v2.pt"     # the untrained v2 network (ModelConfig.sectors > 0): v2 starts from it
LATEST = DIR / "latest.pt"
POOL = DIR / "pool"


def config_of(data):
    """The ModelConfig of a checkpoint dict (unknown fields ignored, missing ones default). The LoRA
    adapters are gone (03.10; they were never on): a checkpoint with lora_rank > 0 is refused."""
    if data["config"].get("lora_rank"):
        raise ValueError("a checkpoint with LoRA adapters (lora_rank > 0): no longer supported")
    names = {f.name for f in dataclasses.fields(model_config.ModelConfig)}
    return model_config.ModelConfig(**{k: v for k, v in data["config"].items() if k in names})


def save(path, actor, critic=None, preset="small", meta=None, train=None):
    """Write a checkpoint atomically (a temporary file, then a rename). train: {"optim": state_dict, "state": dict}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"format": FORMAT, "kinds": list(KINDS), "preset": preset, "config": dataclasses.asdict(actor.cfg),
            "actor": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
            "meta": dict(meta or {}, written=time.strftime("%Y-%m-%d %H:%M:%S"))}
    if critic is not None:
        data["critic"] = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
    if train is not None:
        data["train"] = _to_cpu(train)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)
    return path


def _to_cpu(x):
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(v) for v in x)
    return x.detach().cpu() if hasattr(x, "detach") else x


def train_state(path):
    """The checkpoint's "train" (the optimizer and the trainer's counters), None when it has none."""
    return read(path).get("train")


def read(path, device="cpu"):
    data = torch.load(Path(path), map_location=device, weights_only=True)
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: checkpoint format {data.get('format')!r}, expected {FORMAT}")
    return data


def load_policy(path, device="cpu"):
    """The actor of a checkpoint, in eval mode, on `device`."""
    data = read(path, device)
    actor = model_policy.Actor(config_of(data))
    actor.load_state_dict(data["actor"])
    return actor.to(device).eval()


def load_critic(path, device="cpu"):
    """The critic of a checkpoint (None when it has none)."""
    data = read(path, device)
    if "critic" not in data:
        return None
    critic = model_critic.Critic(config_of(data))
    critic.load(data["critic"])
    return critic.to(device)


def meta(path):
    return read(path).get("meta", {})


def random_for(cfg):
    """The untrained network of cfg's kind: random_v2.pt for v2 (written if missing), else random.pt."""
    if not cfg.sectors:
        return RANDOM
    if not RANDOM_V2.exists() or config_of(read(RANDOM_V2)) != cfg:
        write_random(RANDOM_V2, "v2")
    return RANDOM_V2


def write_random(path=RANDOM, preset="small", seed=0):
    """An untrained network: the same format, random weights (seeded)."""
    torch.manual_seed(seed)
    cfg = model_config.preset(preset)
    actor = model_policy.Actor(cfg)
    critic = model_critic.Critic(cfg)
    return save(path, actor, critic, preset, {"update": 0, "battles": 0, "note": "untrained, random weights",
                                              "seed": seed})


if __name__ == "__main__":
    print(write_random())
