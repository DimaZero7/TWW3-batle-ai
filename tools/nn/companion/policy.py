"""The companion's policy: the actor from a training checkpoint, or a fresh untrained one (torch).

A checkpoint is loaded with tools/nn/train/checkpoint.py load_policy(path). Without one (not
trained yet, or the file is missing) a freshly initialised small actor with a fixed seed is used:
its orders are random, but they are the network's orders, given through the whole path.
"""
from pathlib import Path

import torch

from tools import config as project
from tools.nn.model import config as model_config
from tools.nn.model.policy import Actor

DEFAULT = project.BUILD / "nn-train" / "random.pt"


def fresh(seed=0, preset="small"):
    torch.manual_seed(seed)
    return Actor(model_config.preset(preset)).eval()


def load(path=None, seed=0):
    """(actor, description). path None: the default checkpoint if it exists, else a fresh actor.
    An explicit path that cannot be loaded is an error."""
    explicit = path is not None
    path = Path(path) if path else DEFAULT
    if path.exists():
        try:
            from tools.nn.train.checkpoint import load_policy
            actor = load_policy(path)
            actor = actor[0] if isinstance(actor, tuple) else actor
            if not hasattr(actor, "cfg"):
                raise TypeError(f"load_policy({path}) gave {type(actor).__name__}, not an actor")
            return actor.eval(), f"checkpoint {path}"
        except Exception as e:           # noqa: BLE001 - reported, then the fallback
            if explicit:
                raise
            reason = f"{path} not loaded ({type(e).__name__}: {e})"
    elif explicit:
        raise FileNotFoundError(path)
    else:
        reason = f"no checkpoint at {path}"
    return fresh(seed), f"fresh small actor, seed {seed} ({reason})"
