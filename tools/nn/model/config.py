"""Sizes of the network (docs/en/training/model.md). Presets: small (tests, first training), wide (small x 2 in
every width: what tools/nn/model/widen.py makes of a trained small network) and target."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    d: int = 128             # token width
    layers: int = 3          # attention layers of the actor
    heads: int = 4
    ff: int = 4              # feed-forward width = ff * d
    memory: bool = True      # a GRU per token (before the last attention layer)
    dist_bins: int = 16      # buckets of the distance between two units: a learned attention bias
    n_dir: int = 16          # move point: 16 directions in the side's frame (0 = towards the enemy) ...
    n_dist: int = 8          # ... x 8 distances from the unit, geometric dist_min..dist_max
    dist_min: float = 10.0
    dist_max: float = 400.0
    pointer: int = 64        # width of the target pointer's query and key
    critic_d: int = 128      # the centralised critic (training only)
    critic_layers: int = 3
    critic_heads: int = 4

    @property
    def points(self):
        return self.n_dir * self.n_dist


SMALL = ModelConfig()
# small x 2 in width: token width, attention heads (the head width stays 32), feed-forward, GRU, pointer
# and the critic's; the depth the same (tools/nn/model/widen.py widens a trained small network to it)
WIDE = ModelConfig(d=256, heads=8, pointer=128, critic_d=256, critic_heads=8)
TARGET = ModelConfig(d=512, layers=4, heads=8, pointer=128, critic_d=512, critic_layers=6, critic_heads=8)
PRESETS = {"small": SMALL, "wide": WIDE, "target": TARGET}


def preset(name, **changes):
    return replace(PRESETS[name], **changes)
