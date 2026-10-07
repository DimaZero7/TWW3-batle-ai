"""Sizes of the network (docs/en/training/model.md). Presets: small (tests, first training), wide (small x 2 in
every width: what tools/nn/model/widen.py makes of a trained small network), target, and v2 (wide's base with the
map's sector tokens, the chained heads and the commitment: tools/nn/model/sectors.py, chain.py, commit.py)."""
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
    grid: int = 0            # > 0: the move point is a cell of a grid x grid map grid in the side's frame (fixed for
    #                          the battle, so a cell is one place on the map), not a bin from the unit (heads.py)
    grid_half: float = 800.0  # ... the grid covers +-grid_half m forward and lateral from the map's centre
    pointer: int = 64        # width of the target pointer's query and key
    sectors: int = 0         # > 0 (v2): the map as sectors x sectors tokens in the side's frame (+-grid_half m), seen by
    #                          the units (sectors.py); the chained heads (chain.py): kind -> target -> sector -> fine
    #                          cell -> commitment (commit.py). 0: the heads of heads.py
    fine: int = 4            # ... a sector's fine x fine cells: the move point (sector 100 m / 4 = 25 m)
    sector_d: int = 64       # ... the sector tokens' width
    sector_heads: int = 2    # ... heads of the units -> sectors attention (head width sector_d / sector_heads)
    sector_at: int = 1       # ... it comes after this many of the units' attention blocks
    eyes: bool = False       # v2: the eyes, auxiliary heads on the simulator's truth fed back into the network (eyes.py)
    critic_d: int = 128      # the centralised critic (training only)
    critic_layers: int = 3
    critic_heads: int = 4

    @property
    def points(self):
        if self.sectors:
            return (self.sectors * self.fine) ** 2
        return self.grid * self.grid if self.grid else self.n_dir * self.n_dist


SMALL = ModelConfig()
# small x 2 in width: token width, attention heads (the head width stays 32), feed-forward, GRU, pointer
# and the critic's; the depth the same (tools/nn/model/widen.py widens a trained small network to it)
WIDE = ModelConfig(d=256, heads=8, pointer=128, critic_d=256, critic_heads=8)
TARGET = ModelConfig(d=512, layers=4, heads=8, pointer=128, critic_d=512, critic_layers=6, critic_heads=8)
# v2 (08.10.2026): wide's base (unit tokens by passport, attention, a GRU per token) + 16 x 16 map sectors of 100 m
# + the eyes
V2 = ModelConfig(d=256, heads=8, pointer=128, critic_d=256, critic_heads=8, sectors=16, eyes=True)
PRESETS = {"small": SMALL, "wide": WIDE, "target": TARGET, "v2": V2}


def preset(name, **changes):
    return replace(PRESETS[name], **changes)
