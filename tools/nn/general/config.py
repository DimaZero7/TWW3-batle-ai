"""The general's settings (S28 skeleton, docs/en/training/general.md; design/general/index.html).

Saved with a checkpoint (checkpoint.py "general": the dataclass's fields), so a network with a general rebuilds with
the same sizes and rules. The skeleton's simplifications are fields too (docs "Simplifications of the skeleton"): each
is a value of its field, never code that cannot be switched; a value whose full version is not built yet is refused
when the config is made (NotBuilt), so a run never thinks it has what it has not.
"""
from dataclasses import asdict, dataclass, fields


class NotBuilt(ValueError):
    """A setting whose full version is not built yet (the skeleton's simplification is the only one there is)."""


@dataclass(frozen=True)
class GeneralConfig:
    card: int = 64             # D: the width of every card (units, tiles, groups, mission kinds)
    reader_hidden: int = 128   # the army reader's first layer (army_reader.ArmyReader hidden)
    council_layers: int = 2    # "Совещание": units with tiles, each asks each (simplification 3: small)
    council_heads: int = 4
    group_layers: int = 1      # "Совещание групп": groups listen to the tiles and all groups
    ff: int = 2                # feed-forward width = ff x card in both councils
    places: int = 10           # G: group places a side ("Сбор групп": 10 ours + 10 the enemy's)
    rounds: int = 3            # slot attention's rounds inside one decision
    kinds: int = 10            # K: abstract mission kinds ("Виды заданий": a learned table)
    center: float = 1.0        # the group collection's keys less this x their side's mean key (a learned share, starts
    #                            here): the units compete for the places by what tells them apart - without it a fresh
    #                            general put nearly every unit into one group (1.0-1.3 groups a side); 0: plain slot
    #                            attention
    # --- the skeleton's simplifications (docs "Simplifications of the skeleton") ---
    tiles: str = "sectors"     # 1: the map tiles are the unit network's 16 x 16 sectors (8 features each) projected to
    #                            the card's width; "cnn" (the map CNN's 32 x 32 tiles, map_cnn.py) is not built yet
    effects: str = "off"       # 2: the effect cards are zeros ("off"); the effects' analysis is not built yet
    zone: str = "place"        # 5: the deployment zone is the area tools/nn/armies/place.py deploys an army in (the
    #                            arena's rule); "map" (the map's zone layers, map_input.py) is not built yet
    deploy: str = "net"        # 5: "net" the general picks the groups' places, a rule lays the units out; "script":
    #                            place.py's placement as before (the general still forms groups and gives missions)
    # --- re-planning ("Пересмотр замысла") ---
    replan_s: float = 30.0     # the plan is made again every this many seconds of battle ...
    replan_min_s: float = 3.0  # ... and on an event (a group broken, a lord fallen), but not sooner than this after
    #                            the last plan
    broken: float = 0.5        # a group is broken when more than this share of its units routs or is destroyed
    # --- the executor's task input ("Задание отряда") ---
    dist_m: float = 100.0      # a target's distance from the unit / this ...
    dist_cap: float = 20.0     # ... at most this

    def __post_init__(self):
        if self.tiles != "sectors":
            raise NotBuilt(f"general tiles {self.tiles!r}: only 'sectors' is built (the map CNN's tiles come later)")
        if self.effects != "off":
            raise NotBuilt(f"general effects {self.effects!r}: only 'off' is built (the effects' analysis comes later)")
        if self.zone != "place":
            raise NotBuilt(f"general zone {self.zone!r}: only 'place' is built (the map's zone layers come later)")
        if self.deploy not in ("net", "script"):
            raise ValueError(f"general deploy {self.deploy!r}: 'net' or 'script'")
        assert self.card % self.council_heads == 0, "the card's width must split into the heads"

    @property
    def task_dim(self):
        """The executor's task input a unit: its group card (the places' mix), its mission kind's card, 2 targets x
        (direction cos, sin from the unit's facing, distance, "none")."""
        return 2 * self.card + 8


def to_dict(cfg):
    return asdict(cfg)


def of_dict(d):
    """A GeneralConfig of a checkpoint's dict (unknown fields ignored, missing ones default)."""
    names = {f.name for f in fields(GeneralConfig)}
    return GeneralConfig(**{k: v for k, v in (d or {}).items() if k in names})
