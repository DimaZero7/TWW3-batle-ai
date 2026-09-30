"""The units each faction may field in training (config/nn/pools.json joined with the passports
of config/nn/units.json).

    pool = load()["wh2_main_skv_skaven"]
    pool.lord.cost, [u.key for u in pool.units], pool.cap_of(n_units=10)

Plain python: no numpy, no torch.
"""
import dataclasses
import json
import math
from dataclasses import dataclass

from tools import config as project

POOLS = project.CONFIG_DIR / "nn" / "pools.json"
PASSPORTS = project.CONFIG_DIR / "nn" / "units.json"
SPACING_M = 1.5          # a man's place in a formation (config/nn/sim.json formation.spacing_m)


@dataclass(frozen=True)
class Unit:
    key: str
    slot: str            # name in an army: slot_1, slot_2, ...
    cost: int            # multiplayer_cost
    category: str        # passport category (inf_melee, inf_ranged, ...)
    missile: bool        # has a missile weapon: deployed behind the melee lines
    men: int
    width: float         # frontage at deployment, m
    depth: float         # formation depth at that frontage, m


@dataclass(frozen=True)
class Pool:
    faction: str
    lord: Unit
    units: tuple         # (Unit, ...) the side buys from
    caps: dict           # {category: share of the army's units, lord counted}

    def cap(self, category, n_units):
        """Most units of a category in an army of the lord and n_units units (None = no cap)."""
        share = self.caps.get(category)
        return None if share is None else max(1, math.floor(share * (n_units + 1)))

    def groups(self):
        """Capped categories in a fixed order."""
        return tuple(sorted(self.caps))


def depth(men, width, spacing=SPACING_M):
    """Depth of a formation of `men` with a `width` front, m (0 for one man)."""
    if men <= 1:
        return 0.0
    files = max(1, min(men, round(width / spacing)))
    return math.ceil(men / files) * spacing


def _unit(spec, passports, slot):
    key = spec["key"]
    assert key in passports, f"{key}: no passport in {PASSPORTS.name}"
    p = passports[key]
    width = float(spec.get("width") or round(p["men"] / max(1, p["rank_depth"])) * SPACING_M)
    return Unit(key=key, slot=spec.get("slot", slot), cost=int(p["multiplayer_cost"]), category=p["category"],
                missile=p.get("missile") is not None, men=int(p["men"]), width=width, depth=depth(p["men"], width))


def load(path=None, passports_path=None):
    """{faction: Pool}."""
    doc = json.loads((path or POOLS).read_text(encoding="utf-8"))
    passports = json.loads((passports_path or PASSPORTS).read_text(encoding="utf-8"))["units"]
    caps = {k: float(v) for k, v in doc.get("caps", {}).items() if not k.startswith("_")}
    out = {}
    for faction, spec in doc["factions"].items():
        lord = dataclasses.replace(_unit(spec["lord"], passports, "lord"), slot="lord")
        units = tuple(_unit(u, passports, u["key"]) for u in spec["units"])
        assert units, f"{faction}: an empty pool"
        slots = [u.slot for u in units]
        assert len(slots) == len(set(slots)) and "lord" not in slots, f"{faction}: slots must be unique"
        out[faction] = Pool(faction=faction, lord=lord, units=units, caps=caps)
    return out
