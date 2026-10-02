"""The units each faction may field in training: config/nn/pools.json joined with the passports
(config/nn/units.json) and the campaign's army templates (config/nn/army_templates.json).

    pool = load()["wh2_main_skv_skaven"]
    pool.lord.cost, [u.key for u in pool.units], pool.cap("inf_ranged", 10)
    pool.templates        # ((name, weight, (share of each pool unit, ...)), ...)
    pool.budget_factor    # 0.8: the faction gets 0.8 of the other side's gold (1.0 default)

Plain python: no numpy, no torch.
"""
import dataclasses
import json
import math
from dataclasses import dataclass

from tools import config as project

POOLS = project.CONFIG_DIR / "nn" / "pools.json"
PASSPORTS = project.CONFIG_DIR / "nn" / "units.json"
TEMPLATES = project.CONFIG_DIR / "nn" / "army_templates.json"
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
    family: str = ""     # root of its army generator group (army_templates.json), "" unknown


@dataclass(frozen=True)
class Pool:
    faction: str
    lord: Unit
    units: tuple         # (Unit, ...) the side buys from
    caps: dict           # {category: share of the army's units, lord counted}
    generator: str = ""  # the faction's army generator config (WH_Empire, ...)
    templates: tuple = ()  # ((template, weight, (share per unit of `units`)), ...)
    budget_factor: float = 1.0  # gold relative to the other side's after the equal-budget draw

    def cap(self, category, n_units):
        """Most units of a category in an army of the lord and n_units units (None = no cap)."""
        share = self.caps.get(category)
        return None if share is None else max(1, math.floor(share * (n_units + 1)))


def depth(men, width, spacing=SPACING_M):
    """Depth of a formation of `men` with a `width` front, m (0 for one man)."""
    if men <= 1:
        return 0.0
    files = max(1, min(men, round(width / spacing)))
    return math.ceil(men / files) * spacing


def root(group, parents):
    """The last group of a group's parent chain."""
    seen = {group}
    while group in parents:
        group = parents[group]
        assert group not in seen, f"a loop of parents at {group}"
        seen.add(group)
    return group


def family_shares(ratios, families, parents):
    """A template's ratios {group: ratio} summed by family (root group) over the families present
    in the pool, then split evenly among the pool's units of a family: (share per unit), sum 1;
    None when the template names none of the pool's families."""
    total = {}
    for group, ratio in ratios.items():
        fam = root(group, parents)
        if fam in families and ratio > 0:
            total[fam] = total.get(fam, 0) + ratio
    s = sum(total.values())
    if not s:
        return None
    return tuple(total.get(f, 0) / s / families.count(f) for f in families)


def _unit(spec, passports, slot):
    key = spec["key"]
    assert key in passports, f"{key}: no passport in {PASSPORTS.name}"
    p = passports[key]
    width = float(spec.get("width") or round(p["men"] / max(1, p["rank_depth"])) * SPACING_M)
    return Unit(key=key, slot=spec.get("slot", slot), cost=int(p["multiplayer_cost"]), category=p["category"],
                missile=p.get("missile") is not None, men=int(p["men"]), width=width, depth=depth(p["men"], width))


def _templates(faction, units, doc):
    """(units with their families, templates) of a faction from army_templates.json."""
    f = doc["factions"].get(faction)
    if f is None:
        return units, ()
    parents = doc["parents"]
    units = tuple(dataclasses.replace(u, family=root(f["units"][u.key][0]["group"], parents))
                  if f["units"].get(u.key) else u for u in units)
    families = [u.family for u in units]
    out = []
    for name, t in f["templates"].items():
        shares = family_shares(t["ratios"], families, parents)
        if shares is not None and t["priority"] > 0:
            out.append((name, float(t["priority"]), shares))
    return units, tuple(out)


def load(path=None, passports_path=None, templates_path=None):
    """{faction: Pool}. templates_path=False: without the army templates."""
    doc = json.loads((path or POOLS).read_text(encoding="utf-8"))
    passports = json.loads((passports_path or PASSPORTS).read_text(encoding="utf-8"))["units"]
    tdoc = None
    if templates_path is not False and (templates_path or TEMPLATES).exists():
        tdoc = json.loads((templates_path or TEMPLATES).read_text(encoding="utf-8"))
    caps = {k: float(v) for k, v in doc.get("caps", {}).items() if not k.startswith("_")}
    out = {}
    for faction, spec in doc["factions"].items():
        lord = dataclasses.replace(_unit(spec["lord"], passports, "lord"), slot="lord")
        units = tuple(_unit(u, passports, u["key"]) for u in spec["units"])
        assert units, f"{faction}: an empty pool"
        slots = [u.slot for u in units]
        assert len(slots) == len(set(slots)) and "lord" not in slots, f"{faction}: slots must be unique"
        templates = ()
        if tdoc is not None:
            units, templates = _templates(faction, units, tdoc)
        factor = float(spec.get("budget_factor", 1.0))
        assert factor > 0, f"{faction}: budget_factor {factor} must be positive"
        out[faction] = Pool(faction=faction, lord=lord, units=units, caps=caps,
                            generator=spec.get("generator", ""), templates=templates, budget_factor=factor)
    return out


def mix(path=None):
    """{"template": share, "random": share} of armies (config/nn/pools.json "mix")."""
    doc = json.loads((path or POOLS).read_text(encoding="utf-8"))
    m = {k: float(v) for k, v in doc.get("mix", {"template": 1.0}).items() if not k.startswith("_")}
    assert set(m) <= {"template", "random"} and abs(sum(m.values()) - 1) < 1e-9, f"bad mix {m}"
    return m
