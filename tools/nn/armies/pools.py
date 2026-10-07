"""The units each faction may field in training: config/nn/pools.json joined with the passports
(config/nn/units.json) and the campaign's army templates (config/nn/army_templates.json).

    pool = load()["wh2_main_skv_skaven"]
    pool.lord.cost, [u.key for u in pool.units], pool.cap("inf_ranged", 10)
    pool.templates        # (Template(name, weight, shares, asks), ...)
    pool.budget_factor    # 0.8: the faction gets 0.8 of the other side's gold (1.0 default)

How a template's ratios reach the pool's units (pools.json "template_rule"):
  "group" (07.10.2026, the default): a ratio on a generator group goes to the pool's units of that group or of a
      group below it (the groups form a tree through cdir_military_generator_unit_group_overrides, child -> parent:
      the templates ask groups that hold no unit themselves - melee_infantry 54 times, ranged_infantry_main 21 times
      in the database, every unit sits in a leaf such as ..._frontline_spears_high_quality - so a group stands for
      its subtree); when none of those can be bought (the pool lacks them, or the budget no longer fits them) the
      ratio goes to the parent group's subtree, and so on up (our pool is a subset of the faction's roster: the
      stand-in for the units it lacks); a ratio whose chain reaches no unit is dropped (artillery, cavalry,
      monsters: the shares of the rest renormalise). Within the group reached, an even split: the database's
      quality (cdir_military_generator_unit_qualities, "the recruitment priority within a group" by the modders)
      has no known rule in numbers.
  "family" (before 07.10.2026): a template's ratios summed by family (the root of a group's chain), split evenly
      among the pool's units of the family (family_shares).

numpy, no torch.
"""
import dataclasses
import json
import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

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


class Template(NamedTuple):
    """One army template of a faction for its pool: name, weight (its priority), shares (per pool unit, sum 1: what
    the template asks when every unit can be bought) and asks ((ratio, levels), ...): levels - the pool units
    (indices) under the asked group, then under its parent, and so on (Template.weights)."""
    name: str
    weight: float
    shares: tuple
    asks: tuple

    def weights(self, ok):
        """[n units] the template's weights of the units that can be bought now (ok: [n] bool): each ratio goes to
        the first level of its chain with a unit that can be bought, split evenly among those."""
        w = np.zeros(len(ok))
        for ratio, levels in self.asks:
            for level in levels:
                can = [i for i in level if ok[i]]
                if can:
                    w[can] += ratio / len(can)
                    break
        return w


@dataclass(frozen=True)
class Pool:
    faction: str
    lord: Unit
    units: tuple         # (Unit, ...) the side buys from
    caps: dict           # {category: share of the army's units, lord counted}
    generator: str = ""  # the faction's army generator config (WH_Empire, ...)
    templates: tuple = ()  # (Template, ...)
    budget_factor: float = 1.0  # gold relative to the other side's after the equal-budget draw
    budget_max: float = None    # the most this faction's side spends (pools.json faction "budget_max"; None: no cap)

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


def chain(group, parents):
    """[group, its parent, ...] up the generator's group tree."""
    out, seen = [group], {group}
    while group in parents:
        group = parents[group]
        assert group not in seen, f"a loop of parents at {group}"
        seen.add(group)
        out.append(group)
    return out


def group_asks(ratios, unit_groups, parents):
    """((ratio, levels), ...) of a template under the "group" rule (module doc): levels - per group of the asked
    group's chain, the units (indices) whose own groups lie under it; empty levels dropped; asks reaching no unit
    dropped."""
    under = [set(g for own in groups for g in chain(own, parents)) for groups in unit_groups]
    out = []
    for group, ratio in sorted(ratios.items()):
        if ratio <= 0:
            continue
        levels = tuple(lv for lv in (tuple(i for i, u in enumerate(under) if g in u) for g in chain(group, parents))
                       if lv)
        if levels:
            out.append((float(ratio), levels))
    return tuple(out)


def family_asks(ratios, families, parents):
    """The "family" rule (family_shares) as asks: one per unit, its share, a single level."""
    shares = family_shares(ratios, families, parents)
    return None if shares is None else tuple((s, ((i,),)) for i, s in enumerate(shares) if s > 0)


def _templates(faction, units, doc, rule="group"):
    """(units with their families, templates) of a faction from army_templates.json."""
    f = doc["factions"].get(faction)
    if f is None:
        return units, ()
    parents = doc["parents"]
    units = tuple(dataclasses.replace(u, family=root(f["units"][u.key][0]["group"], parents))
                  if f["units"].get(u.key) else u for u in units)
    families = [u.family for u in units]
    unit_groups = [[g["group"] for g in f["units"].get(u.key) or ()] for u in units]
    assert rule in ("group", "family"), f"unknown template_rule {rule!r}"
    out = []
    for name, t in f["templates"].items():
        if t["priority"] <= 0:
            continue
        asks = (group_asks(t["ratios"], unit_groups, parents) if rule == "group"
                else family_asks(t["ratios"], families, parents))
        if not asks:
            continue
        template = Template(name, float(t["priority"]), (), asks)
        w = template.weights(np.ones(len(units), bool))
        out.append(template._replace(shares=tuple(float(x) for x in w / w.sum())))
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
            units, templates = _templates(faction, units, tdoc, doc.get("template_rule", "family"))
        factor = float(spec.get("budget_factor", 1.0))
        assert factor > 0, f"{faction}: budget_factor {factor} must be positive"
        cap = spec.get("budget_max")
        out[faction] = Pool(faction=faction, lord=lord, units=units, caps=caps,
                            generator=spec.get("generator", ""), templates=templates, budget_factor=factor,
                            budget_max=None if cap is None else float(cap))
    return out


def budget_max(path=None):
    """The largest budget B drawn (config/nn/pools.json "budget_max"; None: what the pools can field).
    Keeps the budgets where they were when dear units join a pool (02.10.2026: 7725, the Empire's
    most before the greatswords: 19 swordsmen and the general)."""
    doc = json.loads((path or POOLS).read_text(encoding="utf-8"))
    v = doc.get("budget_max")
    return None if v is None else float(v)


def budget_rare(path=None):
    """(share, most) of config/nn/pools.json "budget_rare": that share of battles draws its budget B above the usual
    top (budget_max and the factions' budget_max), up to `most` (and what both sides can field with 19 units and the
    lord); (0.0, None): never (the default). Raised step by step for rich battles of elite units."""
    doc = json.loads((path or POOLS).read_text(encoding="utf-8"))
    r = doc.get("budget_rare") or {}
    share, most = float(r.get("share", 0.0)), r.get("max")
    assert 0.0 <= share <= 1.0, f"budget_rare share {share}"
    return (share, None if most is None else float(most)) if share > 0 and most is not None else (0.0, None)


def mix(path=None):
    """{"template": share, "random": share} of armies (config/nn/pools.json "mix")."""
    doc = json.loads((path or POOLS).read_text(encoding="utf-8"))
    m = {k: float(v) for k, v in doc.get("mix", {"template": 1.0}).items() if not k.startswith("_")}
    assert set(m) <= {"template", "random"} and abs(sum(m.values()) - 1) < 1e-9, f"bad mix {m}"
    return m
