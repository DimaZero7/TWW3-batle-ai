"""Ability passports: what each ability of our units does, from the game's database.

Reads data/db.pack (tools/nn/dbtables.py) and writes config/nn/abilities.json, one passport per
ability key the units of config/nn/units.json own (their `abilities`). The network's input
(tools/nn/model/abilities.py) and the simulator (tools/nn/sim/abilities.py) both read this file:
an ability is known by what it does (times, reach, targets, effects), never by its name, so a new
lord's ability needs a passport, not a new network.

    py -3.14 -m tools.nn.abilities                  # the abilities of the units in units.json
    py -3.14 -m tools.nn.abilities --abilities k1,k2   # more keys

Needs Python 3.14+ (compression.zstd) to read the game. A lord with a card from battle
(data/roster/<key>.json) is checked against it: which of his abilities are passive.
"""
import argparse
import json
import sys
from pathlib import Path

from tools import config as project

OUT = project.CONFIG_DIR / "nn" / "abilities.json"
UNITS = project.CONFIG_DIR / "nn" / "units.json"
ROSTER = project.ROOT / "data" / "roster"
TABLES = ("special_ability_to_special_ability_phase_junctions", "special_ability_phase_stat_effects",
          "special_ability_phase_attribute_effects")
PREFIX = "unit_special_abilities"

# Where each passport field comes from (inferred: the column's meaning is ours, from its values).
FIELDS = {
    "active_s": "unit_special_abilities.active_time (inferred), s; -1: no duration (passive or instant)",
    "recharge_s": "unit_special_abilities.recharge_time (inferred), s; -1: none",
    "uses": "unit_special_abilities.num_uses (inferred); -1: unlimited",
    "range_m": "unit_special_abilities.effect_range (inferred), m: the reach around the caster (0: itself)",
    "passive": "active_s and recharge_s both -1 (inferred); checked against the cards' owned_passive lists",
    "targets_own": "unit_special_abilities field 5 (inferred): 1 for abilities cast on the caster or a friend",
    "friendly_units": "unit_special_abilities.num_effected_friendly_units (inferred): -1 all in range, n, 0 none",
    "enemy_units": "unit_special_abilities.num_effected_enemy_units (inferred): -1 all in range, n, 0 none",
    "self_cast": "active, targets_own, friendly_units and enemy_units 0 or -1 (ours): used on the caster "
                 "himself, no target to choose (perform_special_ability(key, the lord))",
    "phases": "special_ability_to_special_ability_phase_junctions for the ability: the phase named as the "
              "ability when there is one, else all its phases",
    "targets": "the phases' target_self / target_friends / target_enemies (inferred: self-only abilities "
               "have 1 0 0, auras 1 1 0, hexes 0 0 1)",
    "effects": "special_ability_phase_stat_effects of the phases: stat, how (add or mult), value, and on "
               "whom (the phase's targets)",
    "attributes": "special_ability_phase_attribute_effects of the phases (attribute, positive/negative)",
}
MISSING = {
    "area damage": "battle_vortexs / projectile bombardments are not decoded: no damage numbers yet "
                   "(Verminous Valour's 25 m, 1 s blast has damage 0 and AP 0 by an earlier hand decode)",
    "conditions": "when a passive is on (Single Entity, Scurry Away, Strength in Numbers have conditions) is "
                  "in tables not decoded; a passive's effects are what it gives while on",
    "ui type": "unit_abilities (hex, augment, ...) is not decoded: the targets say the same",
    "initial recharge": "not found among the decoded fields: abilities start ready (as the simulator assumed)",
}


def phases_of(key, junctions):
    """The ability's phases with their targets: the phase named as the ability when it has one (the
    others are variants that skills or other units add), else all."""
    rows = [r for r in junctions if r["special_ability"] == key]
    own = [r for r in rows if r["phase"] == key]
    return own or rows


def passport(key, prefix, t):
    """One ability's passport from its unit_special_abilities prefix and the decoded tables t."""
    p = prefix[key]
    phases = phases_of(key, t["special_ability_to_special_ability_phase_junctions"])
    targets = {k: any(r[f"target_{k}"] for r in phases) for k in ("self", "friends", "enemies")}
    effects, attributes = [], []
    for ph in phases:
        on = [k for k in ("self", "friends", "enemies") if ph[f"target_{k}"]]
        for r in t["special_ability_phase_stat_effects"]:
            if r["phase"] == ph["phase"]:
                effects.append({"stat": r["stat"], "how": r["how"], "value": r["value"], "on": on,
                                "phase": ph["phase"]})
        for r in t["special_ability_phase_attribute_effects"]:
            if r["phase"] == ph["phase"]:
                attributes.append({"attribute": r["attribute"], "effect": r["effect"], "on": on,
                                   "phase": ph["phase"]})
    passive = p["active_time"] < 0 and p["recharge_time"] < 0
    friendly, enemy = p["num_effected_friendly_units"], p["num_effected_enemy_units"]
    return {
        "active_s": p["active_time"], "recharge_s": p["recharge_time"], "uses": p["num_uses"],
        "range_m": p["effect_range"], "passive": passive, "targets_own": p["targets_own"],
        "friendly_units": friendly, "enemy_units": enemy,
        "self_cast": (not passive) and p["targets_own"] and friendly in (0, -1) and enemy in (0, -1),
        "phases": [r["phase"] for r in phases], "targets": targets,
        "effects": sorted(effects, key=lambda e: (e["phase"], e["stat"], e["how"])),
        "attributes": sorted(attributes, key=lambda a: (a["phase"], a["attribute"])),
    }


def card_passives(unit_key, roster=ROSTER):
    """(non-passive, passive) ability keys on the unit's card from battle, or None without a card."""
    path = Path(roster) / f"{unit_key}.json"
    if not path.exists():
        return None
    prof = json.loads(path.read_text(encoding="utf-8"))["card"]["profile"]
    return set(prof["owned_non_passive_special_abilities"]), set(prof["owned_passive_special_abilities"])


def check_cards(abilities, units, roster=ROSTER):
    """Differences between the passports' passive flag and the cards: [text]."""
    bad = []
    for unit_key in units:
        card = card_passives(unit_key, roster)
        if card is None:
            continue
        active, passive = card
        for k in active:
            if k in abilities and abilities[k]["passive"]:
                bad.append(f"{unit_key}: {k} is passive in the passport, active on the card")
        for k in passive:
            if k in abilities and not abilities[k]["passive"]:
                bad.append(f"{unit_key}: {k} is active in the passport, passive on the card")
    return bad


def owned(units):
    """Every ability key the units own, sorted."""
    return sorted({k for u in units.values() for k in u.get("abilities") or ()})


def build(raw, tables, keys, units, roster=ROSTER):
    """{key: passport}; raw: the unit_special_abilities bytes, tables: the decoded others."""
    from tools.nn import dbtables
    prefix = dbtables.decode_prefix(raw, PREFIX, keys)
    out = {k: passport(k, prefix, tables) for k in keys}
    bad = check_cards(out, units, roster)
    assert not bad, "passports differ from the cards:\n" + "\n".join(bad)
    return out


def document(abilities, source):
    return {"_description": "Ability passports (docs/en/training/model.md, abilities): what each ability of our "
                            "units does, from the game's database. Written by py -3.14 -m tools.nn.abilities.",
            "_source": str(source), "_fields": FIELDS, "_missing": MISSING, "abilities": abilities}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--abilities", help="comma-separated extra ability keys")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    from tools.nn import dbtables
    from tools.nn.gamedb import read_entries
    game = Path(project.load()["game_dir"]) / "data" / "db.pack"
    tables = dbtables.read_tables(game, TABLES)
    raw = read_entries(game, {dbtables.entry_name(PREFIX)})[dbtables.entry_name(PREFIX)]
    units = json.loads(UNITS.read_text(encoding="utf-8"))["units"]
    keys = sorted(set(owned(units)) | set(args.abilities.split(",") if args.abilities else ()))
    abilities = build(raw, tables, keys, units)
    args.out.write_text(json.dumps(document(abilities, game), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
    for k, a in abilities.items():
        eff = ", ".join(f"{e['stat']} {e['how']} {e['value']:g}" for e in a["effects"])
        kind = "passive" if a["passive"] else f"{a['active_s']:g} s / {a['recharge_s']:g} s"
        print(f"{k}: {kind}, range {a['range_m']:g} m, self_cast {a['self_cast']}: {eff}")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
