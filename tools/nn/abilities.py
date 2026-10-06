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
          "special_ability_phase_attribute_effects", "special_ability_to_recharge_contexts",
          "special_ability_to_auto_deactivate_flags")
PREFIX = "unit_special_abilities"
PHASES = "special_ability_phases"
# The abilities' combat potential (unit_special_abilities.additional_melee_cp / additional_missile_cp): the two
# float32 values stand together in each ability's row of the game's db.pack (checked 06.10.2026 for every key
# below), but the row's other fields before them are not decoded (variable-length), so the values are copied here
# from twwstats' mirror of the same table (build/conform/cards_cp.json). A unit's strength for the army destruction
# is main_units melee_cp + missile_cp + these (the recorded strategic value at the start: General 950, Warlord 900,
# build/movelords). An ability missing here counts 0.
ABILITY_CP = {
    "wh2_main_character_abilities_verminous_valour": (50, 50),
    "wh_dlc04_unit_passive_strength_of_the_penitent": (25, 0),
    "wh_main_character_abilities_deadly_onslaught": (150, 0),
    "wh_main_character_abilities_foe_seeker": (50, 50),
    "wh_main_character_abilities_rally": (50, 50),
    "wh_main_character_abilities_stand_your_ground": (100, 50),
    "wh_main_lord_passive_hold_the_line": (75, 25),
}

# Where each passport field comes from (inferred: the column's meaning is ours, from its values).
FIELDS = {
    "active_s": "unit_special_abilities.active_time (inferred), s; -1: no duration (passive or instant)",
    "recharge_s": "unit_special_abilities.recharge_time (inferred), s; -1: none",
    "initial_s": "unit_special_abilities.initial_recharge (inferred), s: the ability is on cooldown this long from "
                 "the battle's start (-1 and 0: ready at once; Steam guide 1698960734 'most abilities start at 0'; "
                 "Gate of Khorne 60 = fandom)",
    "uses": "unit_special_abilities.num_uses (inferred); -1: unlimited",
    "vigour_per_s": "special_ability_phases.fatigue_change_ratio of the ability's phases (the column name: the "
                    "community, twwstats; its place: the first float after the key, -0.01 for Foe-Seeker = fandom "
                    "'Vigour per second: -1%'): the share of the maximum fatigue (30000) the owner loses each second "
                    "while it is active (negative: recovers; the in-game probe build/movelords: -300 points/s)",
    "cp": "unit_special_abilities.additional_melee_cp / additional_missile_cp (ABILITY_CP: copied from twwstats' "
          "mirror of the table, the values stand in the db rows): the ability's combat potential",
    "range_m": "unit_special_abilities.effect_range (inferred), m: the reach around the caster (0: itself)",
    "passive": "active_s and recharge_s both -1 (inferred); checked against the cards' owned_passive lists",
    "targets_own": "unit_special_abilities field 5 (inferred): 1 for abilities cast on the caster or a friend",
    "friendly_units": "unit_special_abilities.num_effected_friendly_units (inferred): -1 all in range, n, 0 none",
    "enemy_units": "unit_special_abilities.num_effected_enemy_units (inferred): -1 all in range, n, 0 none",
    "update_targets": "unit_special_abilities field 8, after num_effected_enemy_units (inferred; the community's "
                      "schemas call it update_targets_every_frame): 1 - the ability picks its targets in range every "
                      "frame, an aura that follows the caster (Rally, Hold the Line); 0 - it is laid once at the cast "
                      "on the units in range, and they keep it for active_s wherever they go, later arrivals get "
                      "nothing (Stand Your Ground; 142 recorded battles, build/effects/spec.md 3.2)",
    "self_cast": "active, not auto, targets_own, friendly_units and enemy_units 0 or -1 (ours): used on the "
                 "caster himself, no target to choose (perform_special_ability(key, the lord))",
    "recharge_when": "special_ability_to_recharge_contexts.context of the ability: its recharge runs only while "
                     "this holds (Strength of the Penitent: losing_melee_combat - 3 s while losing, CA hotfix 6.2.2; as the "
                     "game acts on it: while not winning its melee, out of melee too - build/effectsimpl; "
                     "Wounds: health_below_25% - its 5 s initial recharge runs once below a quarter); a passive "
                     "without an active time is on only once it has recharged so",
    "off_when": "special_ability_to_auto_deactivate_flags.flag (inferred): the effect switches off while this "
                "holds (Frenzy: morale_is_lower_than_half_of_base_morale; Penitent: out_of_melee)",
    "auto": "a timed passive (ours): an active time, a key with '_passive_' (the game lists it among the "
            "passives, unit_abilities) - the game fires it by itself whenever it is ready and nothing holds it off "
            "(off_when; the recordings: Strength of the Penitent on at the contact, 53 of 54, without losses too); "
            "no player or network order",
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
                   "(Verminous Valour's 25 m, 1 s blast has damage 0 and AP 0 by an earlier hand decode; it throws the "
                   "men around the Warlord back: probe T-E3, 2 s after the cast no swordsman within 6 m, the nearest "
                   "6.5-8.6 m away, in contact again after ~8 s - its force is not decoded, the simulator has no knockback)",
    "conditions": "decoded: recharge_when (special_ability_to_recharge_contexts) and off_when "
                  "(special_ability_to_auto_deactivate_flags); a passive's effects are what it gives while on; "
                  "passive and auto abilities are innate effects (config/nn/effects.json, python -m tools.nn.effects)",
    "ui type": "unit_abilities (hex, augment, ...) is not decoded: the targets say the same",
}


def phases_of(key, junctions):
    """The ability's phases with their targets: the phase named as the ability when it has one (the
    others are variants that skills or other units add), else all."""
    rows = [r for r in junctions if r["special_ability"] == key]
    own = [r for r in rows if r["phase"] == key]
    return own or rows


def phase_fatigue(b, phases):
    """{phase: fatigue_change_ratio} from special_ability_phases (bytes b): a phase's row starts with its key (the
    table's rows are sorted by key: the first place the key is written), then three flags, then the ratio (float).
    A phase whose row does not read so raises: never a guess."""
    import struct
    out = {}
    for key in phases:
        raw = key.encode("utf-8")
        at = b.find(struct.pack("<H", len(raw)) + raw)
        if at < 0:
            raise ValueError(f"{PHASES}: no row for {key}")
        p = at + 2 + len(raw)
        flags, ratio = b[p:p + 3], struct.unpack("<f", b[p + 3:p + 7])[0]
        if any(x not in (0, 1) for x in flags) or not -1.0 <= ratio <= 1.0:
            raise ValueError(f"{PHASES}: {key} does not read as flags and a ratio")
        out[key] = round(ratio, 6)
    return out


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
    auto = not passive and "_passive_" in key
    friendly, enemy = p["num_effected_friendly_units"], p["num_effected_enemy_units"]
    return {
        "active_s": p["active_time"], "recharge_s": p["recharge_time"],
        "initial_s": max(0.0, p.get("initial_recharge", 0.0)),
        "uses": p["num_uses"],
        # the owner's own phases (targets self): what his fatigue does while it is active
        "vigour_per_s": round(sum(t.get("phase_fatigue", {}).get(ph["phase"], 0.0) for ph in phases
                                  if ph["target_self"]), 6),
        "cp": list(ABILITY_CP.get(key, (0, 0))),
        "range_m": p["effect_range"], "passive": passive, "targets_own": p["targets_own"],
        "friendly_units": friendly, "enemy_units": enemy,
        "self_cast": (not passive) and not auto and p["targets_own"] and friendly in (0, -1) and enemy in (0, -1),
        "update_targets": bool(p["update_targets"]),
        "auto": auto,
        "recharge_when": sorted(r["context"] for r in t.get("special_ability_to_recharge_contexts", ())
                                if r["special_ability"] == key),
        "off_when": sorted(r["flag"] for r in t.get("special_ability_to_auto_deactivate_flags", ())
                           if r["special_ability"] == key),
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
            if k in abilities and not (abilities[k]["passive"] or abilities[k].get("auto")):
                bad.append(f"{unit_key}: {k} is active in the passport, passive on the card")
    return bad


def owned(units):
    """Every ability key the units own, sorted."""
    return sorted({k for u in units.values() for k in u.get("abilities") or ()})


def build(raw, tables, keys, units, roster=ROSTER, phases_raw=None):
    """{key: passport}; raw: the unit_special_abilities bytes, tables: the decoded others."""
    from tools.nn import dbtables
    prefix = dbtables.decode_prefix(raw, PREFIX, keys)
    if phases_raw is not None:
        names = sorted({r["phase"] for k in keys
                        for r in phases_of(k, tables["special_ability_to_special_ability_phase_junctions"])})
        tables = dict(tables, phase_fatigue=phase_fatigue(phases_raw, names))
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
    got = read_entries(game, {dbtables.entry_name(PREFIX), dbtables.entry_name(PHASES)})
    raw, phases_raw = got[dbtables.entry_name(PREFIX)], got[dbtables.entry_name(PHASES)]
    units = json.loads(UNITS.read_text(encoding="utf-8"))["units"]
    keys = sorted(set(owned(units)) | set(args.abilities.split(",") if args.abilities else ()))
    abilities = build(raw, tables, keys, units, phases_raw=phases_raw)
    args.out.write_text(json.dumps(document(abilities, game), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
    for k, a in abilities.items():
        eff = ", ".join(f"{e['stat']} {e['how']} {e['value']:g}" for e in a["effects"])
        kind = "passive" if a["passive"] else f"{a['active_s']:g} s / {a['recharge_s']:g} s" + (
            f" auto, recharges {a['recharge_when']}" if a["auto"] else "")
        kind += f" off {a['off_when']}" if a["off_when"] else ""
        print(f"{k}: {kind}, range {a['range_m']:g} m, self_cast {a['self_cast']}, update_targets "
              f"{a['update_targets']}: {eff}")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
