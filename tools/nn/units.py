"""Unit passports: for each unit, what decides a fight, from the game's database.

Reads data/db.pack (tools/nn/dbtables.py), joins the tables by unit and writes
config/nn/units.json, one passport per main unit key. The simulator and the
network's input both read this file. Every value comes from a database table
(`_fields` in the file names the table and column); a unit with a card from
battle (data/roster/<key>.json) is checked against it before writing.

    py -3.14 -m tools.nn.units                 # the v1 units below
    py -3.14 -m tools.nn.units --units k1,k2   # other units

Needs Python 3.14+ (compression.zstd) to read the game; the card check alone
(`check_card`) runs on any Python.
"""
import argparse
import json
import sys
from pathlib import Path

from tools import config as project

OUT = project.CONFIG_DIR / "nn" / "units.json"
ROSTER = project.ROOT / "data" / "roster"
# v1 test build (user, 30.09.2026): Empire against Skaven, a lord, spears without shields, missile infantry.
UNITS = ("wh_main_emp_cha_general_0", "wh_main_emp_inf_spearmen_0", "wh2_dlc13_emp_inf_archers_0",
         "wh2_main_skv_cha_warlord_0", "wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0",
         "wh2_main_skv_inf_skavenslave_slingers_0",
         # armour-piercing halberds of the lord swarm probe (tools/nn/lord_swarm.py, 01.10.2026)
         "wh_main_emp_inf_halberdiers", "wh2_main_skv_inf_stormvermin_0",
         # spearmen with shields (02.10.2026): the Empire's answer to Skaven slings and arrows
         "wh_main_emp_inf_spearmen_1",
         # swordsmen with sword and shield (02.10.2026): the Empire's melee line against Skaven infantry
         "wh_main_emp_inf_swordsmen",
         # 02.10.2026, the Empire's second wave: unbreakable frenzied flagellants (cheap chaff that never
         # routs), heavy armour-piercing greatswords, and the first DIRECT-fire shooters (pistols, fire
         # whilst moving, decent melee): units.md "Flagellants, Greatswords, Free Company Militia"
         "wh_dlc04_emp_inf_flagellants_0", "wh_main_emp_inf_greatswords", "wh_dlc04_emp_inf_free_company_militia_0",
         # 02.10.2026, the Skaven wave: the cheapest chaff (plain skavenslaves, expendable), clanrats with
         # sword and shield (the shield blocks arrows and pistols from the front) and Night Runners with
         # slings (fast, vanguard, long-ranged): units.md "Skavenslaves, Clanrats with shields, Night Runners"
         "wh2_main_skv_inf_skavenslaves_0", "wh2_main_skv_inf_clanrats_1", "wh2_main_skv_inf_night_runners_1",
         # 07.10.2026, the third wave: the Empire's handgunners (in-game name "Пистольеры": flat armour-piercing fire
         # at 145 m) and crossbowmen (160 m); the Skaven's clanrats with swords (no shield), clanrat spearmen with
         # shields, stormvermin with sword and shield and Night Runners with throwing stars (70 m, a 360 deg fire arc,
         # fire whilst moving): units.md "Handgunners, crossbowmen, clanrats, stormvermin, throwing stars"
         "wh_main_emp_inf_handgunners", "wh_main_emp_inf_crossbowmen", "wh2_main_skv_inf_clanrats_0",
         "wh2_main_skv_inf_clanrat_spearmen_1", "wh2_main_skv_inf_stormvermin_1", "wh2_main_skv_inf_night_runners_0",
         # 11.10.2026, the new units (build/newunits/plan.md; the probes of commit bd0ad4e): the Skaven's Ratling Gun (a
         # burst of 18 bullets a man, suppressing fire), Plague Monks and Plague Censer Bearers (frenzy; the censer's
         # contagion), Giant Rats (war beasts, 9.5 m/s; _1 with poison). Only the v2 army scheme fields them
         # (config/nn/pools.json units_v2): units.md "Ratling Gun, Plague Monks, Censer Bearers, Giant Rats"
         "wh2_dlc12_skv_inf_ratling_gun_0", "wh2_main_skv_inf_plague_monks", "wh2_main_skv_inf_plague_monk_censer_bearer",
         "wh2_dlc16_skv_mon_wolf_rats_0", "wh2_dlc16_skv_mon_wolf_rats_1")
TABLES = ("main_units", "land_units", "battle_entities", "melee_weapons", "missile_weapons", "projectiles",
          "unit_armour_types", "unit_shield_types", "unit_attributes_to_groups_junctions",
          "land_units_to_unit_abilites_junctions", "unit_spacings", "special_ability_phase_stat_effects",
          "special_ability_phases")

# Where each passport field comes from: table.column by the RPFM schema's names (tools/nn/dbschema.py). checked:
# compared with the unit card from battle (check_card); "ours": what we make of a named column.
FIELDS = {
    "land_unit": ("main_units.land_unit", None),
    "caste": ("main_units.caste", None),
    "category": ("land_units.category", None),
    "class": ("land_units.class", "card profile.unit_class"),
    "men": ("main_units.num_men", "card NumEntitiesInitial"),
    "mount": ("land_units.mount", None),
    "num_mounts": ("land_units.num_mounts", None),
    "engine": ("land_units.engine", None),
    "rank_depth": ("land_units.rank_depth", None),
    "spacing.template": ("land_units.spacing (the formation template)", None),
    "spacing.h": ("unit_spacings.close_formation_spacing_horizontal: a man's place across the front in close "
                  "formation, m",
                  "data/roster widths: front = (files - 1) h, files = floor(width / h)"),
    "spacing.v": ("unit_spacings.close_formation_spacing_vertical: between ranks, m",
                  "data/roster: depth = (ranks - 1) v"),
    "spacing.scatter": ("unit_spacings.close_formation_spacing_variation, m", None),
    "spacing.chaotic": ("unit_spacings.is_chaotic_formation: a chaotic (loose, irregular) formation", None),
    "entity": ("land_units.man_entity", None),
    "hp_per_man": ("battle_entities.hit_points + land_units.bonus_hit_points", "card HealthMax / men"),
    "hp_total": ("hp_per_man * men", "card HealthMax"),
    "mass": ("battle_entities.mass", "card Mass"),
    "size": ("battle_entities.size", None),
    "height_m": ("battle_entities.height", None),
    "radius_m": ("battle_entities.radius", None),
    "radii_ratio": ("battle_entities.radii_ratio: an ellipse body (shape 'ellipse') is radius long and radius x this "
                    "across; 1 a circle (every man)", None),
    "speed.walk": ("battle_entities.walk_speed, m/s", "card slow_speed"),
    "speed.run": ("battle_entities.run_speed, m/s", "card fast_speed; scalar_speed = run x 10"),
    "speed.charge": ("battle_entities.charge_speed, m/s", None),
    "speed.charge_distance": ("battle_entities.charge_distance_commence_run, m: an attacking unit closes "
                              "the last this many metres to its target at its charge speed (30 infantry, 35 lords; "
                              "the melee probe: the last 30 m at 3.65-3.88 m/s, run 3.0)", None),
    "speed.charge_pose_distance": ("battle_entities.charge_distance_adopt_charge_pose, m: an attacking unit "
                                   "takes its charge pose this far from its target (25 infantry, 30 lords; the morale "
                                   "probe: the charge's +15 morale starts at a gap of 24.6-27.5 m)", None),
    "speed.acceleration": ("battle_entities.acceleration", None),
    "speed.turn_deg_s": ("battle_entities.turn_speed, deg/s: how fast a man (a model) turns (patch 6.1: horse "
                         "cavalry 100 -> 120 deg/s, and 120 in the DB)", None),
    "speed.deceleration": ("battle_entities.deceleration", None),
    "melee.attack": ("land_units.melee_attack", "card stat_melee_attack"),
    "melee.defence": ("land_units.melee_defence", "card stat_melee_defence"),
    "melee.charge_bonus": ("land_units.charge_bonus", "card stat_charge_bonus"),
    "melee.weapon": ("land_units.primary_melee_weapon", None),
    "melee.damage": ("melee_weapons.damage", "damage + ap_damage = card stat_weapon_damage"),
    "melee.ap_damage": ("melee_weapons.ap_damage", "damage + ap_damage = card stat_weapon_damage"),
    "melee.bonus_v_large": ("melee_weapons.bonus_v_large", None),
    "melee.bonus_v_infantry": ("melee_weapons.bonus_v_infantry", None),
    "melee.attack_interval_s": ("melee_weapons.melee_attack_interval", None),
    "melee.splash_target_size": ("melee_weapons.splash_attack_target_size", None),
    "melee.splash_max_attacks": ("melee_weapons.splash_attack_max_attacks", None),
    "melee.splash_power_multiplier": ("melee_weapons.splash_attack_power_multiplier", None),
    "melee.magic": ("melee_weapons.is_magical: magical attacks - the censer, the plain Giant Rats' claws", None),
    "melee.contact_phase": ("melee_weapons.contact_phase: the contact effect a blow lays on its target (its "
                            "numbers in the document's contact_phases)", None),
    "armour": ("unit_armour_types.armour_value of land_units.armour", "card stat_armour"),
    "armour_material": ("unit_armour_types.audio_type: the armour's sound type (body, leather, chainmail, ...)", None),
    "shield.key": ("land_units.shield", None),
    "shield.missile_block_chance": ("unit_shield_types.missile_block_chance, %", None),
    "leadership": ("land_units.morale", None),
    "damage_resist": ("land_units.damage_mod_all/physical/missile/magic/flame, %", None),
    "can_skirmish": ("land_units.can_skirmish", None),
    "can_brace": ("land_units.can_brace", None),
    "hiding_scalar": ("land_units.hiding_scalar", None),
    "attributes": ("unit_attributes_to_groups_junctions for land_units.attribute_group", None),
    "abilities": ("land_units_to_unit_abilites_junctions.ability", "card owned abilities"),
    "multiplayer_cost": ("main_units.multiplayer_cost", None),
    "combat_potential.melee": ("main_units.melee_cp: the unit's melee combat potential (the army-destruction "
                               "strength with its abilities' cp, config/nn/abilities.json; build/movelords)", None),
    "combat_potential.missile": ("main_units.missile_cp: its missile combat potential (counts by the ammunition "
                                 "left)", None),
    "missile.weapon": ("land_units.primary_missile_weapon", None),
    "missile.projectile": ("missile_weapons.default_projectile", None),
    "missile.ammo": ("land_units.primary_ammo, per man", "card stat_ammo"),
    "missile.accuracy": ("land_units.accuracy", None),
    "missile.category": ("projectiles.category", None),
    "missile.range_m": ("projectiles.effective_range", "card scalar_missile_range"),
    "missile.min_range_m": ("projectiles.minimum_range", None),
    "missile.reload_s": ("projectiles.base_reload_time",
                         "card stat_missile_damage_over_time = (damage + ap_damage) x 10 / reload_s"),
    "missile.damage": ("projectiles.damage", "see reload_s"),
    "missile.ap_damage": ("projectiles.ap_damage", "see reload_s"),
    "missile.projectile_number": ("projectiles.projectile_number", None),
    "missile.shots_per_volley": ("projectiles.shots_per_volley", None),
    "missile.trajectory": ("projectiles.trajectory_sight", None),
    "missile.direct": ("projectiles.trajectory_sight == 'low' (ours): flat, fixed-speed fire (guns) that needs a clear line "
                       "past friendly units (docs/en/game/mechanics/missiles.md: `low` flat; `dual_low_fixed` "
                       "switches to an arc when blocked, `fixed` arcs)", None),
    "missile.muzzle_velocity": ("projectiles.muzzle_velocity, m/s", None),
    "missile.max_elevation": ("projectiles.max_elevation, degrees", None),
    "missile.calibration_distance_m": ("projectiles.calibration_distance", None),
    "missile.calibration_area_m": ("projectiles.calibration_area", None),
    "missile.marksmanship": ("projectiles.marksmanship_bonus: 10 for our arrows, slings and pistol, 15 for the Night "
                             "Runners' sling; added to land_units.accuracy in the spread (ours, tools/nn/sim/missile.py)",
                             None),
    "missile.fire_arc_deg": ("battle_entities.fire_arc_close / 2 (ours: the column is the full angle, 60 / 70 for our "
                             "shooters, 360 for throwing stars), degrees each side", None),
    "missile.explosion": ("projectiles.explosion_type", None),
    "missile.penetration": ("projectiles.projectile_penetration", None),
    "missile.shot_type": ("projectiles.shot_type", None),
    "missile.contact_phase": ("projectiles.contact_stat_effect: the contact effect a hit lays on its target "
                              "(the Ratling Gun's suppressing fire)", None),
}
# The document's contact_phases: per contact phase a unit's weapon lays (melee.contact_phase, missile.contact_phase).
PHASE_FIELDS = {
    "duration_s": "special_ability_phases.duration, s (10 for poison, the censer's contagion and suppressing fire - "
                  "fandom: contact effects last 10 s)",
    "effect": "special_ability_phases.effect_type (positive / negative)",
    "stats": "special_ability_phase_stat_effects rows of the phase: stat, how (add / mult), value",
}
# Asked for, but not in the tables read (or not decoded): why (docs/*/training/units.md).
MISSING = {
    "missile.bonus_v_large / bonus_v_infantry": "projectiles.bonus_v_large / bonus_v_infantry: 0 for every projectile "
                                                "of our units (not in the passport)",
    "missile.can_fire_over_friendlies": "no column of its own: missile.direct (trajectory 'low') says the unit "
                                        "cannot; the others arc over friends (knowledge base, not measured)",
    "fatigue modifiers per unit": "none in land_units / battle_entities; fatigue rules are global "
                                  "(config/nn/game_rules.json, fatigue)",
    "lord aura per unit": "no per-unit radius found; the aura is global (_kv_morale_tables: general_aura_radius, "
                          "general_inspire_effect_amount_*); a lord is marked by the attribute 'encourages'",
    "ability and attribute effects": "only the keys here: the abilities' numbers are their passports "
                                     "(config/nn/abilities.json), the attributes' rules and every innate effect "
                                     "the catalogue config/nn/effects.json (python -m tools.nn.effects)",
    "experience": "per rank, global: config/nn/game_rules.json (experience_bonus, experience_levels)",
}


def _index(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r[key], r)
    return out


def passport(key, t):
    """One unit's passport from decoded tables t ({table: rows})."""
    main = _index(t["main_units"], "unit")[key]
    land = _index(t["land_units"], "key")[main["land_unit"]]
    entity = _index(t["battle_entities"], "key")[land["man_entity"]]
    weapon = _index(t["melee_weapons"], "key")[land["primary_melee_weapon"]]
    armour = _index(t["unit_armour_types"], "key")[land["armour"]]
    shield = _index(t["unit_shield_types"], "key")[land["shield"]]
    sp = _index(t["unit_spacings"], "key")[land["spacing"]]
    hp = entity["hit_points"] + land["bonus_hit_points"]
    out = {
        "land_unit": land["key"], "caste": main["caste"], "category": land["category"], "class": land["class"],
        "men": main["num_men"], "mount": land["mount"], "num_mounts": land["num_mounts"], "engine": land["engine"],
        "rank_depth": land["rank_depth"], "entity": land["man_entity"],
        "spacing": {"template": land["spacing"], "h": sp["close_formation_spacing_horizontal"],
                    "v": sp["close_formation_spacing_vertical"], "scatter": sp["close_formation_spacing_variation"],
                    "chaotic": sp["is_chaotic_formation"]},
        "hp_per_man": hp, "hp_total": hp * main["num_men"], "mass": entity["mass"], "size": entity["size"],
        "height_m": entity["height"], "radius_m": entity["radius"],
        "radii_ratio": entity["radii_ratio"] if entity["shape"] == "ellipse" else 1.0,
        "speed": {"walk": entity["walk_speed"], "run": entity["run_speed"], "charge": entity["charge_speed"],
                  "charge_distance": entity["charge_distance_commence_run"],
                  "charge_pose_distance": entity["charge_distance_adopt_charge_pose"],
                  "acceleration": entity["acceleration"], "deceleration": entity["deceleration"],
                  "turn_deg_s": entity["turn_speed"]},
        "melee": {"attack": land["melee_attack"], "defence": land["melee_defence"],
                  "charge_bonus": land["charge_bonus"], "weapon": weapon["key"], "damage": weapon["damage"],
                  "ap_damage": weapon["ap_damage"], "bonus_v_large": weapon["bonus_v_large"],
                  "bonus_v_infantry": weapon["bonus_v_infantry"],
                  "attack_interval_s": weapon["melee_attack_interval"],
                  "splash_target_size": weapon["splash_attack_target_size"],
                  "splash_max_attacks": weapon["splash_attack_max_attacks"],
                  "splash_power_multiplier": weapon["splash_attack_power_multiplier"],
                  "magic": weapon["is_magical"], "contact_phase": weapon["contact_phase"]},
        "armour": armour["armour_value"], "armour_material": armour["audio_type"],
        "shield": {"key": shield["key"], "missile_block_chance": shield["missile_block_chance"]},
        "leadership": land["morale"],
        "damage_resist": {k: land[f"damage_mod_{k}"] for k in ("all", "physical", "missile", "magic", "flame")},
        "can_skirmish": land["can_skirmish"], "can_brace": land["can_brace"], "hiding_scalar": land["hiding_scalar"],
        "attributes": sorted(r["attribute"] for r in t["unit_attributes_to_groups_junctions"]
                             if land["attribute_group"] and r["attribute_group"] == land["attribute_group"]),
        "abilities": sorted(r["ability"] for r in t["land_units_to_unit_abilites_junctions"]
                            if r["land_unit"] == land["key"]),
        "multiplayer_cost": main["multiplayer_cost"],
        "combat_potential": {"melee": main["melee_cp"], "missile": main["missile_cp"]},
        "missile": None,
    }
    if land["primary_missile_weapon"]:
        mw = _index(t["missile_weapons"], "key")[land["primary_missile_weapon"]]
        pr = _index(t["projectiles"], "key")[mw["default_projectile"]]
        out["missile"] = {
            "weapon": mw["key"], "projectile": pr["key"], "ammo": land["primary_ammo"], "accuracy": land["accuracy"],
            "category": pr["category"], "range_m": pr["effective_range"], "min_range_m": pr["minimum_range"],
            "reload_s": pr["base_reload_time"], "damage": pr["damage"], "ap_damage": pr["ap_damage"],
            "projectile_number": pr["projectile_number"], "shots_per_volley": pr["shots_per_volley"],
            "trajectory": pr["trajectory_sight"], "direct": pr["trajectory_sight"] == "low",
            "muzzle_velocity": pr["muzzle_velocity"],
            "max_elevation": pr["max_elevation"], "calibration_distance_m": pr["calibration_distance"],
            "calibration_area_m": pr["calibration_area"], "marksmanship": pr["marksmanship_bonus"],
            "fire_arc_deg": entity["fire_arc_close"] / 2, "explosion": pr["explosion_type"],
            "penetration": pr["projectile_penetration"], "shot_type": pr["shot_type"],
            "contact_phase": pr["contact_stat_effect"]}
    return out


def phase_keys(units):
    """The contact phases the units' weapons lay (melee.contact_phase, missile.contact_phase), sorted."""
    return sorted({p for u in units.values() for p in ((u.get("melee") or {}).get("contact_phase"),
                                                       (u.get("missile") or {}).get("contact_phase")) if p})


def contact_phases(units, t):
    """{phase key: {"duration_s", "effect", "stats"}} of the contact phases the units' weapons lay (PHASE_FIELDS)."""
    out = {}
    heads = _index(t["special_ability_phases"], "id")
    for k in phase_keys(units):
        stats = sorted(({"stat": r["stat"], "how": r["how"], "value": r["value"]}
                        for r in t["special_ability_phase_stat_effects"] if r["phase"] == k),
                       key=lambda r: (r["stat"], r["how"]))
        out[k] = {"duration_s": heads[k]["duration"], "effect": heads[k]["effect_type"], "stats": stats}
    return out


def card(key, roster=ROSTER):
    path = Path(roster) / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8"))["card"] if path.exists() else None


def check_card(p, c):
    """Differences between a passport and the unit's card from battle: [text]."""
    s = {k: v["base"] for k, v in c["stats"].items()}
    d, prof = c["details"], c["profile"]
    pairs = [("men", p["men"], d["NumEntitiesInitial"]), ("hp_total", p["hp_total"], d["HealthMax"]),
             ("mass", p["mass"], d["Mass"]), ("class", p["class"], prof["unit_class"]),
             ("speed.walk", p["speed"]["walk"], round(prof["slow_speed"], 2)),
             ("speed.run", p["speed"]["run"], round(prof["fast_speed"], 2)),
             ("scalar_speed", round(p["speed"]["run"] * 10), s["scalar_speed"]),
             ("melee.attack", p["melee"]["attack"], s["stat_melee_attack"]),
             ("melee.defence", p["melee"]["defence"], s["stat_melee_defence"]),
             ("melee.charge_bonus", p["melee"]["charge_bonus"], s["stat_charge_bonus"]),
             ("melee damage + ap", p["melee"]["damage"] + p["melee"]["ap_damage"], s["stat_weapon_damage"]),
             ("armour", p["armour"], s["stat_armour"]),
             ("abilities", p["abilities"], sorted(list(prof["owned_non_passive_special_abilities"])
                                                  + list(prof["owned_passive_special_abilities"])))]
    m = p["missile"]
    if m or "stat_ammo" in s:
        m = m or {}
        # (the card's damage over 10 s counts every shot of a volley: the Ratling Gun's 18 - twdatabase 288 = 8 x 18 x
        # 10 / 5)
        pairs += [("missile.ammo", m.get("ammo"), s.get("stat_ammo")),
                  ("missile.range_m", m.get("range_m"), s.get("scalar_missile_range")),
                  ("missile damage per 10 s", round((m.get("damage", 0) + m.get("ap_damage", 0))
                                                    * (m.get("shots_per_volley") or 1) * 10 / m.get("reload_s", 1), 2),
                   round(s.get("stat_missile_damage_over_time", -1), 2))]
    return [f"{name}: passport {a!r}, card {b!r}" for name, a, b in pairs if a != b]


def build(tables, keys=UNITS, roster=ROSTER):
    """{key: passport}, each checked against its card when it has one; raises on a difference."""
    units, bad = {}, []
    for key in keys:
        p = passport(key, tables)
        c = card(key, roster)
        p["card_check"] = "no card" if c is None else ("equal" if not check_card(p, c) else "DIFFERENT")
        bad += [f"{key}: {x}" for x in (check_card(p, c) if c else [])]
        units[key] = p
    assert not bad, "passport differs from the card:\n" + "\n".join(bad)
    return units


def document(units, source, phases=None):
    return {"_description": "Unit passports (step 1 of the battle simulator, docs/en/training/units.md): what "
                            "decides a fight, from the game's database. Written by py -3.14 -m tools.nn.units.",
            "_source": str(source),
            "_fields": {k: {"source": s, "checked": c} for k, (s, c) in FIELDS.items()},
            "_missing": MISSING,
            "_contact_phases": PHASE_FIELDS,
            "contact_phases": phases or {},
            "units": units}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units", help="comma-separated main unit keys (default: the v1 build)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    from tools.nn import dbtables
    game = Path(project.load()["game_dir"]) / "data" / "db.pack"
    tables = dbtables.read_tables(game, TABLES)
    units = build(tables, tuple(args.units.split(",")) if args.units else UNITS)
    phases = contact_phases(units, tables)
    args.out.write_text(json.dumps(document(units, game, phases), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
    for key, p in units.items():
        m = p["missile"]
        print(f"{key}: men {p['men']}, hp {p['hp_per_man']}, ma/md {p['melee']['attack']}/{p['melee']['defence']}, "
              f"armour {p['armour']}, card {p['card_check']}" + (f", missile {m['range_m']} m" if m else ""))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
