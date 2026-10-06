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
         "wh2_main_skv_inf_skavenslaves_0", "wh2_main_skv_inf_clanrats_1", "wh2_main_skv_inf_night_runners_1")
TABLES = ("main_units", "land_units", "battle_entities", "melee_weapons", "missile_weapons", "projectiles",
          "unit_armour_types", "unit_shield_types", "unit_attributes_to_groups_junctions",
          "land_units_to_unit_abilites_junctions", "unit_spacings")

# Where each passport field comes from. checked: compared with the unit card from battle
# (check_card); inferred: the column's meaning is ours, from its values (docs/*/game/database.md).
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
    "spacing.h": ("unit_spacings.close_h (inferred): a man's place across the front in close formation, m",
                  "data/roster widths: front = (files - 1) h, files = floor(width / h)"),
    "spacing.v": ("unit_spacings.close_v (inferred): between ranks, m", "data/roster: depth = (ranks - 1) v"),
    "spacing.scatter": ("unit_spacings.close_scatter (inferred), m", None),
    "spacing.chaotic": ("unit_spacings.chaotic (inferred): a chaotic (loose, irregular) formation", None),
    "entity": ("land_units.man_entity", None),
    "hp_per_man": ("battle_entities.hit_points + land_units.bonus_hit_points", "card HealthMax / men"),
    "hp_total": ("hp_per_man * men", "card HealthMax"),
    "mass": ("battle_entities.mass", "card Mass"),
    "size": ("battle_entities.size (inferred)", None),
    "height_m": ("battle_entities.height (inferred)", None),
    "radius_m": ("battle_entities.radius (inferred)", None),
    "speed.walk": ("battle_entities.walk_speed, m/s", "card slow_speed"),
    "speed.run": ("battle_entities.run_speed, m/s", "card fast_speed; scalar_speed = run x 10"),
    "speed.charge": ("battle_entities.charge_speed (inferred), m/s", None),
    "speed.charge_distance": ("battle_entities.charge_distance_commence_run (inferred), m: an attacking unit closes "
                              "the last this many metres to its target at its charge speed (30 infantry, 35 lords; "
                              "the melee probe: the last 30 m at 3.65-3.88 m/s, run 3.0)", None),
    "speed.acceleration": ("battle_entities.acceleration (inferred)", None),
    "speed.deceleration": ("battle_entities.deceleration (inferred)", None),
    "melee.attack": ("land_units.melee_attack", "card stat_melee_attack"),
    "melee.defence": ("land_units.melee_defence", "card stat_melee_defence"),
    "melee.charge_bonus": ("land_units.charge_bonus", "card stat_charge_bonus"),
    "melee.weapon": ("land_units.primary_melee_weapon", None),
    "melee.damage": ("melee_weapons.damage", "damage + ap_damage = card stat_weapon_damage"),
    "melee.ap_damage": ("melee_weapons.ap_damage", "damage + ap_damage = card stat_weapon_damage"),
    "melee.bonus_v_large": ("melee_weapons.bonus_v_large (inferred)", None),
    "melee.bonus_v_infantry": ("melee_weapons.bonus_v_infantry (inferred)", None),
    "melee.attack_interval_s": ("melee_weapons.melee_attack_interval (inferred)", None),
    "melee.splash_target_size": ("melee_weapons.splash_attack_target_size (inferred)", None),
    "melee.splash_max_attacks": ("melee_weapons.splash_attack_max_attacks (inferred)", None),
    "melee.splash_power_multiplier": ("melee_weapons.splash_attack_power_multiplier (inferred)", None),
    "armour": ("unit_armour_types.armour_value of land_units.armour", "card stat_armour"),
    "armour_material": ("unit_armour_types.material (inferred)", None),
    "shield.key": ("land_units.shield", None),
    "shield.missile_block_chance": ("unit_shield_types.missile_block_chance (inferred), %", None),
    "leadership": ("land_units.morale", None),
    "damage_resist": ("land_units.damage_mod_all/physical/missile/magic/flame, %", None),
    "can_skirmish": ("land_units.can_skirmish", None),
    "can_brace": ("land_units.can_brace", None),
    "hiding_scalar": ("land_units.hiding_scalar", None),
    "attributes": ("unit_attributes_to_groups_junctions for land_units.attribute_group", None),
    "abilities": ("land_units_to_unit_abilites_junctions.ability", "card owned abilities"),
    "multiplayer_cost": ("main_units.multiplayer_cost", None),
    "missile.weapon": ("land_units.primary_missile_weapon", None),
    "missile.projectile": ("missile_weapons.default_projectile", None),
    "missile.ammo": ("land_units.primary_ammo, per man", "card stat_ammo"),
    "missile.accuracy": ("land_units.accuracy", None),
    "missile.category": ("projectiles.category", None),
    "missile.range_m": ("projectiles.effective_range", "card scalar_missile_range"),
    "missile.min_range_m": ("projectiles.minimum_range (inferred)", None),
    "missile.reload_s": ("projectiles.base_reload_time",
                         "card stat_missile_damage_over_time = (damage + ap_damage) x 10 / reload_s"),
    "missile.damage": ("projectiles.damage", "see reload_s"),
    "missile.ap_damage": ("projectiles.ap_damage", "see reload_s"),
    "missile.projectile_number": ("projectiles.projectile_number (inferred)", None),
    "missile.shots_per_volley": ("projectiles.shots_per_volley (inferred)", None),
    "missile.trajectory": ("projectiles.trajectory (inferred)", None),
    "missile.direct": ("projectiles.trajectory == 'low' (ours): flat, fixed-speed fire (guns) that needs a clear line "
                       "past friendly units (docs/en/game/mechanics/missiles.md: `low` flat; `dual_low_fixed` "
                       "switches to an arc when blocked, `fixed` arcs)", None),
    "missile.muzzle_velocity": ("projectiles.muzzle_velocity (inferred), m/s", None),
    "missile.max_elevation": ("projectiles.max_elevation (inferred), degrees", None),
    "missile.calibration_distance_m": ("projectiles.calibration_distance (inferred)", None),
    "missile.calibration_area_m": ("projectiles.calibration_area (inferred)", None),
    "missile.explosion": ("projectiles.explosion_type (inferred)", None),
    "missile.penetration": ("projectiles.penetration (inferred)", None),
    "missile.shot_type": ("projectiles.shot_type (inferred)", None),
}
# Asked for, but not in the tables read (or not decoded): why (docs/*/training/units.md).
MISSING = {
    "missile.bonus_v_large / bonus_v_infantry": "no such column found: the anti-large arrow row equals the plain arrow",
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
        "spacing": {"template": land["spacing"], "h": sp["close_h"], "v": sp["close_v"],
                    "scatter": sp["close_scatter"], "chaotic": sp["chaotic"]},
        "hp_per_man": hp, "hp_total": hp * main["num_men"], "mass": entity["mass"], "size": entity["size"],
        "height_m": entity["height"], "radius_m": entity["radius"],
        "speed": {"walk": entity["walk_speed"], "run": entity["run_speed"], "charge": entity["charge_speed"],
                  "charge_distance": entity["charge_distance_commence_run"],
                  "acceleration": entity["acceleration"], "deceleration": entity["deceleration"]},
        "melee": {"attack": land["melee_attack"], "defence": land["melee_defence"],
                  "charge_bonus": land["charge_bonus"], "weapon": weapon["key"], "damage": weapon["damage"],
                  "ap_damage": weapon["ap_damage"], "bonus_v_large": weapon["bonus_v_large"],
                  "bonus_v_infantry": weapon["bonus_v_infantry"],
                  "attack_interval_s": weapon["melee_attack_interval"],
                  "splash_target_size": weapon["splash_attack_target_size"],
                  "splash_max_attacks": weapon["splash_attack_max_attacks"],
                  "splash_power_multiplier": weapon["splash_attack_power_multiplier"]},
        "armour": armour["armour_value"], "armour_material": armour["material"],
        "shield": {"key": shield["key"], "missile_block_chance": shield["missile_block_chance"]},
        "leadership": land["morale"],
        "damage_resist": {k: land[f"damage_mod_{k}"] for k in ("all", "physical", "missile", "magic", "flame")},
        "can_skirmish": land["can_skirmish"], "can_brace": land["can_brace"], "hiding_scalar": land["hiding_scalar"],
        "attributes": sorted(r["attribute"] for r in t["unit_attributes_to_groups_junctions"]
                             if land["attribute_group"] and r["attribute_group"] == land["attribute_group"]),
        "abilities": sorted(r["ability"] for r in t["land_units_to_unit_abilites_junctions"]
                            if r["land_unit"] == land["key"]),
        "multiplayer_cost": main["multiplayer_cost"],
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
            "trajectory": pr["trajectory"], "direct": pr["trajectory"] == "low",
            "muzzle_velocity": pr["muzzle_velocity"],
            "max_elevation": pr["max_elevation"], "calibration_distance_m": pr["calibration_distance"],
            "calibration_area_m": pr["calibration_area"], "explosion": pr["explosion_type"],
            "penetration": pr["penetration"], "shot_type": pr["shot_type"]}
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
        pairs += [("missile.ammo", m.get("ammo"), s.get("stat_ammo")),
                  ("missile.range_m", m.get("range_m"), s.get("scalar_missile_range")),
                  ("missile damage per 10 s", round((m.get("damage", 0) + m.get("ap_damage", 0)) * 10
                                                    / m.get("reload_s", 1), 2),
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


def document(units, source):
    return {"_description": "Unit passports (step 1 of the battle simulator, docs/en/training/units.md): what "
                            "decides a fight, from the game's database. Written by py -3.14 -m tools.nn.units.",
            "_source": str(source),
            "_fields": {k: {"source": s, "checked": c} for k, (s, c) in FIELDS.items()},
            "_missing": MISSING,
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
    args.out.write_text(json.dumps(document(units, game), indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                        newline="\n")
    for key, p in units.items():
        m = p["missile"]
        print(f"{key}: men {p['men']}, hp {p['hp_per_man']}, ma/md {p['melee']['attack']}/{p['melee']['defence']}, "
              f"armour {p['armour']}, card {p['card_check']}" + (f", missile {m['range_m']} m" if m else ""))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
