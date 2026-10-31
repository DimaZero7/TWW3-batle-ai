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
         "wh2_dlc16_skv_mon_wolf_rats_0", "wh2_dlc16_skv_mon_wolf_rats_1",
         # 25.10.2026, line g43 (build/diag_units/new_units.md): the Empire's Huntsmen (bow 160 m, stalk), Nuln Ironsides
         # (Blackpowder Discipline) and Hochland Long Rifles (260 m, a bullet through two men); the Skaven's Gutter
         # Runners (stalk, Snare Net; throwing stars or slings, plain or poisoned) and Warplock Jezzails (275 m, a pavise,
         # the shield breaker, spotting 160 m). Only the v3 army scheme fields them (config/nn/pools.json units_v3):
         # units.md "Huntsmen, Ironsides, Long Rifles, Gutter Runners, Jezzails"
         "wh2_dlc13_emp_inf_huntsmen_0", "wh3_dlc25_emp_inf_nuln_ironsides", "wh3_dlc25_emp_inf_hochland_long_rifles",
         "wh2_main_skv_inf_gutter_runners_0", "wh2_main_skv_inf_gutter_runners_1",
         "wh2_main_skv_inf_gutter_runner_slingers_0", "wh2_main_skv_inf_gutter_runner_slingers_1",
         "wh2_dlc12_skv_inf_warplock_jezzails_0",
         # 29.10.2026, the Empire's cavalry (build/diag_cav/units.md; the simulator's large bodies: tools/nn/sim/impact.py):
         # Empire Knights, Reiksguard, Knights of the Blazing Sun, Knights of the Black Rose, Demigryph Knights (lances,
         # halberds). No army scheme fields them yet (config/nn/pools.json): units.md "Cavalry"
         "wh_main_emp_cav_empire_knights", "wh_main_emp_cav_reiksguard", "wh_dlc04_emp_cav_knights_blazing_sun_0",
         "wh3_dlc25_emp_cav_knights_of_the_black_rose", "wh_main_emp_cav_demigryph_knights_0",
         "wh_main_emp_cav_demigryph_knights_1",
         # 29.10.2026, artillery step 1 (build/diag_arty/empire.md, skaven.md; the simulator's guns: tools/nn/sim/artillery.py):
         # the Empire's Mortar, Great Cannon, Helblaster Volley Gun and Helstorm Rocket Battery, the Skaven's Plagueclaw
         # Catapult and Warp Lightning Cannon - 4 engines and their crew each. Only the v3 army scheme fields them
         # (config/nn/pools.json units_v3): units.md "Artillery"
         "wh_main_emp_art_mortar", "wh_main_emp_art_great_cannon", "wh_main_emp_art_helblaster_volley_gun",
         "wh_main_emp_art_helstorm_rocket_battery", "wh2_main_skv_art_plagueclaw_catapult",
         "wh2_main_skv_art_warp_lightning_cannon",
         # 29.10.2026, the Skaven monsters (build/diag_skv_mon/units.md, mechanic.md, sim_gap.md; the simulator's very large
         # bodies: the ring of attackers, regeneration, terror - docs/ru/training/simulator.md "Монстры"): Rat Ogres (16
         # large men, brace), the Brood Horror (one very large body: poison, regeneration, terror) and the Rat Ogre Mutant
         # (one very large body: magical attacks, 20 % physical resistance). Only the v3 army scheme fields them
         # (config/nn/pools.json units_v3): units.md "Монстры скавенов"
         "wh2_main_skv_mon_rat_ogres", "wh2_dlc16_skv_mon_brood_horror_0", "wh2_dlc16_skv_mon_rat_ogre_mutant",
         # 31.10.2026, the Pusbags (build/diag_roster/wh3_dlc29_skv_inf_pusbags/units.md, mechanic.md): the skavenslaves'
         # body and weapon with poison, defence 5, 43 HP a man and two passives - Unshakable Pathogen (direct damage to
         # itself in melee: tools/nn/sim/direct.py) and Pestilent Perfection (a vortex at every man's death:
         # tools/nn/sim/vortex.py). Only the v3 army scheme fields them (config/nn/pools.json units_v3)
         "wh3_dlc29_skv_inf_pusbags",
         # 31.10.2026, the Warp Grinders (build/diag_roster/wh2_dlc14_skv_inf_warp_grinder_0/units.md, mechanic.md): 24
         # crews of 184 HP, armour 70, a magical drill 18 + 78 sweeping 4 men; two active abilities without a target -
         # Warp-Quake (direct damage to every enemy within 35 m: tools/nn/sim/abilities.py, direct.py) and Seismic Snare
         # (no movement for the enemies within 35 m). Only the v3 army scheme fields them (config/nn/pools.json units_v3)
         "wh2_dlc14_skv_inf_warp_grinder_0")
# The spells our casters cast (the battle XML's <unit_capabilities>, tools/nn/scenario.py; build/magic/plan.md 1): a
# hero's rank-1 spell - a free skill at his creation (character_skill_nodes.points_on_creation 1: the Gold Wizard's
# hidden Searing Doom node, the Warlock Engineer's first Ruin node). The list replaces the unit's default abilities in
# battle (without it a caster gets his whole lore: the magic probe, build/magic/probe1.md). The spells' numbers are
# their passports (config/nn/abilities.json "spells", python -m tools.nn.abilities).
SPELLS = {"wh3_dlc25_emp_cha_wizard_metal": ("wh_main_spell_metal_searing_doom",),
          "wh2_main_skv_cha_warlock_engineer": ("wh2_main_spell_ruin_warp_lightning",)}
# The database's entity sizes in order (battle_entities.size; tools/nn/sim/state.py SIZE_RANK the same).
SIZES = ("very_small", "small", "medium", "large", "very_large")
TABLES = ("main_units", "land_units", "battle_entities", "mounts", "melee_weapons", "missile_weapons", "projectiles",
          "unit_armour_types", "unit_shield_types", "unit_attributes_to_groups_junctions",
          "land_units_to_unit_abilites_junctions", "unit_spacings", "special_ability_phase_stat_effects",
          "special_ability_phases", "projectile_penetration_junctions", "battlefield_engines", "projectiles_explosions")

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
    "engines.n": ("an artillery unit's guns (land_units.num_engines with land_units.engine; no `engines` for a unit "
                  "without)", None),
    "engines.type": ("battlefield_engines.engine_type (Generic_3_Crew: the animation set with 3 crew places, R-work "
                     "artillery_rules.md 5)", None),
    "engines.crew_per_engine": ("ours: the number in engines.type (3) - the men a gun needs to fire (the game's code "
                                "names no number: the probe 'crew under fire' to check)", None),
    "engines.entity": ("battlefield_engines.battle_entity: the gun's body", None),
    "engines.hp": ("the gun's battle_entities.hit_points + land_units.bonus_hit_points (425 + 48 the Empire's)", None),
    "engines.mass": ("the gun's battle_entities.mass (2000)", None),
    "engines.size": ("the gun's battle_entities.size (large)", None),
    "engines.radius_m": ("the gun's battle_entities.radius", None),
    "engines.radii_ratio": ("the gun's battle_entities.radii_ratio (a capsule's or an ellipse's; 1 a circle)", None),
    "engines.height_m": ("the gun's battle_entities.height", None),
    "engines.turn_deg_s": ("the gun's battle_entities.turn_speed, deg/s (45): the unit's speed.turn_deg_s", None),
    "engines.fire_arc_deg": ("the gun's battle_entities.fire_arc_close / 2 (25): the unit's missile.fire_arc_deg", None),
    "engines.projectile_resistance": ("the gun's battle_entities.projectile_penetration_resistance (5)", None),
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
    "hp_per_man": ("battle_entities.hit_points + land_units.bonus_hit_points (+ the mount's battle_entities.hit_points "
                   "of a mounted unit: the charge probe 28.10, Empire Knights 6480 = 60 x (8 + 8 + 92))",
                   "card HealthMax / men"),
    "hp_total": ("hp_per_man * men (+ engines.n x engines.hp of an artillery unit: one number for the crew and the guns, "
                 "as the game's card)", "card HealthMax"),
    "mass": ("battle_entities.mass (a mounted unit: its mount's, mounts.entity - as radius_m, radii_ratio, speed.*)",
             "card Mass"),
    "size": ("battle_entities.size; a mounted unit the larger of the rider's and the mount's (medium and above is "
             "'large': bonus v large, charge defence v large; build/diag_cav/mechanic.md M1)", None),
    "height_m": ("battle_entities.height; a mounted unit the taller of the rider and the mount (the warhorse 2.0, the "
                 "demigryph 2.3; the rider 1.8)", None),
    "radius_m": ("battle_entities.radius", None),
    "radii_ratio": ("battle_entities.radii_ratio: an ellipse or a capsule body (shape 'ellipse', 'capsule') is radius long "
                    "and radius x this across (the Brood Horror's capsule 3.8 x 0.36: 7.6 m long, 2.74 m across); 1 a circle "
                    "(every man)", None),
    "body_shape": ("battle_entities.shape (a mounted unit its mount's): circle, ellipse or capsule (a capsule: two half "
                   "circles of radius x radii_ratio joined by straight sides 2 radius (1 - radii_ratio) long)", None),
    "knock_ignore": ("battle_entities.knock_interrupts_ignore_chance, % (a mounted unit its mount's): the chance a man stays "
                     "on his feet when a blow or a body would knock him back, down or flying (the game's program, "
                     "build/diag_skv_mon/mechanic.md M5: a roll 0-100 below it; the Brood Horror 60, the Rat Ogre Mutant 65, "
                     "Rat Ogres 20, infantry 0)", None),
    "hit_reaction_ignore": ("battle_entities.hit_reactions_ignore_chance, %: the chance a man takes no hit reaction (the "
                            "Brood Horror 70, the Mutant 65, Rat Ogres 20; the simulator has no hit reactions)", None),
    "melee.weapon_length": ("melee_weapons.weapon_length, m (the Brood Horror's 7, the Mutant's 6, Rat Ogres' 2, a spear 1): "
                            "no rule of the simulator reads it (build/diag_skv_mon/units.md 4.9)", None),
    "speed.walk": ("battle_entities.walk_speed, m/s (a mounted unit its mount's, an artillery unit its gun's: 2.0, run 2.0 - "
                   "the attribute cant_run; the speed.* and turn_deg_s alike)", "card slow_speed"),
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
    "melee.flaming": ("melee_weapons.ignition_amount > 0: flaming attacks (the RPFM schema: 'if set to a value higher "
                      "than 0, it will deal fire damage'; Knights of the Blazing Sun's lance 1.0)", None),
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
    "hiding_scalar": ("land_units.hiding_scalar: the unit's own share of an enemy's spotting distance (stealth: "
                      "a Gutter Runner is seen within 0.8 of it)", None),
    "spotting.scrub": ("land_units.spot_dist_scrub, m: how far the unit sees a hidden enemy out of woods (80; the "
                       "Jezzails 160) - the simulator's open ground (tools/nn/sim/stealth.py)", None),
    "spotting.tree": ("land_units.spot_dist_tree, m: the same into woods (60; the Jezzails 160); no woods in the "
                      "simulator", None),
    "projectile_resistance": ("battle_entities.projectile_penetration_resistance: what a man adds to a penetrating "
                              "projectile's counter (1 infantry, 2 the Giant Rats; the RPFM schema: 'higher number "
                              "means this entity can stop projectiles more easily'); a mounted unit the mount's (the "
                              "warhorse 2, the demigryph 3)", None),
    "projectile_speed_keep": ("battle_entities.projectile_penetration_speed_change: the share of its speed a projectile "
                              "keeps passing through the man (0.9 men, 0.95 Skaven); a mounted unit the mount's (0.85)",
                              None),
    "attributes": ("unit_attributes_to_groups_junctions for land_units.attribute_group", None),
    "abilities": ("land_units_to_unit_abilites_junctions.ability", "card owned abilities"),
    "spells": ("ours: SPELLS - the caster's rank-1 spell (character_skill_nodes.points_on_creation; build/magic/plan.md "
               "1); only on casters", None),
    "multiplayer_cost": ("main_units.multiplayer_cost", None),
    "combat_potential.melee": ("main_units.melee_cp: the unit's melee combat potential (the army-destruction "
                               "strength with its abilities' cp, config/nn/abilities.json; build/movelords)", None),
    "combat_potential.missile": ("main_units.missile_cp: its missile combat potential (counts by the ammunition "
                                 "left)", None),
    "missile.weapon": ("land_units.primary_missile_weapon", None),
    "missile.projectile": ("missile_weapons.default_projectile", None),
    "missile.ammo": ("land_units.primary_ammo, per man; per gun of an artillery unit (missile.ammo_per 'engine': the "
                     "projectiles - 198 Helblaster = 22 x 9 bullets, R-work artillery_rules.md 4)", "card stat_ammo"),
    "missile.ammo_per": ("ours: 'man', or 'engine' for an artillery unit", None),
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
    "missile.penetration_max": ("projectile_penetration_junctions.max_penetration of missile.penetration: the counter "
                                "a projectile goes on through (each man struck adds his projectile_resistance; -1 "
                                "no limit, absent 0: it stops in the first)", None),
    "missile.penetration_cap": ("projectile_penetration_junctions.entity_size_cap of missile.penetration: it passes "
                                "only a man of a smaller size (very_small < small < medium < large < very_large)",
                                None),
    "missile.shot_type": ("projectiles.shot_type", None),
    "missile.contact_phase": ("projectiles.contact_stat_effect: the contact effect a hit lays on its target "
                              "(the Ratling Gun's suppressing fire)", None),
    "missile.fixed_elevation": ("projectiles.fixed_elevation, degrees: the launch angle of a 'fixed' trajectory (the "
                                "Mortar 50; read only for 'fixed')", None),
    "missile.magic": ("projectiles.is_magical", None),
    "missile.bonus_v_large": ("projectiles.bonus_v_large (the Warp Lightning Cannon +70)", None),
    "missile.bonus_v_infantry": ("projectiles.bonus_v_infantry", None),
    "missile.can_bounce": ("projectiles.can_bounce (the Great Cannon): no number of the bounce in the database", None),
    "missile.can_roll": ("projectiles.can_roll", None),
    "missile.mass": ("projectiles.mass, kg", None),
    "missile.collision_radius_m": ("projectiles.collision_radius, m", None),
    "missile.prefer_central": ("projectiles.prefer_central_targets: the shooter's target man costs + 5 x his distance "
                               "from his unit's centre - the guns aim at its most central man (build/diag_arty/"
                               "mechanic.md 1; the Mortar, the cannons, the catapult), else at the man nearest them "
                               "(the rockets, the Helblaster)", None),
    "missile.blast": ("projectiles_explosions of missile.explosion (null: none): radius_m detonation_radius, speed "
                      "detonation_speed (the ring's growth, m/s), damage / ap_damage detonation_damage / _ap, force "
                      "detonation_force, allies affects_allies, magic is_magical, contact_phase contact_phase_effect, "
                      "type detonation_type", None),
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
    "collision attack": "melee_weapons.collision_attack_max_targets (0 for every unit of the pool; the Hell Pit Abomination "
                        "5): not read",
    "projectile intersection": "battle_entities.projectile_intersection_radius_ratio (Rat Ogres 0.8, the Brood Horror 1.25, "
                               "the Mutant 1.0): whether the program's projectile hit reads it is not known (build/diag_skv_mon/"
                               "sim_gap.md 21)",
    "mount turning": "battle_entities.min_turning_speed (the warhorse 1.0 m/s, infantry 0) and strafe_speed (3.0): what "
                     "they do in numbers is not known (build/diag_cav/units.md 4.6)",
    "charge target pick": "battle_entities.charge_distance_pick_target (35 m the warhorse, 30 the demigryph): no reader "
                          "found in the game's program (build/diag_cav/mechanic.md 8)",
    "melee weapon type": "melee_weapons.melee_weapon_type (lance / sword): no mechanic known to depend on it for our units "
                         "(weapon_length is in the passport, melee.weapon_length, unread)",
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
    pen = _index(t.get("projectile_penetration_junctions") or (), "key")
    # A mounted unit (land_units.mount: cavalry, chariots' beasts) moves and collides as its mount: the mount's
    # battle_entities (mounts.entity) give the speeds, the charge distances, the body (radius, ellipse) and the mass
    # (before: the rider's for all - Empire Knights ran at 3.3 m/s, the warhorse 7.8).
    mount_row = _index(t.get("mounts") or (), "key").get(land["mount"]) if land["mount"] else None
    body = _index(t["battle_entities"], "key")[mount_row["entity"]] if mount_row else entity
    # ... and its health is the rider's and the mount's together, with the unit's bonus (the charge probe 28.10: Empire
    # Knights 6480 = 60 x (8 rider + 8 warhorse + 92); the rider's alone gave 100 a man); its size the larger of the two
    # (medium and above is 'large', build/diag_cav/mechanic.md M1), its height the taller (the shadow a projectile hits
    # and the sight), its resistance to a penetrating projectile the mount's (the demigryph 3, the rider 2)
    hp = entity["hit_points"] + land["bonus_hit_points"] + (body["hit_points"] if mount_row else 0)
    size = max(entity["size"], body["size"], key=lambda s: SIZES.index(s) if s in SIZES else 1)
    height = max(entity["height"], body["height"])
    # An artillery unit (land_units.engine, num_engines > 0): its men are the crew (their body, size and health as any
    # infantry's), its guns move it (the gun's battle_entities give the speeds and the turn: 2.0 m/s, cant_run, 45 deg/s)
    # and fire its missile weapon (battlefield_engines.missile_weapon); the unit's health is the crew's and the guns'
    # together (one number, as the game's card: build/diag_arty/empire.md 1)
    eng_row = (_index(t.get("battlefield_engines") or (), "key").get(land.get("engine"))
               if land.get("engine") and (land.get("num_engines") or 0) > 0 else None)
    gun = _index(t["battle_entities"], "key")[eng_row["battle_entity"]] if eng_row else None
    mover = gun if gun else body
    engines = None
    if gun:
        import re
        crew = re.search(r"(\d+)_Crew", eng_row["engine_type"] or "")
        engines = {"n": land["num_engines"], "type": eng_row["engine_type"], "entity": gun["key"],
                   "crew_per_engine": int(crew.group(1)) if crew else 1,
                   "hp": gun["hit_points"] + land["bonus_hit_points"], "mass": gun["mass"], "size": gun["size"],
                   "radius_m": gun["radius"], "radii_ratio": gun["radii_ratio"] if gun["shape"] in ("ellipse", "capsule")
                   else 1.0, "height_m": gun["height"], "turn_deg_s": gun["turn_speed"],
                   "fire_arc_deg": gun["fire_arc_close"] / 2,
                   "projectile_resistance": gun["projectile_penetration_resistance"]}
    out = {
        "land_unit": land["key"], "caste": main["caste"], "category": land["category"], "class": land["class"],
        "men": main["num_men"], "mount": land["mount"], "num_mounts": land["num_mounts"], "engine": land["engine"],
        **({"engines": engines} if engines else {}),
        "rank_depth": land["rank_depth"], "entity": land["man_entity"],
        "spacing": {"template": land["spacing"], "h": sp["close_formation_spacing_horizontal"],
                    "v": sp["close_formation_spacing_vertical"], "scatter": sp["close_formation_spacing_variation"],
                    "chaotic": sp["is_chaotic_formation"]},
        "hp_per_man": hp, "hp_total": hp * main["num_men"] + (engines["n"] * engines["hp"] if engines else 0),
        "mass": body["mass"], "size": size,
        "height_m": height, "radius_m": body["radius"],
        # (a capsule too: the Brood Horror's body 7.6 x 2.74 m, not a circle 7.6 m across - build/diag_skv_mon/units.md 4.1)
        "radii_ratio": body["radii_ratio"] if body["shape"] in ("ellipse", "capsule") else 1.0,
        "body_shape": body["shape"],
        "knock_ignore": body.get("knock_interrupts_ignore_chance") or 0,
        "hit_reaction_ignore": body.get("hit_reactions_ignore_chance") or 0,
        "speed": {"walk": mover["walk_speed"], "run": mover["run_speed"], "charge": mover["charge_speed"],
                  "charge_distance": mover["charge_distance_commence_run"],
                  "charge_pose_distance": mover["charge_distance_adopt_charge_pose"],
                  "acceleration": mover["acceleration"], "deceleration": mover["deceleration"],
                  "turn_deg_s": mover["turn_speed"]},
        "melee": {"attack": land["melee_attack"], "defence": land["melee_defence"],
                  "charge_bonus": land["charge_bonus"], "weapon": weapon["key"], "damage": weapon["damage"],
                  "ap_damage": weapon["ap_damage"], "bonus_v_large": weapon["bonus_v_large"],
                  "bonus_v_infantry": weapon["bonus_v_infantry"],
                  "attack_interval_s": weapon["melee_attack_interval"],
                  "splash_target_size": weapon["splash_attack_target_size"],
                  "splash_max_attacks": weapon["splash_attack_max_attacks"],
                  "splash_power_multiplier": weapon["splash_attack_power_multiplier"],
                  "magic": weapon["is_magical"], "flaming": (weapon.get("ignition_amount") or 0) > 0,
                  "contact_phase": weapon["contact_phase"], "weapon_length": weapon.get("weapon_length") or 0},
        "armour": armour["armour_value"], "armour_material": armour["audio_type"],
        "shield": {"key": shield["key"], "missile_block_chance": shield["missile_block_chance"]},
        "leadership": land["morale"],
        "damage_resist": {k: land[f"damage_mod_{k}"] for k in ("all", "physical", "missile", "magic", "flame")},
        "can_skirmish": land["can_skirmish"], "can_brace": land["can_brace"], "hiding_scalar": land["hiding_scalar"],
        "spotting": {"scrub": land["spot_dist_scrub"], "tree": land["spot_dist_tree"]},
        "projectile_resistance": body["projectile_penetration_resistance"],
        "projectile_speed_keep": body["projectile_penetration_speed_change"],
        "attributes": sorted(r["attribute"] for r in t["unit_attributes_to_groups_junctions"]
                             if land["attribute_group"] and r["attribute_group"] == land["attribute_group"]),
        "abilities": sorted(r["ability"] for r in t["land_units_to_unit_abilites_junctions"]
                            if r["land_unit"] == land["key"]),
        "multiplayer_cost": main["multiplayer_cost"],
        "combat_potential": {"melee": main["melee_cp"], "missile": main["missile_cp"]},
        "missile": None,
    }
    weapon_key = land["primary_missile_weapon"] or (eng_row["missile_weapon"] if eng_row else None)
    if weapon_key:
        mw = _index(t["missile_weapons"], "key")[weapon_key]
        pr = _index(t["projectiles"], "key")[mw["default_projectile"]]
        ex = (_index(t.get("projectiles_explosions") or (), "key").get(pr["explosion_type"])
              if pr["explosion_type"] else None)
        out["missile"] = {
            "weapon": mw["key"], "projectile": pr["key"], "ammo": land["primary_ammo"],
            "ammo_per": "engine" if engines else "man", "accuracy": land["accuracy"],
            "category": pr["category"], "range_m": pr["effective_range"], "min_range_m": pr["minimum_range"],
            "reload_s": pr["base_reload_time"], "damage": pr["damage"], "ap_damage": pr["ap_damage"],
            "projectile_number": pr["projectile_number"], "shots_per_volley": pr["shots_per_volley"],
            "trajectory": pr["trajectory_sight"], "direct": pr["trajectory_sight"] == "low",
            "muzzle_velocity": pr["muzzle_velocity"],
            "max_elevation": pr["max_elevation"], "calibration_distance_m": pr["calibration_distance"],
            "calibration_area_m": pr["calibration_area"], "marksmanship": pr["marksmanship_bonus"],
            "fire_arc_deg": (gun or entity)["fire_arc_close"] / 2, "explosion": pr["explosion_type"],
            "penetration": pr["projectile_penetration"], "shot_type": pr["shot_type"],
            "penetration_max": (pen.get(pr["projectile_penetration"]) or {}).get("max_penetration") or 0,
            "penetration_cap": (pen.get(pr["projectile_penetration"]) or {}).get("entity_size_cap"),
            "contact_phase": pr["contact_stat_effect"],
            "fixed_elevation": pr["fixed_elevation"], "magic": bool(pr["is_magical"]),
            "bonus_v_large": pr["bonus_v_large"], "bonus_v_infantry": pr["bonus_v_infantry"],
            "can_bounce": bool(pr["can_bounce"]), "can_roll": bool(pr["can_roll"]), "mass": pr["mass"],
            "collision_radius_m": pr["collision_radius"], "prefer_central": bool(pr["prefer_central_targets"]),
            "blast": {"key": ex["key"], "type": ex["detonation_type"], "radius_m": ex["detonation_radius"],
                      "speed": ex["detonation_speed"], "damage": ex["detonation_damage"],
                      "ap_damage": ex["detonation_damage_ap"], "force": ex["detonation_force"],
                      "allies": bool(ex["affects_allies"]), "magic": bool(ex["is_magical"]),
                      "contact_phase": ex["contact_phase_effect"]} if ex else None}
    if key in SPELLS:
        out["spells"] = list(SPELLS[key])
    return out


def phase_keys(units):
    """The contact phases the units' weapons lay (melee.contact_phase, missile.contact_phase), sorted."""
    return sorted({p for u in units.values() for p in ((u.get("melee") or {}).get("contact_phase"),
                                                       (u.get("missile") or {}).get("contact_phase"),
                                                       ((u.get("missile") or {}).get("blast") or {}).get("contact_phase"))
                   if p})


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
