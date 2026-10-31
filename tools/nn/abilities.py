"""Ability passports: what each ability of our units does, from the game's database.

Reads data/db.pack (tools/nn/dbtables.py) and writes config/nn/abilities.json, one passport per
ability key the units of config/nn/units.json own (their `abilities`). The network's input
(tools/nn/model/abilities.py) and the simulator (tools/nn/sim/abilities.py) both read this file:
an ability is known by what it does (times, reach, targets, effects), never by its name, so a new
lord's ability needs a passport, not a new network.
The casters' spells (their passports' `spells`, config/nn/units.json and data/units/probe-units.json) get a spell
passport in the file's own section `spells` (spell_passport, SPELL_FIELDS: cost, wind-up, range, targets, the
bombardment with its projectile and explosion); the simulator reads it (tools/nn/sim/magic.py), the network's ability
table does not.

    py -3.14 -m tools.nn.abilities                  # the abilities of the units in units.json
    py -3.14 -m tools.nn.abilities --abilities k1,k2   # more keys

Needs Python 3.14+ (compression.zstd) to read the game. A lord with a card from battle
(data/roster/<key>.json) is checked against it: which of his abilities are passive.
"""
import argparse
import json
import math
import sys
from pathlib import Path

from tools import config as project

OUT = project.CONFIG_DIR / "nn" / "abilities.json"
UNITS = project.CONFIG_DIR / "nn" / "units.json"
ROSTER = project.ROOT / "data" / "roster"
TABLES = ("unit_special_abilities", "special_ability_to_special_ability_phase_junctions",
          "special_ability_phase_stat_effects", "special_ability_phase_attribute_effects",
          "special_ability_to_recharge_contexts", "special_ability_to_auto_deactivate_flags",
          # (31.10.2026, the Pusbags) the vortices (unit_special_abilities.vortex) and the direct damage's scale by the
          # battle's unit size (passport `vortex`, `direct`)
          "battle_vortexs", "_kv_unit_ability_scaling_rules")
# The battle's unit size of our battles (the pools' men: 180 skavenslaves, 120 swordsmen - the game's 'ultra'): the
# direct damage's scale direct_damage_<size> of _kv_unit_ability_scaling_rules (build/diag_roster/wh3_dlc29_skv_inf_pusbags/
# mechanic.md B2: small 0.25, medium 0.5, large 0.75, ultra 1.0)
UNIT_SIZE = "ultra"
# The behaviours (unit_special_abilities.behaviour) whose vortex the death of the owner's men fires (mechanic.md P1: bit
# 14 of the ability's behaviours, at every man's death of any cause) - such an ability is innate, whatever its active time
DEATH_VORTEX = ("vortex_on_entity_death",)
# The direct damage's hold of its targets: an enemy struck stays a target until it is this far beyond the ability's
# effect_range (the game's program, build/diag_roster/wh2_dlc14_skv_inf_warp_grinder_0/mechanic.md 1, 5: the Warp Grinder's
# 35 + 5 m)
DIRECT_HOLD_M = 5.0
PHASES = "special_ability_phases"      # read with floats as written (fatigue_change_ratio to 6 digits)
# the spells' own tables (spell_passport): the ability's kind, the bombardment, its projectile, explosion and penetration
SPELL_TABLES = ("unit_abilities", "projectile_bombardments", "projectiles", "projectiles_explosions",
                "projectile_penetration_junctions")
PROBE_UNITS = project.ROOT / "data" / "units" / "probe-units.json"

# Where each spell passport field comes from (spell_passport; the spells of the casters' `spells`, tools/nn/units.py)
SPELL_FIELDS = {
    "kind": "unit_abilities.type (wh_type_bombardment, ...)",
    "mana": "unit_special_abilities.mana_cost: the winds of magic a cast takes from its side's current pool",
    "wind_up_s": "unit_special_abilities.wind_up_time, s: the caster stands casting this long after the order",
    "range_m": "unit_special_abilities.target_intercept_range, m: the cast's reach, caster's centre to the target's "
               "(out of it the caster walks in first: the magic probe, build/magic/probe1.md)",
    "active_s": "unit_special_abilities.active_time, s; the recharge runs from its end (the probe: casts 40 s / 37 s "
                "apart = wind-up + active + recharge)",
    "recharge_s": "unit_special_abilities.recharge_time, s",
    "initial_s": "unit_special_abilities.initial_recharge, s (-1 and 0: ready at once)",
    "targets": "unit_special_abilities.target_enemies / target_friends / target_self / target_ground",
    "ai_usage": "unit_special_abilities.ai_usage: the game's AI usage template",
    "overcast": "unit_abilities.overpower_option: the overcast spell's key (not modelled yet)",
    "bombardment": "projectile_bombardments of unit_special_abilities.bombardment: projectiles = num_projectiles, "
                   "start_s = start_time (after the wind-up), window_s = arrival_window, spread_m = radius_spread (the "
                   "projectiles land within this radius of the aim), launch_height_m = launch_height, launch = "
                   "launch_source",
    "bombardment.projectile": "projectiles of its projectile_type: damage, ap = ap_damage, collision_m = "
                              "collision_radius, penetration = projectile_penetration_junctions.max_penetration of its "
                              "projectile_penetration (the men one projectile can strike), penetration_cap = its "
                              "entity_size_cap, elevation_deg = fixed_elevation, muzzle_v = muzzle_velocity, gravity, "
                              "magical = is_magical, allies = can_damage_allies, explosion = explosion_type, mass "
                              "(its knock on the man it strikes: mass x impact_v / his mass against the game's "
                              "collision_speed_delta_threshold_knocked_*), impact_v = ours: its speed at the end of the "
                              "fall (flight_s)",
    "bombardment.explosion": "projectiles_explosions of the projectile's explosion_type: radius_m = detonation_radius, "
                             "damage = detonation_damage, ap = detonation_damage_ap, allies = affects_allies, force = "
                             "detonation_force (its shockwave's knock: force / a man's mass against "
                             "shockwave_speed_delta_threshold_knocked_*)",
    "bombardment.flight_s": "ours: the fall from launch_height at muzzle_velocity aimed fixed_elevation below the "
                            "horizontal under gravity, h = v sin(e) t + g t^2 / 2 (Searing Doom 2.96 s, Warp Lightning "
                            "1.17 s: with the wind-up and start the first hit 5.96 / 5.17 s after the order, the probe "
                            "6.0 / 5.0-5.5 s)",
}


def flight_s(height, v, elevation_deg, gravity):
    """Seconds of the fall from `height` m at muzzle velocity v aimed elevation_deg below the horizontal under gravity
    (h = v sin(e) t + g t^2 / 2); 0 without a height."""
    if not height or height <= 0:
        return 0.0
    vy = max(float(v), 0.0) * math.sin(math.radians(float(elevation_deg or 0.0)))
    g = float(gravity or 0.0)
    if g <= 0:
        return height / vy if vy > 0 else 0.0
    return (-vy + math.sqrt(vy * vy + 2 * g * height)) / g


def impact_v(height, v, elevation_deg, gravity):
    """Speed, m/s, at the end of the fall of flight_s (the same throw): the horizontal v cos(e) and the vertical
    v sin(e) + g t."""
    t = flight_s(height, v, elevation_deg, gravity)
    e = math.radians(float(elevation_deg or 0.0))
    v = max(float(v), 0.0)
    return math.hypot(v * math.cos(e), v * math.sin(e) + float(gravity or 0.0) * t)


def spell_passport(key, t):
    """One spell's passport (SPELL_FIELDS) from the decoded tables t (TABLES and SPELL_TABLES)."""
    rows = {r["key"]: r for r in t["unit_special_abilities"]}
    if key not in rows:
        raise ValueError(f"unit_special_abilities: no row for {key}")
    p = rows[key]
    ua = {r["key"]: r for r in t["unit_abilities"]}.get(key) or {}
    out = {"kind": ua.get("type"), "mana": p["mana_cost"], "wind_up_s": p["wind_up_time"],
           "range_m": p["target_intercept_range"], "active_s": p["active_time"], "recharge_s": p["recharge_time"],
           "initial_s": max(0.0, p.get("initial_recharge", 0.0)),
           "targets": {k: bool(p[f"target_{k}"]) for k in ("enemies", "friends", "self", "ground")},
           "ai_usage": p.get("ai_usage"), "overcast": ua.get("overpower_option"), "bombardment": None}
    if p.get("bombardment"):
        b = {r["bombardment_key"]: r for r in t["projectile_bombardments"]}[p["bombardment"]]
        pr = {r["key"]: r for r in t["projectiles"]}[b["projectile_type"]]
        pen = {r["key"]: r for r in t["projectile_penetration_junctions"]}.get(pr["projectile_penetration"]) or {}
        ex = {r["key"]: r for r in t["projectiles_explosions"]}.get(pr["explosion_type"])
        proj = {"key": pr["key"], "damage": pr["damage"], "ap": pr["ap_damage"], "collision_m": pr["collision_radius"],
                "penetration": pen.get("max_penetration"), "penetration_cap": pen.get("entity_size_cap"),
                "elevation_deg": pr["fixed_elevation"], "muzzle_v": pr["muzzle_velocity"], "gravity": pr["gravity"],
                "magical": bool(pr["is_magical"]), "allies": bool(pr["can_damage_allies"]),
                "explosion": pr["explosion_type"] or None, "mass": pr["mass"],
                "impact_v": round(impact_v(b["launch_height"], pr["muzzle_velocity"], pr["fixed_elevation"],
                                           pr["gravity"]), 2)}
        out["bombardment"] = {
            "key": b["bombardment_key"], "projectiles": b["num_projectiles"], "start_s": b["start_time"],
            "window_s": b["arrival_window"], "spread_m": b["radius_spread"], "launch_height_m": b["launch_height"],
            "launch": b["launch_source"], "projectile": proj,
            "explosion": None if ex is None else {"key": ex["key"], "radius_m": ex["detonation_radius"],
                                                  "damage": ex["detonation_damage"], "ap": ex["detonation_damage_ap"],
                                                  "allies": bool(ex["affects_allies"]),
                                                  "force": ex["detonation_force"]},
            "flight_s": round(flight_s(b["launch_height"], pr["muzzle_velocity"], pr["fixed_elevation"], pr["gravity"]),
                              3)}
    return out


def spells_of(*docs):
    """Every spell key the casters of the unit passport documents own (their `spells`), sorted."""
    return sorted({k for d in docs for u in d.values() for k in u.get("spells") or ()})

# Where each passport field comes from: table.column by the RPFM schema's names (tools/nn/dbschema.py); "ours": what
# we make of the columns.
FIELDS = {
    "active_s": "unit_special_abilities.active_time, s; -1: no duration (passive or instant)",
    "recharge_s": "unit_special_abilities.recharge_time, s; -1: none",
    "initial_s": "unit_special_abilities.initial_recharge, s: the ability is on cooldown this long from "
                 "the battle's start (-1 and 0: ready at once; Steam guide 1698960734 'most abilities start at 0'; "
                 "Gate of Khorne 60 = fandom)",
    "uses": "unit_special_abilities.num_uses; -1: unlimited",
    "vigour_per_s": "special_ability_phases.fatigue_change_ratio of the ability's phases (-0.01 for Foe-Seeker = fandom "
                    "'Vigour per second: -1%'): the share of the maximum fatigue (30000) the owner loses each second "
                    "while it is active (negative: recovers; the in-game probe build/movelords: -300 points/s)",
    "cp": "unit_special_abilities.additional_melee_cp / additional_missile_cp: the ability's combat potential",
    "range_m": "unit_special_abilities.effect_range, m: the reach around the caster (0: itself)",
    "passive": "active_s and recharge_s both -1 (ours; the table's own passive column also marks Strength of the "
               "Penitent, our auto); checked against the cards' owned_passive lists",
    "targets_own": "unit_special_abilities.affect_self: the ability acts on its caster too",
    "friendly_units": "unit_special_abilities.num_effected_friendly_units: -1 all in range, n, 0 none",
    "enemy_units": "unit_special_abilities.num_effected_enemy_units: -1 all in range, n, 0 none",
    "update_targets": "unit_special_abilities.update_targets_every_frame: 1 - the ability picks its targets in range every "
                      "frame, an aura that follows the caster (Rally, Hold the Line); 0 - it is laid once at the cast "
                      "on the units in range, and they keep it for active_s wherever they go, later arrivals get "
                      "nothing (Stand Your Ground; 142 recorded battles, build/effects/spec.md 3.2)",
    "self_cast": "active, not auto, affect_self, friendly_units and enemy_units 0 or -1 (ours): used on the "
                 "caster himself, no target to choose (perform_special_ability(key, the lord))",
    "recharge_when": "special_ability_to_recharge_contexts.recharge_context of the ability: its recharge runs only while "
                     "this holds (Strength of the Penitent: losing_melee_combat - 3 s while losing, CA hotfix 6.2.2; as the "
                     "game acts on it: while not winning its melee, out of melee too - build/effectsimpl; "
                     "Wounds: health_below_25% - its 5 s initial recharge runs once below a quarter); a passive "
                     "without an active time is on only once it has recharged so",
    "off_when": "special_ability_to_auto_deactivate_flags.deactivate_flag: the effect switches off while this "
                "holds (Frenzy: morale_is_lower_than_half_of_base_morale; Penitent: out_of_melee)",
    "auto": "a timed passive (ours): an active time, a key with '_passive_' (the game lists it among the "
            "passives, unit_abilities) - the game fires it by itself whenever it is ready and nothing holds it off "
            "(off_when; the recordings: Strength of the Penitent on at the contact, 53 of 54, without losses too); "
            "no player or network order",
    "phases": "special_ability_to_special_ability_phase_junctions for the ability: the phase named as the "
              "ability when there is one, else all its phases",
    "targets": "the phases' target_self / target_friends / target_enemies (self-only abilities have 1 0 0, auras "
               "1 1 0, hexes 0 0 1)",
    "effects": "special_ability_phase_stat_effects of the phases: stat, how (add or mult), value, and on "
               "whom (the phase's targets)",
    "attributes": "special_ability_phase_attribute_effects of the phases (attribute, attribute_type: positive/negative)",
    "heal": "only an ability whose phases heal (special_ability_phases heal_amount != 0: Regeneration): share - "
            "heal_amount, the share of the unit's largest health healed each tick; every_s - hp_change_frequency, the tick, "
            "s (Regeneration 0.001 every 1.0 s: the Brood Horror's 6009 HP heal 6 HP/s - the game's program, "
            "build/diag_skv_mon/mechanic.md M11: round(heal_amount x the power x the largest health) a tick)",
    "direct": "only an ability whose phase deals direct damage (special_ability_phases damage_amount != 0: the Pusbags' "
              "Unshakable Pathogen): amount - damage_amount, entities - max_damaged_entities (-1: all its men), every_s - "
              "hp_change_frequency (the tick: at the phase's start and every every_s), on - the phase's targets, scale - "
              "_kv_unit_ability_scaling_rules direct_damage_<UNIT_SIZE> (ultra 1.0), min_unary - "
              "direct_damage_damage_scale_min_unary (0.5): a tick strikes min(entities, alive) different random men, each "
              "trunc(trunc(max(1, amount x scale)) x U[min_unary, 1]), at least 1 HP, past armour and shields (the game's "
              "program, build/diag_roster/wh3_dlc29_skv_inf_pusbags/mechanic.md B1-B4); hit_hp - the mean of that (3.2), "
              "hit_dist - [[HP, share]]; magical - the hit is magical (past armour and shields, the spell resistance and the "
              "ward cut it: round(HP x (100 - resistance) / 100), the physical does not); range_m - effect_range (0: the "
              "owner itself), hold_m - range_m + DIRECT_HOLD_M 5 (a target stays one to there), n_enemy / n_friend - "
              "num_effected_enemy_units / _friendly_units (-1 all), every_frame - update_targets_every_frame (the targets "
              "re-picked every tick, one out of range dropped at once, back in: a tick at once), duration_s - the phase's "
              "duration, else the ability's active_time (-1: while on); the random men are picked again at every tick "
              "(build/diag_roster/wh2_dlc14_skv_inf_warp_grinder_0/mechanic.md W2-W5, 5)",
    "behaviour": "unit_special_abilities.behaviour when set (vortex_on_entity_death: a vortex at every man's death; such "
                 "an ability without a recharge is passive whatever its active time)",
    "vortex": "battle_vortexs of unit_special_abilities.vortex: delay_s - delay (+ i x between_s for the i-th of n - "
              "num_vortexes / delay_between_vortexes), duration_s - max(duration, goal_radius / expansion_speed), "
              "start_radius_m / goal_radius_m / expansion_speed - the circle growing, damage / damage_ap - one strike of a "
              "man (at most once a second a vortex: once in its 1 s), magical - is_magical (the physical resistance off), "
              "spell - is_spell (the spell resistance on), flaming - ignition_amount != 0, force - detonation_force (the "
              "knock: dv = 1.3 F S / (F + m), S the expansion speed), contact - contact_effect (laid on the whole unit of a "
              "man struck), allies / enemies - affects_allies / affects_enemies, move_m_s - movement_speed, launch - "
              "launch_source, follow - follow_target, next - subsequent_vortex (mechanic.md P1-P8)",
    "delay_s": "only a passive of several phases whose first phases lay nothing (Blackpowder Discipline: phase i 10 s, "
               "no stats, then phase ii): the special_ability_phases durations of those first phases, in the junctions' "
               "order - its effects come on once its conditions have held that long (ours: the phases run in order)",
}
MISSING = {
    "area damage": "battle_vortexs decoded since 31.10.2026 (the passport's `vortex`); the simulator acts on the vortex "
                   "at a man's death only (tools/nn/sim/vortex.py). Verminous Valour's vortex (25 m, 1 s, damage 0, force "
                   "200) throws the men around the Warlord back: probe T-E3, 2 s after the cast no swordsman within 6 m, "
                   "the nearest 6.5-8.6 m away, in contact again after ~8 s - no knockback of an active ability's vortex "
                   "in the simulator",
    "conditions": "decoded: recharge_when (special_ability_to_recharge_contexts) and off_when "
                  "(special_ability_to_auto_deactivate_flags); a passive's effects are what it gives while on; "
                  "passive and auto abilities are innate effects (config/nn/effects.json, python -m tools.nn.effects)",
    "ui type": "unit_abilities (hex, augment, ...) is not decoded: the targets say the same",
}


def phases_of(key, junctions):
    """The ability's phases with their targets (the junction's rows with special_ability = key): the phase named as
    the ability when it has one, else all. Every ability of our units has exactly one phase, named as itself; an
    ability with more phases would lose the others here (kept as it always worked: tools/nn/dbschema.py notes)."""
    rows = [r for r in junctions if r["special_ability"] == key]
    own = [r for r in rows if r["phase"] == key]
    return own or rows


def phase_fatigue(rows, phases):
    """{phase: fatigue_change_ratio rounded to 6 digits} of the phases (special_ability_phases rows read with floats as
    written, by their id); a phase with no row raises."""
    by_id = {r["id"]: r for r in rows}
    missing = [k for k in phases if k not in by_id]
    if missing:
        raise ValueError(f"{PHASES}: no row for {missing}")
    return {k: round(by_id[k]["fatigue_change_ratio"], 6) for k in phases}


def _number(v):
    """A float that is a whole number as an int (the passports' cp have always been written so)."""
    return int(v) if float(v).is_integer() else v


def passport(key, rows, t):
    """One ability's passport from its unit_special_abilities row (rows: {key: row}) and the decoded tables t."""
    p = rows[key]
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
                attributes.append({"attribute": r["attribute"], "effect": r["attribute_type"], "on": on,
                                   "phase": ph["phase"]})
    # (DEATH_VORTEX) an ability the death of the owner's men fires is innate: the Pusbags' Pestilent Perfection has
    # active_time 1 (the vortex's life) and no recharge
    passive = (p["active_time"] < 0 and p["recharge_time"] < 0) or (p.get("behaviour") in DEATH_VORTEX
                                                                    and p["recharge_time"] < 0)
    auto = not passive and "_passive_" in key
    friendly, enemy = p["num_effected_friendly_units"], p["num_effected_enemy_units"]
    extra = {}
    durations = t.get("phase_duration") or {}
    if passive and len(phases) > 1 and durations:
        # (FIELDS delay_s) the first phases that lay no stat run their durations before the rest come on
        delay = 0.0
        for ph in sorted(phases, key=lambda r: r.get("order", 0)):
            if any(e["phase"] == ph["phase"] for e in effects):
                break
            delay += max(0.0, float(durations.get(ph["phase"], 0.0)))
        if delay > 0:
            extra["delay_s"] = round(delay, 6)
    heals = t.get("phase_heal") or {}
    healing = [heals[ph["phase"]] for ph in phases if (heals.get(ph["phase"]) or (0, 0))[0]]
    if healing:
        # (FIELDS heal) the phase's heal_amount and hp_change_frequency
        extra["heal"] = {"share": healing[0][0], "every_s": healing[0][1]}
    direct = [(ph, (t.get("phase_direct") or {}).get(ph["phase"])) for ph in phases]
    direct = [(ph, d) for ph, d in direct if d and d[0]]
    if direct:
        # (FIELDS direct) the phase's damage_amount, max_damaged_entities, hp_change_frequency and the unit size's scale
        ph, (amount, entities, every) = direct[0]
        kv = t.get("ability_scaling") or {}
        scale = float(kv.get(f"direct_damage_{UNIT_SIZE}", 1.0))
        low = float(kv.get("direct_damage_damage_scale_min_unary", 1.0))
        dur = float((t.get("phase_duration") or {}).get(ph["phase"], -1.0))
        extra["direct"] = {"amount": amount, "entities": entities, "every_s": every,
                           "on": [k for k in ("self", "friends", "enemies") if ph[f"target_{k}"]],
                           "scale": scale, "min_unary": low, "hit_hp": direct_hit_mean(amount, scale, low),
                           "hit_dist": direct_hit_dist(amount, scale, low),
                           # who: the owner itself, or the units within range_m of its centre (held to range_m + hold),
                           # at most n_enemy / n_friend of them (-1 all), re-picked every tick (every_frame)
                           "range_m": float(p["effect_range"]), "hold_m": float(p["effect_range"]) + DIRECT_HOLD_M,
                           "n_enemy": int(p["num_effected_enemy_units"]), "n_friend": int(p["num_effected_friendly_units"]),
                           "every_frame": bool(p["update_targets_every_frame"]),
                           "duration_s": dur if dur > 0 else float(p["active_time"]),
                           # magical (the program's damage flags 6): past armour and shields, the spell resistance and
                           # the ward cut it, the physical does not
                           "magical": True}
    if p.get("behaviour"):
        extra["behaviour"] = p["behaviour"]
    vx = (t.get("vortices") or {}).get(p.get("vortex") or "")
    if vx:
        extra["vortex"] = vortex_passport(vx)
    return {
        "active_s": p["active_time"], "recharge_s": p["recharge_time"],
        "initial_s": max(0.0, p.get("initial_recharge", 0.0)),
        "uses": p["num_uses"],
        # the owner's own phases (targets self): what his fatigue does while it is active
        "vigour_per_s": round(sum(t.get("phase_fatigue", {}).get(ph["phase"], 0.0) for ph in phases
                                  if ph["target_self"]), 6),
        "cp": [_number(p["additional_melee_cp"]), _number(p["additional_missile_cp"])],
        "range_m": p["effect_range"], "passive": passive, "targets_own": p["affect_self"],
        "friendly_units": friendly, "enemy_units": enemy,
        "self_cast": (not passive) and not auto and p["affect_self"] and friendly in (0, -1) and enemy in (0, -1),
        "update_targets": bool(p["update_targets_every_frame"]),
        "auto": auto,
        "recharge_when": sorted(r["recharge_context"] for r in t.get("special_ability_to_recharge_contexts", ())
                                if r["special_ability"] == key),
        "off_when": sorted(r["deactivate_flag"] for r in t.get("special_ability_to_auto_deactivate_flags", ())
                           if r["special_ability"] == key),
        "phases": [r["phase"] for r in phases], "targets": targets,
        "effects": sorted(effects, key=lambda e: (e["phase"], e["stat"], e["how"])),
        "attributes": sorted(attributes, key=lambda a: (a["phase"], a["attribute"])),
        **extra,
    }


def direct_hit_mean(amount, scale, low):
    """The mean HP of one man's hit of a direct-damage tick (FIELDS direct): max(1, trunc(base x U[low, 1])), base =
    trunc(max(1, amount x scale)) - 5, 1.0, 0.5: 2 (0.2), 3 (0.4), 4 (0.4) -> 3.2."""
    base = math.floor(max(1.0, float(amount) * float(scale)))
    lo, hi = base * float(low), float(base)
    if hi <= lo:
        return float(max(1, math.floor(hi)))
    # E[max(1, floor X)], X ~ U[lo, hi]: 1 + the sum over k >= 2 of P(X >= k)
    return round(1.0 + sum(min(1.0, max(0.0, (hi - k) / (hi - lo))) for k in range(2, base + 1)), 6)


def direct_hit_dist(amount, scale, low):
    """[[HP, share]] of one man's hit of a direct-damage tick (FIELDS direct): max(1, trunc(base x U[low, 1])) - 5, 1.0,
    0.5: [[2, 0.2], [3, 0.4], [4, 0.4]]."""
    base = math.floor(max(1.0, float(amount) * float(scale)))
    lo, hi = base * float(low), float(base)
    if hi <= lo:
        return [[max(1, math.floor(hi)), 1.0]]
    out = {}
    for k in range(math.floor(lo), base + 1):
        share = (min(hi, k + 1) - max(lo, k)) / (hi - lo)
        if share > 1e-9:
            out[max(1, k)] = out.get(max(1, k), 0.0) + share
    return [[k, round(v, 6)] for k, v in sorted(out.items())]


def vortex_passport(r):
    """A vortex's passport (FIELDS vortex) from its battle_vortexs row."""
    exp = float(r["expansion_speed"])
    return {"delay_s": float(r["delay"]), "n": int(r["num_vortexes"]), "between_s": float(r["delay_between_vortexes"]),
            "duration_s": max(float(r["duration"]), float(r["goal_radius"]) / exp if exp > 0 else 0.0),
            "start_radius_m": float(r["start_radius"]), "goal_radius_m": float(r["goal_radius"]), "expansion_speed": exp,
            "damage": int(r["damage"]), "damage_ap": int(r["damage_ap"]), "magical": bool(r["is_magical"]),
            "spell": bool(r["is_spell"]), "flaming": float(r["ignition_amount"]) != 0,
            "force": float(r["detonation_force"]), "contact": r.get("contact_effect"), "allies": bool(r["affects_allies"]),
            "enemies": bool(r["affects_enemies"]), "move_m_s": float(r["movement_speed"]), "launch": r["launch_source"],
            "follow": bool(r["follow_target"]), "next": r.get("subsequent_vortex"), "key": r["vortex_key"]}


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


def build(tables, keys, units, roster=ROSTER):
    """{key: passport} from the decoded tables (TABLES; PHASES too, read with floats as written, for vigour_per_s)."""
    rows = {}
    for r in tables["unit_special_abilities"]:
        rows.setdefault(r["key"], r)
    missing = [k for k in keys if k not in rows]
    if missing:
        raise ValueError(f"unit_special_abilities: no row for {missing}")
    if PHASES in tables:
        names = sorted({r["phase"] for k in keys
                        for r in phases_of(k, tables["special_ability_to_special_ability_phase_junctions"])})
        tables = dict(tables, phase_fatigue=phase_fatigue(tables[PHASES], names),
                      phase_duration={r["id"]: r.get("duration", -1.0) for r in tables[PHASES] if r["id"] in names},
                      phase_heal={r["id"]: (round(float(r.get("heal_amount") or 0.0), 6),
                                            round(float(r.get("hp_change_frequency") or 0.0), 6))
                                  for r in tables[PHASES] if r["id"] in names},
                      phase_direct={r["id"]: (int(r.get("damage_amount") or 0), int(r.get("max_damaged_entities") or 0),
                                              round(float(r.get("hp_change_frequency") or 0.0), 6))
                                    for r in tables[PHASES] if r["id"] in names})
    if "battle_vortexs" in tables:
        tables = dict(tables, vortices={r["vortex_key"]: r for r in tables["battle_vortexs"]})
    if "_kv_unit_ability_scaling_rules" in tables:
        tables = dict(tables, ability_scaling={r["key"]: round(float(r["value"]), 6)
                                               for r in tables["_kv_unit_ability_scaling_rules"]})
    out = {k: passport(k, rows, tables) for k in keys}
    bad = check_cards(out, units, roster)
    assert not bad, "passports differ from the cards:\n" + "\n".join(bad)
    return out


def document(abilities, source, spells=None):
    """The file's document; spells ({key: spell passport}, the casters') in their own section, after the abilities:
    the network's ability table (tools/nn/model/abilities.py) reads `abilities` only."""
    doc = {"_description": "Ability passports (docs/en/training/model.md, abilities): what each ability of our "
                           "units does, from the game's database. Written by py -3.14 -m tools.nn.abilities.",
           "_source": str(source), "_fields": FIELDS, "_missing": MISSING, "abilities": abilities}
    if spells:
        doc["_spell_fields"] = SPELL_FIELDS
        doc["spells"] = spells
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--abilities", help="comma-separated extra ability keys")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    from tools.nn import dbtables
    game = Path(project.load()["game_dir"]) / "data" / "db.pack"
    tables = dict(dbtables.read_tables(game, TABLES + SPELL_TABLES),
                  **dbtables.read_tables(game, (PHASES,), digits=None))
    units = json.loads(UNITS.read_text(encoding="utf-8"))["units"]
    keys = sorted(set(owned(units)) | set(args.abilities.split(",") if args.abilities else ()))
    abilities = build(tables, keys, units)
    probe = json.loads(PROBE_UNITS.read_text(encoding="utf-8"))["units"] if PROBE_UNITS.exists() else {}
    spells = {k: spell_passport(k, tables) for k in spells_of(units, probe)}
    args.out.write_text(json.dumps(document(abilities, game, spells), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
    for k, s in spells.items():
        b = s["bombardment"] or {}
        print(f"spell {k}: {s['kind']}, mana {s['mana']:g}, wind-up {s['wind_up_s']:g} s, range {s['range_m']:g} m, "
              f"{b.get('projectiles')} projectiles over {b.get('window_s')} s from {b.get('start_s')} s, flight "
              f"{b.get('flight_s')} s")
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
