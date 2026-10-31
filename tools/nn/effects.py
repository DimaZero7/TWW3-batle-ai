"""Innate unit effects: every attribute and passive (or game-fired) ability our units own, as one
catalogue (docs/en/training/units.md "Innate effects"). Writes config/nn/effects.json.

An effect is known by what it does, from the game's database:
* a unit ATTRIBUTE (unit_attributes_to_groups_junctions, the passport's `attributes`): a rule flag of
  the engine (unbreakable, expendable, ...). The database gives the key only; what the rule does and
  its numbers (the _kv tables, config/nn/game_rules.json) are in ATTRIBUTES below, with the source;
* a PASSIVE ability (the ability passport's `passive`, config/nn/abilities.json): stat modifiers on
  the owner (and on friends within range_m: an aura), on while its conditions hold (the passport's
  off_when: special_ability_to_auto_deactivate_flags; recharge_when: special_ability_to_recharge_contexts -
  a passive's recharge runs only there, so it is on only there: Wounds below a quarter of health);
* a TIMED passive (the passport's `auto`): the game fires it by itself whenever it is ready and no
  off_when holds (Strength of the Penitent: in melee), it lasts active_s, ends at once while an off_when
  holds, and its recharge_s runs only while its recharge context holds (recharge_when: losing the
  melee; CA hotfix 6.2.2 "recharges when losing"; as the game acts on it - the probe T-E1 and the
  recordings: not winning its melee, out of melee too; gaps of exactly 3 s in 92 %).

Which unit owns which effect comes from the database rows (the passports), never from a hand list.
An effect is `modelled` by the simulator (tools/nn/sim/effects.py) when the simulator has every
stat, rule and condition it needs (SIM_STATS, SIM_RULES, CONDITIONS); otherwise the file says what
is missing. `order` is append-only (an effect keeps its place when the file is rebuilt): the
network's input (tools/nn/model/effects.py) has one pair of columns per effect in this order.

    python -m tools.nn.effects        # from config/nn/units.json and abilities.json (any Python)

Only json; no torch, no db.pack (the passports were read from it).
"""
import argparse
import json
import sys
from pathlib import Path

from tools import config as project

OUT = project.CONFIG_DIR / "nn" / "effects.json"
UNITS = project.CONFIG_DIR / "nn" / "units.json"
ABILITIES = project.CONFIG_DIR / "nn" / "abilities.json"
KB = "docs/en/game/mechanics/"

# The database's condition flags (special_ability_to_auto_deactivate_flags.flag,
# special_ability_to_recharge_contexts.context) -> the simulator's predicate of the same meaning
# (tools/nn/sim/effects.py PREDICATES).
CONDITIONS = {
    "out_of_melee": "out_of_melee",
    "engaged_in_melee": "in_melee",
    "losing_melee_combat": "losing_melee",
    "morale_is_lower_than_half_of_base_morale": "morale_below_half",
    "morale_is_higher_than_wavering": "not_wavering",
    "health_below_50%_base": "hp_below_half",
    "health_below_25%": "hp_below_quarter",
    # (25.10.2026, Blackpowder Discipline) moving: the unit moves (the observed `mv`); climbing or manning siege
    # equipment on a platform: never on our flat open maps (no walls, towers or ladders)
    "moving": "moving",
    "climbing_manning_eq_on_platform": "climbing",
}
# Words for the predicates (the `when` text).
SAY = {"out_of_melee": "out of melee", "in_melee": "in melee",
       "losing_melee": "not winning its melee (out of melee too: the database's losing_melee_combat as the game acts on it)",
       "morale_below_half": "morale below half of leadership", "not_wavering": "steady (not wavering or routing)",
       "hp_below_half": "health below 50 % of the start", "hp_below_quarter": "health below 25 %",
       "moving": "moving", "climbing": "climbing or manning equipment on a platform (never on our maps)"}
# (stat, how) of the database the simulator lays on a unit (the same as tools/nn/sim/abilities.SIM_STATS).
SIM_STATS = {("scalar_speed", "mult"): "speed", ("scalar_charge_speed", "mult"): "charge_speed",
             ("stat_melee_attack", "add"): "attack", ("stat_melee_defence", "add"): "defence",
             ("stat_melee_damage_base", "mult"): "damage", ("stat_melee_damage_ap", "mult"): "ap",
             ("stat_charge_bonus", "mult"): "charge", ("stat_morale", "add"): "leadership",
             ("stat_resistance_physical", "add"): "resist_physical",
             # the missile damage (poison's x0.85: the contact effects only, tools/nn/sim/effects.py E_STATS)
             ("scalar_missile_damage_base", "mult"): "m_damage", ("scalar_missile_damage_ap", "mult"): "m_ap",
             # 25.10.2026 (line g43): the shield breaker's -24 to the shield's block chance (percentage points of it), the
             # Blackpowder Discipline's accuracy (added to land_units accuracy + marksmanship: the spread) and reload
             # skill (the reload max(4 s, base x (100 - skill) %)); tools/nn/sim/effects.py
             ("stat_missile_block_chance", "add"): "shield", ("stat_accuracy", "add"): "accuracy",
             ("stat_reloading", "add"): "reloading",
             # 29.10.2026 (the Skaven monsters): the fire's resistance - Regeneration's weakness -20 (a flaming blow x1.2;
             # the game's program sums the resistances, build/diag_skv_mon/mechanic.md M14, M20; tools/nn/sim/regen.py)
             ("stat_weakness_flame", "add"): "resist_flame", ("stat_resistance_flame", "add"): "resist_flame"}
# Rules the simulator acts on (tools/nn/sim/effects.py RULES): rule -> what the simulator does.
SIM_RULES = {
    "unbreakable": "morale points never below leadership: never wavers or routs (morale.py)",
    "expendable": "its rout does not count among the routing friends of others (battle.py)",
    "encourages": "the lord's aura: +4 morale points to friends within 70 m, fading to 0 at 105 m (battle.py)",
    "charge_reflection": "braced (slower than melee.braced_speed at contact), meets a frontal infantry charge as "
                         "a charge (battle.py)",
    "fire_while_moving": "aims and shoots while moving (battle.py)",
    "fatigue_immune": "gains no fatigue (fatigue.py)",
    "stalk": "hides anywhere: any ground is cover for it (tools/nn/sim/stealth.py; config/nn/sim.json stealth)",
    "charge_defence_vs_large": "braced (slower than melee.braced_speed), cancels the charge bonus of a large charger "
                               "within 45 deg of its front (melee.py; config/nn/sim.json contact.charge_defence)",
    "charge_defence": "braced, cancels the charge bonus of any charger within 45 deg of its front (melee.py; "
                      "config/nn/sim.json contact.charge_defence)",
    "causes_fear": "-8 morale points (ume_concerned_unit_frightened) to the enemies within fear_effect_range 20 m of its "
                   "formation's edge, not stacking (morale.py; config/nn/sim.json morale.fear)",
    "immune_to_psychology": "no fear, no terror (morale.py; config/nn/sim.json morale.fear, morale.terror)",
    "causes_terror": "an enemy within terror_effect_range 5 m of its formation's edge with morale points below "
                     "morale_shock_terror_morale_threshold 13 runs for morale_shock_rout_timer 14 s (no rout counted), then "
                     "is immune to it for morale_shock_rout_immunity_timer 85 s; not one immune to psychology, unbreakable or "
                     "causing terror itself (morale.py terror; config/nn/sim.json morale.terror; build/diag_skv_mon/"
                     "mechanic.md M17)",
}
# Unit attributes: the database gives the key only (unit_attributes_tables); the engine's rule, its
# numbers and where they come from. `rules`: the flags the attribute sets; `schema_why`: why the
# simulator does not act on a rule (shown in the file; the network still sees the effect).
ATTRIBUTES = {
    "unbreakable": {"name": "Unbreakable", "rules": ["unbreakable"],
                    "does": "never loses leadership, never routs (also not when the army is destroyed)",
                    "kb": KB + "abilities.md#attributes-selection"},
    "expendable": {"name": "Expendable", "rules": ["expendable"],
                   "does": "its rout scares only other expendable units (routing_friends_effect_weighting -3 each)",
                   "kb": KB + "morale.md#waver-rout-rally-shatter"},
    "encourages": {"name": "Encourage", "rules": ["encourages"],
                   "does": "+4 leadership to friends nearby (_kv_morale general_aura_radius 70 m, "
                           "general_inspire_effect_amount_max 4; not stacking with the lord's aura since 5.3)",
                   "kb": KB + "morale.md#modifiers-points"},
    "charge_reflection": {"name": "Charge Reflection", "rules": ["charge_reflection"],
                          "does": "braced, reflects a frontal charge (_kv_rules charge_reflect_damage_multiplier 2, "
                                  "charge_reflect_min_charge_factor_threshold 0.7)",
                          "kb": KB + "melee.md#bracing-and-charge-defence"},
    "charge_defense_vs_large": {"name": "Charge Defence vs. Large", "rules": ["charge_defence_vs_large"],
                                "does": "braced, cancels the charge bonus of large chargers from the front",
                                "kb": KB + "melee.md#bracing-and-charge-defence"},
    "charge_defense": {"name": "Charge Defence", "rules": ["charge_defence"],
                       "does": "braced, cancels the charge bonus of any charger from the front",
                       "kb": KB + "melee.md#bracing-and-charge-defence"},
    "mounted_fire_move": {"name": "Fire Whilst Moving", "rules": ["fire_while_moving"],
                          "does": "shoots on the move", "kb": KB + "missiles.md"},
    "guerrilla_deploy": {"name": "Vanguard Deployment", "rules": ["vanguard"],
                         "does": "may deploy outside the deployment zone, not in the enemy's",
                         "kb": KB + "abilities.md#attributes-selection",
                         "schema_why": "deployment comes from the army generator (tools/nn/armies/place.py)"},
    "hide_forest": {"name": "Hide (forest)", "rules": ["hide_forest"], "does": "hidden while in woods",
                    "kb": KB + "abilities.md#attributes-selection",
                    "schema_why": "the simulator's maps have no woods and no hiding"},
    "stalk": {"name": "Stalk", "rules": ["stalk"],
              "does": "hides in any terrain, open ground too (fandom 'Stalk'); seen by an enemy within its spotting "
                      "distance x the unit's hiding_scalar, revealed by firing and melee, not by running",
              "kb": KB + "abilities.md#attributes-selection"},
    "immune_to_psychology": {"name": "Immune to Psychology", "rules": ["immune_to_psychology"],
                             "does": "immune to fear and terror", "kb": KB + "abilities.md#attributes-selection"},
    "causes_fear": {"name": "Causes Fear", "rules": ["causes_fear"],
                    "does": "-8 leadership to enemies within 20 m (fear_effect_range)", "kb": KB + "morale.md"},
    "causes_terror": {"name": "Causes Terror", "rules": ["causes_terror"],
                      "does": "an enemy within 5 m (terror_effect_range) below 13 morale points runs for 14 s, then is "
                              "immune to it for 85 s (the game's program: build/diag_skv_mon/mechanic.md M17)",
                      "kb": KB + "morale.md"},
    "fatigue_immune": {"name": "Perfect Vigour", "rules": ["fatigue_immune"], "does": "never tires",
                       "kb": KB + "fatigue.md"},
    "ignore_trees": {"name": "Ignore Trees", "rules": ["ignore_trees"], "does": "shoots through trees",
                     "kb": KB + "abilities.md#attributes-selection", "schema_why": "flat maps only: no trees"},
    "snipe": {"name": "Snipe", "rules": ["snipe"], "does": "firing does not reveal the hidden unit (fandom 'Snipe')",
              "kb": KB + "abilities.md#attributes-selection",
              "schema_why": "the unit that has it (Hochland Long Rifles) hides only in woods (hide_forest): no woods on "
                            "our maps, it is never hidden"},
    "strider": {"name": "Strider", "rules": ["strider"], "does": "ignores terrain penalties, passes trees",
                "kb": KB + "abilities.md#attributes-selection", "schema_why": "flat maps only"},
    "moulder_monster": {"name": "Moulder Monster", "rules": ["moulder_monster"],
                        "does": "a mark for the campaign abilities of Clan Moulder's lords (Throt, the Packmaster); "
                                "nothing in a battle by itself",
                        "kb": KB + "abilities.md#attributes-selection", "schema_why": "nothing to act on in battle"},
    # (31.10.2026, the Warp Grinders: build/diag_roster/wh2_dlc14_skv_inf_warp_grinder_0/units.md)
    "wallbreaker": {"name": "Wallbreaker", "rules": ["wallbreaker"],
                    "does": "damages walls and gates (building damage x1.5 on top of the weapon's own); sieges only",
                    "kb": KB + "abilities.md#attributes-selection",
                    "schema_why": "no walls or buildings in our battles"},
}
# Contact effects (fandom "Contact Effect"; docs/en/game/mechanics/abilities.md "Contact effects"): a weapon's blow
# (melee_weapons.contact_phase) or projectile hit (projectiles.contact_stat_effect) lays its phase on the enemy it hits
# for the phase's duration (special_ability_phases, 10 s), each hit starting it again; since Update 8.1 different
# contact effects stack on a unit, the same one does not (CA, "Stacking Contact Effects - Update 8.1"); friendly fire
# lays none. The game's probes (build/newunits/newunits_game.json): the censer's and the poison's phase gone 9.75-10.75 s
# after the attacker began to leave melee, the suppressing fire's 10.5-11 s after the gun's last shot.
CONTACT_NAMES = {"wh_main_unit_contact_poison": "Poison!",
                 "wh2_main_unit_contact_plague_monk_censer_morale": "Contaminated!",
                 "wh2_dlc12_unit_contact_suppressive_fire": "Suppressed!",
                 "wh2_dlc09_unit_contact_shieldbreaker": "Shieldbreaker"}
KB_CONTACT = KB + "abilities.md#contact-effects"
# The ability attributes (special_ability_phase_attribute_effects) that are rules of the simulator.
ABILITY_ATTRIBUTE_RULES = {"unbreakable": "unbreakable", "immune_to_psychology": "immune_to_psychology",
                           "fatigue_immune": "fatigue_immune"}
# The knowledge base's page for known passives (a link only; the numbers are the passport's).
KB_ABILITIES = {"strength_in_numbers": KB + "abilities.md#attributes-selection",
                "scurry_away": KB + "morale.md#waver-rout-rally-shatter",
                "frenzy": KB + "abilities.md#attributes-selection",
                "hold_the_line": KB + "morale.md#modifiers-points",
                "strength_of_the_penitent": "docs/en/training/units.md#flagellants-greatswords-free-company-militia",
                "single_entity": KB + "single-entities.md",
                "snare_net": KB + "abilities.md#attributes-selection",
                "blackpowder_discipline": KB + "abilities.md#attributes-selection",
                "unshakable_pathogen": "docs/ru/training/units.md", "pestilent_perfection": "docs/ru/training/units.md"}
MISSING = {
    "effect bundles": "effect_bundles / effect_bonus_value_*: campaign effects (skills, techs, the contexts of "
                      "battle_context_unit_attribute_junctions such as causes_fear vs a culture); custom battles have none",
    "attribute numbers": "unit_attributes_tables holds the keys only: an attribute's rule and numbers are the "
                         "engine's (_kv tables, game_rules.json) and the knowledge base's (ATTRIBUTES in tools/nn/effects.py)",
    "stubborn, hatred": "not in the WH3 database as unit attributes or passive abilities (WH2 had them)",
    "phase flags": "special_ability_phases (unbreakable, cant_move, ... per phase) is not decoded: the phases of "
                   "our passives carry their effects in the stat and attribute tables",
    "direct damage on others": "an innate effect's direct damage (the passport's direct) on friends or enemies (the "
                               "Cauldron's miasma: not in the pools) - the simulator acts on an innate one on the owner "
                               "itself only (tools/nn/sim/direct.py); an ACTIVE ability's on the enemies within its range "
                               "(the Warp Grinders' Warp-Quake, since 31.10.2026) is no innate effect: "
                               "tools/nn/sim/abilities.py",
    "other vortices": "a vortex of another behaviour (vortex_per_entity, vortex_on_death), of an active ability or a spell, "
                      "a moving one, one striking its own side, several in a row or a subsequent one: the simulator acts on "
                      "a standing vortex at a man's death striking enemies only (tools/nn/sim/vortex.py)",
}
# (31.10.2026, the Pusbags) the vortex at a man's death the simulator acts on (tools/nn/sim/vortex.py;
# build/diag_roster/wh3_dlc29_skv_inf_pusbags/mechanic.md P1-P8): what else such a vortex must be
VORTEX_BEHAVIOURS = ("vortex_on_entity_death",)


def name_of(key):
    """A readable name from an ability key: wh2_main_unit_passive_scurry_away -> Scurry Away."""
    tail = key.split("_passive_", 1)[-1] if "_passive_" in key else key
    return tail.replace("_", " ").title()


def _say(preds, joiner=" and "):
    return joiner.join(SAY.get(p, p) for p in preds)


def attribute_effect(key):
    a = ATTRIBUTES.get(key)
    if a is None:
        return {"kind": "attribute", "name": key.replace("_", " ").title(), "rules": [key], "stats": [],
                "range_m": 0.0, "needs": [], "off_when": [], "timed": None, "when": "always",
                "source": f"db: unit_attributes_to_groups_junctions ({key})", "kb": None, "does": None,
                "modelled": False, "sim": None, "why": "no rule known for this attribute yet (ATTRIBUTES)"}
    rules = list(a["rules"])
    sim_rules = [r for r in rules if r in SIM_RULES]
    modelled = bool(sim_rules) and not a.get("schema_why")
    return {"kind": "attribute", "name": a["name"], "rules": rules, "stats": [], "range_m": 0.0, "needs": [],
            "off_when": [], "timed": None, "when": "always",
            "source": f"db: unit_attributes_to_groups_junctions ({key})", "kb": a["kb"], "does": a["does"],
            "modelled": modelled, "sim": "; ".join(SIM_RULES[r] for r in sim_rules) if modelled else None,
            "why": a.get("schema_why") or (None if modelled else "no simulator rule")}


def ability_effect(key, p):
    """An innate effect from a passive or timed (auto) ability passport."""
    stats, unknown = [], []
    for e in p.get("effects") or ():
        sim = SIM_STATS.get((e["stat"], e["how"]))
        on = [g for g in e.get("on") or () if g in ("self", "friends", "enemies")]
        stats.append({"stat": e["stat"], "how": e["how"], "value": e["value"], "on": on, "sim": sim})
        if sim is None:
            unknown.append(f"{e['stat']} {e['how']}")
    # the charge speed of an effect that lays both scalar_speed and scalar_charge_speed on the same units (the Snare Net
    # on enemies): the game's charge is the speed's x the charge's - the simulator's rule takes a charge multiplier as
    # the whole of it, so the catalogue writes the product as its sim_value (0.9 x 0.9 = 0.81)
    for c in stats:
        if (c["stat"], c["how"]) != ("scalar_charge_speed", "mult"):
            continue
        sp = [s for s in stats if (s["stat"], s["how"]) == ("scalar_speed", "mult") and s["on"] == c["on"]]
        if sp:
            c["sim_value"] = round(float(c["value"]) * float(sp[0]["value"]), 6)
            c["sim_why"] = ("the charge speed as the game moves it: scalar_speed x scalar_charge_speed (the simulator "
                            "takes an effect's charge multiplier as the whole charge)")
    rules = []
    for a in p.get("attributes") or ():
        rule = ABILITY_ATTRIBUTE_RULES.get(a["attribute"], a["attribute"])
        rules.append(rule)
    flags = sorted(set(p.get("off_when") or ()))
    contexts = sorted(set(p.get("recharge_when") or ()))
    bad = [f for f in flags + contexts if f not in CONDITIONS]
    off = [CONDITIONS[f] for f in flags if f in CONDITIONS]
    ctx = [CONDITIONS[c] for c in contexts if c in CONDITIONS]
    timed = None
    if p.get("auto"):
        # fired whenever ready (fires_when: none); its recharge context is when the recharge runs
        timed = {"active_s": p["active_s"], "recharge_s": p["recharge_s"], "fires_when": [], "recharge_needs": ctx}
        needs = []
        when = (f"the game fires it whenever ready: {p['active_s']:g} s, ready again after {p['recharge_s']:g} s "
                + (f"of {_say(ctx)}" if ctx else "") + " (the recharge runs only then)" * bool(ctx))
    else:
        needs = ctx
        when = "always" if not needs else "while " + _say(needs)
    delay = float(p.get("delay_s") or 0.0)
    if delay > 0 and not timed:
        # (the passport's delay_s: its first phases lay nothing - Blackpowder Discipline's 10 s) on once its conditions
        # have held that long without a break
        when += f", once that has held {delay:g} s"
    if any("enemies" in s["on"] for s in stats):
        when += f"; on enemies whose centre is within {float(p.get('range_m') or 0.0):g} m of its centre"
    if off:
        when += ("; " if timed else ", ") + "off while " + _say(off, " or ")
    when = when.replace("always, off while", "off while")
    tail = key.split("_passive_", 1)[-1]
    sources = ["unit_special_abilities", "special_ability_phase_stat_effects"]
    if p.get("attributes"):
        sources.append("special_ability_phase_attribute_effects")
    if flags:
        sources.append("special_ability_to_auto_deactivate_flags")
    if contexts:
        sources.append("special_ability_to_recharge_contexts")
    if p.get("direct"):
        sources += ["special_ability_phases", "_kv_unit_ability_scaling_rules"]
    if p.get("vortex"):
        sources.append("battle_vortexs")
    sim_rules = [r for r in rules if r in SIM_RULES]
    schema_rules = [r for r in rules if r not in SIM_RULES]
    why = []
    if unknown:
        why.append("stats the simulator lacks: " + ", ".join(unknown))
    if bad:
        why.append("conditions the simulator lacks: " + ", ".join(bad))
    if schema_rules:
        why.append("rules with nothing to act on: " + ", ".join(schema_rules))
    modelled = not unknown and not bad
    parts = [f"{s['stat']} {s['how']} {s['value']:g}" for s in stats if s["sim"]] + [f"rule {r}" for r in sim_rules]
    direct, direct_why = direct_of(p)
    if direct:
        parts.append(f"direct damage {direct['amount']} x {direct['scale']:g} to {direct['entities']} random own men every "
                     f"{direct['every_s']:g} s ({direct['hit_hp']:g} HP a man)")
        if direct_why:
            why.append(direct_why)
    vortex, vortex_why = vortex_of(p)
    if vortex:
        parts.append(f"at every own man's death a vortex {vortex['delay_s']:g} s later: {vortex['goal_radius_m']:g} m, "
                     f"{vortex['damage']} + {vortex['damage_ap']} once a man, force {vortex['force']:g}"
                     + (f", lays {vortex['contact']}" if vortex.get("contact") else ""))
        if vortex_why:
            why.append(vortex_why)
    modelled = modelled and not direct_why and not vortex_why
    heal = None
    if p.get("heal"):
        # (the ability passport's heal: Regeneration's 0.001 of the largest health every 1 s; tools/nn/sim/regen.py)
        h = p["heal"]
        heal = {"share": float(h["share"]), "every_s": float(h["every_s"])}
        parts.append(f"heals {h['share']:g} of its largest health every {h['every_s']:g} s")
    return {"kind": "timed" if timed else "passive", "name": name_of(key), "rules": rules, "stats": stats,
            "range_m": float(p.get("range_m") or 0.0), "needs": needs, "off_when": off, "timed": timed,
            **({"delay_s": delay} if delay > 0 and not timed else {}),
            **({"heal": heal} if heal else {}),
            **({"direct": direct} if direct else {}),
            **({"vortex": vortex} if vortex else {}),
            "when": when, "source": "db: " + ", ".join(sources) + f" ({key}); config/nn/abilities.json",
            "kb": KB_ABILITIES.get(tail), "does": None,
            "modelled": modelled, "sim": ("; ".join(parts) + f" ({when})") if modelled else None,
            "why": "; ".join(why) or None}


def direct_of(p):
    """(direct, why not) of a passport's direct damage (abilities.py FIELDS direct): the simulator acts on the one on the
    owner itself (tools/nn/sim/direct.py); None, None without one."""
    d = p.get("direct")
    if not d:
        return None, None
    out = {k: d[k] for k in ("amount", "entities", "every_s", "on", "scale", "min_unary", "hit_hp", "hit_dist",
                             "range_m", "hold_m", "n_enemy", "n_friend", "every_frame", "duration_s", "magical") if k in d}
    # (the ability's auto-deactivate flag out_of_melee: only in melee - the Pusbags' Pathogen)
    out["melee_only"] = "out_of_melee" in (p.get("off_when") or ())
    if list(d.get("on") or ()) != ["self"]:
        return out, "direct damage on " + ", ".join(d.get("on") or ["nobody"]) + " (the owner itself only is modelled)"
    if float(d.get("every_s") or 0) <= 0 or int(d.get("entities") or 0) == 0:
        return out, "direct damage without a tick or men"
    return out, None


def vortex_of(p):
    """(vortex, why not) of a passport's vortex (abilities.py FIELDS vortex): the simulator acts on a standing vortex at a
    man's death that strikes enemies only, once, without a next one (tools/nn/sim/vortex.py); None, None without one."""
    v = p.get("vortex")
    if not v:
        return None, None
    out = dict(v, behaviour=p.get("behaviour"))
    bad = []
    if p.get("behaviour") not in VORTEX_BEHAVIOURS:
        bad.append(f"behaviour {p.get('behaviour')}")
    if float(v.get("move_m_s") or 0) != 0 or v.get("follow"):
        bad.append("a moving vortex")
    if v.get("allies") or not v.get("enemies"):
        bad.append("a vortex striking its own side")
    if int(v.get("n") or 1) != 1 or v.get("next"):
        bad.append("several vortices")
    return out, ("vortex not modelled: " + ", ".join(bad)) if bad else None


def innate(passport):
    """Is an ability innate (passive or fired by the game itself), not cast by order?"""
    return bool(passport.get("passive") or passport.get("auto"))


def contact_effect(key, phase):
    """A contact effect from the passports' contact phase (config/nn/units.json contact_phases): its stats lie on the
    unit a weapon of it hit, for duration_s from the last hit (CONTACT_NAMES' comment)."""
    stats, unknown = [], []
    for e in phase.get("stats") or ():
        sim = SIM_STATS.get((e["stat"], e["how"]))
        stats.append({"stat": e["stat"], "how": e["how"], "value": e["value"], "on": ["hit"], "sim": sim})
        if sim is None:
            unknown.append(f"{e['stat']} {e['how']}")
    d = float(phase.get("duration_s") or 0.0)
    modelled = not unknown and d > 0
    parts = [f"{s['stat']} {s['how']} {s['value']:g}" for s in stats if s["sim"]]
    when = f"for {d:g} s after each hit of a weapon that lays it (refreshed by every hit; another contact effect stacks)"
    return {"kind": "contact", "name": CONTACT_NAMES.get(key, name_of(key)), "rules": [], "stats": stats,
            "range_m": 0.0, "needs": [], "off_when": [], "timed": None, "duration_s": d, "when": when,
            "source": f"db: melee_weapons.contact_phase / projectiles.contact_stat_effect, special_ability_phases "
                      f"(duration), special_ability_phase_stat_effects ({key}); config/nn/units.json contact_phases",
            "kb": KB_CONTACT, "does": None, "modelled": modelled,
            "sim": ("; ".join(parts) + f" ({when})") if modelled else None,
            "why": ("stats the simulator lacks: " + ", ".join(unknown)) if unknown else None}


def inflicted(unit):
    """The contact effects a unit's weapons lay: [phase keys] (melee, then missile; none: [])."""
    return sorted({p for p in ((unit.get("melee") or {}).get("contact_phase"),
                               (unit.get("missile") or {}).get("contact_phase")) if p})


def build(units, abilities, previous_order=(), phases=None):
    """(effects {key: effect}, unit links {unit key: [effect keys]}, order [keys], inflicts {unit key: [contact effect
    keys its weapons lay]}). phases: the passports' contact phases (config/nn/units.json contact_phases)."""
    effects, links, inflicts = {}, {}, {}
    phases = phases or {}
    for ukey, u in sorted(units.items()):
        own = []
        for a in u.get("attributes") or ():
            effects.setdefault(a, attribute_effect(a))
            own.append(a)
        for a in u.get("abilities") or ():
            p = abilities.get(a)
            if p is not None and innate(p):
                effects.setdefault(a, ability_effect(a, p))
                own.append(a)
        links[ukey] = sorted(own)
        # (31.10.2026) and the contact effects its men's vortices lay (the passport's vortex.contact: the Pusbags' -10
        # morale of the censer)
        vx = sorted({(abilities.get(a) or {}).get("vortex", {}).get("contact") for a in u.get("abilities") or ()} - {None})
        lays = sorted(set(k for k in inflicted(u) + vx if k in phases))
        for k in lays:
            effects.setdefault(k, contact_effect(k, phases[k]))
        if lays:
            inflicts[ukey] = lays
    # append-only: an effect keeps its place (one no unit owns any more keeps it too, empty), new ones go last
    order = list(previous_order) + sorted(k for k in effects if k not in previous_order)
    return effects, links, order, inflicts


def document(effects, links, order, inflicts=None):
    return {"_description": "Innate unit effects (docs/en/training/units.md, Innate effects): every attribute and "
                            "passive or game-fired ability of the units in config/nn/units.json, from the game's "
                            "database. Written by python -m tools.nn.effects; the simulator "
                            "(tools/nn/sim/effects.py) lays on the `modelled` ones, the network "
                            "(tools/nn/model/effects.py) sees each by `order` (append-only).",
            "_fields": {
                "kind": "attribute (a rule flag of the engine), passive (on while its conditions hold), timed "
                        "(fired by the game itself, active_s then recharge_s)",
                "rules": "rule flags it sets (attributes; ability attribute effects)",
                "stats": "the ability passport's stat effects: stat, how (add / mult), value, on (self, friends "
                         "within range_m), sim (the simulator's stat; null: not modelled)",
                "needs": "predicates that must hold for it to be on (the database's recharge contexts of a passive: "
                         "its recharge runs only there)",
                "off_when": "predicates that hold it off (the database's auto-deactivate flags)",
                "timed": "a timed passive: active_s, recharge_s, fires_when (predicates that must hold to fire "
                         "besides being ready and nothing holding it off; the database has none: []), "
                         "recharge_needs (the recharge contexts: its recharge runs only while they hold)",
                "when": "the condition in words", "source": "database tables and rows", "kb": "knowledge base page",
                "modelled": "the simulator acts on it", "sim": "what the simulator does", "why": "why not (all) of it",
                "predicates": "the conditions the simulator evaluates: " + ", ".join(
                    f"{p} = {SAY[p]}" for p in SAY) + " (db flag -> predicate: " + ", ".join(
                    f"{k} -> {v}" for k, v in CONDITIONS.items()) + ")"},
            "_missing": MISSING,
            "_inflicts": "contact effects (kind contact) a unit's weapons lay on the enemies they hit (the passport's "
                         "melee.contact_phase, missile.contact_phase); the unit does not own them (units), the network "
                         "sees them as its `owned` column of the effect, `on` while the unit itself is under one",
            "order": order,
            "effects": {k: effects[k] for k in order if k in effects},
            "units": links,
            "inflicts": inflicts or {}}


def load(path=None):
    """The catalogue (config/nn/effects.json): {"order", "effects", "units"}."""
    return json.loads(Path(path or OUT).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    udoc = json.loads(UNITS.read_text(encoding="utf-8"))
    units = udoc["units"]
    abilities = json.loads(ABILITIES.read_text(encoding="utf-8"))["abilities"]
    previous = load(args.out)["order"] if args.out.exists() else ()
    effects, links, order, inflicts = build(units, abilities, previous, udoc.get("contact_phases"))
    args.out.write_text(json.dumps(document(effects, links, order, inflicts), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
    for k in order:
        e = effects.get(k)
        if e:
            owners = sum(k in v for v in links.values()) + sum(k in v for v in inflicts.values())
            print(f"{k}: {e['kind']}, {'modelled' if e['modelled'] else 'schema only'} ({e['when']}), "
                  f"{owners} units" + (f" - {e['why']}" if e["why"] else ""))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
