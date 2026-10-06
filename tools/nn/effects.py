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
}
# Words for the predicates (the `when` text).
SAY = {"out_of_melee": "out of melee", "in_melee": "in melee",
       "losing_melee": "not winning its melee (out of melee too: the database's losing_melee_combat as the game acts on it)",
       "morale_below_half": "morale below half of leadership", "not_wavering": "steady (not wavering or routing)",
       "hp_below_half": "health below 50 % of the start", "hp_below_quarter": "health below 25 %"}
# (stat, how) of the database the simulator lays on a unit (the same as tools/nn/sim/abilities.SIM_STATS).
SIM_STATS = {("scalar_speed", "mult"): "speed", ("scalar_charge_speed", "mult"): "charge_speed",
             ("stat_melee_attack", "add"): "attack", ("stat_melee_defence", "add"): "defence",
             ("stat_melee_damage_base", "mult"): "damage", ("stat_melee_damage_ap", "mult"): "ap",
             ("stat_charge_bonus", "mult"): "charge", ("stat_morale", "add"): "leadership",
             ("stat_resistance_physical", "add"): "resist_physical"}
# Rules the simulator acts on (tools/nn/sim/effects.py RULES): rule -> what the simulator does.
SIM_RULES = {
    "unbreakable": "morale points never below leadership: never wavers or routs (morale.py)",
    "expendable": "its rout does not count among the routing friends of others (battle.py)",
    "encourages": "the lord's aura: +4 morale points to friends within 70 m, fading to 0 at 105 m (battle.py)",
    "charge_reflection": "braced (slower than melee.braced_speed at contact), meets a frontal infantry charge as "
                         "a charge (battle.py)",
    "fire_while_moving": "aims and shoots while moving (battle.py)",
    "fatigue_immune": "gains no fatigue (fatigue.py)",
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
                                "kb": KB + "melee.md#bracing-and-charge-defence",
                                "schema_why": "no large units in our pools (no cavalry or monsters yet)"},
    "charge_defense": {"name": "Charge Defence", "rules": ["charge_defence"],
                       "does": "braced, cancels the charge bonus of any charger from the front",
                       "kb": KB + "melee.md#bracing-and-charge-defence",
                       "schema_why": "the simulator's charge rule has no per-attacker cancel yet"},
    "mounted_fire_move": {"name": "Fire Whilst Moving", "rules": ["fire_while_moving"],
                          "does": "shoots on the move", "kb": KB + "missiles.md"},
    "guerrilla_deploy": {"name": "Vanguard Deployment", "rules": ["vanguard"],
                         "does": "may deploy outside the deployment zone, not in the enemy's",
                         "kb": KB + "abilities.md#attributes-selection",
                         "schema_why": "deployment comes from the army generator (tools/nn/armies/place.py)"},
    "hide_forest": {"name": "Hide (forest)", "rules": ["hide_forest"], "does": "hidden while in woods",
                    "kb": KB + "abilities.md#attributes-selection",
                    "schema_why": "the simulator's maps have no woods and no hiding"},
    "stalk": {"name": "Stalk", "rules": ["stalk"], "does": "hidden while moving, anywhere",
              "kb": KB + "abilities.md#attributes-selection", "schema_why": "the simulator has no hiding"},
    "immune_to_psychology": {"name": "Immune to Psychology", "rules": ["immune_to_psychology"],
                             "does": "immune to fear and terror", "kb": KB + "abilities.md#attributes-selection",
                             "schema_why": "no fear or terror in the simulator: nothing to act on"},
    "causes_fear": {"name": "Causes Fear", "rules": ["causes_fear"],
                    "does": "-8 leadership to enemies within 20 m (fear_effect_range)",
                    "kb": KB + "morale.md", "schema_why": "fear is not modelled"},
    "causes_terror": {"name": "Causes Terror", "rules": ["causes_terror"],
                      "does": "a melee hit routs an enemy within 5 m at 13 morale points or less for 14 s",
                      "kb": KB + "morale.md", "schema_why": "terror is not modelled"},
    "fatigue_immune": {"name": "Perfect Vigour", "rules": ["fatigue_immune"], "does": "never tires",
                       "kb": KB + "fatigue.md"},
    "strider": {"name": "Strider", "rules": ["strider"], "does": "ignores terrain penalties, passes trees",
                "kb": KB + "abilities.md#attributes-selection", "schema_why": "flat maps only"},
}
# The ability attributes (special_ability_phase_attribute_effects) that are rules of the simulator.
ABILITY_ATTRIBUTE_RULES = {"unbreakable": "unbreakable", "immune_to_psychology": "immune_to_psychology",
                           "fatigue_immune": "fatigue_immune"}
# The knowledge base's page for known passives (a link only; the numbers are the passport's).
KB_ABILITIES = {"strength_in_numbers": KB + "abilities.md#attributes-selection",
                "scurry_away": KB + "morale.md#waver-rout-rally-shatter",
                "frenzy": KB + "abilities.md#attributes-selection",
                "hold_the_line": KB + "morale.md#modifiers-points",
                "strength_of_the_penitent": "docs/en/training/units.md#flagellants-greatswords-free-company-militia",
                "single_entity": KB + "single-entities.md"}
MISSING = {
    "effect bundles": "effect_bundles / effect_bonus_value_*: campaign effects (skills, techs, the contexts of "
                      "battle_context_unit_attribute_junctions such as causes_fear vs a culture); custom battles have none",
    "attribute numbers": "unit_attributes_tables holds the keys only: an attribute's rule and numbers are the "
                         "engine's (_kv tables, game_rules.json) and the knowledge base's (ATTRIBUTES in tools/nn/effects.py)",
    "stubborn, hatred": "not in the WH3 database as unit attributes or passive abilities (WH2 had them)",
    "phase flags": "special_ability_phases (unbreakable, cant_move, ... per phase) is not decoded: the phases of "
                   "our passives carry their effects in the stat and attribute tables",
}


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
        if "enemies" in on:
            unknown.append("effects on enemies")
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
    return {"kind": "timed" if timed else "passive", "name": name_of(key), "rules": rules, "stats": stats,
            "range_m": float(p.get("range_m") or 0.0), "needs": needs, "off_when": off, "timed": timed,
            "when": when, "source": "db: " + ", ".join(sources) + f" ({key}); config/nn/abilities.json",
            "kb": KB_ABILITIES.get(tail), "does": None,
            "modelled": modelled, "sim": ("; ".join(parts) + f" ({when})") if modelled else None,
            "why": "; ".join(why) or None}


def innate(passport):
    """Is an ability innate (passive or fired by the game itself), not cast by order?"""
    return bool(passport.get("passive") or passport.get("auto"))


def build(units, abilities, previous_order=()):
    """(effects {key: effect}, unit links {unit key: [effect keys]}, order [keys])."""
    effects, links = {}, {}
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
    # append-only: an effect keeps its place (one no unit owns any more keeps it too, empty), new ones go last
    order = list(previous_order) + sorted(k for k in effects if k not in previous_order)
    return effects, links, order


def document(effects, links, order):
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
            "order": order,
            "effects": {k: effects[k] for k in order if k in effects},
            "units": links}


def load(path=None):
    """The catalogue (config/nn/effects.json): {"order", "effects", "units"}."""
    return json.loads(Path(path or OUT).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    units = json.loads(UNITS.read_text(encoding="utf-8"))["units"]
    abilities = json.loads(ABILITIES.read_text(encoding="utf-8"))["abilities"]
    previous = load(args.out)["order"] if args.out.exists() else ()
    effects, links, order = build(units, abilities, previous)
    args.out.write_text(json.dumps(document(effects, links, order), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
    for k in order:
        e = effects.get(k)
        if e:
            owners = sum(k in v for v in links.values())
            print(f"{k}: {e['kind']}, {'modelled' if e['modelled'] else 'schema only'} ({e['when']}), "
                  f"{owners} units" + (f" - {e['why']}" if e["why"] else ""))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
