"""The simulator's numbers, from three files (docs/en/training/simulator.md):

    config/nn/units.json       unit passports (the game's database)
    config/nn/abilities.json   ability passports (the game's database)
    config/nn/effects.json     innate effects: attributes and passives, linked to units (the game's database)
    config/nn/game_rules.json  the game's battle, morale and fatigue rules (the game's database)
    config/nn/sim.json         measured or calibrated numbers the others lack

Only json; no torch here.
"""
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from tools.nn.sim import abilities, magic
from tools.nn.sim import effects as innate
from tools.nn.sim.state import SIZE_RANK

ROOT = Path(__file__).resolve().parents[3]
# Projectile categories a shield blocks (small arms: tw-modding "Missiles and You", fandom "Shielded"; artillery,
# misc, grenades and spells go through; build/shields/spec.md S4).
SMALL_ARMS = ("arrow", "musket", "sling", "javelin", "axe")
CONFIG = ROOT / "config" / "nn"


def _value(x):
    return x["value"] if isinstance(x, dict) and "value" in x else x


def aim_times(missile, ms):
    """(aim_s, rt_s, quiet_s) of a missile weapon (a passport's `missile`; empty: none) by sim.json's `missile` section
    `ms` (battle.py applies them):
      aim_s    seconds of aim before the first shot, the aim clock starting at a halt - the projectile's own
               (aim_projectile_s: the Ratling Gun's spin-up 4.5), else the category's (aim_s), else aim_s_other; 0
               without a missile weapon;
      rt_s     the aim a new target costs a unit that had one - the projectile's own (retarget_projectile_s), else the
               category's (retarget_cat_s), else -1: the rule of all (a new order resets the aim, a switch costs
               retarget_s);
      quiet_s  arcing fire's silence after a new order or a new target (order_quiet_cat_s), 0 for the rest.
    The simulator's STATIC aim_s, rt_s, quiet_s and the network's AIM columns (tools/nn/model/passport.aim) come from
    here."""
    if not missile:
        return 0, -1.0, 0.0
    cat, proj = missile.get("category"), missile.get("projectile")
    own_aim = (ms.get("aim_projectile_s") or {}).get(proj)
    own_rt = (ms.get("retarget_projectile_s") or {}).get(proj)
    aim_s = float(own_aim) if own_aim is not None else ms["aim_s"].get(cat, ms["aim_s_other"])
    rt_s = float(own_rt) if own_rt is not None else float((ms.get("retarget_cat_s") or {}).get(cat, -1.0))
    quiet_s = float((ms.get("order_quiet_cat_s") or {}).get(cat, 0.0))
    return aim_s, rt_s, quiet_s


def reform_share(cls, rf):
    """STATIC wmin_k of a unit of the database's class `cls` (land_units.class) by sim.json formation.reform `rf`: the
    narrowest formation's files as a share of its men - the cavalry's, monsters' and chariots' (classes starting with
    large_classes) large_share, the others' share; 0 (the width fixed) for a class starting with fixed_classes
    (artillery) or with the rule off."""
    if not rf.get("on"):
        return 0.0
    c = str(cls or "")
    if c.startswith(tuple(rf.get("fixed_classes") or ())):
        return 0.0
    if c.startswith(tuple(rf.get("large_classes") or ())):
        return float(rf["large_share"])
    return float(rf["share"])


def retarget_silence(aim_s, rt_s, quiet_s, ms):
    """Seconds without a shot after an attack order on another target, for a unit that aims at one (battle.py): the
    new order takes the aim back by rt_s (-1: to 0, missile.aim_reset_on_order), the switch of target by rt_s (-1:
    retarget_s; nothing with retarget_s 0), and arcing fire keeps quiet quiet_s (order_quiet_cat_s) - the longest of
    them. The reload is not reset: it runs on (the shot also waits for loaded men). 0 without a missile weapon.
    Arrows and slings 8.5 (the quiet window), the Ratling Gun 4.5 (its spin-up), muskets 0, the rest aim_s."""
    if not aim_s:
        return 0.0
    order = (rt_s if rt_s >= 0 else aim_s) if ms.get("aim_reset_on_order") else 0.0
    rule = ms.get("retarget_s")
    switch = (rt_s if rt_s >= 0 else rule) if rule else 0.0
    return float(max(order, switch, quiet_s))


@dataclass
class Params:
    units: dict      # key -> passport
    rules: dict      # game_rules.json: battle, morale, fatigue
    sim: dict        # sim.json
    abilities: dict = None   # abilities.json: ability key -> passport
    effects: dict = None     # effects.json: {"order", "effects", "units"}
    spells: dict = None      # abilities.json "spells": spell key -> passport (tools/nn/sim/magic.py)
    # worked out once from the passports (__post_init__), read by the step (a constant for torch.compile: no Python
    # loop over the passports inside the compiled step)
    pen_men: int = field(init=False, default=1)       # the most men a projectile strikes (missile.pen_limit)
    can_hide: bool = field(init=False, default=False)  # some unit can hide: the attribute stalk (stealth.on)
    has_art: bool = field(init=False, default=False)   # some unit has guns and artillery is on (artillery.on)

    def __post_init__(self):
        self.pen_men = pen_limit_of(self.units or {}, self.sim or {})
        # (hide_forest too: on a map with forest ground such a unit hides there - stealth.cover; without one it never
        # meets the conditions, so stealth's step changes nothing)
        self.can_hide = any({"stalk", "hide_forest"} & set(u.get("attributes") or ())
                            for u in (self.units or {}).values())
        self.has_art = bool(((self.sim or {}).get("artillery") or {}).get("on")) and any(
            (u.get("engines") or {}).get("n") for u in (self.units or {}).values())

    # --- the game's rules (config/nn/game_rules.json) ---
    @property
    def battle(self):
        return self.rules["battle"]

    @property
    def morale(self):
        return self.rules["morale"]

    @property
    def fatigue(self):
        return self.rules["fatigue"]

    # --- calibration (config/nn/sim.json) ---
    @property
    def dt(self):
        return float(_value(self.sim["step_s"]))

    @property
    def limit_s(self):
        return float(_value(self.sim["battle_limit_s"]))

    @property
    def map_half(self):
        return float(_value(self.sim["map_half_m"]))

    def cal(self, section, name):
        return self.sim[section][name]

    def with_cal(self, section, **values):
        """A copy with some calibrated numbers replaced (for fitting)."""
        sim = json.loads(json.dumps(self.sim))
        sim[section].update(values)
        return Params(self.units, self.rules, sim, self.abilities, self.effects, self.spells)

    # --- per unit ---
    def spacing_of(self, key):
        """(h, v), m: a man's place across the front and between ranks in the unit's close formation (the
        database's unit_spacings of its template, config/nn/units.json spacing); sim.json formation.spacing_m for a
        unit without one."""
        sp = (self.units.get(key) or {}).get("spacing") or {}
        d = float(self.sim["formation"]["spacing_m"])
        return float(sp.get("h") or d), float(sp.get("v") or d)

    def skirmish_gaps(self, key):
        """{"sk_m", "sk_stop_m"} of a unit key (config/nn/sim.json skirmish: start_gap_m / stop_gap_m by key, else
        "default"); both 0 for a unit that cannot skirmish (no can_skirmish or no missile weapon) or with the rule off."""
        u = self.units[key]
        cal = self.sim.get("skirmish") or {}
        if not cal.get("on") or not u.get("can_skirmish") or not u.get("missile"):
            return {"sk_m": 0.0, "sk_stop_m": 0.0}

        def by_key(table):
            return float(table.get(key, table["default"])) if isinstance(table, dict) else float(table)
        return {"sk_m": by_key(cal["start_gap_m"]), "sk_stop_m": by_key(cal["stop_gap_m"])}

    def artillery_static(self, key):
        """The unit's STATIC artillery fields (module artillery_static)."""
        # (the blast's contact effect as the weapons' own: the modelled contact effects, effects.laid_by)
        idx = {k: i for _, i, k in innate.contact_keys(self)}
        return artillery_static(self.units, self.sim, idx, key)

    def static(self, key, faction=None):
        """The STATIC fields (tools/nn/sim/state.py) of a unit key, without its place."""
        u = self.units[key]
        m = u["melee"]
        missile = u.get("missile") or {}
        cat = missile.get("category")
        ms = self.sim["missile"]
        # measured reload of the category, for the passport reload it was measured on (reload_ref_s); another
        # unit of the category scales with its passport (Night Runners' sling 8 s against the slaves' 9 s)
        # a projectile measured on its own (missile.reload_projectile_s) takes its measured cycle
        reload_s = 0
        own = (ms.get("reload_projectile_s") or {}).get(missile.get("projectile")) if missile else None
        if own is not None:
            reload_s = float(own)
        elif missile and cat in (ms.get("reload_db_categories") or ()):
            # (missile.reload_db_why) artillery: the database's reload (no fitted scale) + the gun's measured animations
            # (artillery.cycle_add_s by projectile: a gun's cycle is reload + aim + fire animation, mechanic.md 4)
            reload_s = float(missile.get("reload_s") or 0.0) + float(
                ((self.sim.get("artillery") or {}).get("cycle_add_s") or {}).get(missile.get("projectile"), 0.0)
                if u.get("engines") and (self.sim.get("artillery") or {}).get("on") else 0.0)
        elif missile:
            ref = (ms.get("reload_ref_s") or {}).get(cat)
            reload_s = (ms["reload_s"][cat] * (missile.get("reload_s", ref) / ref if ref else 1.0)
                        if cat in ms["reload_s"] else missile.get("reload_s", 0) * ms["reload_scale_other"])
        aim_s, rt_s, quiet_s = aim_times(missile, ms)
        resist = u.get("damage_resist") or {}
        eng = u.get("engines") or None
        h, v = self.spacing_of(key)
        men = u["men"]
        return {
            "men0": men, "hp_man": u["hp_per_man"], "hp0": u["hp_total"], "mass": u["mass"],
            "radius": u["radius_m"], "sp_h": h, "sp_v": v,
            # the body across: an ellipse body (a beast, a weapon team) is radius long and radius x this across
            "body_ratio": u.get("radii_ratio", 1.0),
            # the default order: men / rank_depth files (the database's ranks), h apart
            # (an artillery unit: its guns side by side, one file a gun - geometry.dims)
            "width": (eng["n"] * h if eng and (self.sim.get("artillery") or {}).get("on")
                      else (men / max(1, u.get("rank_depth") or 1)) * h if men > 1 else 2 * u["radius_m"]),
            # the narrowest formation's files as a share of the men by the database's class (formation.reform; 0: fixed)
            "wmin_k": reform_share(u.get("class"), self.sim["formation"].get("reform") or {}),
            "walk": u["speed"]["walk"], "run": u["speed"]["run"], "charge_speed": u["speed"]["charge"],
            "charge_dist": u["speed"].get("charge_distance", 0.0),
            "charge_pose": u["speed"].get("charge_pose_distance", 0.0),
            "accel": u["speed"]["acceleration"], "decel": u["speed"]["deceleration"],
            "turn": u["speed"].get("turn_deg_s", 120.0),
            "attack": m["attack"], "defence": m["defence"], "charge_bonus": m["charge_bonus"],
            "damage": m["damage"], "ap_damage": m["ap_damage"], "bonus_v_large": m["bonus_v_large"],
            "bonus_v_inf": m["bonus_v_infantry"], "interval": m["attack_interval_s"],
            "splash": max(1, m.get("splash_max_attacks") or 1),
            # the largest target a blow strikes several of (melee.splash_by_size; build/diag_cav/mechanic.md M18): the
            # weapon's splash_attack_target_size, else smaller than the striker
            "splash_cap": float(SIZE_RANK[m["splash_target_size"]] if m.get("splash_target_size") in SIZE_RANK
                                else SIZE_RANK.get(u.get("size", "small"), 1) - 1),
            "armour": u["armour"], "shield": (u.get("shield") or {}).get("missile_block_chance", 0) / 100,
            "leadership": u["leadership"], "resist_missile": resist.get("missile", 0) / 100,
            # damage_mod_all (a ward save) adds to physical: every damage here is physical (melee, and missiles
            # through missile.physical_resist), so it counts once (0 for every unit today)
            "resist_physical": (resist.get("physical", 0) + resist.get("all", 0)) / 100,
            # spell resistance and the ward cut a magical projectile (magic.py; docs/en/game/mechanics/magic.md)
            "resist_magic": min(0.9, (resist.get("magic", 0) + resist.get("all", 0)) / 100),
            # the ward alone: what a magical vortex that is no spell meets (vortex.py; 0 for every unit today)
            "resist_all": min(0.9, resist.get("all", 0) / 100),
            "large": u.get("size", "small") != "small",
            # (29.10.2026, the Skaven monsters; state.py) the knock resistance of its body, its blows' fire and magic, its
            # own fire resistance (the effects add Regeneration's weakness, effects.py) and its regeneration's pool
            "knock_ign": float(u.get("knock_ignore") or 0) / 100,
            "flaming": bool(m.get("flaming")), "mmagic": bool(m.get("magic")),
            "resist_flame": float(resist.get("flame", 0)) / 100,
            # its men brace (the database's can_brace; impact.py) and its run's fatigue (_kv_fatigue: running_cavalry for
            # the category cavalry, else running; fatigue.py)
            "brace_ok": bool(u.get("can_brace")),
            "fat_run": float(self.fatigue.get("running_cavalry", self.fatigue["running"])
                             if u.get("category") == "cavalry" else self.fatigue["running"]),
            # rule flags (expendable, encourages, reflect, unbreakable, fire_move, fatigue_immune), the
            # effects bitmask and the timed effects: from the unit's innate effects (config/nn/effects.json)
            **innate.static(self, key),
            # (regen.py) the share of its starting health it may heal in a battle: sim.json regen.cap with a regeneration
            "heal_cap": float((self.sim.get("regen") or {}).get("cap", 0.0)) if innate.heal_of(self, key) > 0 else 0.0,
            "direct": bool(missile.get("direct")) if missile else False,
            # a volley of several shots a man (the database's shots_per_volley) fired as a burst at the measured rate
            # (missile.burst_rate_s by projectile; missile.py, battle.py)
            "spv": float(missile.get("shots_per_volley") or 1) if missile else 1.0,
            "bst_rate": float((ms.get("burst_rate_s") or {}).get(missile.get("projectile"), 0.0)) if missile else 0.0,
            "melee_fire": bool(missile) and missile.get("projectile") in (ms.get("fire_in_melee") or ()),
            # its hits on a lone man in melee land as out of melee (missile.single_in_melee_projectiles: the Ratling Gun)
            "lone_hit": bool(missile) and missile.get("projectile") in (ms.get("single_in_melee_projectiles") or ()),
            # (an artillery unit's ammunition is per gun: missile.ammo_per 'engine')
            "ammo0": ((eng["n"] if missile.get("ammo_per") == "engine" and eng else men) * missile.get("ammo", 0)
                      if missile else 0),
            "range": missile.get("range_m", 0) if missile else 0,
            "reload": reload_s,
            # the database's reload (the reload skill's base, effects.py: Blackpowder Discipline's +10)
            "reload_db": float(missile.get("reload_s") or 0.0) if missile else 0.0,
            "m_damage": missile.get("damage", 0) if missile else 0,
            "m_ap": missile.get("ap_damage", 0) if missile else 0,
            "hit_rate": ms["hit_rate"].get(cat, ms["hit_rate_other"]) if missile else 0,
            # the spread (missile.py hit_chance): the database's projectile and the unit's accuracy
            "arc": missile.get("fire_arc_deg", 180.0) if missile else 0,
            "acc": (missile.get("accuracy", 0) + missile.get("marksmanship", 0)) if missile else 0,
            "cal_area": missile.get("calibration_area_m", 0.0) if missile else 0,
            "cal_dist": missile.get("calibration_distance_m", 0.0) if missile else 0,
            "muzzle_v": missile.get("muzzle_velocity", 0.0) if missile else 0,
            "height": u.get("height_m", 1.8),
            "small_arms": cat in SMALL_ARMS if missile else False,
            # a projectile measured on its own (missile.aim_projectile_s: the Ratling Gun's spin-up) takes its own aim
            "aim_s": aim_s,
            # by category, measured (missile.order_quiet_why): arcing fire keeps quiet after a new order or target,
            # whatever its reload; direct fire and throwing weapons lose nothing (-1: the rule of all)
            "quiet_s": quiet_s,
            # (missile.reaim_broken_why) arrows and bolts: once the target lost this share of its men, each shot aims
            "broken_at": float((ms.get("reaim_broken_cat_share") or {}).get(cat, 0.0)) if missile else 0.0,
            "rt_s": rt_s,
            # its hits slow the small formations they hit (missile.hit_slow: the hit reactions of gunpowder, by category)
            "hslow": bool(missile) and bool((ms.get("hit_slow") or {}).get("on"))
                     and cat in ((ms.get("hit_slow") or {}).get("categories") or ()),
            "friendly_fire": ms["friendly_fire"].get(cat, ms["friendly_fire_other"]) if missile else 0,
            "morale_bonus": self.sim["morale"]["faction_bonus"].get(faction, 0),
            "rout_death_s": (self.sim["morale"]["lord_fall"].get("rout_death_s") or {}).get(faction, 0),
            "cost": u.get("multiplayer_cost", 0),
            # stealth (stealth.py): its spotting distance (open ground: spot_dist_scrub) and hiding_scalar
            "spot_m": float((u.get("spotting") or {}).get("scrub", 80.0)),
            # ... and of a target mostly on forest ground (spot_dist_tree: 60 m; stealth.py)
            "spot_tree": float((u.get("spotting") or {}).get("tree", 60.0)),
            "hide_k": float(u.get("hiding_scalar") or 1.0),
            # the map's ground and trees (ground.py): hides in woods (hide_forest), no ground penalty and through the
            # trees (strider), through the trees (ignore_trees); its projectile's first metres no tree stops
            # (the database's vegetation_ignore_time x muzzle_velocity: terrain.trees.ignore_s by projectile)
            "hidef": "hide_forest" in (u.get("attributes") or ()),
            "strider": "strider" in (u.get("attributes") or ()),
            "ign_trees": "ignore_trees" in (u.get("attributes") or ()),
            "veg_m": (float(((self.sim.get("terrain") or {}).get("trees") or {}).get("ignore_s", {}).get(
                missile.get("projectile"), 0.0)) * float(missile.get("muzzle_velocity") or 0.0)) if missile else 0.0,
            # the penetrating projectile (missile.py penetration): its counter and size cap, a man's resistance and size
            "pen_max": float(missile.get("penetration_max") or 0) if missile else 0.0,
            "pen_cap": float(SIZE_RANK.get(missile.get("penetration_cap"), 99) if missile.get("penetration_cap")
                             else (99 if (missile.get("penetration_max") or 0) < 0 else 0)) if missile else 0.0,
            "pen_res": float(u.get("projectile_resistance") or 1),
            "size_k": float(SIZE_RANK.get(u.get("size", "small"), 1)),
            # the army destruction's strength (morale.army_collapse): the database's combat potential
            "cp_fixed": (u.get("combat_potential") or {}).get("melee", u.get("multiplayer_cost", 0)) + sum(
                sum((self.abilities.get(k) or {}).get("cp") or (0, 0)) for k in u.get("abilities") or ()),
            "cp_missile": (u.get("combat_potential") or {}).get("missile", 0),
            "ai": False,
            # skirmish mode (skirmish.py, sim.json skirmish): the start / stop gaps of a unit that can skirmish (the
            # database's can_skirmish, a missile unit); 0 when it cannot or the rule is off. `sk` (the toggle) is set
            # by scenario.build / abilities.set_rule
            **self.skirmish_gaps(key),
            "sk": False,
            # artillery (tools/nn/sim/artillery.py; config/nn/sim.json artillery): its guns, their crew, the trajectory,
            # the minimum range, the bonus against large and the projectile's blast - 0 / -1 for every other unit
            **self.artillery_static(key),
            **{f"ab{k}": i for k, i in enumerate(abilities.slots_of(self, key))},
            # its spells (the passport's `spells`, config/nn/abilities.json spells; magic.py)
            **{f"sp{k}": i for k, i in enumerate(magic.slots_of(self, key))},
        }


def artillery_static(units, sim, effects_index, key):
    """The STATIC artillery fields of a unit (state.py eng0 ... c_mom): its guns (engines.n, hp, crew_per_engine), its
    projectile's fixed elevation (trajectory 'fixed'), minimum range, bonus against large, collision radius and
    momentum, its blast (missile.blast) and the blast's contact effect (index in effects_index, -1 none). Every field 0
    (ex_fx -1) for a unit without guns or with sim.json artillery off: the rules then never act on it."""
    u = units[key]
    eng = u.get("engines") or {}
    m = u.get("missile") or {}
    out = {"eng0": 0.0, "eng_hp": 0.0, "crew_k": 0.0, "fix_el": 0.0, "min_range": 0.0, "m_bvl": 0.0, "ex_r": 0.0,
           "ex_d": 0.0, "ex_ap": 0.0, "ex_f": 0.0, "ex_v": 0.0, "ex_allies": False, "ex_magic": False, "ex_fx": -1,
           "c_rad": 0.0, "c_mom": 0.0, "p_magic": False, "central": False, "rl_add": 0.0}
    if not (sim.get("artillery") or {}).get("on") or not eng.get("n"):
        return out
    out.update(eng0=float(eng["n"]), eng_hp=float(eng["hp"]), crew_k=float(eng.get("crew_per_engine") or 1))
    if m:
        bl = m.get("blast") or {}
        out.update(fix_el=float(m.get("fixed_elevation") or 0.0) if m.get("trajectory") == "fixed" else 0.0,
                   min_range=float(m.get("min_range_m") or 0.0), m_bvl=float(m.get("bonus_v_large") or 0.0),
                   c_rad=float(m.get("collision_radius_m") or 0.0),
                   c_mom=float(m.get("mass") or 0.0) * float(m.get("muzzle_velocity") or 0.0),
                   p_magic=bool(m.get("magic")),
                   # (build/diag_arty/mechanic.md 1) the aim: the most central man (prefer_central_targets) or the nearest
                   central=bool(m.get("prefer_central")),
                   # (artillery.cycle_add_s, measured: the aim and fire animations a gun's cycle adds to its reload)
                   rl_add=float(((sim.get("artillery") or {}).get("cycle_add_s") or {}).get(m.get("projectile"), 0.0)))
        if bl and bl.get("type") == "shockwave":
            out.update(ex_r=float(bl["radius_m"]), ex_d=float(bl["damage"]), ex_ap=float(bl["ap_damage"]),
                       ex_f=float(bl["force"]), ex_v=float(bl["speed"]), ex_allies=bool(bl.get("allies")),
                       ex_magic=bool(bl.get("magic")),
                       ex_fx=effects_index.get(bl.get("contact_phase"), -1) if bl.get("contact_phase") else -1)
    return out


def pen_limit_of(units, sim):
    """The most men a projectile of the passports `units` can strike (tools/nn/sim/missile.py men_struck over every
    shooter and every target), at most sim.json missile.penetration.max_men: 1 when none penetrates."""
    limit = int(((sim.get("missile") or {}).get("penetration") or {}).get("max_men", 8))
    sizes = [SIZE_RANK.get(u.get("size", "small"), 1) for u in units.values()] or [1]
    resists = [float(u.get("projectile_resistance") or 1) for u in units.values()] or [1.0]
    small, res = min(sizes), min(resists)
    out = 1
    for u in units.values():
        m = u.get("missile") or {}
        mx = m.get("penetration_max") or 0
        cap = m.get("penetration_cap")
        cap_r = SIZE_RANK.get(cap, 99) if cap else (99 if mx < 0 else 0)
        if cap_r > small:
            out = max(out, limit if mx < 0 else min(limit, 1 + int(mx // max(res, 1e-3))))
    return out


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load(units=None, rules=None, sim=None, abilities=None, effects=None):
    """Params from the five files (defaults: config/nn/); the spells are abilities.json's own section. With the default
    units the casters the armies field (config/nn/pools.json "mages" with a share above 0; their passports in
    data/units/probe-units.json) join them; with the share 0 (the default) nothing joins (and the step skips magic)."""
    ab = _read(abilities or CONFIG / "abilities.json")
    extra = {}
    if units is None:
        from tools.nn.armies import pools
        extra = pools.mage_passports()
    return Params(units={**extra, **_read(units or CONFIG / "units.json")["units"]},
                  rules=_read(rules or CONFIG / "game_rules.json"),
                  sim=_read(sim or CONFIG / "sim.json"),
                  abilities=ab["abilities"],
                  effects=_read(effects or CONFIG / "effects.json"),
                  spells=ab.get("spells") or {})
