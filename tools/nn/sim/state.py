"""The simulator's state layout: the single source of truth (docs/en/training/simulator.md).

A batch of B battles, N unit slots each. A slot is one unit of either side; `side` [B, N] says
whose (1 or 2, 0 = an empty slot). Layout the simulator builds: side 1 in slots [0, H), side 2
in [H, 2H), H = N // 2; a side's lord (if any) is its first slot. Everything goes by `side`, so
other layouts work too; only own_first() assumes this one.

Three groups of per-unit tensors, all [B, N]:

* OBSERVED - the fields and names of the recorded `nn_sample` rows (tools/nn/gamedata.py,
  docs/en/training/README.md "What a recording holds"), so recorded battles and the simulator
  feed the network the same way. `target` is the recording's `t` (current target, a slot index,
  -1 none; as gamedata.Battle.target), `fat` the fatigue state as a number (FATIGUE_LEVELS)
  instead of its string; the battle time is `t` [B] (s), as gamedata.Battle.t.
* STATIC - the unit passport (config/nn/units.json) and the unit's place, fixed for the battle.
* INTERNAL - what the simulator keeps for itself (morale points, fatigue points, timers).

Names are importable without torch; the tensors need it.
"""
from dataclasses import dataclass, field

try:
    import torch
except ImportError:          # the layout (names) is importable without torch
    torch = None

# The database's entity sizes in order (battle_entities size, projectile_penetration_junctions entity_size_cap):
# a penetrating projectile passes a man only of a size below its cap (missile.py penetration).
SIZE_RANK = {"very_small": 0, "small": 1, "medium": 2, "large": 3, "very_large": 4}
# Fatigue states in order, as the game names them (`fat` in recordings; `fat` here is the index).
FATIGUE_LEVELS = ("threshold_fresh", "threshold_active", "threshold_winded", "threshold_tired",
                  "threshold_very_tired", "threshold_exhausted")
# MoraleState (`ms`) as recorded: 1 eager ... 5 wavering, 6 routing, 7 shattered.
MORALE_STATES = {1: "eager", 2: "confident", 3: "steady", 4: "shaken", 5: "wavering", 6: "routing",
                 7: "shattered"}

# name -> (dtype, meaning). dtype: "f" float32, "b" bool, "i" int64.
OBSERVED = {
    "x": ("f", "position east, m (map centre 0)"),
    "z": ("f", "position north, m"),
    "b": ("f", "bearing, degrees: facing (sin b, cos b) in (x, z); 90 = east"),
    "men": ("f", "men alive"),
    "hp": ("f", "share of health left, 0-1"),
    "mp": ("f", "MoralePercent: morale points / leadership; routs at 0; may be below 0 or above 1"),
    "ms": ("f", "MoraleState 1-7 (MORALE_STATES)"),
    "r": ("b", "routing (shattered units too)"),
    "s": ("b", "shattered"),
    "w": ("b", "wavering"),
    "m": ("b", "in melee"),
    "mv": ("b", "moving"),
    "f": ("b", "running"),
    "a": ("f", "projectiles left, whole unit"),
    "fire": ("b", "firing missiles"),
    "target": ("i", "current target: slot of the enemy fought or shot at, -1 none (recording's t)"),
    "fat": ("f", "fatigue state 0-5 (FATIGUE_LEVELS)"),
    "k": ("f", "kills: enemy men killed"),
    "ox": ("f", "the order's point x, m (own position when holding)"),
    "oz": ("f", "the order's point z, m"),
    "lf": ("b", "an enemy threatens the left flank"),
    "rf": ("b", "an enemy threatens the right flank"),
    "bf": ("b", "an enemy threatens the rear"),
    "vis": ("b", "visible to the other side"),
}

STATIC = {
    "side": ("i", "1 or 2; 0 = empty slot"),
    "lord": ("b", "the side's general"),
    "men0": ("f", "men at the start"),
    "hp_man": ("f", "health of a man"),
    "hp0": ("f", "health of the unit at the start"),
    "mass": ("f", "a man's mass"),
    "radius": ("f", "a man's radius, m"),
    "width": ("f", "frontage ordered, m (files = floor(width / sp_h)): the deployment's, then re-formed by a width "
                   "order (Orders.width; tools/nn/sim/reform.py changes it over the re-form's time)"),
    "wmin_k": ("f", "the narrowest formation's files as a share of the men (formation.reform: 0.04 infantry, 0.10 "
                    "cavalry, monsters, chariots by the database's class); 0: the width is fixed (artillery)"),
    "sp_h": ("f", "a man's place across the front, m (the formation template's close spacing, the database)"),
    "sp_v": ("f", "a man's place between ranks, m (the same)"),
    "body_ratio": ("f", "a man's body across over its radius (the database's battle_entities radii_ratio of an ellipse "
                        "body - a beast 0.4, a weapon team 0.6; 1 a circle, every man): melee.body_across"),
    "walk": ("f", "walk speed, m/s"),
    "run": ("f", "run speed, m/s"),
    "charge_speed": ("f", "charge speed, m/s"),
    "charge_dist": ("f", "charge distance, m: under an attack order the unit closes the last this many metres to its "
                         "target at its charge speed (the database's battle_entities: 30 infantry, 35 lords)"),
    "charge_pose": ("f", "charge pose distance, m: under an attack order the unit takes its charge pose this far from "
                         "its target (the database's battle_entities: 25 infantry, 30 lords; the charge's morale)"),
    "accel": ("f", "acceleration, m/s2"),
    "turn": ("f", "a man's turn rate, deg/s (the database's battle_entities turn_rate: 120 infantry and the General, "
                  "180 archers and the Warlord, 240 militia): a unit on the move turns at it (movement.py), a lone man "
                  "standing too"),
    "decel": ("f", "deceleration, m/s2"),
    "attack": ("f", "melee attack"),
    "defence": ("f", "melee defence"),
    "charge_bonus": ("f", "charge bonus"),
    "damage": ("f", "melee weapon damage (base)"),
    "ap_damage": ("f", "melee weapon armour-piercing damage"),
    "bonus_v_large": ("f", "melee bonus against large"),
    "bonus_v_inf": ("f", "melee bonus against infantry"),
    "interval": ("f", "time between a man's blows, s"),
    "splash": ("f", "targets per blow (hits several)"),
    "armour": ("f", "armour"),
    "shield": ("f", "shield: chance to block a projectile from the front, 0-1"),
    "leadership": ("f", "leadership"),
    "resist_missile": ("f", "missile resistance, 0-1"),
    "resist_physical": ("f", "physical resistance, 0-1"),
    "large": ("b", "size class above small"),
    "expendable": ("b", "attribute expendable: its rout does not scare others"),
    "encourages": ("b", "attribute encourages (lords)"),
    "reflect": ("b", "attribute charge_reflection: braced, it meets a frontal infantry charge as a charge"),
    "unbreakable": ("b", "attribute unbreakable: never loses leadership, never wavers or routs"),
    "fire_move": ("b", "attribute mounted_fire_move: shoots while moving (fire whilst moving)"),
    "fatigue_immune": ("b", "attribute fatigue_immune (Perfect Vigour): gains no fatigue"),
    "direct": ("b", "direct (flat, trajectory low) fire: holds fire where friends block the line (missile.py)"),
    "ammo0": ("f", "projectiles of the whole unit at the start"),
    "range": ("f", "missile range, m (0 = no missile)"),
    "reload": ("f", "reload, s (measured where known, config/nn/sim.json)"),
    "m_damage": ("f", "projectile damage (base)"),
    "m_ap": ("f", "projectile armour-piercing damage"),
    "hit_rate": ("f", "projectile hit rate at the edge of range (config/nn/sim.json; missile.accuracy units 'rates' and "
                      "a lone man in melee)"),
    "arc": ("f", "fire arc each side of the facing, degrees: a man fires only at a target centre within it seen from "
                 "where he stands (the database's battle_entities fire arc / 2: 30, militia 35)"),
    "acc": ("f", "accuracy, 0-100: land_units accuracy + the projectile's marksmanship (the spread, missile.py)"),
    "cal_area": ("f", "the projectile's calibration area, m (the spread at its calibration distance)"),
    "cal_dist": ("f", "the projectile's calibration distance, m"),
    "muzzle_v": ("f", "the projectile's muzzle velocity, m/s (the angle it comes down at)"),
    "height": ("f", "a man's height, m (the shadow a falling projectile hits)"),
    "small_arms": ("b", "small-arms projectile (arrow, musket, sling, javelin, axe): shields block it (missile.py)"),
    "aim_s": ("f", "first shot after halting, s (config/nn/sim.json)"),
    "broken_at": ("f", "the share of its target's starting men lost from which every shot waits aim_s after the "
                       "reload (missile.reaim_broken_cat_share by category: arrows and bolts 1/3; 0 never)"),
    "quiet_s": ("f", "seconds without a shot after a new order or target to a shooter already aiming, whatever its "
                     "reload (missile.order_quiet_cat_s by category: arcing arrows, bolts, slings 8.5; 0 none)"),
    "rt_s": ("f", "seconds a new order or a new target takes off the aim clock (missile.retarget_cat_s by category: "
                  "musket - handguns, pistols, throwing stars - 0; -1: the rule of all, aim_s on an order, retarget_s "
                  "on a target)"),
    "friendly_fire": ("f", "share of its hits aimed at a unit in melee that land on its own side (config/nn/sim.json)"),
    "morale_bonus": ("f", "morale points at the start beyond leadership (config/nn/sim.json)"),
    "rout_death_s": ("f", "a lord of this faction who routs counts as killed this long into his rout (the "
                          "faction's lords crumble: Vampire Counts, config/nn/sim.json morale.lord_fall); 0 never"),
    "cost": ("f", "multiplayer cost"),
    "cp_fixed": ("f", "combat potential that does not run out: the database's main_units melee_cp + its abilities' "
                      "additional melee and missile cp (the army-destruction strength, morale.army_collapse)"),
    "cp_missile": ("f", "combat potential of its missiles (main_units missile_cp): counts by the ammunition left"),
    "ai": ("b", "played by the game's AI: its lord fires his active abilities by the AI's rule (the network's "
                "side by order: tools/nn/sim/orders.py ability)"),
    "sk_m": ("f", "skirmish mode: the start gap, m - the nearest standing enemy's formation edge this close (the "
                  "simulator's rectangles, geometry.pairwise gap) starts the run away (skirmish.py; config/nn/sim.json "
                  "skirmish.start_gap_m by unit); 0: the unit cannot skirmish (the database's can_skirmish) or the rule is off"),
    "sk_stop_m": ("f", "skirmish mode: the stop gap, m - the run goes on until the nearest standing enemy is this far "
                       "(skirmish.stop_gap_m)"),
    "sk": ("b", "skirmish mode switched on (the game's toggle): the game's AI keeps it on for its shooters, the bridge "
                "switches it off for the network's (--skirmish off); scenario.build: on for a side the game's AI plays "
                "(`ai`) unless the side says \"skirmish\"; abilities.set_rule sets it with `ai` (the scripts' sides on, "
                "a network's off); skirmish.network in config/nn/sim.json gives it to every unit"),
    "ab0": ("i", "the unit's first ability slot: index in the ability passports (tools/nn/sim/abilities.py keys), "
                 "-1 none"),
    "ab1": ("i", "its second ability (-1 none)"),
    "ab2": ("i", "its third ability (-1 none)"),
    "spv": ("f", "shots a man fires in one volley (the database's projectiles shots_per_volley: 18 the Ratling Gun, 1 "
                 "the others)"),
    "bst_rate": ("f", "shots a second a man fires in a burst (a volley of spv > 1 shots fired over spv / bst_rate s: the "
                      "Ratling Gun 4, measured - config/nn/sim.json missile.burst_rate_s); 0: the volley at once"),
    "melee_fire": ("b", "shoots in melee too (config/nn/sim.json missile.fire_in_melee: the Ratling Gun, measured)"),
    "lone_hit": ("b", "its hits on a lone man (a lord) in melee land as out of melee (config/nn/sim.json "
                      "missile.single_in_melee_projectiles: the Ratling Gun, measured; others: single_entity_in_melee)"),
    "mfx": ("i", "the contact effect its blows lay on the enemy they hit: index in config/nn/effects.json order, -1 "
                 "none (the passport's melee.contact_phase; tools/nn/sim/effects.py lay)"),
    "sfx": ("i", "the contact effect its projectiles lay (missile.contact_phase), -1 none"),
    "hslow": ("b", "its projectiles' hits slow the small formations they hit (config/nn/sim.json missile.hit_slow by "
                   "category: musket - handguns, the Ratling Gun, pistols, throwing stars; missile.py hit_slow)"),
    "fx": ("i", "its innate effects: bitmask over config/nn/effects.json order (tools/nn/sim/effects.py)"),
    "fxt0": ("i", "its first timed effect (index in that order, -1 none)"),
    "fxt1": ("i", "its second timed effect (-1 none)"),
    "sp0": ("i", "the unit's first spell: index in the spell passports (config/nn/abilities.json spells; "
                 "tools/nn/sim/magic.py keys), -1 none"),
    "sp1": ("i", "its second spell (-1 none)"),
    "resist_magic": ("f", "spell resistance, 0-1: the database's damage_mod_magic + damage_mod_all (a ward), what cuts a "
                          "magical projectile (magic.py; docs/en/game/mechanics/magic.md)"),
    # (25.10.2026, line g43: stealth, the penetrating bullet, the reload skill, the delayed passive)
    "stalk": ("b", "attribute stalk: any ground hides it (tools/nn/sim/stealth.py)"),
    "spot_m": ("f", "how far it sees a hidden enemy, m (the database's land_units spot_dist_scrub: 80, the Jezzails 160; "
                    "stealth.py)"),
    "hide_k": ("f", "its hiding_scalar: an enemy sees it hidden within its spotting distance x this (0.8 the Gutter "
                    "Runners)"),
    # (28.10.2026: the map's ground and trees, tools/nn/sim/ground.py)
    "spot_tree": ("f", "how far it sees a hidden enemy standing mostly on forest ground, m (the database's land_units "
                       "spot_dist_tree: 60; stealth.py)"),
    "hidef": ("b", "attribute hide_forest: it hides in woods (stealth.py: its men on forest ground)"),
    "strider": ("b", "attribute strider: no ground penalty, passes through the trees (ground.py)"),
    "ign_trees": ("b", "attribute ignore_trees: passes through the trees, no forest penalty (ground.py)"),
    "veg_m": ("f", "its projectile's first metres of flight no tree stops (the database's vegetation_ignore_time x "
                   "muzzle_velocity, ~42 m; ground.trunk_keep); 0 no missile"),
    "reload_db": ("f", "the database's reload_s of its projectile, s (the reload skill's base: max(4 s, base x (100 - "
                       "skill) %); `reload` is the measured cycle, scaled by it; 0 no missile)"),
    "pen_max": ("f", "its projectile's penetration counter (the database's projectile_penetration_junctions "
                     "max_penetration: each man struck adds his pen_res, the projectile goes on while it is at most this; "
                     "-1 no limit, 0 stops in the first man; missile.py penetration)"),
    "pen_cap": ("f", "its projectile passes only a man of a smaller size than this (entity_size_cap as SIZE_RANK: "
                     "very_small 0 ... very_large 4; 99 none)"),
    "pen_res": ("f", "a man's projectile_penetration_resistance (1 infantry, 2 the Giant Rats)"),
    "size_k": ("f", "a man's size as SIZE_RANK (battle_entities size: small 1)"),
    "fxd": ("i", "its delayed innate effect (the catalogue's delay_s: Blackpowder Discipline), index in the order, -1 "
                 "none (effects.py)"),
    "tag": ("i", "a drill's mark of the unit (tools/nn/train/drills: 1 our unit of the drill's situation, 2 the "
                 "enemy's, inserted into an otherwise normal battle; 0 every other unit); set by scenario.build from "
                 "the description's \"tag\"; the simulator and the observation never read it (only the drill's scripts)"),
}

INTERNAL = {
    "vx": ("f", "velocity x, m/s"),
    "vz": ("f", "velocity z, m/s"),
    "hp_abs": ("f", "health left, HP"),
    "morale": ("f", "morale points"),
    "fatigue": ("f", "fatigue points"),
    "recent": ("f", "HP lost recently (the last morale.casualties_s seconds: recent casualties)"),
    "extended": ("f", "HP lost over the last morale.extended_s seconds (extended casualties)"),
    "contact_s": ("f", "seconds in melee since the contact began (it goes on through gaps shorter than "
                       "contact.reset_s out of contact)"),
    "out_s": ("f", "seconds out of contact since the last contact (contact.reset_s)"),
    "melee_s": ("f", "seconds in melee over the whole battle, never reset: the opening wave's clock and 'it has fought "
                     "already' (contact.opening_once; battle.py, melee.py)"),
    "charge": ("f", "the charge left, 0-1: 1 at the charge's first blow, then down by its own clock over "
                    "charge_decay_duration (13 s), in contact or not"),
    "runup": ("f", "run-up, m: run at charge.min_speed_share of the run speed or faster since the last stop, "
                   "walk or contact (charge.min_runup_m)"),
    "ran_in": ("b", "it came into its present fight moving (or charging), not standing: set when its melee clock "
                    "starts, cleared when the clock resets (battle.py: the first strike; melee.py: men gather round "
                    "a lord only when he ran in)"),
    "aim": ("f", "seconds standing still able to shoot"),
    "quiet": ("f", "seconds left of the quiet window after a new order or target (quiet_s; no shot while > 0)"),
    "aim_tgt": ("i", "the enemy it aimed at last step (-1 none): a change makes it wait missile.retarget_s"),
    "turn_on": ("b", "a standing shooter turning to its target (missile.stand_fire_arc_deg starts it, "
                     "missile.turn_done_deg ends it)"),
    "leave_s": ("f", "seconds leaving melee while still touching an enemy (contact.pin_s)"),
    "order_s": ("f", "seconds since the order in force was given while the unit was in melee (another kind, target or "
                     "point given in melee restarts it; given out of melee it starts at the window, already over): "
                     "melee.py contact.fresh_incidental within the database's melee_breakoff_secs"),
    "exit_s": ("f", "seconds since it began to leave melee (a withdraw or a far move while in contact), in contact or "
                    "not, while it keeps leaving: at the database's melee_breakoff_secs still in contact it drops the "
                    "order (battle.py, contact.breakoff)"),
    "brk_s": ("f", "seconds left of the melee exit's window (melee_breakoff_secs from a leave or an attack on an enemy "
                   "it does not touch, given in melee) that a hold given in melee keeps: no strikes while > 0 under "
                   "HOLD (battle.py, contact.hold_breakoff); 0: none"),
    "low_s": ("f", "seconds since its health first fell below a quarter (0 before; never back: the Wounds of a single "
                   "entity come on its initial cooldown after that and stay to the end, effects.py)"),
    "shots": ("f", "fractional shots carried to the next step"),
    "bst": ("f", "share of its men firing their burst now (a volley of spv shots over spv / bst_rate s; missile.py)"),
    "bst_s": ("f", "seconds left of its burst (0: none)"),
    "bst_p": ("f", "share of its bursting men aiming again since their target man died (missile.burst_share: they lose "
                   "those shots of their burst; back at aim_s)"),
    "cfx0_s": ("f", "seconds left of the first contact effect of config/nn/effects.json on it (tools/nn/sim/effects.py "
                    "contact_keys, lay: duration_s from each hit of a weapon that lays it)"),
    "cfx1_s": ("f", "the second contact effect's (cfx0_s)"),
    "cfx2_s": ("f", "the third contact effect's"),
    "cfx3_s": ("f", "the fourth contact effect's"),
    "unready": ("f", "share of the men still reloading, 0-1 (0: all loaded, the start; missile.py volley)"),
    "late": ("f", "share of a shooter's men whose target man died since their last shot (missile.reaim: they aim "
                  "again before the next shot): 1 - prod(1 - the target's men lost a step / its men), 0 at a volley "
                  "and at a new target"),
    "fw": ("f", "HP its front rank lacks (kills.missile_front: projectile hits on the front rank first; melee.front_kills: "
                "the rest's wounds are the unit's less these)"),
    "loaded_s": ("f", "seconds a shooter has been loaded for its next volley, waiting for its late men to aim "
                      "(missile.reaim; 0 while reloading)"),
    "fc_cap": ("f", "this step's fire it had (bookkeeping, missile.friend_los): men x dt / reload of a shooter that could "
                    "shoot at a target, before the friends' check (tools/nn/train/behaviour.py: the friends' shares)"),
    "fc_skip": ("f", "... of it the friends' men took away: all of it while it holds fire for them, else the share of "
                     "its men they block"),
    "fc_lob": ("f", "... of it shot at 45 deg because its line was closed (arcing fire)"),
    "skr": ("b", "in a skirmish run now (skirmish.py): started at sk_m, kept until sk_stop_m, in melee or routing ended"),
    "sk_in": ("b", "an enemy coming that attacks another unit was within sk_m last step (skirmish.py: such an enemy starts "
                   "a run only as it comes within the gap)"),
    "sk_shown": ("b", "ox, oz show the skirmish run's far point (skirmish.show_far_point, battle.py): the order's own point "
                      "is in sk_ox, sk_oz and comes back at the next step's start"),
    "sk_ox": ("f", "the order's own point x under a shown skirmish far point (sk_shown)"),
    "sk_oz": ("f", "the order's own point z (sk_shown)"),
    "skt": ("b", "a skirmish run ended out of melee: the unit turns back to its target once it stands (battle.py, the "
                 "standing turn of a shooter; as the game's shooters faced back and shot 2.5-7 s after the run)"),
    "rout_count": ("f", "times routed"),
    "rout_s": ("f", "seconds since the rout began"),
    "rally_s": ("f", "seconds since the last rally"),
    "waver_s": ("f", "seconds of the current unbroken wavering (0 on its first step, and when not wavering): the waver "
                     "timer before a rout (morale.waver_timer)"),
    "rpin_s": ("f", "seconds into the rout's exit of a formation that routed from melee (contact.rout_pin_s: its men "
                    "stay in contact that long; 0: not in one)"),
    "rpin_d": ("f", "metres run since its rout's exit began (left out of its contact test while the exit lasts)"),
    "stk_n": ("f", "a router's men stuck among the men of a standing enemy formation it ran into (contact.rout_stuck): "
                   "they stay in melee with it while the rest flee; 0: none"),
    "stk_k": ("i", "the slot of the formation its stuck men are among (stk_n > 0)"),
    "rally_ok_s": ("f", "seconds a routing unit has met the rally's conditions without a break (morale.rally_wait_s)"),
    "flank_hit": ("f", "worst direction attacked from now: 0 front, 1 flank, 2 rear"),
    "under_fire_s": ("f", "seconds since last hit by a projectile"),
    "hs_s": ("f", "seconds left of the slowing from hits of projectiles that slow (STATIC hslow; missile.py hit_slow: "
                  "missile.hit_slow.duration_s from each step such hits land in), 0: none"),
    "lost_worst": ("f", "the worst share of the unit lost so far (tools/nn/train/reward.py track: health, a rout's "
                        "share, out whole), kept by the reward's caller, not the simulator: a rally gives nothing back"),
    "dealt": ("f", "HP dealt in melee recently (decaying)"),
    "taken": ("f", "HP taken in melee recently (decaying)"),
    "shot_dealt": ("f", "HP dealt by its missiles to enemies recently (decaying like dealt: the morale's fight balance)"),
    "shot_taken": ("f", "HP taken from enemy missiles recently (decaying like taken: the morale's fight balance)"),
    "gold_out": ("f", "the gold of enemy health it has destroyed so far (HP dealt to an enemy / its starting HP x its "
                      "cost, summed; bookkeeping only: the network's 'eyes' learn from it, tools/nn/model/eyes.py)"),
    "chm_s": ("f", "seconds left of the charge's morale block (+15; morale.charge, battle.py)"),
    "chm_n": ("f", "the charge's morale blocks so far in this charge (0 when not charging)"),
    "gone": ("b", "left the map (routed off it)"),
    "dead_s": ("f", "seconds since the unit was killed (health 0; a lord whose faction's routed lord crumbles: "
                    "since rout_death_s into his rout); 0 while not (battle.py, the lord-fall morale)"),
    "gone_s": ("f", "seconds since the unit left the map alive (routed off it); 0 while not (battle.py, the "
                    "lord-fall morale)"),
    "order_kind": ("i", "the order in force (tools/nn/sim/orders.py)"),
    "order_target": ("i", "its target slot"),
    "order_run": ("b", "its run flag"),
    "oface": ("f", "its facing to stand at (Orders.face snapped to the game's 64 sectors), degrees; -1 none"),
    "oface_s": ("f", "seconds the unit has stood ready to take that facing (the re-form's clock: it turns from "
                     "turn.face_delay_s on; battle.py)"),
    "fim_s": ("f", "seconds left of the 10 s after a facing order given in melee in which it strikes nobody but its "
                   "order's target (melee_breakoff_total_immunity_secs; battle.py contact.face_breakoff)"),
    "ow": ("f", "the width order in force (Orders.width as given, m; -1 none): the same number re-given is no new order"),
    "wnew": ("f", "the width the unit re-forms to (Orders.width snapped to whole files: tools/nn/sim/reform.py), m; "
                  "-1 none"),
    "w0": ("f", "the width the re-form started from, m (the width goes from it to wnew evenly over wT)"),
    "wT": ("f", "the re-form's time, s (reform.ready_s)"),
    "ws": ("f", "seconds into the re-form"),
    "ab0_on": ("f", "seconds ability 0 stays active"),
    "ab0_cd": ("f", "seconds until ability 0 is ready"),
    "ab1_on": ("f", "seconds ability 1 stays active"),
    "ab1_cd": ("f", "seconds until ability 1 is ready"),
    "ab2_on": ("f", "seconds ability 2 stays active"),
    "ab2_cd": ("f", "seconds until ability 2 is ready"),
    "fxt0_on": ("f", "seconds timed effect 0 stays on"),
    "fxt0_cd": ("f", "seconds until timed effect 0 is ready"),
    "fxt1_on": ("f", "seconds timed effect 1 stays on"),
    "fxt1_cd": ("f", "seconds until timed effect 1 is ready"),
    "fxt0_used": ("b", "timed effect 0 has fired in this battle: its recharge runs only while its recharge context "
                       "holds (the initial one, before the first fire, always; effects.py)"),
    "fxt1_used": ("b", "timed effect 1 has fired in this battle"),
    "wm_cur": ("f", "its side's current winds of magic (the pool the casts pay from; the same on every unit of the side; "
                    "magic.py)"),
    "wm_res": ("f", "its side's winds reserve (it flows into the current pool, magic.winds.flow_per_s)"),
    "sp0_cd": ("f", "seconds until spell 0 is ready (magic.py: from the wind-up's end, active_s + recharge_s)"),
    "sp1_cd": ("f", "seconds until spell 1 is ready"),
    "cq_k": ("i", "a cast asked of it for the next step (magic.request): the spell slot, -1 none"),
    "cq_t": ("i", "the asked cast's target slot"),
    "cast_k": ("i", "the spell slot it is casting (pending out of range or winding up), -1 none"),
    "cast_t": ("i", "the cast's target slot"),
    "cast_s": ("f", "seconds left of the cast's wind-up; < 0: pending, walking into range"),
    "bm0_s": ("f", "seconds since the wind-up's end of spell 0's bombardment, -1 none (magic.bombard)"),
    "bm0_t": ("i", "its target slot (its centre fixes the strike point start_s after the wind-up)"),
    "bm0_x": ("f", "its strike point x, m (fixed: bm0_fix)"),
    "bm0_z": ("f", "its strike point z, m"),
    "bm0_fix": ("b", "its strike point is fixed"),
    "bm1_s": ("f", "the same of spell 1's bombardment"),
    "bm1_t": ("i", "its target slot"),
    "bm1_x": ("f", "its strike point x, m"),
    "bm1_z": ("f", "its strike point z, m"),
    "bm1_fix": ("b", "its strike point is fixed"),
    "sp0_on": ("f", "seconds spell 0's effect stays active (active_s from the wind-up's end: the ability bar's timer; "
                    "magic.py)"),
    "sp1_on": ("f", "seconds spell 1's effect stays active"),
    "bm_hit": ("b", "spell projectiles struck it in the last step (a bombardment falling on it: the game shows it; the "
                    "network's spell_strike)"),
    "bm_gold_e": ("f", "bookkeeping (no effect): the gold of enemy health its bombardments have taken so far "
                       "(HP / starting HP x cost; the spell price's bricks, tools/nn/abilities_price)"),
    "bm_gold_f": ("f", "bookkeeping: the same of its own side's health (friendly fire)"),
    "bm_knock": ("f", "bookkeeping: enemy men its bombardments have knocked down so far (magic.knock: knock "
                      "events, each down for the getting-up delay)"),
    "bm_w": ("f", "HP of its wounds the bombardments left (their HP less the health of the men they killed, at most "
                  "its wounds): the next strike finds them where its dose falls (magic.bombard)"),
    "cmb": ("f", "the morale's fight balance points of the last step (morale.combat_points: < 0 losing, > 0 winning; 0 "
                 "out of a fight or routing): the 'losing its melee' of the effects (effects.py losing_melee)"),
    "snap_s": ("f", "seconds left of an ability laid on it at a friend's cast (passport update_targets false: Stand "
                    "Your Ground keeps on the units in range at the cast wherever they go; abilities.py)"),
    "snap_ab": ("i", "that ability (index in the passports, -1 none)"),
    "fx_on": ("i", "the innate effects on in the last step: bitmask over config/nn/effects.json order"),
    "ab_rx": ("i", "the abilities of other units on it in the last step (a friend's aura or cast, an enemy's): bitmask "
                   "over the passports' order (abilities.keys); the network's effects from others"),
    "fx_rx": ("i", "the innate auras of other units on it in the last step (Hold the Line): the same bitmask "
                   "(effects.py); State.observation() gives ab_rx | fx_rx"),
    "fxd_s": ("f", "seconds its delayed effect's conditions have held without a break (effects.py delayed: on at delay_s)"),
    "hid": ("b", "hidden from the enemy now (stealth.py; vis = present and not hid)"),
    "hid_s": ("f", "the hide clock: seconds it has met the hiding conditions unseen (it hides at stealth.hide_s)"),
    "hid_t": ("f", "seconds since it hid (the enemies' spotting + stealth.fresh_bonus_m for stealth.fresh_s)"),
    "hid_g": ("f", "seconds of the grace used out of cover while hidden (stealth.grace_s)"),
    "hid_r": ("f", "seconds left of the window after it was revealed in which it hides again at once (stealth.quick_s)"),
    "hid_d": ("b", "deployed, not stepped yet: a hide_forest unit in the woods at the first step starts hidden "
                   "(stealth.start_hidden)"),
}

GROUPS = {"observed": OBSERVED, "static": STATIC, "internal": INTERNAL}

# Per-unit histories, [B, K * N] (K steps of N slots, newest step first): kept 2-D so that every [B, ...]
# copy of a State (narrowing, putting back, restarting a battle from the bank) carries them unchanged.
HISTORY = {
    "lost_hist": "HP lost each step over the last K steps (morale.casualties_window \"sliding\": the recent and "
                 "extended casualty windows; K = tools/nn/sim/morale.history_steps, 0 when the windows decay)",
}
# Per pair of units, [B, N * N] (row: the attacker j, column: the unit i it strikes; 2-D for the same reason):
PAIRS = {
    "fbk_s": "seconds left of the melee exit's window a facing order given in melee opened on enemy j for unit i "
             "(melee_breakoff_secs from the order; no blow of i starts on j while > 0; battle.py contact.face_breakoff)",
    "hit_sec": "the side of i that j's contact began on, + 1 (1 front, 2 flank, 3 rear; 0: no contact): held while the "
               "pair stays in contact (morale.attacked_first: the database's 'attacked in the flank / rear' is the "
               "first contact's side)",
}

# The battle's id [B, N] (int64, the same on every slot; empty() sets it, nothing per unit fills it): its row in the
# batch it was built in. The hash draws (morale's rally, the rout's dodge side, the scripts' picks) key on it, so a
# battle draws the same whatever batch it later runs in - narrowed, compacted or restarted from a bank.
BATTLE_ID = "row"
# Per-battle values of the simulator's uncertain numbers (tools/nn/sim/knobs.py): u["k_<name>"] [B, N] (the same on every
# slot), present only in training with domain randomisation on (tools/nn/train/dr.py); empty() makes none.


def _dtype(code):
    return {"f": torch.float32, "b": torch.bool, "i": torch.int64}[code]


@dataclass
class State:
    """A batch of battles. u: every per-unit tensor [B, N] by name (OBSERVED, STATIC, INTERNAL) and the
    per-unit histories [B, K * N] (HISTORY), the per-pair fields [B, N * N] (PAIRS);
    t: battle time [B], s; attacker [B] (1 or 2); done [B]; winner [B] (0 none yet, 1 or 2);
    lord_dead_s [B, 2]: seconds since side 1's / side 2's lord died or shattered (-1: alive or none);
    bounds: the map's half-size, m (square, centre 0); keys [B][N] unit keys ("" empty slot)."""
    u: dict
    t: "torch.Tensor"
    attacker: "torch.Tensor"
    done: "torch.Tensor"
    winner: "torch.Tensor"
    lord_dead_s: "torch.Tensor"
    bounds: float = 1020.0
    keys: list = field(default_factory=list)

    @property
    def B(self):
        return self.t.shape[0]

    @property
    def N(self):
        return self.u["side"].shape[1]

    @property
    def device(self):
        return self.t.device

    def __getitem__(self, name):
        return self.u[name]

    def observation(self):
        """The contract with the network (tools/nn/model/observation.py): the OBSERVED tensors
        [B, N], `side` [B, N], `t` [B] (s), the abilities' timers ab{k}_on / ab{k}_cd [B, N] (s:
        active left, until ready; the observation shows them for own units, `on` for seen enemies),
        `fx_on` [B, N]: the innate effects on now (bitmask, tools/nn/sim/effects.py), `ab_rx` [B, N]: the abilities of
        other units on it (bitmask over the passports' order: abilities.py and effects.py auras) and `gone` [B, N]
        (left the map: its gold is lost whole, the attacker's progress in the observation), the order in force
        `order_kind`, `order_target` [B, N] (the observation's ORDER)."""
        from tools.nn.sim.abilities import SLOTS
        out = {k: self.u[k] for k in OBSERVED}
        for k in range(SLOTS):
            for t in ("on", "cd"):
                out[f"ab{k}_{t}"] = self.u[f"ab{k}_{t}"]
        out["fx_on"] = self.u["fx_on"]
        if "ab_rx" in self.u:
            out["ab_rx"] = self.u["ab_rx"] | self.u.get("fx_rx", torch.zeros_like(self.u["ab_rx"]))
        out["gone"] = self.u["gone"]
        out["side"] = self.u["side"]
        out["order_kind"] = self.u["order_kind"]      # the order in force (observation ORDER)
        out["order_target"] = self.u["order_target"]
        if "wm_cur" in self.u:
            # spells (observation.py, magic): the side's winds, each spell slot's timers, who is casting now (winding up:
            # the staff glows) and who spell projectiles struck in the last step
            from tools.nn.sim.magic import SLOTS as SPELLS
            out["wm_cur"], out["wm_res"] = self.u["wm_cur"], self.u["wm_res"]
            for k in range(SPELLS):
                out[f"sp{k}_cd"] = self.u[f"sp{k}_cd"]
                out[f"sp{k}_on"] = self.u[f"sp{k}_on"]
            out["cast_k"] = self.u["cast_k"]
            out["cast_now"] = (self.u["cast_k"] >= 0) & (self.u["cast_s"] >= 0)
            out["bm_hit"] = self.u["bm_hit"]
        if "shape" in self.u:
            # the units' shape from their men (the fighters lab, tools/nn/fighters/hook.py shape=True: observation SHAPE)
            out["shape"], out["shape_ok"] = self.u["shape"], self.u["shape_ok"]
        out["t"] = self.t
        return out

    def clone(self):
        return State({k: v.clone() for k, v in self.u.items()}, self.t.clone(), self.attacker.clone(),
                     self.done.clone(), self.winner.clone(), self.lord_dead_s.clone(), self.bounds,
                     [list(r) for r in self.keys])


def empty(B, N, device="cpu", history=0):
    """B battles with N empty slots (side 0); tools/nn/sim/scenario.py fills them. history: steps K of the
    per-unit histories (HISTORY: [B, K * N])."""
    u = {}
    for group in GROUPS.values():
        for name, (code, _) in group.items():
            u[name] = torch.zeros((B, N), dtype=_dtype(code), device=device)
    for name in HISTORY:
        u[name] = torch.zeros((B, history * N), device=device)
    for name in PAIRS:
        u[name] = torch.zeros((B, N * N), device=device)
    u["target"].fill_(-1)
    u["order_target"].fill_(-1)
    u["oface"].fill_(-1)
    u["wnew"].fill_(-1)
    u["ow"].fill_(-1)
    u["aim_tgt"].fill_(-1)
    u["fxt0"].fill_(-1)
    u["fxt1"].fill_(-1)
    u["mfx"].fill_(-1)
    u["sfx"].fill_(-1)
    u["fxd"].fill_(-1)
    u["snap_ab"].fill_(-1)
    for name in ("sp0", "sp1", "cq_k", "cq_t", "cast_k", "cast_t", "bm0_s", "bm0_t", "bm1_s", "bm1_t"):
        u[name].fill_(-1)
    u["vis"].fill_(True)
    u[BATTLE_ID] = torch.arange(B, device=device)[:, None].expand(B, N).clone()
    zeros = torch.zeros(B, device=device)
    return State(u=u, t=zeros.clone(), attacker=torch.ones(B, dtype=torch.int64, device=device),
                 done=torch.zeros(B, dtype=torch.bool, device=device),
                 winner=torch.zeros(B, dtype=torch.int64, device=device),
                 lord_dead_s=torch.full((B, 2), -1.0, device=device), keys=[[""] * N for _ in range(B)])


def own_first(x, side):
    """x [B, N, ...] in the simulator's layout (side 1's half, then side 2's) seen from `side`:
    for side 2 the halves swap, so the side's own units come first."""
    if side == 1:
        return x
    H = x.shape[1] // 2
    return torch.cat([x[:, H:], x[:, :H]], dim=1)


def slot_from_own_first(index, side, N):
    """A slot index in own_first order -> the simulator's slot (-1 stays -1)."""
    if side == 1:
        return index
    H = N // 2
    return torch.where(index < 0, index, (index + H) % N)
