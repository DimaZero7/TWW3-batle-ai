"""Fatigue (docs/en/training/simulator.md, docs/en/game/mechanics/fatigue.md).

Points per activity from the game's database (_kv_fatigue_tables): charging +34, melee +19 (a single entity: the
database's +19, measured 186-200/s in the probes; a formation calibration.multi_combat),
shooting +18, running +4 (the cavalry +6: running_cavalry, u fat_run), walking -1, standing ready -7, idle -18; thresholds fresh 0, active 2800,
winded 6600, tired 12600, very tired 18000, exhausted 27000 (max 30000). The calibration
(config/nn/sim.json fatigue.calibration, ON) runs 10 ticks/s, freezes dead and departed slots and
charges melee only to a unit with an attack order: the database's +19 a tick for a single entity, 13.7 for a
formation (measured; a formation's men are not tired one by one - build/movelords); charging (+34) while the unit
sprints at its target under the attack order, up to the contact (calibration.charging "sprint", battle.py); a
formation in melee without the order stands ready (-7), under a move / withdraw order -2.5, a missile formation in
melee -1.2 with the attack order / -4.9 without (measured, build/fatigue2); a single entity without the order pays
combat while he strikes and rests as a formation while he does not (calibration.single_melee; missing: he moves or
rests at idle). A move costs by its order's run flag (run +4, walk -1), a routing unit +4; shooting 11.4 a tick on the seconds
it shoots (measured, build/fatigue2; aiming stands ready); a standing unit
rests at idle with no standing enemy within ready_enemy_m, else stands ready (ready_enemy_m 0: no such rule - a unit
under an attack order or aiming stands ready, any other standing unit rests at idle). A ranked unit (u["rank"], absent
in the simulator for now) adds calibration.rank's points a tick to every activity (the database). OFF is the legacy
model: 5 ticks/s, ready recovery, every melee +19. Tired states cost morale (morale.py) and scale
speed, melee attack and defence, armour, charge bonus, AP damage and reload by the database's
unit_fatigue_effects_tables (effects(); config/nn/sim.json fatigue.effects).
"""
import torch

from tools.nn.sim import knobs

LEVELS = ("threshold_fresh", "threshold_active", "threshold_winded", "threshold_tired", "threshold_very_tired",
          "threshold_exhausted")


def step(u, activity, params, dt):
    """Activity bools [B, N], highest priority charging > melee > shooting > run > walk > idle;
    the rest stands ready. The calibration also reads idle, active (alive and on the field),
    attack (an attack order), single (a single entity), run_order (the order's run flag),
    enemy_near (a standing enemy within calibration.ready_enemy_m), routing, move_order (a move or withdraw order)
    and shooter (a missile unit); a missing one means: not idle, active, attacking, a formation, running by speed,
    no enemy near, not routing, no move order, not a shooter. `charging` is the caller's: with calibration.charging
    "sprint" the charge sprint up to the contact (battle.py). OFF keeps the legacy clock and
    rates, inactive slots included.
    """
    F = params.fatigue
    cal = params.sim["fatigue"]
    trial = cal.get("calibration", {})
    enabled = trial.get("on", False)
    scale = (trial["per_second"] if enabled else cal["per_second"]) * dt
    rated = trial if enabled else {}
    walking = float(rated.get("walking", F["walking"]))
    moving, running = activity["walking"], activity["running"]
    if enabled and trial.get("move") == "order" and "run_order" in activity:
        # A move costs by its order, whatever the speed: run +4, walk recovers (database -1).
        running = moving & activity["run_order"]
    # the run's points a tick: the unit's own (u fat_run: _kv_fatigue running_cavalry 6 for the cavalry, running 4 else)
    run_rate = u["fat_run"] if "fat_run" in u else F["running"]
    move_rate = torch.where(running, run_rate, torch.full_like(u["fatigue"], walking))
    rate = torch.full_like(u["fatigue"], F["ready"])
    if enabled and "idle" in activity:
        # Idle rest needs no standing enemy near; near one the unit stands ready.
        idle = activity["idle"] & ~activity.get("enemy_near", torch.zeros_like(activity["idle"]))
        rate = torch.where(idle, F["idle"], rate)
    rate = torch.where(moving, move_rate, rate)
    rate = torch.where(running, run_rate, rate)
    if enabled and trial.get("routing") == "running" and "routing" in activity:
        # A routing unit flees at a run: it keeps tiring at the running rate, never rests.
        rate = torch.where(activity["routing"], run_rate, rate)
    rate = torch.where(activity["shooting"], float(rated.get("shooting", F["shooting"])), rate)
    melee, charging = activity["melee"], activity["charging"]
    if enabled and trial.get("melee") == "attack_order":
        # Only a unit told to attack pays for melee; a formation pays less than a single entity
        # (its men are not all fighting). Without the order it walks or rests while engaged.
        attack = activity.get("attack", torch.ones_like(melee))
        single = activity.get("single", torch.zeros_like(melee))
        # A single entity: single_combat a tick (missing: the database's melee +19); config/nn/sim.json
        # fatigue.lord_why. (The charge's cost lasts calibration.charge_s: the caller's `charging`.)
        combat = torch.where(single, torch.full_like(rate, float(trial.get("single_combat", F["combat"]))),
                             knobs.full_like(rate, knobs.get(u, "multi_combat", float(trial["multi_combat"]))))
        rest = torch.where(moving, move_rate, torch.full_like(rate, F["idle"]))
        form = ~single
        # A lone man (a lord) in melee without the attack order pays by what he does (calibration.single_melee
        # "strikes"; fatigue.calibration.single_melee_why): striking the enemies he touches - the database's combat, as
        # under the order; not striking (a hold's exit window, a leave) - as a formation, ready / melee_move. Missing:
        # he walks or rests at idle -18 while he fights.
        lone = single & (trial.get("single_melee") == "strikes") & ("strikes" in activity)
        rests_like_form = form | lone
        if "melee_hold" in trial:
            # A formation in melee without the attack order (89 % of it under hold, the engine giving it no target)
            # stands ready: the database's ready -7 (calibration.melee_hold "ready"; fatigue.calibration.why).
            hold = trial["melee_hold"]
            hold = F["ready"] if hold == "ready" else float(hold)
            rest = torch.where(rests_like_form, torch.full_like(rate, hold), rest)
        if "melee_move" in trial and "move_order" in activity:
            # ... under a move or withdraw order: the measured -2.5 a tick (calibration.melee_move)
            rest = torch.where(rests_like_form & activity["move_order"], torch.full_like(rate, float(trial["melee_move"])),
                               rest)
        if "strikes" in activity:
            rest = torch.where(lone & activity["strikes"], combat, rest)
        melee_rate = torch.where(attack, combat, rest)
        if "shooter_melee" in trial and "shooter" in activity:
            # A missile formation in melee: measured, with the attack order / with another (calibration.shooter_melee)
            with_attack, other = (float(v) for v in trial["shooter_melee"])
            melee_rate = torch.where(form & activity["shooter"],
                                     torch.where(attack, torch.full_like(rate, with_attack), torch.full_like(rate, other)),
                                     melee_rate)
        rate = torch.where(melee, melee_rate, rate)
        charging = charging & attack
    else:
        rate = torch.where(melee, F["combat"], rate)
    rate = torch.where(charging, F["charging"], rate)
    if enabled and trial.get("rank") and "rank" in u:
        # A ranked unit: the database's unit_stats_land_experience_bonuses gives each rank 0-9 points a tick on every
        # activity's rate (calibration.rank: 0 / -1 / -2 / -3 / -4 from ranks 0 / 3 / 5 / 7 / 9; rank 9 never tires
        # running and rests at -22). The fatrun probe: rank 5 first active 170.5 s into a run (rank 0 78 / 83.5 s), rank 9
        # fresh all 180 s (fatigue.calibration.rank_why). The simulator's units carry no rank yet (u["rank"] absent).
        table = torch.tensor([float(v) for v in trial["rank"]], dtype=rate.dtype, device=rate.device)
        rate = rate + table[u["rank"].long().clamp(0, len(table) - 1)]
    # Perfect Vigour (attribute fatigue_immune, an innate effect): never tires
    if "fatigue_immune" in u:
        rate = torch.where(u["fatigue_immune"], torch.clamp(rate, max=0.0), rate)
    if enabled and "active" in activity:
        rate = torch.where(activity["active"], rate, torch.zeros_like(rate))
    # an ability's vigour (Foe-Seeker: the database's fatigue_change_ratio -0.01): that share of the maximum a second
    # (abilities.vigour), on top of the activity's rate
    change = rate * scale
    if "vigour" in activity:
        change = change + activity["vigour"] * F["threshold_max"] * dt
    u["fatigue"] = (u["fatigue"] + change).clamp(0, F["threshold_max"])
    th = torch.tensor([F[k] for k in LEVELS], dtype=u["fatigue"].dtype, device=u["fatigue"].device)
    u["fat"] = (torch.bucketize(u["fatigue"], th, right=True) - 1).clamp(min=0).float()


# The stats unit_fatigue_effects_tables scales, and the unit field each one scales.
EFFECT_FIELDS = {"scalar_speed": ("walk", "run", "charge_speed"), "stat_melee_attack": ("attack",),
                 "stat_melee_defence": ("defence",), "stat_armour": ("armour",),
                 "stat_charge_bonus": ("charge_bonus",), "stat_melee_damage_ap": ("ap_damage",)}


def stat_mult(u, params, stat):
    """[B, N] the fatigue state's multiplier of one stat (effects() below lays it on u in the step; 1 where the table has
    none): a rule that takes the stat without the fatigue divides by it (the pursuit's charge, lone_rout.charge)."""
    table = params.sim["fatigue"].get("effects") or {}
    if "fat" not in u:
        return torch.ones_like(u["attack"])
    mult = [float(table.get(lv, {}).get(stat, 1.0)) for lv in LEVELS]
    level = u["fat"].long().clamp(0, len(LEVELS) - 1)
    return torch.tensor(mult, dtype=u["attack"].dtype, device=u["attack"].device)[level]


def effects(u, params):
    """Lay the fatigue state's stat multipliers on u for this step (the game's database,
    unit_fatigue_effects_tables, copied to config/nn/sim.json fatigue.effects; missing rows x1).
    stat_reloading scales the reload skill: the reload time is divided by it. Returns the fields
    it changed with their old values (restore with u.update)."""
    table = params.sim["fatigue"].get("effects")
    if not table:
        return {}
    old = {}
    level = u["fat"].long().clamp(0, len(LEVELS) - 1)
    for stat, fields in list(EFFECT_FIELDS.items()) + [("stat_reloading", ("reload",))]:
        mult = [float(table.get(lv, {}).get(stat, 1.0)) for lv in LEVELS]
        if all(m == 1.0 for m in mult):
            continue
        m = torch.tensor(mult, dtype=u["attack"].dtype, device=u["attack"].device)[level]
        for f in fields:
            old.setdefault(f, u[f])
            u[f] = u[f] / m if stat == "stat_reloading" else u[f] * m
    return old
