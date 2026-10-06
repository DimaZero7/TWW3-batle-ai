"""One step of every battle in the batch, and running battles to the end
(docs/en/training/simulator.md).

A step (0.5 s): take the orders -> contacts -> melee blows and shots -> damage, men, kills ->
morale (wavering, rout, rally) -> fatigue -> movement -> the observed flags -> is the battle over.
A side loses when none of its units stands (all dead, routing, shattered or gone); at the time
limit the defender wins (the side that is not the attacker).
"""
import math

import torch

from tools.nn.sim import abilities, effects, fatigue, geometry, melee, missile, morale, movement
from tools.nn.sim import orders as O
from tools.nn.sim.params import load


def threat_flags(u, pw, params, *, alive=None, standing=None):
    """Native lf/rf/bf proxies, shared by observation and exposed-flank morale.

    The optional geometry trial requires the threatening enemy to face the unit.
    It does not redefine the companion's native flags or the direction of a hit.
    """
    cal = params.sim["threat"]
    trial = cal.get("calibration", {})
    enabled = trial.get("on", False)
    radius = trial["range_m"] if enabled else cal["range_m"]
    front = trial["front_deg"] if enabled else 60
    rear = trial["rear_deg"] if enabled else 120
    if alive is None:
        alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    if standing is None:
        standing = alive & ~u["r"]
    threat = pw["enemy"] & standing[:, None, :] & (pw["dist"] <= radius)
    if enabled:
        threat = threat & (pw["rel_j"].abs() <= trial["facing_deg"] * geometry.DEG)
    rel = pw["rel_i"] / geometry.DEG
    return {"lf": alive & (threat & (rel < -front) & (rel > -rear)).any(2),
            "rf": alive & (threat & (rel > front) & (rel < rear)).any(2),
            "bf": alive & (threat & (rel.abs() >= rear)).any(2)}


def step(st, orders, params=None, dt=None):
    """Advance every unfinished battle of st by one step (in place). orders: tools/nn/sim/orders.Orders."""
    params = params or load()
    dt = dt or params.dt
    u = st.u
    old = dict(u)
    live = ~st.done
    cal = params.sim
    R = params.battle
    spacing = cal["formation"]["spacing_m"]
    reach = cal["contact"]["reach_m"]

    present = u["side"] > 0
    alive = present & (u["men"] > 0) & ~u["gone"]
    standing = alive & ~u["r"]
    same_side = u["side"][:, :, None] == u["side"][:, None, :]
    N = st.N
    eye = torch.eye(N, dtype=torch.bool, device=st.device)[None]

    # --- orders --- (KEEP: the order in force goes on)
    take = standing & (orders.kind != O.KEEP)
    kind = torch.where(take, orders.kind, u["order_kind"])
    tgt = torch.where(take, orders.target, u["order_target"])
    run = torch.where(take, orders.run, u["order_run"])
    t_ok = (tgt >= 0) & alive.gather(1, tgt.clamp(min=0)) & ~same_side.gather(2, tgt.clamp(min=0)[:, :, None]).squeeze(2)
    kind = torch.where((kind == O.ATTACK) & ~t_ok, torch.full_like(kind, O.HOLD), kind)
    tgt = torch.where(kind == O.ATTACK, tgt, torch.full_like(tgt, -1))
    point = (kind == O.MOVE) | (kind == O.WITHDRAW)
    u["ox"] = torch.where(take & point, orders.x, u["ox"])
    u["oz"] = torch.where(take & point, orders.z, u["oz"])
    # A new order (another kind, or another attack target) makes a shooter aim again (missile.aim_reset_on_order;
    # measured: a firing unit given one in the game shoots its next 10 s at 0.6-0.75 of the rate of one given none).
    if cal["missile"].get("aim_reset_on_order"):
        changed = (kind != u["order_kind"]) | (tgt != u["order_target"])
        u["aim"] = torch.where(changed, torch.zeros_like(u["aim"]), u["aim"])
    u["order_kind"], u["order_target"], u["order_run"] = kind, tgt, run

    # --- contacts ---
    pw = geometry.pairwise(u, spacing)
    both = alive[:, :, None] & alive[:, None, :]
    # Units already fighting stay in contact a little longer (formations thin as men fall).
    held = (u["m"][:, :, None] | u["m"][:, None, :]).float() * cal["contact"]["hold_m"]
    # Formations touch when their edges overlap by -reach_m (the game's formations step into each other at the
    # contact); a lone man (a lord) stops at a formation's edge (contact.lord_reach_m).
    lone = (u["men0"] <= 1)
    reach_ij = torch.where(lone[:, :, None] | lone[:, None, :], float(cal["contact"].get("lord_reach_m", reach)), reach)
    touch = pw["enemy"] & both & (pw["gap"] <= reach_ij + held)
    # Leaving melee: a withdraw order, or any unit told to move contact.leave_m or more away (0: off).
    # It walks out of the fight: it strikes nobody and is not held in place, the enemies in contact
    # still strike it (measured in the game, config/nn/sim.json contact.why).
    leave_m = float(cal["contact"].get("leave_m", 0.0))
    far_point = torch.sqrt((u["ox"] - u["x"]) ** 2 + (u["oz"] - u["z"]) ** 2) >= leave_m
    leaving = (kind == O.WITHDRAW) | ((kind == O.MOVE) & far_point & (leave_m > 0))
    striker = standing & ~leaving
    strike = touch & striker[:, :, None]
    engaged = standing & (touch & standing[:, None, :]).any(2)
    # Pinned: a unit leaving melee stays held where it is (not striking, struck) until it has been leaving in
    # contact for contact.pin_s seconds (a missile unit) or contact.pin_melee_s (a unit without a missile weapon);
    # a lone man (a lord) is not held (measured in the game, config/nn/sim.json contact.pin_why).
    pin_s = float(cal["contact"].get("pin_s", 0.0))
    pin_melee_s = float(cal["contact"].get("pin_melee_s", 0.0))
    stuck = leaving & engaged & (u["men0"] > 1)
    hold_for = torch.where(u["range"] > 0, torch.full_like(u["leave_s"], pin_s), torch.full_like(u["leave_s"], pin_melee_s))
    stuck = stuck & (hold_for > 0)
    pinned = stuck & (u["leave_s"] < hold_for)
    u["leave_s"] = torch.where(stuck, u["leave_s"] + dt, torch.zeros_like(u["leave_s"]))
    speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
    # --- innate effects (attributes, passives, game-fired timed passives: config/nn/effects.json): their
    # stats and rule flags hold for this step ---
    innate = effects.apply(u, params, dt, standing, engaged, pw["dist"], same_side)
    # The melee clock: seconds since the contact began; it goes on through gaps out of contact shorter than
    # contact.reset_s (the game: pull-outs of 4-7 s do not restart a fight, build/cyclecharge).
    reset_s = float(cal["contact"].get("reset_s", 0.0))
    fresh = engaged & (u["contact_s"] <= 0)              # its fight starts this step
    u["out_s"] = torch.where(engaged, torch.zeros_like(u["out_s"]), u["out_s"] + dt)
    u["contact_s"] = torch.where(engaged, u["contact_s"] + dt,
                                 torch.where(u["out_s"] < reset_s, u["contact_s"], torch.zeros_like(u["contact_s"])))
    # --- the charge (CA Feature Focus #2, build/charge/spec.md): an attack order and a run-up - charge.min_runup_m
    # run at charge.min_speed_share of the run speed or faster - give the full charge bonus at the contact (an
    # attack order given within charge.order_grace_s after the contact still counts); it fades on its own clock,
    # linear to 0 over charge_decay_duration from the first blow, in contact or out of it; a new charge restarts it.
    ccal = cal["charge"]
    decay = R["charge_decay_duration"]
    u["charge"] = (u["charge"] - dt / decay).clamp(min=0)
    fire = engaged & (kind == O.ATTACK) & (u["runup"] >= float(ccal["min_runup_m"]))
    u["charge"] = torch.where(fire, torch.ones_like(u["charge"]), u["charge"])
    charge_now = u["charge"]
    fast = (speed >= float(ccal["min_speed_share"]) * u["run"]) & (speed > 0.1)
    grace = engaged & (u["contact_s"] < float(ccal["order_grace_s"])) & ~fire
    u["runup"] = torch.where(engaged, torch.where(grace, u["runup"], torch.zeros_like(u["runup"])),
                             torch.where(fast, u["runup"] + speed * dt, torch.zeros_like(u["runup"])))
    # The first strike (the game: a man's attack interval starts only after his blow, CA Feature Focus #2): a unit
    # that comes into a fight moving - or whose charge lands - strikes once at once with every man in contact (the
    # charge's full bonus with it), then at the steady rate; a standing unit it reaches does not (its men are
    # struck, not striking: the melee probe, build/meleetests). ran_in keeps that it came in moving for the rest of
    # the fight (melee.py: men gather round a lord only when he ran in).
    # (an attack order given within order_grace_s of a contact it ran into charges then: no second first strike)
    arrive = fresh & (speed > 0.3)
    first = arrive | (fire & ~(u["ran_in"] & (u["contact_s"] <= float(ccal["order_grace_s"]) + dt)))
    u["ran_in"] = (u["ran_in"] | arrive | fire) & (u["contact_s"] > 0)

    # --- lord abilities cast by the game's AI by its rule or by the network's order: their effects hold
    # for this step ---
    base = abilities.apply(u, params, dt, standing, engaged, pw["dist"], same_side, orders.ability)
    # --- fatigue's stat multipliers (the database's unit_fatigue_effects_tables), for this step ---
    tired = fatigue.effects(u, params)

    # --- melee ---
    rate, mhit, sector, _, swing = melee.strikes(u, pw, strike, params, charge_now, first=True)
    hp_melee = rate * dt + torch.where(first[:, :, None], swing, torch.zeros_like(swing))
    # A unit leaving melee (held or walking out) still in contact takes contact.leave_taken of the blows: melee
    # units more (they turn their backs), missile units less (measured, config/nn/sim.json contact.pin_why).
    taken_cal = cal["contact"].get("leave_taken")
    if taken_cal:
        out = (leaving & engaged & (u["men0"] > 1))[:, None, :]
        mult = torch.where((u["range"] > 0)[:, None, :], float(taken_cal["missile"]), float(taken_cal["melee"]))
        hp_melee = torch.where(out, hp_melee * mult, hp_melee)

    # --- shooting ---
    # (fire whilst moving, attribute mounted_fire_move: shoots and aims on the move too)
    still = (speed < 0.2) | u["fire_move"]
    ready = standing & ~engaged & (u["a"] > 0) & (u["range"] > 0) & still
    # On the move (fire whilst moving) a unit shoots only at targets within missile.move_fire_arc_deg of its
    # facing (the way it walks): walking away it holds fire (measured, config/nn/sim.json missile.move_fire_why).
    arc = float(cal["missile"].get("move_fire_arc_deg", 180.0))
    on_move = u["fire_move"] & (speed >= 0.2)
    behind = (on_move[:, :, None] & (pw["rel_i"].abs() > arc * geometry.DEG)) if arc < 180 else None
    # Standing, a unit keeps its facing while its target is within missile.stand_fire_arc_deg of it (measured,
    # missile.stand_fire_why) and turns to a target beyond it first (the facing below, at turn.*_deg_s) until the
    # target is within missile.turn_done_deg (then it keeps that facing again); it aims only once the turn is done.
    # Without an ordered target it keeps its target while that stays in range and stands (missile.sticky_target),
    # else takes the nearest within stand_fire_arc_deg, else turns to the nearest beyond it.
    ms_cal = cal["missile"]
    prev = u["aim_tgt"] if ms_cal.get("sticky_target") else None
    aim_at = missile.choose_target(u, pw, ready, tgt, kind == O.ATTACK, exclude=behind, prev=prev)
    s_arc = float(ms_cal.get("stand_fire_arc_deg", 180.0))
    turning = torch.zeros_like(ready)
    done_deg = ms_cal.get("turn_done_deg")
    if s_arc < 180:
        on_stand = ready & (speed < 0.2)
        out = on_stand[:, :, None] & (pw["rel_i"].abs() > s_arc * geometry.DEG)
        ahead = missile.choose_target(u, pw, ready, tgt, kind == O.ATTACK,
                                      exclude=out if behind is None else (out | behind), prev=prev)
        ordered = (kind == O.ATTACK) & (aim_at == tgt) & (aim_at >= 0)
        aim_at = torch.where(ordered | (ahead < 0), aim_at, ahead)
        beyond = (aim_at >= 0) & out.gather(2, aim_at.clamp(min=0)[:, :, None]).squeeze(2)
        if ms_cal.get("turn_ordered_only"):
            # firing at will a unit never turns (the probe: a target 50 deg off, no turn, 6 % of the men fired);
            # it turns only to the target of an attack order
            beyond = beyond & (kind == O.ATTACK) & (aim_at == tgt)
        if done_deg is not None:
            off = pw["rel_i"].gather(2, aim_at.clamp(min=0)[:, :, None]).squeeze(2).abs()
            go_on = u["turn_on"] & on_stand & (aim_at >= 0) & (off > float(done_deg) * geometry.DEG)
            turning = beyond | go_on
            u["turn_on"] = turning
        else:
            turning = beyond
    # A new target (another enemy than last step's, an order's or its own) costs missile.retarget_s without fire
    # (measured, missile.retarget_why): the aim clock goes back to retarget_s before aim_s.
    retarget_s = ms_cal.get("retarget_s")
    if retarget_s:
        switched = ready & (u["aim_tgt"] >= 0) & (aim_at >= 0) & (aim_at != u["aim_tgt"])
        u["aim"] = torch.where(switched, torch.minimum(u["aim"], u["aim_s"] - float(retarget_s)), u["aim"])
    u["aim_tgt"] = torch.where(ready, aim_at, torch.full_like(aim_at, -1))
    u["aim"] = torch.where(ready & ~turning, u["aim"] + dt, torch.zeros_like(u["aim"]))
    can = ready & ~turning & (u["aim"] >= u["aim_s"])
    m_target = torch.where(can, aim_at, torch.full_like(aim_at, -1))
    # direct fire needs a clear line past friends (missile.py clear_shot); arcing fire: unchanged
    m_target, clear = missile.clear_shot(u, pw, m_target, can, tgt, kind == O.ATTACK, params)
    # a standing unit's men fire only at a target centre within their fire arc (missile.arc_share; the database's
    # battle_entities fire arc): out-of-arc men lose the volley like blocked ones (they reload too)
    if ms_cal.get("per_man_arc"):
        clear = torch.where(speed < 0.2, clear * missile.arc_share(u, pw, m_target), clear)
    if behind is not None:
        back = behind.gather(2, m_target.clamp(min=0)[:, :, None]).squeeze(2) & (m_target >= 0)
        m_target = torch.where(back, torch.full_like(m_target, -1), m_target)
    # the men reload all the time (moving too); the unit shoots when it can and enough of its men are loaded
    # (missile.py: whole-unit volleys one reload apart). When the unit fires, every loaded man
    # starts reloading, those whose line is blocked too (clear_shot): else the blocked men stayed
    # loaded and fired on the next steps, and a screen that blocks half the men blocked nothing.
    unready = (u["unready"] - dt / u["reload"].clamp(min=1e-6)).clamp(min=0)
    loaded = 1 - unready
    # missile.volley_load: the unit shoots only once this share of its men is loaded (1: whole-unit volleys one
    # reload apart, as the game; 0: every loaded man at once, the volley then a steady trickle).
    load_min = float(cal["missile"].get("volley_load", 0.0))
    # (between volleys the unit is still shooting at its target: the game's IsFiringMissiles, the `fire` flag)
    volley_target = torch.where(loaded >= load_min - 1e-3, m_target, torch.full_like(m_target, -1))
    shots, hp_missile, shit = missile.volley(u, pw, volley_target, dt, params, contact=touch, clear=clear,
                                             loaded=loaded)
    u["unready"] = torch.where(shots > 0, torch.ones_like(unready), unready)
    u["a"] = (u["a"] - shots).clamp(min=0)
    firing = (m_target >= 0) & ((shots > 0) | (load_min > 0))

    # --- damage, men, kills ---
    dmg = hp_melee + hp_missile
    hit = torch.where(hp_missile > 0, shit, mhit)
    taken = dmg.sum(1)
    scale = torch.where(taken > u["hp_abs"], u["hp_abs"] / taken.clamp(min=1e-9), torch.ones_like(taken))
    dmg = dmg * scale[:, None, :]
    taken = dmg.sum(1)
    share = melee.kill_share(u["hp_man"][:, None, :], hit, cal["kills"]["exponent"])
    kills = dmg / u["hp_man"][:, None, :].clamp(min=1e-6) * share
    hp_new = (u["hp_abs"] - taken).clamp(min=0)
    floor = torch.ceil(hp_new / u["hp_man"].clamp(min=1e-6) - 1e-6)
    men_new = torch.maximum(u["men"] - kills.sum(1), floor)
    men_new = torch.minimum(men_new, u["men"])
    men_new = torch.where(u["men0"] <= 1, (hp_new > 0).float(), men_new)
    men_new = torch.where(hp_new <= 0, torch.zeros_like(men_new), men_new)
    drop = u["men"] - men_new
    want = kills.sum(1)
    credit = kills * (drop / want.clamp(min=1e-9)).clamp(max=10)[:, None, :] * pw["enemy"]
    u["k"] = u["k"] + credit.sum(2)
    u["hp_abs"], u["men"] = hp_new, men_new
    u["hp"] = torch.where(present, hp_new / u["hp0"].clamp(min=1e-6), torch.zeros_like(hp_new))
    # Casualty windows (recent casualties: the last 4 s, extended: the last 60 s; morale.casualty_windows);
    # recent_s: the melee balance (dealt / taken; reward.py reads the same).
    morale.casualty_windows(u, taken, params, dt)
    fade = math.exp(-dt / cal["morale"]["recent_s"])
    melee_scaled = hp_melee * scale[:, None, :]
    u["dealt"] = u["dealt"] * fade + melee_scaled.sum(2)
    u["taken"] = u["taken"] * fade + melee_scaled.sum(1)
    shot_at = (hp_missile * pw["enemy"]).sum(1) > 0      # friendly fire does not count (ume_concerned_under_friendly_fire 0)
    u["under_fire_s"] = torch.where(shot_at, torch.zeros_like(u["under_fire_s"]), u["under_fire_s"] + dt)
    attackers = strike & standing[:, :, None]
    u["flank_hit"] = torch.where(attackers, sector, torch.zeros_like(sector)).amax(1).float()
    # First struck from a worse side (flank, rear) this step: the database's was_attacked_in_flank / _rear.
    worse = u["flank_hit"] > old["flank_hit"]
    flank_event = torch.where(worse & (u["flank_hit"] >= 2), params.morale["was_attacked_in_rear"],
                              torch.where(worse & (u["flank_hit"] >= 1), params.morale["was_attacked_in_flank"], 0.0))

    alive = present & (u["men"] > 0) & ~u["gone"]
    standing = alive & ~u["r"]

    # --- the lords ---
    # A lord who falls (dead or shattered) takes his aura with him (the aura below counts standing
    # lords only); st.lord_dead_s times the fall (the metrics' "lord lost").
    has_lord = torch.stack([(u["lord"] & (u["side"] == s)).any(1) for s in (1, 2)], 1)
    lord_alive = torch.stack([(u["lord"] & (u["side"] == s) & alive & ~u["s"]).any(1) for s in (1, 2)], 1)
    dead = has_lord & ~lord_alive
    st.lord_dead_s = torch.where(live[:, None] & dead, torch.where(st.lord_dead_s < 0, torch.zeros_like(st.lord_dead_s),
                                                                    st.lord_dead_s + dt), st.lord_dead_s)
    # sim.json morale.lord_fall: a KILLED lord (health 0) costs the rest of his army the database's
    # general_died_recently (-16) for recent_s, then general_dead (-10) to the end; one who routed OFF
    # THE MAP (gone, men left) general_fled_recently (-16) for fled_s, then nothing; a routed or
    # shattered one still on the field only his aura (+ `routed`, 0) - unless his faction's routed
    # lord crumbles to death (rout_death_s: Vampire Counts, counted as killed that long into his rout).
    fall = cal["morale"]["lord_fall"]
    killed = present & (u["men"] <= 0) & ~u["gone"]
    crumbled = alive & u["r"] & (u["rout_death_s"] > 0) & (u["rout_s"] >= u["rout_death_s"])
    u["dead_s"] = torch.where(live[:, None] & (killed | crumbled | (u["dead_s"] > 0)), u["dead_s"] + dt, u["dead_s"])
    left = present & u["gone"] & (u["men"] > 0) & (u["dead_s"] <= 0)
    u["gone_s"] = torch.where(live[:, None] & (left | (u["gone_s"] > 0)), u["gone_s"] + dt, u["gone_s"])
    side_idx = (u["side"] - 1).clamp(min=0)

    def lords_max(x):           # [B, N]: the side's lords' largest x, for each unit of the side
        return torch.stack([torch.where(u["lord"] & (u["side"] == s), x, torch.zeros_like(x)).amax(1)
                            for s in (1, 2)], 1).gather(1, side_idx)
    killed_s, fled_s = lords_max(u["dead_s"]), lords_max(u["gone_s"])
    fallen = st.lord_dead_s.gather(1, side_idx) >= 0
    M = params.morale
    lord_pts = torch.where(fallen, float(fall.get("routed", 0.0)), 0.0)
    if fall.get("fled"):
        lord_pts = torch.where((fled_s > 0) & (fled_s <= float(fall["fled_s"])),
                               M["ume_concerned_general_fled_recently"], lord_pts)
    if fall.get("killed"):
        lord_pts = torch.where(killed_s > 0, torch.where(killed_s <= float(fall["recent_s"]),
                                                          M["ume_concerned_general_died_recently"],
                                                          M["ume_concerned_general_dead"]), lord_pts)

    # --- morale ---
    d = pw["dist"]
    friends = same_side & ~eye & alive[:, None, :]
    foes = pw["enemy"] & alive[:, None, :]
    # The lord's aura: full within general_aura_radius, fading linearly to 0 at
    # x inspiration_radius_max_effect_range_modifier (database: 70 m, 105 m).
    r0 = M["general_aura_radius"]
    r1 = r0 * (M["inspiration_radius_max_effect_range_modifier"] if cal["morale"].get("aura_fade") else 1.0)
    reach_share = ((r1 - d) / max(r1 - r0, 1e-6)).clamp(0, 1) if r1 > r0 else (d <= r0).float()
    # morale.lord_own_aura false: the aura reaches the lord's units, not the lord himself (measured: a lord at full
    # health with a friend near stands at leadership + flanks secure, no +4; config/nn/sim.json morale.lord_why).
    giver = same_side & standing[:, None, :] & u["encourages"][:, None, :]
    if not cal["morale"].get("lord_own_aura", True):
        giver = giver & ~eye
    aura = torch.where(giver, reach_share, torch.zeros_like(d)).amax(2)
    worth = u["cost"] * u["hp"]
    collapse = morale.army_collapse(u, params)
    ctx = {
        "aura": aura * standing.float(),
        "flank_event": flank_event,
        "collapse": collapse,
        "lord_dead_points": lord_pts,
        "neighbour": (friends & standing[:, None, :] & (d <= M["neighbour_effect_range"])).any(2),
        "in_melee": engaged,
        # A routing expendable unit scares nobody (morale.expendable_scares_expendable: only other expendable units;
        # the database's attribute text, measured: config/nn/sim.json morale.expendable_why).
        "routing_friends": (friends & u["r"][:, None, :] & ~(u["expendable"][:, None, :] & ~(
            u["expendable"][:, :, None] & bool(cal["morale"].get("expendable_scares_expendable"))))
                            & (d <= M["routing_unit_effect_distance_front"])).float().sum(2),
        "routing_enemies": (foes & u["r"][:, None, :] & (d <= M["routing_unit_effect_distance_front"])).float().sum(2),
        # The game's "under missile attack" holds morale.under_fire_s after the last projectile hit (measured).
        "under_fire": u["under_fire_s"] < float(cal["morale"].get("under_fire_s", 2.0)),
        "strong_enemy": (foes & standing[:, None, :] & (d <= M["enemy_effect_range"])
                         & (worth[:, None, :] > worth[:, :, None])).any(2),
        "enemy_near": (foes & standing[:, None, :] & (d <= cal["morale"]["rally_free_m"])).any(2),
    }
    morale.step(u, ctx, params, dt)
    standing = alive & ~u["r"]

    # --- fatigue ---
    # An ended move is rest too: KEEP preserves the order, not an eternal ready stance.
    # Explicit attacks and aiming/turning shooters remain ready between active bouts.
    # The recordings do not expose a combat-stance flag; pending work is our proxy.
    # movement.velocity brakes to rest within 0.5 m of the destination.
    pending_move = point & (((u["ox"] - u["x"]) ** 2 + (u["oz"] - u["z"]) ** 2) > 0.5 ** 2)
    idle = standing & ~pending_move & (kind != O.ATTACK) & (aim_at < 0)
    # The charge's fatigue (+34) only in the first fatigue.calibration.charge_s seconds after its first blow.
    charge_s = float(cal["fatigue"].get("calibration", {}).get("charge_s", decay))
    activity = {"melee": engaged, "charging": engaged & (charge_now > 1 - charge_s / decay), "shooting": firing,
                "running": speed > u["walk"] + 0.3, "walking": speed > 0.3,
                "idle": idle, "active": alive, "attack": kind == O.ATTACK, "single": u["men0"] <= 1,
                "run_order": u["order_run"].bool(), "routing": alive & u["r"]}
    near_m = float(cal["fatigue"].get("calibration", {}).get("ready_enemy_m", 0.0))
    if near_m > 0:
        activity["enemy_near"] = (foes & standing[:, None, :] & (d <= near_m)).any(2)
    fatigue.step(u, activity, params, dt)

    # --- movement ---
    gx, gz = u["x"].clone(), u["z"].clone()
    gx = torch.where(point, u["ox"], gx)
    gz = torch.where(point, u["oz"], gz)
    ti = tgt.clamp(min=0)
    tx, tz = u["x"].gather(1, ti), u["z"].gather(1, ti)
    attack = kind == O.ATTACK
    shooter = (u["range"] > 0) & (u["a"] > 0)
    t_reach = pw["gap"].gather(2, ti[:, :, None]).squeeze(2)
    # a shooter under an attack order walks until its target's centre is within range and stops there (the scripted
    # range: archers stopped ~131 m centre to centre, range 130; missile.range_why); range_centre false: the old
    # edge-to-edge 0.95 x range
    if cal["missile"].get("range_centre"):
        in_range = pw["dist"].gather(2, ti[:, :, None]).squeeze(2) <= u["range"]
    else:
        in_range = t_reach <= 0.95 * u["range"]
    close_in = attack & ~(shooter & in_range)
    gx = torch.where(close_in, tx, gx)
    gz = torch.where(close_in, tz, gz)
    want = torch.where(run, u["run"], u["walk"])
    # The charge sprint (the database's battle_entities charge distance and charge speed): under an attack order, at
    # a run or a walk, a unit closes the last charge_dist metres (30 infantry, 35 lords) to its target at its charge
    # speed - a charge then (the run-up above); a move order gives none (the melee probe: the last 30 m at 3.65-3.88
    # m/s at a run of 3.0, the last 10 m 3.9-4.7, at a walk the same).
    sprint = close_in & (t_reach <= u["charge_dist"])
    want = torch.where(sprint, u["charge_speed"], want)
    moving = standing & (point | close_in)
    fx, fz = movement.flee_goal(u, pw, alive, st.bounds)
    routing = alive & u["r"]
    gx = torch.where(routing, fx, gx)
    gz = torch.where(routing, fz, gz)
    want = torch.where(routing, u["run"] * cal["morale"]["rout_speed"], want)
    moving = moving | routing
    # A unit in melee closes up to its nearest opponent (walking) until the formations touch.
    foe_gap = torch.where(touch & standing[:, None, :], pw["gap"], torch.full_like(pw["gap"], 1e9))
    close_to = foe_gap.argmin(2)
    close_gap = foe_gap.min(2).values
    closing = engaged & ~leaving & (close_gap > 0) & (close_gap < 1e9)
    cx, cz = u["x"].gather(1, close_to), u["z"].gather(1, close_to)
    gx = torch.where(closing, cx, gx)
    gz = torch.where(closing, cz, gz)
    want = torch.where(closing, u["walk"], want)
    moving = moving | closing
    vx, vz = movement.velocity(u, gx, gz, want, moving, dt)
    locked = engaged & (~leaving | pinned) & ~closing
    stop = locked | ~alive
    vx = torch.where(stop, torch.zeros_like(vx), vx)
    vz = torch.where(stop, torch.zeros_like(vz), vz)
    u["vx"], u["vz"] = vx, vz
    u["x"] = u["x"] + vx * dt
    u["z"] = u["z"] + vz * dt
    pw2 = geometry.pairwise(u, spacing)
    px, pz = movement.separate(pw2, standing[:, :, None] & standing[:, None, :] & same_side & ~eye)
    u["x"] = u["x"] + px
    u["z"] = u["z"] + pz
    movement.clamp_to_map(u, st.bounds)

    # --- facing and the observed flags ---
    spd = torch.sqrt(vx * vx + vz * vz)
    mv = alive & (spd > 0.3)
    # A formation steps a few metres back or aside to its point without turning round.
    goal_d = torch.sqrt((gx - u["x"]) ** 2 + (gz - u["z"]) ** 2)
    shuffle = standing & point & ~routing & (goal_d < cal["contact"]["step_m"]) & (u["men0"] > 1)
    movement.face(u, vx, vz, mv & ~shuffle)
    near_foe = torch.where(touch & standing[:, None, :], d, torch.full_like(d, 1e9))
    opp = near_foe.argmin(2)
    has_opp = near_foe.min(2).values < 1e9
    ox_, oz_ = u["x"].gather(1, opp), u["z"].gather(1, opp)
    movement.face(u, ox_ - u["x"], oz_ - u["z"], has_opp & ~mv & standing)
    wt = aim_at.clamp(min=0)
    if done_deg is not None:
        # a standing shooter keeps its facing; it turns only to a target beyond stand_fire_arc_deg (turning above)
        movement.face(u, u["x"].gather(1, wt) - u["x"], u["z"].gather(1, wt) - u["z"],
                      turning & ~mv & ~has_opp & standing)
    else:
        mt = m_target.clamp(min=0)
        movement.face(u, u["x"].gather(1, mt) - u["x"], u["z"].gather(1, mt) - u["z"], firing & ~mv & ~has_opp)
        # A shooter that aims or turns to its target (not yet shooting) faces it too.
        movement.face(u, u["x"].gather(1, wt) - u["x"], u["z"].gather(1, wt) - u["z"],
                      (aim_at >= 0) & ~firing & ~mv & ~has_opp & standing)
    # A formation in melee turns slowly (measured): an enemy on its flank or rear stays there.
    movement.limit_turn(u, old["b"], engaged & ~leaving & (u["men0"] > 1), cal["contact"]["melee_turn_deg_s"] * dt)
    # Out of melee a standing unit turns in place at turn.formation_deg_s (a lord or another single entity:
    # turn.single_deg_s); measured, config/nn/sim.json turn.why. A walking unit faces the way it walks.
    t_cal = cal.get("turn") or {}
    if t_cal.get("formation_deg_s"):
        rate = torch.where(u["men0"] > 1, torch.full_like(u["b"], float(t_cal["formation_deg_s"])),
                           torch.full_like(u["b"], float(t_cal.get("single_deg_s", t_cal["formation_deg_s"]))))
        movement.limit_turn(u, old["b"], standing & ~mv & ~engaged, rate * dt)
    router_hit = torch.where(strike & u["r"][:, None, :], d, torch.full_like(d, 1e9))
    router = router_hit.argmin(2)
    has_router = router_hit.min(2).values < 1e9
    target = torch.where(has_opp & standing, opp, torch.where(firing, m_target,
                                                               torch.where(has_router, router, torch.full_like(opp, -1))))
    u["target"] = torch.where(alive, target, torch.full_like(target, -1))
    u["m"] = alive & (engaged | has_router)
    u["mv"] = mv
    u["f"] = mv & (spd > u["walk"] + 0.3)
    u["fire"] = firing
    hold = standing & (kind == O.HOLD)
    u["ox"] = torch.where(hold, u["x"], torch.where(attack & standing, tx, u["ox"]))
    u["oz"] = torch.where(hold, u["z"], torch.where(attack & standing, tz, u["oz"]))
    threat_cal = cal["threat"].get("calibration", {})
    # A calibrated flag uses the position and bearing visible in this observation.
    # OFF preserves the legacy pre-move geometry for exact baseline comparison.
    threat_pw = geometry.pairwise(u, spacing) if threat_cal.get("on") else pw
    masks = {} if threat_cal.get("on") else {"alive": alive, "standing": standing}
    u.update(threat_flags(u, threat_pw, params, **masks))
    dead = present & ((u["men"] <= 0) | u["gone"])
    for k in ("m", "mv", "f", "fire", "w", "lf", "rf", "bf"):
        u[k] = u[k] & ~dead

    u.update(tired)
    abilities.restore(u, base)
    effects.restore(u, innate)
    effects.refresh_on(u, params)

    # --- is the battle over ---
    standing = present & (u["men"] > 0) & ~u["gone"] & ~u["r"]
    stand1 = (standing & (u["side"] == 1)).any(1)
    stand2 = (standing & (u["side"] == 2)).any(1)
    t_new = st.t + dt
    defender = 3 - st.attacker
    over = ~stand1 | ~stand2 | (t_new >= params.limit_s)
    winner = torch.where(stand1 & ~stand2, 1, torch.where(stand2 & ~stand1, 2, defender))
    finish = live & over
    # Frozen battles keep their last state.
    keep = live[:, None]
    for k in list(u):
        if u[k] is not old[k]:
            u[k] = torch.where(keep, u[k], old[k])
    st.t = torch.where(live, t_new, st.t)
    st.winner = torch.where(finish, winner, st.winner)
    st.done = st.done | finish
    return st


_COMPILED = {}


def stepper(device, compile=None):
    """The step function to use: compiled by torch.compile on CUDA (about 10x faster on the GPU;
    the container has no C++ compiler for the CPU), plain elsewhere or when compile is False."""
    use = compile if compile is not None else torch.device(device).type == "cuda"
    if not use:
        return step
    if "step" not in _COMPILED:
        _COMPILED["step"] = torch.compile(step, dynamic=False)
    return _COMPILED["step"]


def run(st, policy, params=None, until_s=None, every_s=None, record=None, compile=None):
    """Run the batch until every battle is over (or until_s): policy(st) -> Orders, asked every
    every_s seconds (default: every step); record(st), if given, after every step."""
    params = params or load()
    dt = params.dt
    limit = until_s or params.limit_s
    every = max(1, int(round((every_s or dt) / dt)))
    advance = stepper(st.device, compile)
    n = 0
    orders = None
    while not bool((st.done | (st.t >= limit)).all()):
        if orders is None or n % every == 0:
            orders = policy(st)
        advance(st, orders, params, dt)
        if record is not None:
            record(st)
        n += 1
    return st
