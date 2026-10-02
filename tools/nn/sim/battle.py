"""One step of every battle in the batch, and running battles to the end
(docs/en/training/simulator.md).

A step (0.5 s): take the orders -> contacts -> melee blows and shots -> damage, men, kills ->
morale (wavering, rout, rally) -> fatigue -> movement -> the observed flags -> is the battle over.
A side loses when none of its units stands (all dead, routing, shattered or gone); at the time
limit the defender wins (the side that is not the attacker).
"""
import math

import torch

from tools.nn.sim import abilities, fatigue, geometry, melee, missile, morale, movement
from tools.nn.sim import orders as O
from tools.nn.sim.params import load


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
    u["order_kind"], u["order_target"], u["order_run"] = kind, tgt, run

    # --- contacts ---
    pw = geometry.pairwise(u, spacing)
    both = alive[:, :, None] & alive[:, None, :]
    # Units already fighting stay in contact a little longer (formations thin as men fall).
    held = (u["m"][:, :, None] | u["m"][:, None, :]).float() * cal["contact"]["hold_m"]
    touch = pw["enemy"] & both & (pw["gap"] <= reach + held)
    leaving = kind == O.WITHDRAW
    striker = standing & ~leaving
    strike = touch & striker[:, :, None]
    engaged = standing & (touch & standing[:, None, :]).any(2)
    speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
    new = engaged & (u["contact_s"] <= 0)
    fast = speed >= 0.5 * u["run"]
    factor = torch.where(fast, (speed / u["run"].clamp(min=0.1)).clamp(max=1), torch.zeros_like(speed))
    # Bracing: a unit with charge_reflection that stands still meets an infantry charge coming within
    # bracing_attack_angle of its front as if it charged too (measured in the whole battles: the
    # charger gains nothing). The database's charge_reflect_min_charge_factor_threshold (0.7) is not
    # applied: chargers here meet their target at 0.55-0.9 of their run (median 0.75).
    braced = new & u["reflect"] & (speed < cal["melee"]["braced_speed"])
    incoming = torch.where(touch & (new & (u["men0"] > 1))[:, None, :]
                           & (pw["rel_i"].abs() <= R["bracing_attack_angle"] * geometry.DEG),
                           factor[:, None, :], torch.zeros_like(pw["gap"])).amax(2)
    factor = torch.where(braced & (incoming > 0), incoming, factor)
    u["charge"] = torch.where(new, factor, torch.where(engaged, u["charge"], torch.zeros_like(u["charge"])))
    decay = R["charge_decay_duration"]
    charge_now = u["charge"] * (1 - u["contact_s"] / decay).clamp(min=0)
    u["contact_s"] = torch.where(engaged, u["contact_s"] + dt, torch.zeros_like(u["contact_s"]))

    # --- lord abilities (the game's AI by its rule, the network by order; passives for all): their
    # effects hold for this step ---
    base = abilities.apply(u, params, dt, standing, engaged, pw["dist"], same_side, orders.ability)

    # --- melee ---
    rate, mhit, sector, _ = melee.strikes(u, pw, strike, params, charge_now, u["contact_s"])
    hp_melee = rate * dt

    # --- shooting ---
    still = speed < 0.2
    ready = standing & ~engaged & (u["a"] > 0) & (u["range"] > 0) & still
    u["aim"] = torch.where(ready, u["aim"] + dt, torch.zeros_like(u["aim"]))
    can = ready & (u["aim"] >= u["aim_s"])
    m_target = missile.choose_target(u, pw, can, tgt, kind == O.ATTACK)
    shots, hp_missile, shit = missile.volley(u, pw, m_target, dt, params, contact=touch)
    u["a"] = (u["a"] - shots).clamp(min=0)
    firing = shots > 0

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
    tau = cal["morale"]["recent_s"]
    fade = math.exp(-dt / tau)
    u["recent"] = u["recent"] * fade + taken
    melee_scaled = hp_melee * scale[:, None, :]
    u["dealt"] = u["dealt"] * fade + melee_scaled.sum(2)
    u["taken"] = u["taken"] * fade + melee_scaled.sum(1)
    shot_at = (hp_missile * pw["enemy"]).sum(1) > 0      # friendly fire does not count (ume_concerned_under_friendly_fire 0)
    u["under_fire_s"] = torch.where(shot_at, torch.zeros_like(u["under_fire_s"]), u["under_fire_s"] + dt)
    attackers = strike & standing[:, :, None]
    u["flank_hit"] = torch.where(attackers, sector, torch.zeros_like(sector)).amax(1).float()

    alive = present & (u["men"] > 0) & ~u["gone"]
    standing = alive & ~u["r"]

    # --- the lords ---
    # A shattered lord is lost as if slain: in the game the whole army drops 0.5-0.6 of its
    # leadership in the second he shatters (gate runs 02.10.2026), the same −16 then −10.
    has_lord = torch.stack([(u["lord"] & (u["side"] == s)).any(1) for s in (1, 2)], 1)
    lord_alive = torch.stack([(u["lord"] & (u["side"] == s) & alive & ~u["s"]).any(1) for s in (1, 2)], 1)
    dead = has_lord & ~lord_alive
    st.lord_dead_s = torch.where(live[:, None] & dead, torch.where(st.lord_dead_s < 0, torch.zeros_like(st.lord_dead_s),
                                                                    st.lord_dead_s + dt), st.lord_dead_s)
    side_idx = (u["side"] - 1).clamp(min=0)
    since = st.lord_dead_s.gather(1, side_idx)
    M = params.morale
    lord_pts = torch.where(since < 0, 0.0, torch.where(since < tau, M["ume_concerned_general_died_recently"],
                                                       M["ume_concerned_general_dead"]))

    # --- morale ---
    d = pw["dist"]
    friends = same_side & ~eye & alive[:, None, :]
    foes = pw["enemy"] & alive[:, None, :]
    aura = (same_side & standing[:, None, :] & u["encourages"][:, None, :] & (d <= M["general_aura_radius"])).any(2)
    worth = u["cost"] * u["hp"]
    ctx = {
        "aura": aura & standing,
        "lord_dead_points": lord_pts,
        "neighbour": (friends & standing[:, None, :] & (d <= M["neighbour_effect_range"])).any(2),
        "in_melee": engaged,
        "routing_friends": (friends & u["r"][:, None, :] & ~u["expendable"][:, None, :]
                            & (d <= M["routing_unit_effect_distance_front"])).float().sum(2),
        "routing_enemies": (foes & u["r"][:, None, :] & (d <= M["routing_unit_effect_distance_front"])).float().sum(2),
        "under_fire": u["under_fire_s"] < 2.0,
        "strong_enemy": (foes & standing[:, None, :] & (d <= M["enemy_effect_range"])
                         & (worth[:, None, :] > worth[:, :, None])).any(2),
        "enemy_near": (foes & standing[:, None, :] & (d <= cal["morale"]["rally_free_m"])).any(2),
    }
    morale.step(u, ctx, params, dt)
    standing = alive & ~u["r"]

    # --- fatigue ---
    activity = {"melee": engaged, "charging": engaged & (charge_now > 0), "shooting": firing,
                "running": speed > u["walk"] + 0.3, "walking": speed > 0.3}
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
    in_range = t_reach <= 0.95 * u["range"]
    close_in = attack & ~(shooter & in_range)
    gx = torch.where(close_in, tx, gx)
    gz = torch.where(close_in, tz, gz)
    want = torch.where(run, u["run"], u["walk"])
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
    locked = engaged & ~leaving & ~closing
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
    mt = m_target.clamp(min=0)
    movement.face(u, u["x"].gather(1, mt) - u["x"], u["z"].gather(1, mt) - u["z"], firing & ~mv & ~has_opp)
    # A formation in melee turns slowly (measured): an enemy on its flank or rear stays there.
    movement.limit_turn(u, old["b"], engaged & ~leaving & (u["men0"] > 1), cal["contact"]["melee_turn_deg_s"] * dt)
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
    tr = cal["threat"]["range_m"]
    threat = foes & standing[:, None, :] & (d <= tr)
    rel = pw["rel_i"] / geometry.DEG
    u["lf"] = alive & (threat & (rel < -60) & (rel > -120)).any(2)
    u["rf"] = alive & (threat & (rel > 60) & (rel < 120)).any(2)
    u["bf"] = alive & (threat & (rel.abs() >= 120)).any(2)
    dead = present & ((u["men"] <= 0) | u["gone"])
    for k in ("m", "mv", "f", "fire", "w", "lf", "rf", "bf"):
        u[k] = u[k] & ~dead

    abilities.restore(u, base)

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
