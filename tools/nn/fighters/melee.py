"""Melee man by man (docs/en/training/fighters_lab.md).

Each man looks at the men round him (cells.py). He may strike an enemy man of a unit his own unit may strike this step
(the simulator's `strike`: its orders, leaving, routing - battle.py decides that for the units) whose centre is within
the database's formed combat distance (entity_action_attack_formed_combat_distance, 2.5 m) plus that man's radius. He
keeps the man he fights while that one stays in reach, else takes the nearest (one of his unit's attack target first).
His blow (CA Feature Focus #2, the database; the same formulas as the simulator's melee.py, per man):
  hit chance   35 + attack + charge bonus x his unit's charge left + bonus v the target's size - defence x direction,
               within 8-90 %; direction: the struck man's facing (to the man he fights, else his unit's), the striker
               within 45 deg of it the front, beyond 135 deg the rear (defence x0.6 flank, x0.3 rear; a routing man: rear;
               a lone man: contact.lord_direction)
  clock        the passport's attack interval after every swing, hit or miss (lab.json melee.swing_per_interval; off: the
               simulator's melee.miss_s after a miss); it runs only while
               an enemy man is in reach (lab.json melee)
  damage       the armour-piercing part whole, the base part less his own armour roll U(0.5, 1) x armour (at most all
               of it), then resistance (at most 90 %), at least 1; the charge bonus and the bonus v infantry / large split
               by the weapon's armour-piercing share; a braced man of a charge_reflection unit x2 on a charger (the
               simulator's rule); a lord's splash blow (splash_max_attacks) goes to as many of the nearest men in reach,
               each taking its share (CA 5.1.0: divided)
  blow share   a man in reach lands lab.json melee.blow_share of the formula's blows (measured; the clock runs as for
               a swing)
  the unit's share  the simulator's measured unit-level rates stay: a held formation hold_rate, a router struck at
               pursuit_rate (rout_pin_rate in its exit), a unit attacking another enemy unit_incidental /
               fresh_incidental / lord_incidental, infantry on a lord also fought by a lord lord_rival_others - a blow
               lands with that chance
A blow's damage goes to the man struck: beyond his health it is lost (overkill), so whole men die and wounds stay
on the living. Returned per pair of units: the HP dealt [B, i, j] and the men striking [B, i, j]."""
import torch

from tools.nn.fighters import cells, state as S
from tools.nn.sim import geometry
from tools.nn.sim import orders as O


def armour_roll(base, ap, armour, resist, gen, shape):
    """HP of a blow / hit: ap whole, base less U(0.5, 1) x armour % of it (at most all), resistance, at least 1."""
    roll = 0.5 + 0.5 * torch.rand(shape, generator=gen, device=base.device)
    blocked = (roll * armour / 100).clamp(max=1.0)
    return ((ap + base * (1 - blocked)) * (1 - resist.clamp(max=0.9))).clamp(min=1.0)


def gates(u, params, permit, pw, charge_now, incidental=None):
    """[B, i, j] the chance a blow of i on j lands (the simulator's unit-level rates, module docstring) and [B, i, j]
    the charge reflection's damage multiplier."""
    cc = params.sim["contact"]
    R = params.battle
    N = u["men"].shape[1]
    single_i = (u["men0"] <= 1)[:, :, None]
    single_j = (u["men0"] <= 1)[:, None, :]
    g = torch.ones_like(pw["dist"])
    routing_j = u["r"][:, None, :]
    if cc.get("pursuit_rate") is not None:
        share = torch.full_like(g, float(cc["pursuit_rate"]))
        if cc.get("rout_pin_s") and cc.get("rout_pin_rate") is not None and "rpin_s" in u:
            share = torch.where((u["rpin_s"] > 0)[:, None, :], float(cc["rout_pin_rate"]), share)
        g = torch.where(routing_j, share, g)
    slot = torch.arange(N, device=g.device)[None, None, :]
    order_t = u["order_target"][:, :, None]
    busy = (u["order_kind"][:, :, None] == O.ATTACK) & (order_t >= 0) & (order_t != slot)
    g = torch.where(single_j & ~single_i & busy, g * float(cc.get("lord_incidental", 1.0)), g)
    inc = torch.full_like(g, float(cc.get("unit_incidental", 1.0) if incidental is None else incidental))
    if cc.get("fresh_incidental") is not None and "order_s" in u:
        fresh = (u["order_s"] < float(R.get("melee_breakoff_secs", 0.0)))[:, :, None]
        inc = torch.where(fresh, float(cc["fresh_incidental"]), inc)
    g = torch.where(~single_j & ~single_i & busy, g * inc, g)
    rival = (single_i & permit).any(dim=1, keepdim=True)
    g = torch.where(single_j & rival & ~single_i, g * float(cc.get("lord_rival_others", 1.0)), g)
    # reflection (the simulator's melee.py): braced, facing the charger, the charger's charge still on
    cal = params.sim["melee"]
    speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
    braced = (u["reflect"] & (speed < cal["braced_speed"]))[:, :, None]
    facing = pw["rel_i"].abs() <= R["bracing_attack_angle"] * geometry.DEG
    charging_j = (charge_now >= R["charge_reflect_min_charge_factor_threshold"])[:, None, :]
    on = braced & facing & charging_j & ~routing_j
    reflect = torch.where(on, torch.full_like(g, float(R["charge_reflect_damage_multiplier"])), torch.ones_like(g))
    # held formations (melee.py: hold_rate; a hold reflecting a charge is not cut)
    held_kind = u["order_kind"] == O.HOLD
    if cc.get("hold_move"):
        move = u["order_kind"] == O.MOVE
        if cc.get("move_far_full") and cc.get("leave_m"):
            far = torch.sqrt((u["ox"] - u["x"]) ** 2 + (u["oz"] - u["z"]) ** 2) >= float(cc["leave_m"])
            move = move & ~far
        held_kind = held_kind | move
    held = held_kind & ((u["range"] <= 0) | bool(cc.get("hold_missile"))) & (u["men0"] > 1)
    cut = held[:, :, None]
    if cc.get("hold_reflect_full"):
        cut = cut & ~on
    g = torch.where(cut, g * float(cc.get("hold_rate", 1.0)), g)
    return g, reflect


def standoff(lab, t):
    s = lab["standoff"]
    return float(s["steady_m"]) - (float(s["steady_m"]) - float(s["start_m"])) * torch.exp(-t / float(s["tau_s"]))


def resolve(F, u, params, lab, permit, pw, charge_now, dt, near):
    """One step of melee (module docstring). near: the neighbour pass of this step (hook.Near). Sets F.tgt (the man
    each fights), F.eng (the enemy man each closes on: within reach + the tether), F.pend (+ the HP each man takes);
    returns (hp [B, i, j] dealt, men striking [B, i, j])."""
    B, M = F.B, F.M
    N = u["men"].shape[1]
    R = params.battle
    live = S.alive(F)
    lying = F.down > 0                                         # knocked down (knock.py): strikes nobody ...
    up = live & ~lying
    cand, d, c_uid = near.cand, near.d, near.c_uid
    # ... and nobody strikes him (lab.json knock.invulnerable)
    c_live = (cand >= 0) & cells.gather2(up if lab.get("knock", {}).get("invulnerable", True) else live, cand)
    pair = (F.uid[:, :, None] * N + c_uid).reshape(B, -1)
    ok_pair = permit.reshape(B, N * N).gather(1, pair).reshape(cand.shape)
    side_m = S.ugather(u, F, "side")
    ok_pair = ok_pair & (cells.gather2(side_m, cand) != side_m[:, :, None])          # enemies only
    combat = float(R.get("entity_action_attack_formed_combat_distance", 2.5))
    tether = float(R.get("entity_action_attack_formed_tether_distance", 2.0))
    r_c = cells.gather2(S.ugather(u, F, "radius"), cand)
    reach = combat + r_c
    me = up[:, :, None]
    inreach = me & c_live & ok_pair & (d <= reach)
    engage = me & c_live & ok_pair & (d <= reach + tether)
    # the target: kept while in reach, else the nearest (his unit's attack target first)
    keep = (inreach & (cand == F.tgt[:, :, None])).any(2)
    ot = S.ugather(u, F, "order_target")
    pref = (u["order_kind"] == O.ATTACK).gather(1, F.uid)[:, :, None] & (c_uid == ot[:, :, None])
    big = torch.full_like(d, 1e9)
    score = torch.where(inreach, d - 2.0 * pref.float(), big)
    best = score.argmin(2)
    has = inreach.any(2)
    new_t = cand.gather(2, best[:, :, None]).squeeze(2)
    tgt = torch.where(keep, F.tgt, torch.where(has, new_t, torch.full_like(F.tgt, -1)))
    e_score = torch.where(engage, d - 2.0 * pref.float(), big)
    eng = torch.where(engage.any(2), cand.gather(2, e_score.argmin(2)[:, :, None]).squeeze(2), torch.full_like(tgt, -1))
    F.eng = torch.where(has, tgt, eng)
    # the clock
    if lab["melee"].get("cooldown_in_reach_only", True):
        F.cd = torch.where(has, F.cd - dt, F.cd)
    else:
        F.cd = F.cd - dt
    fire = has & (F.cd <= 1e-6)
    t_c = tgt.clamp(min=0)
    tu = F.uid.gather(1, t_c)                                   # the struck man's unit
    # hit chance
    gat = lambda name: S.ugather(u, F, name)
    tgu = lambda name: u[name].gather(1, tu)
    large_t = tgu("large")
    bonus = torch.where(large_t, gat("bonus_v_large"), gat("bonus_v_inf"))
    ch = charge_now.gather(1, F.uid)
    routing_t = tgu("r")
    if params.sim["contact"].get("pursuit_rate") is not None:
        ch = torch.where(routing_t, torch.zeros_like(ch), ch)
    extra = gat("charge_bonus") * ch + bonus
    attack = gat("attack") + extra
    # direction: the striker seen from the struck man against his facing
    px, pz = near.px, near.pz
    tx, tz = px.gather(1, t_c), pz.gather(1, t_c)
    t_tgt = F.tgt.gather(1, t_c)                                # whom the struck man fights (last step)
    t_fx = torch.where(t_tgt >= 0, px.gather(1, t_tgt.clamp(min=0)) - tx, torch.sin(tgu("b") * geometry.DEG))
    t_fz = torch.where(t_tgt >= 0, pz.gather(1, t_tgt.clamp(min=0)) - tz, torch.cos(tgu("b") * geometry.DEG))
    ax, az = px - tx, pz - tz
    d_mode = lab["melee"].get("direction", "man")
    if d_mode != "man":
        # (lab.json melee.direction) the struck unit's facing instead of the struck man's; "unit_centre": the striker's
        # unit centre seen from the struck unit's centre (as the rectangle)
        t_fx, t_fz = torch.sin(tgu("b") * geometry.DEG), torch.cos(tgu("b") * geometry.DEG)
        if d_mode == "unit_centre":
            ax, az = gat("x") - tgu("x"), gat("z") - tgu("z")
    cosang = (ax * t_fx + az * t_fz) / (torch.sqrt(ax * ax + az * az) * torch.sqrt(t_fx * t_fx + t_fz * t_fz)).clamp(min=1e-6)
    cc = params.sim["contact"]
    front_c = torch.cos(torch.tensor(float(cc["front_deg"]) * geometry.DEG))
    rear_c = torch.cos(torch.tensor(float(cc["rear_deg"]) * geometry.DEG))
    coef = torch.where(cosang >= front_c, 1.0, torch.where(cosang <= rear_c,
                                                          float(R["melee_defence_direction_penalty_coefficient_rear"]),
                                                          float(R["melee_defence_direction_penalty_coefficient_flank"])))
    coef = torch.where(routing_t, float(R["melee_defence_direction_penalty_coefficient_rear"]), coef)
    F.dir_k = torch.where(fire, (cosang < front_c).long() + (cosang <= rear_c).long(), torch.full_like(F.tgt, -1))
    lone_t = tgu("men0") <= 1
    w = torch.where(lone_t, torch.full_like(coef, float(cc.get("lord_direction", 1.0))), torch.ones_like(coef))
    defence = tgu("defence") * (1 - w * (1 - coef))
    p = torch.clamp(float(R["melee_hit_chance_base"]) + float(R["melee_hit_chance_normalisation_coefficient"])
                    * (attack - defence), float(R["melee_hit_chance_min"]), float(R["melee_hit_chance_max"])) / 100
    gen = F.gen
    hit = fire & (torch.rand((B, M), generator=gen, device=p.device) < p)
    # the unit-level share of blows that land, the reflection
    g, reflect = gates(u, params, permit, pw, charge_now, incidental=lab["melee"].get("unit_incidental"))
    gp = (F.uid * N + tu)
    g_m = g.reshape(B, N * N).gather(1, gp) * float(lab["melee"].get("blow_share", 1.0))
    refl = reflect.reshape(B, N * N).gather(1, gp)
    lands = hit & (torch.rand((B, M), generator=gen, device=p.device) < g_m)
    # the clock after the swing: the interval after a hit, miss_s after a miss (the remainder kept)
    # (lab.json melee.swing_per_interval: a miss costs the whole interval too - every swing, hit or miss, comes an
    # interval after the last; else the simulator's melee.miss_s)
    miss = gat("interval") if lab["melee"].get("swing_per_interval") else torch.full_like(F.cd, float(params.sim["melee"]["miss_s"]))
    F.cd = torch.where(fire, F.cd.clamp(min=-dt) + torch.where(hit, gat("interval"), miss),
                       F.cd)
    F.cd = torch.where(has, F.cd, F.cd.clamp(min=0))
    # damage
    dmg, ap = gat("damage"), gat("ap_damage")
    share = dmg / (dmg + ap).clamp(min=1e-6)
    base = (dmg + extra * share) * refl
    app = (ap + extra * (1 - share)) * refl
    n_max = gat("splash").clamp(min=1)
    splash = bool((u["splash"] > 1).any())
    if splash:
        # a splash blow: the nearest n_max men in reach (the target first), each its 1 / n_max of the blow
        S_ = int(u["splash"].max().clamp(max=8))
        dd = torch.where(inreach, d, big)
        dd = torch.where(cand == tgt[:, :, None], torch.full_like(dd, -1.0), dd)
        val, idx = torch.topk(-dd, S_, dim=2)
        t_all = cand.gather(2, idx)
        ok_s = (val > -1e8) & (torch.arange(S_, device=d.device)[None, None, :] < n_max[:, :, None])
        t_all = torch.where(ok_s, t_all, torch.full_like(t_all, -1))
        div = torch.where(lone_t, torch.ones_like(n_max), n_max)          # a lone target takes the whole blow
        t_all = torch.where(lone_t[:, :, None], torch.cat([tgt[:, :, None], torch.full_like(t_all[..., 1:], -1)], 2),
                            t_all)
    else:
        S_ = 1
        t_all = tgt[:, :, None]
        div = torch.ones_like(n_max)
    amounts = []
    for s in range(S_):
        ts = t_all[..., s]
        tsc = ts.clamp(min=0)
        su = F.uid.gather(1, tsc)
        arm = u["armour"].gather(1, su)
        res = u["resist_physical"].gather(1, su)
        hp_hit = armour_roll(base / div, app / div, arm, res, gen, (B, M))
        amounts.append((ts, torch.where(lands & (ts >= 0), hp_hit, torch.zeros_like(hp_hit))))
    # overkill: what each man takes is at most his health left
    raw = torch.zeros((B, M), device=p.device)
    for ts, a in amounts:
        raw = raw.scatter_add(1, ts.clamp(min=0), a)
    room = (F.hp - F.pend).clamp(min=0)
    k = torch.where(raw > 0, torch.minimum(raw, room) / raw.clamp(min=1e-9), torch.zeros_like(raw))
    hp_pair = torch.zeros((B, N * N), device=p.device)
    for ts, a in amounts:
        eff = a * k.gather(1, ts.clamp(min=0))
        F.pend = F.pend.scatter_add(1, ts.clamp(min=0), eff)
        hp_pair = hp_pair.scatter_add(1, F.uid * N + F.uid.gather(1, ts.clamp(min=0)), eff)
    men = torch.zeros((B, N * N), device=p.device).scatter_add(1, F.uid * N + tu, has.float())
    F.tgt = tgt
    F.fired = fire
    return hp_pair.reshape(B, N, N), men.reshape(B, N, N)
