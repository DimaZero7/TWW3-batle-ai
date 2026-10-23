"""Melee man by man (docs/en/training/fighters_lab.md).

Each man looks at the men round him (cells.py). He may strike an enemy man of a unit his own unit may strike this step
(the simulator's `strike`: its orders, leaving, routing - battle.py decides that for the units) whose centre is within
the database's formed combat distance (entity_action_attack_formed_combat_distance, 2.5 m) plus that man's radius. He
keeps the man he fights while that one stays in reach, else takes the nearest (one of his unit's attack target first).
His blow (CA Feature Focus #2, the database; the same formulas as the simulator's melee.py, per man):
  hit chance   35 + attack + charge bonus x his unit's charge left + bonus v the target's size - defence x direction
               (rounded half away from zero), within 8-90, less the simulator's melee.hit_roll_offset, / 100; direction: the struck man's facing (to the man he fights, else his unit's), the striker
               within 45 deg of it the front, beyond 135 deg the rear (defence x0.6 flank, x0.3 rear; a routing man: rear;
               a lone man: contact.lord_direction, a routing lone man the simulator's lone_rout.direction)
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
               pursuit_rate (rout_pin_rate in its exit; a router struck by a lone man lone_rout.share, his
               blows on his routing attack target with the whole charge: lone_rout.charge), a unit attacking another enemy unit_incidental /
               fresh_incidental / lord_incidental, infantry on a lord also fought by a lord lord_rival_others - a blow
               lands with that chance
  pick         (lab.json melee.pick, "nearest" by default) whom a man strikes: "nearest" - the man he fights while in
               reach, else the nearest (his unit's attack target 2 m nearer); "score" - the scored pick (pick_score)
  pairs        (lab.json melee.pairs, off) an enemy man is fought by at most K men at a time (not a lone man); a man
               left over takes another enemy man in reach with room, else waits (pair_up)
  attackers    (lab.json melee.max_attackers, off) a man takes the blows of at most K of the men fighting him
A blow's damage goes to the man struck: beyond his health it is lost (overkill), so whole men die and wounds stay
on the living. Returned per pair of units: the HP dealt [B, i, j] and the men striking [B, i, j]."""
import math

import torch

from tools.nn.fighters import scans as SC, state as S
from tools.nn.sim import fatigue, geometry, lone_rout
from tools.nn.sim import melee as SM
from tools.nn.sim import orders as O


def armour_roll(base, ap, armour, resist, gen, shape, u01=None):
    """HP of a blow / hit: ap whole, base less U(0.5, 1) x armour % of it (at most all), resistance, at least 1.
    u01: the uniform draws (state.rnd; else from the generator gen)."""
    if u01 is None:
        u01 = torch.rand(shape, generator=gen, device=base.device)
    roll = 0.5 + 0.5 * u01
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
        # a lone man on a lone router (the simulator's lone_rout.share; config/nn/sim.json contact.lone_rout)
        share = lone_rout.share(cc, share, single_i)
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
    if cal.get("reflect_any_angle"):
        # (sim.json melee.reflect_any_angle) from any side (the brace probe; the simulator's melee.py)
        facing = torch.ones_like(facing)
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


def blocked_line(F, u, near, tgt, live):
    """[B, M] a man whose line to the man he fights (tgt, -1 none) passes through a living man of his own side: that
    man's body (his radius) crosses the segment between them, strictly between the two (lab.json melee.clear_line)."""
    from tools.nn.fighters import cells
    px, pz = near.px, near.pz
    t_c = tgt.clamp(min=0)
    sx, sz = px.gather(1, t_c) - px, pz.gather(1, t_c) - pz
    L = torch.sqrt(sx * sx + sz * sz).clamp(min=1e-3)
    ex, ez = sx / L, sz / L
    cand = near.cand                                                    # [B, M, K], -1 none, himself left out
    cc = cand.clamp(min=0)
    rx = cells.gather2(px, cand) - px[:, :, None]
    rz = cells.gather2(pz, cand) - pz[:, :, None]
    along = rx * ex[:, :, None] + rz * ez[:, :, None]
    perp = (rx * ez[:, :, None] - rz * ex[:, :, None]).abs()
    side_m = u["side"].gather(1, F.uid)
    rad_m = u["radius"].gather(1, F.uid)
    same = side_m.gather(1, cc.reshape(F.B, -1)).reshape(cand.shape) == side_m[:, :, None]
    ok = (cand >= 0) & same & live.gather(1, cc.reshape(F.B, -1)).reshape(cand.shape)
    r_c = rad_m.gather(1, cc.reshape(F.B, -1)).reshape(cand.shape)
    inway = ok & (along > 0) & (along < L[:, :, None]) & (perp < r_c)
    return inway.any(2) & (tgt >= 0)


def pick_score(F, u, near, up, c_live, rad, permit, combat, tether, cfg):
    """(lab.json melee.pick "score") [B, M] whom each man strikes (-1: no enemy man in reach): of the `cands` nearest
    enemy men in reach, the highest score = 1 / his rank by distance (1 the nearest) + `front` if he is within front_deg
    of the striker's facing (+ `side` within side_deg) + `attacker` if his unit's attack order is on the striker's unit;
    the man he fought last step x `keep`. The striker's facing (`facing`): "man" - to the man he fought last step while
    that one lives, else his unit's; "unit" - his unit's."""
    cand, d, c_uid, inr, _ = SC.melee_rows(near, None, up, c_live, F.uid, u["side"], rad, permit, combat, tether)
    B, M, Kc = cand.shape
    dd = torch.where(inr, d, torch.full_like(d, 1e9))
    order = dd.argsort(dim=2)
    rank = torch.empty_like(order).scatter_(2, order, torch.arange(1, Kc + 1, device=d.device).expand_as(order))
    ok = inr & (rank <= int(cfg.get("cands", 16)))
    score = 1.0 / rank.float()
    px, pz = near.px, near.pz
    live = S.alive(F)
    t = F.tgt
    tc = t.clamp(min=0)
    seen = (t >= 0) & live.gather(1, tc) if cfg.get("facing", "man") == "man" else torch.zeros_like(live)
    b = S.ugather(u, F, "b") * geometry.DEG
    fx = torch.where(seen, px.gather(1, tc) - px, torch.sin(b))
    fz = torch.where(seen, pz.gather(1, tc) - pz, torch.cos(b))
    from tools.nn.fighters import cells
    ex = cells.gather2(px, cand) - px[:, :, None]
    ez = cells.gather2(pz, cand) - pz[:, :, None]
    cosang = ((ex * fx[:, :, None] + ez * fz[:, :, None])
              / (torch.sqrt(ex * ex + ez * ez) * torch.sqrt(fx * fx + fz * fz)[:, :, None]).clamp(min=1e-6))
    score = score + torch.where(cosang >= math.cos(float(cfg.get("front_deg", 30)) * geometry.DEG),
                                float(cfg.get("front", 1.5)),
                                torch.where(cosang >= math.cos(float(cfg.get("side_deg", 60)) * geometry.DEG),
                                            float(cfg.get("side", 1.0)), 0.0))
    cu = c_uid.clamp(min=0).reshape(B, -1)
    att = ((u["order_kind"].gather(1, cu) == O.ATTACK) & (u["order_target"].gather(1, cu) == F.uid.repeat_interleave(Kc, 1)))
    score = score + float(cfg.get("attacker", 3.0)) * att.reshape(cand.shape).float()
    score = torch.where(cand == t[:, :, None], score * float(cfg.get("keep", 1.25)), score)
    best = cand.gather(2, torch.where(ok, score, torch.full_like(score, -1e9)).argmax(2, keepdim=True)).squeeze(2)
    return torch.where(ok.any(2), best, torch.full_like(best, -1))


def pair_up(F, u, near, tgt, has, keep, pref_unit, up, c_live, rad, permit, combat, tether, K, repair=True):
    """(lab.json melee.pairs) An enemy man is fought by at most K men at a time (a lone man - a lord - by any number).
    Each victim keeps the men who fought him last step (F.won) first, then the men of units whose attack target is his
    unit, then the nearest; a man left over takes the nearest enemy man in reach who still has room (his unit's attack
    target counting 2 m nearer), else he waits: no swing, his clock stands (as out of reach), he stays where he is.
    Returns (has, tgt, won) - has: the men who strike this step, tgt: whom (re-paired), won [B, M]."""
    B, M = tgt.shape
    dev = tgt.device
    t_c = tgt.clamp(min=0)
    striking = has & (tgt >= 0)
    lone = (u["men0"] <= 1).gather(1, F.uid)                                   # [B, M] a man of a lone unit
    capped = striking & ~lone.gather(1, t_c)
    px, pz = near.px, near.pz
    d0 = torch.sqrt((px.gather(1, t_c) - px) ** 2 + (pz.gather(1, t_c) - pz) ** 2)
    won_prev = getattr(F, "won", None)
    held = keep & won_prev if won_prev is not None else torch.zeros_like(keep)
    own = (F.uid.gather(1, t_c) == pref_unit)
    noise = S.rnd(F, M, 131) * 1e-3
    key = held.float() * 1000 + own.float() * 100 - d0 + noise                   # higher first
    count = torch.zeros((B, M), device=dev)
    win = torch.zeros_like(striking)
    left = capped.clone()
    for _ in range(int(K)):
        k_m = torch.where(left, key, torch.full_like(key, -1e9))
        best = torch.full((B, M), -1e9, device=dev).scatter_reduce(1, t_c, k_m, "amax", include_self=True)
        w = left & (k_m >= best.gather(1, t_c))
        win = win | w
        left = left & ~w
        count = count.scatter_add(1, t_c, w.float())
    # the men left over: the nearest enemy man in reach who still has room (the scan of the step over the men with
    # room), two of them on one man - the one with the higher key (his unit's target, nearer) has him; K rounds
    tgt_out = tgt
    none = torch.full_like(tgt, -1)
    for r in range(int(K) if repair else 0):
        room = (count < K) | lone
        _, got, new_v, _, _ = SC.melee_scan(near, up & left, c_live & room, F.uid, u["side"], rad, permit, none,
                                            pref_unit, combat, tether)
        got = got & left
        nv = new_v.clamp(min=0)
        d2 = torch.sqrt((px.gather(1, nv) - px) ** 2 + (pz.gather(1, nv) - pz) ** 2)
        own2 = F.uid.gather(1, nv) == pref_unit
        key2 = torch.where(got, own2.float() * 100 - d2 + S.rnd(F, M, 133 + r) * 1e-3, torch.full_like(d2, -1e9))
        best2 = torch.full((B, M), -1e9, device=dev).scatter_reduce(1, nv, key2, "amax", include_self=True)
        w2 = got & (key2 >= best2.gather(1, nv))
        win = win | w2
        left = left & ~w2
        tgt_out = torch.where(w2, nv, tgt_out)
        count = count.scatter_add(1, nv, w2.float())
    has_out = (has & ~(striking & ~lone.gather(1, t_c))) | win
    return has_out, tgt_out, win


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
    # ... and nobody strikes him (lab.json knock.invulnerable)
    c_live = up if lab.get("knock", {}).get("invulnerable", True) else live
    combat = float(R.get("entity_action_attack_formed_combat_distance", 2.5))
    tether = float(R.get("entity_action_attack_formed_tether_distance", 2.0))
    # the target: kept while in reach, else the nearest (his unit's attack target first); the man he closes on
    ot = S.ugather(u, F, "order_target")
    pref_unit = torch.where((u["order_kind"] == O.ATTACK).gather(1, F.uid), ot, torch.full_like(ot, -1))
    if not lab["melee"].get("prefer_target", True):
        # (lab.json melee.prefer_target off) the nearest enemy man in reach, whatever his unit
        pref_unit = torch.full_like(ot, -1)
    # the reach: the formed combat distance plus the struck man's radius (lab.json melee.reach_radius; off: the distance
    # alone, centre to centre)
    rad = u["radius"] if lab["melee"].get("reach_radius", True) else torch.zeros_like(u["radius"])
    scan = (near, up, c_live, F.uid, u["side"], rad, permit, F.tgt, pref_unit, combat, tether)
    keep, has, new_t, has_e, eng = SC.melee_scan(*scan)
    if lab["melee"].get("pick", "nearest") == "score":
        # (lab.json melee.pick "score") the scored pick among the enemy men in reach (pick_score)
        tgt = pick_score(F, u, near, up, c_live, rad, permit, combat, tether, lab["melee"].get("pick_score") or {})
        keep = (tgt >= 0) & (tgt == F.tgt)
    else:
        tgt = torch.where(keep, F.tgt, torch.where(has, new_t, torch.full_like(F.tgt, -1)))
    eng = torch.where(has_e, eng, torch.full_like(tgt, -1))
    F.eng = torch.where(has, tgt, eng)
    if lab["melee"].get("clear_line"):
        # (lab.json melee.clear_line) a man strikes only when no man of his own side stands in the way to the man he
        # fights (a body within its radius of the line between them): a second rank in reach does not strike past its
        # front rank; he keeps his target and waits (his clock stands, as out of reach)
        has = has & ~blocked_line(F, u, near, tgt, live)
    K_pair = lab["melee"].get("pairs")
    if K_pair:
        # (lab.json melee.pairs) an enemy man is fought by at most K men at a time; the rest re-pair or wait
        has, tgt, F.won = pair_up(F, u, near, tgt, has, keep, pref_unit, up, c_live, rad, permit, combat, tether,
                                  K_pair, lab["melee"].get("pairs_repair", True))
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
        chase = lab["melee"].get("chase_charge")
        if chase is not None:
            # (lab.json melee.chase_charge) a blow on a routing man carries the striker's whole charge bonus with this
            # chance (the pursuitfast probe: full-charge blows or none, no decayed ones), else none
            full = (S.rnd(F, M, 103) < float(chase)).float()
            ch = torch.where(routing_t, full, ch)
        else:
            ch = torch.where(routing_t, torch.zeros_like(ch), ch)
        k_chase = lone_rout.charge(params.sim["contact"], R)
        if k_chase is not None:
            # (the simulator's lone_rout.charge) a lone man's blow on his routing attack target: the whole charge
            chase = lone_rout.chasing(gat("men0") <= 1, gat("order_kind"), gat("order_target"), tu, routing_t, O.ATTACK)
            # (the stat without the fatigue: the simulator's fatigue.effects has laid its multiplier on u)
            full = (k_chase / fatigue.stat_mult(u, params, "stat_charge_bonus")).gather(1, F.uid)
            ch = torch.where(chase, full, ch)
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
    front_c = math.cos(float(cc["front_deg"]) * geometry.DEG)
    rear_c = math.cos(float(cc["rear_deg"]) * geometry.DEG)
    coef = torch.where(cosang >= front_c, 1.0, torch.where(cosang <= rear_c,
                                                          float(R["melee_defence_direction_penalty_coefficient_rear"]),
                                                          float(R["melee_defence_direction_penalty_coefficient_flank"])))
    coef = torch.where(routing_t, float(R["melee_defence_direction_penalty_coefficient_rear"]), coef)
    F.dir_k = torch.where(fire, (cosang < front_c).long() + (cosang <= rear_c).long(), torch.full_like(F.tgt, -1))
    lone_t = tgu("men0") <= 1
    # (a routing lone man does not turn to his chasers: the simulator's lone_rout.direction)
    w = torch.where(lone_t, lone_rout.direction(cc, routing_t, coef.dtype), torch.ones_like(coef))
    defence = SM.directional_defence(tgu("defence"), 1 - w * (1 - coef))
    p = SM.hit_chance(attack, defence, float(R["melee_hit_chance_normalisation_coefficient"]),
                      float(R["melee_hit_chance_base"]), float(R["melee_hit_chance_min"]),
                      float(R["melee_hit_chance_max"]), float(params.sim["melee"].get("hit_roll_offset", 0.0)))
    gen = None                     # (the draws: state.rnd)
    hit = fire & (S.rnd(F, M, 101) < p)
    # the unit-level share of blows that land, the reflection
    g, reflect = gates(u, params, permit, pw, charge_now, incidental=lab["melee"].get("unit_incidental"))
    gp = (F.uid * N + tu)
    g_m = g.reshape(B, N * N).gather(1, gp) * float(lab["melee"].get("blow_share", 1.0))
    refl = reflect.reshape(B, N * N).gather(1, gp)
    K = lab["melee"].get("max_attackers")
    if K:
        # (lab.json melee.max_attackers) a man takes the blows of at most K of the men fighting him at a time: with n
        # in reach on him, each one's swing lands on him with the chance K / n (the swing is spent either way)
        # (not on a lone man - a lord: the wolf rats' probe lanes on the General, base 0.22 / with it 0.02 of the
        # game's damage on him at 5-15 s)
        n_on = torch.zeros((B, M), device=g_m.device).scatter_add(1, t_c, (has & (tgt >= 0)).float()).gather(1, t_c)
        g_m = torch.where(lone_t, g_m, g_m * (float(K) / n_on.clamp(min=1.0)).clamp(max=1.0))
    lands = hit & (S.rnd(F, M, 102) < g_m)
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
    S_ = int(getattr(F, "splash_S", 1))                  # static: the batch's most men a blow splashes (state.build)
    div = torch.where(lone_t, torch.ones_like(n_max), n_max) if S_ > 1 else torch.ones_like(n_max)
    # the blows: (striker [B, X] or None for every man, the man struck [B, X], HP [B, X]); first every man's blow on
    # the man he fights
    hp_hit = armour_roll(base / div, app / div, tgu("armour"), tgu("resist_physical"), gen, (B, M),
                         u01=S.rnd(F, M, 110))
    amounts = [(None, tgt, torch.where(lands & (tgt >= 0), hp_hit, torch.zeros_like(hp_hit)))]
    if S_ > 1:
        # a splash blow: the nearest n_max men in reach (the target first), each its 1 / n_max of the blow; a lone
        # target takes the whole blow. Only the men of splash units (state.build's splash_men [B, L], -1 padding)
        sm = F.splash_men
        smc = sm.clamp(min=0)
        L = sm.shape[1]
        at = lambda t: t.gather(1, smc)
        c_s, d_s, _, in_s, _ = SC.melee_rows(near, smc, up, c_live, F.uid, u["side"], rad, permit, combat,
                                             tether)
        dd = torch.where(in_s, d_s, torch.full_like(d_s, 1e9))
        dd = torch.where(c_s == at(tgt)[:, :, None], torch.full_like(dd, -1.0), dd)
        val, idx = torch.topk(-dd, S_, dim=2)
        t_all = c_s.gather(2, idx)
        ok_s = ((val > -1e8) & (torch.arange(S_, device=d_s.device)[None, None, :] < at(n_max)[:, :, None])
                & ~at(lone_t)[:, :, None] & (sm >= 0)[:, :, None])
        t_all = torch.where(ok_s, t_all, torch.full_like(t_all, -1))
        b_s, a_s, l_s, dv = at(base), at(app), at(lands), at(div)
        for s in range(1, S_):
            ts = t_all[..., s]
            su = F.uid.gather(1, ts.clamp(min=0))
            h_s = armour_roll(b_s / dv, a_s / dv, u["armour"].gather(1, su), u["resist_physical"].gather(1, su), gen,
                              (B, L), u01=S.rnd(F, L, 110 + s))
            amounts.append((smc, ts, torch.where(l_s & (ts >= 0), h_s, torch.zeros_like(h_s))))
    # overkill: what each man takes is at most his health left
    raw = torch.zeros((B, M), device=p.device)
    for _, ts, a in amounts:
        raw = raw.scatter_add(1, ts.clamp(min=0), a)
    room = (F.hp - F.pend).clamp(min=0)
    k = torch.where(raw > 0, torch.minimum(raw, room) / raw.clamp(min=1e-9), torch.zeros_like(raw))
    hp_pair = torch.zeros((B, N * N), device=p.device)
    for who, ts, a in amounts:
        eff = a * k.gather(1, ts.clamp(min=0))
        F.pend = F.pend.scatter_add(1, ts.clamp(min=0), eff)
        su = F.uid if who is None else F.uid.gather(1, who)
        hp_pair = hp_pair.scatter_add(1, su * N + F.uid.gather(1, ts.clamp(min=0)), eff)
    men = torch.zeros((B, N * N), device=p.device).scatter_add(1, F.uid * N + tu, has.float())
    F.tgt = tgt
    F.fired = fire
    return hp_pair.reshape(B, N, N), men.reshape(B, N, N)
