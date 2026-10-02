"""Scripted opponents in the simulator (docs/en/training/training.md). The game's AI takes no part
in training: it stays an independent check.

Each is policy(state) -> Orders for every unit of both sides; the rollout keeps only the orders
of the side it plays.

    nearest     every unit attacks the nearest standing enemy, running (missile units close to
                their range and shoot)
    hold_shoot  hold and shoot at will; a melee unit counter-charges the nearest enemy once it is
                within `react_m`; as the attacker, after `patience_s` everyone attacks (or the
                attacker would just lose on time)
    hold        every unit holds (shoots at will, fights back when attacked)
    ai_like     modelled on the game's AI in the recorded gate battles (build/gate-analysis): a
                line that keeps together (at a run), missile units that halt at range and shoot (the
                enemy lord first, in melee too), the lord in the line, never first, a charge or counter-charge from ~80-100 m,
                free units sent against enemy missile units and enemies already in melee (Line
                below has the numbers and where they come from)
"""
from dataclasses import dataclass

import torch

from tools.nn.sim import orders as O
from tools.nn.sim import replay

REACT_M = 80.0
PATIENCE_S = 300.0
BIG = 1e9


def _nearest(st):
    """(index [B, N] of the nearest standing enemy, its distance [B, N]; 1e9 when none)."""
    u = st.u
    standing = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"] & ~u["r"]
    dx = u["x"][:, None, :] - u["x"][:, :, None]
    dz = u["z"][:, None, :] - u["z"][:, :, None]
    d = torch.sqrt(dx * dx + dz * dz)
    enemy = (u["side"][:, :, None] != u["side"][:, None, :]) & standing[:, None, :]
    d = torch.where(enemy, d, torch.full_like(d, 1e9))
    return d.argmin(2), d.min(2).values


def nearest(st):
    return replay.nearest_attack(st)


def hold(st):
    return O.hold(st.B, st.N, st.device)


def hold_shoot(st, react_m=REACT_M, patience_s=PATIENCE_S):
    u = st.u
    near, dist = _nearest(st)
    has = dist < 1e9
    missile = (u["range"] > 0) & (u["a"] > 0)
    attacker = u["side"] == st.attacker[:, None]
    late = attacker & (st.t[:, None] >= patience_s)
    charge = has & ((~missile & ((dist <= react_m) | u["m"])) | late)
    o = O.hold(st.B, st.N, st.device)
    o.kind = torch.where(charge, torch.full_like(o.kind, O.ATTACK), o.kind)
    o.target = torch.where(charge, near, o.target)
    o.run = charge
    return o


@dataclass(frozen=True)
class Line:
    """The numbers of `ai_like`: from the four gate battles against the game's AI (build/gate-analysis:
    timeline.py, missile_duty.py, target_rank.py) where they show it, else ours ("ours")."""
    counter_m: float = 100.0      # defender: a melee unit counter-charges an enemy this close (battles 1 and 3: the
    #                               defender's line and lord went out at ~100 m, 78-87 s into the battle)
    charge_m: float = 80.0        # attacker: charges from this far (battle 4: targets given at ~80 m, contact 11 s later)
    shaken_m: float = 150.0       # ours: an enemy that wavers is charged from this far
    join_m: float = 150.0         # ours: once own units fight, the rest of the line joins enemies this close
    missile_stop: float = 0.9     # attacking missile units halt at this share of their range (battles 3 and 4: the
    #                               game's missile units halted and shot first at 108-134 m)
    lord_back_m: float = 10.0     # the lord keeps this far behind the line's centre (battle 4: 25 m behind the
    #                               front while advancing, in melee with the line from contact on)
    lord_join_m: float = 50.0     # ours: the lord attacks an enemy this close that is already fighting own units
    lord_retreat_hp: float = 0.3  # ours: a lord in melee below this share of health withdraws behind the line
    cover_m: float = 60.0         # ours: an enemy this close to an own missile unit draws the free melee units
    skirmish_m: float = 40.0      # missile units step back from an enemy melee unit this close (battle 3: the
    #                               game's slings stepped 30 m back and aside as the enemy line came)
    step_m: float = 30.0          # the advance: a point this far ahead
    advance_run: bool = True      # the advance runs: the game's attacker covered 110 m in its first 30 s (battle 4,
    #                               3.6 m/s; the simulator's infantry walks 1.5, runs 3-3.4), 83 m in the next 30
    #                               (the line keeping together); the defender of battle 2 2.7 m/s
    slack_m: float = 15.0         # a unit more than this ahead of its line waits for it
    switch_m: float = 15.0        # target hysteresis: a new target must score this much better
    engaged_bonus_m: float = 30.0   # prefer enemies already in melee (their flanks), battle 3-4 retargets
    missile_bonus_m: float = 25.0   # prefer enemy missile units
    cover_bonus_m: float = 60.0     # prefer enemies that threaten own missile units
    withdraw_m: float = 50.0      # how far a missile unit steps back
    flank_m: float = 12.0         # a free unit goes for an engaged enemy via a point this far beside its flank
    #                               (0: straight at it); the simulator's tactic scan (simulator.md)
    focus_lord: bool = True       # missile units shoot the enemy lord when it is in range (battles 1-3)
    lord_in_melee: bool = True    # ... in melee too: in the 63 network gate battles the game's missile units, the
    #                               network's lord in range, shot him 89 % of their firing seconds, 91 % while he
    #                               was in melee, 86 % while free (02.10.2026; ai_like without it: 34 % of its
    #                               firing on him, the game's AI 57 % of all its firing)
    advance_without_missiles: bool = True   # battle 2: a defender with no missile units met the attacker half-way


def _centroid(x, z, mask):
    w = mask.float()
    n = w.sum(1).clamp(min=1)
    return (x * w).sum(1) / n, (z * w).sum(1) / n


def ai_like(st, p=Line()):
    """Orders [B, N] of the game-like line for both sides (see Line)."""
    u = st.u
    side, x, z = u["side"], u["x"], u["z"]
    alive = (side > 0) & (u["men"] > 0) & ~u["gone"]
    standing = alive & ~u["r"]
    dx = x[:, None, :] - x[:, :, None]                         # [b, i, j]: from i to j
    dz = z[:, None, :] - z[:, :, None]
    d = torch.sqrt(dx * dx + dz * dz)
    foe = (side[:, :, None] != side[:, None, :]) & (side[:, :, None] > 0) & standing[:, None, :]
    d_foe = torch.where(foe, d, torch.full_like(d, BIG))
    near_d = d_foe.min(2).values
    has = near_d < BIG
    missile = (u["range"] > 0) & (u["a"] > 0)
    lord = u["lord"]
    body = standing & ~lord
    line = body & ~missile
    attacker = side == st.attacker[:, None]
    fighting = u["m"] & standing

    # The side's forward (its line's centre to the enemy's centre), the line's centre, per unit.
    fx, fz, cx, cz = (torch.zeros_like(x) for _ in range(4))
    side_fights = torch.zeros_like(standing)
    has_line = torch.zeros_like(standing)
    has_missile = torch.zeros_like(standing)
    has_body = torch.zeros_like(standing)
    threat = torch.zeros_like(d, dtype=torch.bool)             # [b, i, j]: j threatens a missile unit of i's side
    near_missile = (missile & standing & ~fighting)[:, :, None] & foe & (d <= p.cover_m)
    for s in (1, 2):
        mine = side == s
        ref = line & mine
        ref = torch.where(ref.any(1, keepdim=True), ref, body & mine)
        ref = torch.where(ref.any(1, keepdim=True), ref, standing & mine)
        ox, oz = _centroid(x, z, ref)
        ex, ez = _centroid(x, z, standing & (side == 3 - s))
        vx, vz = ex - ox, ez - oz
        n = torch.sqrt(vx * vx + vz * vz).clamp(min=1e-6)
        fx = torch.where(mine, (vx / n)[:, None], fx)
        fz = torch.where(mine, (vz / n)[:, None], fz)
        cx = torch.where(mine, ox[:, None], cx)
        cz = torch.where(mine, oz[:, None], cz)
        side_fights = torch.where(mine, (fighting & mine).any(1, keepdim=True), side_fights)
        has_line = torch.where(mine, (line & mine).any(1, keepdim=True), has_line)
        has_missile = torch.where(mine, (missile & standing & mine).any(1, keepdim=True), has_missile)
        has_body = torch.where(mine, (body & mine).any(1, keepdim=True), has_body)
        thr = (near_missile & mine[:, :, None]).any(1)          # [b, j]
        threat = threat | (mine[:, :, None] & thr[:, None, :])
    ahead = (x - cx) * fx + (z - cz) * fz                     # along forward, from the line's centre

    # Melee targets: nearest, with a pull towards enemies already fighting, missile units and
    # enemies on own missile units; the order in force stays unless another is clearly better.
    e_fight = fighting[:, None, :].float()
    e_missile = missile[:, None, :].float()
    score = d_foe - p.engaged_bonus_m * e_fight - p.missile_bonus_m * e_missile - p.cover_bonus_m * threat.float()
    score = torch.where(foe, score, torch.full_like(score, BIG))
    best_sc, best = score.min(2)
    cur = u["order_target"].clamp(min=0)
    cur_ok = (u["order_kind"] == O.ATTACK) & (u["order_target"] >= 0) & foe.gather(2, cur[:, :, None]).squeeze(2)
    cur_sc = score.gather(2, cur[:, :, None]).squeeze(2)
    tgt = torch.where(cur_ok & (cur_sc <= best_sc + p.switch_m), cur, best)
    opp = u["target"].clamp(min=0)                            # the enemy a unit fights now
    opp_ok = fighting & (u["target"] >= 0) & foe.gather(2, opp[:, :, None]).squeeze(2)
    tgt = torch.where(opp_ok, opp, tgt)
    tgt_d = d_foe.gather(2, tgt[:, :, None]).squeeze(2)
    t_wavers = u["w"].gather(1, tgt)
    t_threat = threat.gather(2, tgt[:, :, None]).squeeze(2)

    trigger = torch.where(attacker, torch.full_like(x, p.charge_m), torch.full_like(x, p.counter_m))
    trigger = torch.where(t_wavers, torch.clamp(trigger, min=p.shaken_m), trigger)
    trigger = torch.where(side_fights | t_threat, torch.clamp(trigger, min=p.join_m), trigger)
    trigger = torch.where(cur_ok | fighting, torch.full_like(trigger, BIG), trigger)
    lone_lord = lord & standing & ~has_body
    charge = (line | lone_lord) & has & (tgt_d <= trigger)

    advance_side = attacker | (~has_missile & p.advance_without_missiles)
    behind = ahead <= p.slack_m
    o = O.hold(st.B, st.N, st.device)
    kind, tx, tz, target, run = o.kind, x.clone(), z.clone(), o.target, o.run

    def put(mask, k, px=None, pz=None, tg=None, r=False):
        nonlocal kind, tx, tz, target, run
        kind = torch.where(mask, torch.full_like(kind, k), kind)
        if px is not None:
            tx, tz = torch.where(mask, px, tx), torch.where(mask, pz, tz)
        if tg is not None:
            target = torch.where(mask, tg, target)
        run = torch.where(mask, torch.full_like(run, r), run)

    fwd_x, fwd_z = x + fx * p.step_m, z + fz * p.step_m
    # The line: charge, or advance with the line (attacker), or hold.
    walk = (line | lone_lord) & has & ~charge & advance_side & behind
    put(walk, O.MOVE, fwd_x, fwd_z, r=p.advance_run)
    put(charge, O.ATTACK, tg=tgt, r=True)
    if p.flank_m > 0:
        # A free unit going for an enemy that already fights ours runs first to a point beside the
        # enemy's flank (the nearer side), then attacks.
        bj = torch.deg2rad(u["b"])
        rx, rz = torch.cos(bj), -torch.sin(bj)                # the right of a unit facing (sin b, cos b)
        off = u["width"] / 2 + p.flank_m
        take = lambda a: a.gather(1, tgt)
        p1x, p1z = take(x + rx * off), take(z + rz * off)
        p2x, p2z = take(x - rx * off), take(z - rz * off)
        d1 = torch.sqrt((p1x - x) ** 2 + (p1z - z) ** 2)
        d2 = torch.sqrt((p2x - x) ** 2 + (p2z - z) ** 2)
        gx, gz = torch.where(d1 <= d2, p1x, p2x), torch.where(d1 <= d2, p1z, p2z)
        via = charge & ~fighting & take(fighting) & (torch.minimum(d1, d2) > 8.0) & (tgt_d > p.flank_m)
        put(via, O.MOVE, gx, gz, r=True)

    # Missile units: the enemy lord when in range; else shoot at will; attackers walk up to range;
    # step back from enemy melee units that come close.
    shooter = missile & standing & ~lord
    aim = lord if p.lord_in_melee else lord & ~fighting
    lord_d = torch.where(foe & aim[:, None, :], d, torch.full_like(d, BIG))
    ld, li = lord_d.min(2)
    closer = shooter & has & ~fighting & advance_side & behind & (near_d > p.missile_stop * u["range"])
    put(closer, O.MOVE, fwd_x, fwd_z, r=p.advance_run)
    if p.focus_lord:
        put(shooter & (ld <= u["range"]) & ~fighting, O.ATTACK, tg=li)
    melee_foe = foe & ~missile[:, None, :] & ~fighting[:, None, :]
    close_melee = torch.where(melee_foe, d, torch.full_like(d, BIG)).min(2).values <= p.skirmish_m
    back = shooter & ~fighting & close_melee & has_line
    put(back, O.WITHDRAW, x - fx * p.withdraw_m, z - fz * p.withdraw_m, r=True)

    # The lord: in the line, a little behind its centre; never charges first: goes in with the
    # line (or at a fight that comes to it); withdraws when hurt.
    the_lord = lord & standing & ~lone_lord
    post_x, post_z = cx - fx * p.lord_back_m, cz - fz * p.lord_back_m
    far = torch.sqrt((post_x - x) ** 2 + (post_z - z) ** 2) > 8.0
    put(the_lord & far & ~fighting, O.MOVE, post_x, post_z, r=p.advance_run)
    going = line & (charge | cur_ok | fighting)
    line_goes = torch.zeros_like(standing)
    for s in (1, 2):
        mine = side == s
        line_goes = torch.where(mine, (going & mine).any(1, keepdim=True), line_goes)
    busy = torch.where(foe & (fighting[:, None, :] | fighting[:, :, None]), d, torch.full_like(d, BIG))
    bd, bi = busy.min(2)
    join = the_lord & ((bd <= p.lord_join_m) | (line_goes & (tgt_d <= p.join_m)) | cur_ok)
    put(join, O.ATTACK, tg=torch.where(bd <= p.lord_join_m, bi, tgt), r=True)
    hurt = the_lord & fighting & (u["hp"] < p.lord_retreat_hp) & has_line
    put(hurt, O.WITHDRAW, post_x - fx * p.lord_back_m, post_z - fz * p.lord_back_m, r=True)

    ok = standing & has
    kind = torch.where(ok | (kind == O.HOLD), kind, torch.full_like(kind, O.HOLD))
    target = torch.where(kind == O.ATTACK, target, torch.full_like(target, -1))
    return O.Orders(kind=kind, x=tx, z=tz, target=target, run=run & ((kind == O.ATTACK) | (kind == O.MOVE)
                                                                       | (kind == O.WITHDRAW)))


SCRIPTS = {"nearest": nearest, "hold_shoot": hold_shoot, "hold": hold, "ai_like": ai_like}
