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
    ai_like     modelled on the game's AI in the recorded gate battles (build/gate-analysis, build/ail): a
                line that marches together at its slowest unit's pace (at a run), missile units that halt
                at range and shoot (the enemy lord first, in melee too), step back only from enemy melee units
                within 20 m and break off melee, after the first fight walk up only to a post 25 m behind the
                line, the lord and the nearest free melee unit cut off an enemy closing on own missile units
                (the guard), the lord 20 m behind the line, never first, a charge or
                counter-charge from ~95-100 m, melee targets chosen as the game AI does (distance with a
                fixed random part, enemies already fighting elsewhere, not the lord, spread over the
                enemies), after the first fight every unit goes in; before contact the line widens to lap
                the enemy's line, and free units go round to the rear of enemies already fighting (Line
                below has the numbers and where they come from)
"""
import math
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
    timeline.py, missile_duty.py, target_rank.py) and the game AI's play in the network's gate battles
    (03.10.2026, build/ail: the 16 battles of the two a1_simbatch/m25 gates and all 110 net-vs-game-AI
    recordings, "pool") where they show it, else ours ("ours")."""
    counter_m: float = 100.0      # defender: a melee unit counter-charges an enemy this close (battles 1 and 3: the
    #                               defender's line and lord went out at ~100 m, 78-87 s into the battle; pool: the
    #                               defender's first targets at 100 m median)
    charge_m: float = 95.0        # attacker: charges from this far (pool: an attacking melee unit's first target taken
    #                               at 87 m median, q10-q90 68-105 m, against 73 m for ai_like at 80)
    shaken_m: float = 150.0       # ours: an enemy that wavers is charged from this far
    join_m: float = 150.0         # ours: once own units have fought, the rest of the line joins enemies this close
    missile_stop: float = 0.9     # attacking missile units halt at this share of their range (battles 3 and 4: the
    #                               game's missile units halted and shot first at 108-134 m)
    lord_back_m: float = 20.0     # the lord keeps this far behind the line's centre (pool: 20-22 m behind the centre
    #                               before contact, 45-50 m from the first contact when it happens, his first melee
    #                               12 s after it; ai_like at 10 m: 16 m, 29 m, 3 s; at 20 m: 27 m, 50 m, 6 s)
    lord_join_m: float = 50.0     # ours: the lord attacks an enemy this close that is already fighting own units
    lord_charge_m: float = 90.0   # ... or, once his line goes in, his target this close (pool: the lord's first target
    #                               at 90 m median, 6 s before the first contact; ai_like at join_m 150: 136 m, 10 s)
    lord_retreat_hp: float = 0.3  # ours: a lord in melee below this share of health withdraws behind the line
    cover_m: float = 60.0         # ours: an enemy this close to an own missile unit draws the free melee units
    skirmish_targeted: bool = True  # the trigger of the step back: an enemy melee unit (the lord too) out of melee
    #                               whose attack order is on this shooter and that closes (>= guard_closing m/s), within
    #                               targeted_m; one hop of hop_m at a run, kept until it arrives, a new one only if the
    #                               trigger still holds then. Measured on the game AI's shooters (build/missile/skirmish,
    #                               one approach definition on the 28 gate battles of build/evade and the 123-battle
    #                               pool of build/missile2): they back off in 39 % / 53 % of approaches, starting 48 / 60 m
    #                               from the enemy, for 7 s at 2.7-3.0 m/s (~19-21 m), shooting on the way 4-10 %; the
    #                               chance to start a second is 31-57 % with an enemy that attacks them and closes within
    #                               50 m, 3-7 % at 60-80 m, 1-7 % with one closing on someone else. As a predictor of a
    #                               start this trigger hits 0.15-0.20 of them, the old one (any enemy melee unit within
    #                               20 m) 0.03-0.05, at the same false alarms. The ai_like twin on the 28 battles' starts
    #                               (6 copies, with this rule and the shooting rules of 06.10.2026): backs off in 25 % of
    #                               approaches (old rule 17 %, game 39 %: OPEN), from 50 m (game 48), hops of 5 s (7),
    #                               fires 54 / 59 % of the seconds at 20-40 / 40-60 m (old rule 87 / 85 %, game 33 / 54 %).
    #                               False: the old rule (any enemy melee unit within skirmish_m, withdraw_m every call)
    targeted_m: float = 50.0
    hop_m: float = 19.0
    skirmish_m: float = 20.0      # the old rule (skirmish_targeted False; its 79 / 63 % firing shares below came from
    #                               a looser approach definition: with build/missile/skirmish's one the game fires 33 / 54
    #                               %): missile units keep shooting at a closing enemy melee unit and step back only from
    #                               one this close (build/evade, 28 gate battles of the 5 ownloss gates, ai_like x 6
    #                               copies: the game's approached missile units fire 79 % of the seconds at 20-40 m
    #                               (n 7 battles) and 63 % at 40-60 m (n 14), move away in 24 % of the approaches;
    #                               ai_like at 50 m fired 9 / 44 % and moved away in 56 %, with these rules 58 / 77 %
    #                               and 18 %. Was 50 from the pool's "moves away 52-58 % within 40 m". The step back
    #                               stays "away from it and back": straight away sent the units in front of their
    #                               line (their depth behind it -55 m against -5, game +27; build/evade "shoot")
    step_m: float = 30.0          # the advance: a point this far ahead
    advance_run: bool = True      # the advance runs: the game's attacker covered 110 m in its first 30 s (battle 4,
    #                               3.6 m/s; the simulator's infantry walks 1.5, runs 3-3.4), 83 m in the next 30
    #                               (the line keeping together); the defender of battle 2 2.7 m/s
    slack_m: float = 15.0         # a unit more than this ahead of its line's rearmost free melee unit waits for it
    #                               (pool, the game's attacking Empire line: every unit marched at 2.8-2.9 m/s, the
    #                               greatswords' pace, 16-17 m deep; ai_like waiting on the line's centre: 3.0-3.6 m/s,
    #                               27-29 m deep, contact 8 s sooner)
    switch_m: float = 15.0        # target hysteresis: a new target must score this much better
    # Melee targets: score = distance - bonuses + penalties - pick_noise_m * Gumbel (lowest wins). The numbers are
    # a conditional logit fitted on the game AI's 3,397 new targets of melee units out of melee in the pool
    # (build/ail/choice.py; utility -d / 27 m + features, each worth these metres): the game AI takes the
    # nearest only 41-48 % of the time (ai_like before: 76-83 %), prefers enemies already in melee somewhere
    # else, avoids the lord, does not prefer missile units nor enemies near its own missile units.
    engaged_bonus_m: float = 20.0   # an enemy in melee (fit: +20 m; was 30, battle 3-4 retargets)
    crowd_m: float = 30.0           # ... unless the unit stands within 40 m of an own fight (fit: -32 m)
    missile_bonus_m: float = 0.0    # enemy missile units (fit: -2 m; was 25)
    cover_bonus_m: float = 0.0      # enemies within cover_m of own missile units (fit: -1 m; was 60)
    lord_penalty_m: float = 25.0    # the enemy lord, for units but the lord (fit: -26 m; the lords' duels stay: the game
    #                                 AI's lord on ours 28 % (16 battles) / 50 % (pool) of his targeted time, ai_like 33-36 %)
    taken_m: float = 6.0            # each other own unit already on that enemy, up to 3 (fit: -6 m a unit)
    ahead_bonus_m: float = 10.0     # times the cosine between the side's forward and the enemy (fit: +10 m)
    wavering_m: float = 15.0        # a wavering enemy (fit: -15 m; the trigger still reaches it from shaken_m)
    pick_noise_m: float = 27.0      # the fit's distance scale: a fixed Gumbel draw per (battle row, unit, enemy) times
    #                                 this (the game AI's picks are that spread; 0: deterministic)
    withdraw_m: float = 50.0      # how far a missile unit steps back
    escape_s: float = 15.0        # a missile unit caught in melee breaks off (WITHDRAW, away from its enemy) for this
    #                               long, then fights; (pool: the game's missile units in melee move 52 % of their
    #                               first 5 s, 34 % at 5-10 s, 23 % later; spells 11 s median, 23 s mean when alone,
    #                               against 16 / 44 s for ai_like that never left)
    flank_m: float = 12.0         # a free unit goes for an engaged enemy via a point this far beside its flank
    #                               (0: straight at it); the simulator's tactic scan (simulator.md)
    wrap_back_m: float = 35.0     # ... and this far behind its centre (its own facing), straight at it once the unit
    #                               stands wrap_deg or more off the enemy's facing (behind it); build/wrap: the
    #                               game AI's melee entries from the rear / flank 0.224 / 0.266, ai_like with
    #                               the point beside the flank only 0.126 / 0.416, with this 0.187 / 0.360
    wrap_deg: float = 140.0
    # The line widens before contact to lap the enemy's line by overlap_m at both ends (each unit shifts aside
    # in proportion to its place in the line, at most lap_space_m between neighbouring units' centres).
    # build/wrap (the 162 fair net-vs-game-AI recordings): the game AI's melee line widened from 170 m (60 s
    # before contact) to 191 m at contact against the network's 157-165 m, 2.4 units beyond the network's end
    # unit at contact (31 m beyond, mean), which then went round (level with the network's line 40 s later);
    # its centres 33-35 m apart. ai_like before: 175 -> 162 m against 166, 0-1 units beyond.
    overlap_m: float = 30.0
    lap_space_m: float = 40.0
    focus_lord: bool = True       # missile units shoot the enemy lord when it is in range (battles 1-3)
    lord_in_melee: bool = True    # ... in melee too: in the 63 network gate battles the game's missile units, the
    #                               network's lord in range, shot him 89 % of their firing seconds, 91 % while he
    #                               was in melee, 86 % while free (02.10.2026; ai_like without it: 34 % of its
    #                               firing on him, the game's AI 57 % of all its firing)
    advance_without_missiles: bool = True   # battle 2: a defender with no missile units met the attacker half-way
    advance_after_contact: bool = True      # once own units have fought (in melee now, or a melee unit has kills),
    #                                         the free units go forward and join enemies within join_m (missile
    #                                         units up to their range): the game AI's units after the first contact
    #                                         stand idle 1.6 % (melee) / 8.7 % (missile, enemy 61 m away: reloading) of
    #                                         the time, ai_like's without it 5.5-6.3 % / 20-27 % (enemy 150-175 m away)
    missile_post_m: float = 25.0  # ... but a missile unit walks up to range only as far as a post this far behind its
    #                               line's centre (0: off). build/evade + build/archers (28 gate battles x 6 copies):
    #                               the game's free missile units stand 27 m behind their melee centre after the first
    #                               fight (battle medians; 56-63 m in its first 120 s), ai_like's 5 m IN FRONT: they
    #                               walked on to range past their own line (the walk-up was half of the orders of
    #                               those in front; 43 m in front after 120 s). With this and the guard: 19 m behind
    #                               (post 55: 20 m, the same; post 25 kept ai_like's trade and routs where they were,
    #                               55 raised its routs 1.13 -> 1.21, game 0.68)
    # The guard: an enemy melee unit (the lord too) out of melee that closes (>= guard_closing m/s) on an own free
    # missile unit within guard_m of it is counter-charged by the own lord when within guard_lord_m of him (out of
    # melee, health >= lord_retreat_hp) and by the own free melee unit nearest to it within guard_unit_m; only once
    # the side has fought (the lord never goes first). build/evade/guard.py (28 gate battles): the game AI takes an
    # enemy that chases one of its missile units when it is 73 m (lord) / 82 m (melee units) from it; our melee
    # units' chases of its missile units last 13 s (median), an approacher ends in melee with an AI unit 80 % (with
    # the AI lord 37 %); ai_like without the guard (it took them at 45-50 m, its lord at 25 m): 31 s, 62 %, 20 %;
    # with the guard and the rules above (build/archers): 17 s, 69 %, 21 %
    guard_m: float = 80.0         # 0: off
    guard_lord_m: float = 90.0
    guard_unit_m: float = 100.0
    guard_closing: float = 0.5
    flee_straight: bool = False   # a missile unit out of melee steps back straight away from the enemy (False: away
    #                               from it and back, towards its own side)


def _centroid(x, z, mask):
    w = mask.float()
    n = w.sum(1).clamp(min=1)
    return (x * w).sum(1) / n, (z * w).sum(1) / n


def pick_noise(B, N, device):
    """[B, N, N] a fixed standard Gumbel draw per (battle row, unit, enemy slot): an integer hash, the same
    every step (no flip-flop between steps, nothing random to replay; a restarted row puts other units in
    the slots)."""
    b = torch.arange(B, device=device)[:, None, None]
    i = torch.arange(N, device=device)[None, :, None]
    j = torch.arange(N, device=device)[None, None, :]
    h = (b * 1000003 + i * 7919 + j * 104729 + 12345) & 0x7FFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0x7FFFFFFF
    h = ((h ^ (h >> 16)) * 668265263) & 0x7FFFFFFF
    h = h ^ (h >> 15)
    q = ((h & 0xFFFFFF).float() + 0.5) / float(1 << 24)
    return -torch.log(-torch.log(q))


def lap_shift(st, p, fx, fz, cx, cz, line, foe, missile, lord):
    """[B, N] the shift aside (m, along the side's left (-fz, fx)) that takes a unit of the line to its place
    in a line lapping the enemy's line (its melee units, else all its standing units) by p.overlap_m at both
    ends: each line unit keeps its share of the way along its line, the line at most p.lap_space_m between
    neighbouring centres; 0 for other units, a line of one unit or one narrower than 5 m, no enemy."""
    u = st.u
    x, z, side = u["x"], u["z"], u["side"]
    lx, lz = -fz, fx
    lat = (x - cx) * lx + (z - cz) * lz                         # across the side's forward, from its centre
    elat = (x[:, None, :] - cx[:, :, None]) * lx[:, :, None] + (z[:, None, :] - cz[:, :, None]) * lz[:, :, None]
    eb = foe & ~(missile | lord)[:, None, :]
    eb = torch.where(eb.any(2, keepdim=True), eb, foe)
    has_e = eb.any(2)
    elo = torch.where(eb, elat, torch.full_like(elat, BIG)).min(2).values
    ehi = torch.where(eb, elat, torch.full_like(elat, -BIG)).max(2).values
    lo, hi, n = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
    for s in (1, 2):
        mine = side == s
        ln = line & mine
        lo = torch.where(mine, torch.where(ln, lat, torch.full_like(lat, BIG)).min(1, keepdim=True).values, lo)
        hi = torch.where(mine, torch.where(ln, lat, torch.full_like(lat, -BIG)).max(1, keepdim=True).values, hi)
        n = torch.where(mine, ln.sum(1, keepdim=True).to(x.dtype), n)
    ok = line & has_e & (n >= 2) & (hi - lo >= 5.0)
    lo, hi = torch.where(ok, lo, torch.zeros_like(lo)), torch.where(ok, hi, torch.ones_like(hi))
    elo, ehi = torch.where(ok, elo, lo), torch.where(ok, ehi, hi)
    out_lo = (lo - (elo - p.overlap_m)).clamp(min=0.0)          # how far each end has to go out
    out_hi = ((ehi + p.overlap_m) - hi).clamp(min=0.0)
    room = ((n - 1) * p.lap_space_m - (hi - lo)).clamp(min=0.0)
    k = (room / (out_lo + out_hi).clamp(min=1e-6)).clamp(max=1.0)
    dlo, dhi = lo - out_lo * k, hi + out_hi * k
    frac = (lat - lo) / (hi - lo)
    return torch.where(ok, dlo + frac * (dhi - dlo) - lat, torch.zeros_like(lat))


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
    side_fought = torch.zeros_like(standing)                   # own melee units have killed (the fight began)
    melee_kills = alive & ~missile & ~lord & (u["k"] > 0)
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
        side_fought = torch.where(mine, ((fighting | melee_kills) & mine).any(1, keepdim=True), side_fought)
        has_line = torch.where(mine, (line & mine).any(1, keepdim=True), has_line)
        has_missile = torch.where(mine, (missile & standing & mine).any(1, keepdim=True), has_missile)
        has_body = torch.where(mine, (body & mine).any(1, keepdim=True), has_body)
        thr = (near_missile & mine[:, :, None]).any(1)          # [b, j]
        threat = threat | (mine[:, :, None] & thr[:, None, :])
    ahead = (x - cx) * fx + (z - cz) * fz                     # along forward, from the line's centre

    # Melee targets: the game AI's choice (Line: distance, enemies in melee elsewhere, not the lord, spread
    # over the enemies, ahead, a fixed random part); the order in force stays unless another is clearly better.
    e_fight = fighting[:, None, :].float()
    e_missile = missile[:, None, :].float()
    same = (side[:, :, None] == side[:, None, :]) & (side[:, :, None] > 0)
    eye = torch.eye(st.N, dtype=torch.bool, device=x.device)[None]
    on = ((u["order_kind"] == O.ATTACK) & (u["order_target"] >= 0)) | (fighting & (u["target"] >= 0))
    on_j = torch.where(u["order_kind"] == O.ATTACK, u["order_target"], u["target"]).clamp(min=0)
    att = torch.zeros_like(d).scatter_(2, on_j[:, :, None], on[:, :, None].float())        # [b, i', j]: i' is on j
    taken = torch.bmm((same & ~eye & standing[:, None, :]).float(), att).clamp(max=3.0)   # [b, i, j]
    own_fight = same & ~eye & fighting[:, None, :] & (d <= 40.0)
    crowded = own_fight.any(2)                                 # [b, i]: an own fight within 40 m
    cos_ahead = (dx * fx[:, :, None] + dz * fz[:, :, None]) / d.clamp(min=1e-6)
    score = (d_foe - p.engaged_bonus_m * e_fight + p.crowd_m * (crowded[:, :, None].float() * e_fight)
             - p.missile_bonus_m * e_missile - p.cover_bonus_m * threat.float()
             + p.lord_penalty_m * (~lord[:, :, None] & lord[:, None, :]).float() + p.taken_m * taken - p.ahead_bonus_m * cos_ahead
             + p.wavering_m * u["w"][:, None, :].float())
    if p.pick_noise_m > 0:
        score = score - p.pick_noise_m * pick_noise(st.B, st.N, x.device)
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
    trigger = torch.where(side_fought | t_threat, torch.clamp(trigger, min=p.join_m), trigger)
    trigger = torch.where(cur_ok | fighting, torch.full_like(trigger, BIG), trigger)
    lone_lord = lord & standing & ~has_body
    charge = (line | lone_lord) & has & (tgt_d <= trigger)

    advance_side = attacker | (~has_missile & p.advance_without_missiles) | (side_fought & p.advance_after_contact)
    # The march: a unit more than slack_m ahead of its line's rearmost free melee unit waits (the line goes at
    # the pace of its slowest unit).
    free_line = line & ~fighting & ~cur_ok                     # (not those already sent at an enemy)
    rear = torch.where(free_line, ahead, torch.full_like(ahead, BIG))
    rear_s = torch.zeros_like(ahead)
    for s in (1, 2):
        mine = side == s
        r = torch.where(mine, rear, torch.full_like(rear, BIG)).min(1, keepdim=True).values
        rear_s = torch.where(mine, torch.where(r < BIG, r, torch.zeros_like(r)), rear_s)
    behind = (ahead - rear_s <= p.slack_m) | side_fought        # (after the first fight nobody waits for the line)
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
    if p.overlap_m > 0:
        # Before contact the line widens to lap the enemy's line at both ends: a marching unit adds the shift
        # aside to its point (at most half a step), a holding one sidesteps (within the simulator's step_m it
        # does not turn: a formation steps aside facing the enemy).
        shift = lap_shift(st, p, fx, fz, cx, cz, line, foe, missile, lord)
        free = line & has & ~charge & ~fighting & ~cur_ok & ~side_fought
        sh = shift.clamp(-p.step_m / 2, p.step_m / 2)
        put(walk & free, O.MOVE, fwd_x - fz * sh, fwd_z + fx * sh, r=p.advance_run)
        side_step = free & ~walk & (shift.abs() > 3.0)
        sh = shift.clamp(-8.0, 8.0)
        put(side_step, O.MOVE, x - fz * sh, z + fx * sh, r=False)
    put(charge, O.ATTACK, tg=tgt, r=True)
    if p.flank_m > 0:
        # A free unit going for an enemy that already fights ours runs first to a point beside the
        # enemy's flank (the nearer side) and wrap_back_m behind it, then attacks once it is behind
        # (wrap_deg off the enemy's facing) or at that point.
        bj = torch.deg2rad(u["b"])
        rx, rz = torch.cos(bj), -torch.sin(bj)                # the right of a unit facing (sin b, cos b)
        hx, hz = torch.sin(bj), torch.cos(bj)                 # its facing
        off = u["width"] / 2 + p.flank_m
        take = lambda a: a.gather(1, tgt)
        bx, bz = x - hx * p.wrap_back_m, z - hz * p.wrap_back_m
        p1x, p1z = take(bx + rx * off), take(bz + rz * off)
        p2x, p2z = take(bx - rx * off), take(bz - rz * off)
        d1 = torch.sqrt((p1x - x) ** 2 + (p1z - z) ** 2)
        d2 = torch.sqrt((p2x - x) ** 2 + (p2z - z) ** 2)
        gx, gz = torch.where(d1 <= d2, p1x, p2x), torch.where(d1 <= d2, p1z, p2z)
        vx_, vz_ = x - take(x), z - take(z)                    # from the enemy to the unit
        cos_off = (vx_ * take(hx) + vz_ * take(hz)) / torch.sqrt(vx_ * vx_ + vz_ * vz_).clamp(min=1e-6)
        behind_it = cos_off <= math.cos(math.radians(p.wrap_deg))
        via = (charge & ~fighting & take(fighting) & (torch.minimum(d1, d2) > 8.0) & (tgt_d > p.flank_m)
               & ~behind_it)
        put(via, O.MOVE, gx, gz, r=True)

    # Missile units: the enemy lord when in range; else shoot at will; attackers walk up to range;
    # step back from enemy melee units that come close.
    shooter = missile & standing & ~lord
    aim = lord if p.lord_in_melee else lord & ~fighting
    lord_d = torch.where(foe & aim[:, None, :], d, torch.full_like(d, BIG))
    ld, li = lord_d.min(2)
    closer = shooter & has & ~fighting & advance_side & behind & (near_d > p.missile_stop * u["range"])
    if p.missile_post_m > 0:                                   # after the first fight: not past the post
        closer = closer & ~(side_fought & (ahead >= -p.missile_post_m))
    put(closer, O.MOVE, fwd_x, fwd_z, r=p.advance_run)
    if p.focus_lord:
        put(shooter & (ld <= u["range"]) & ~fighting, O.ATTACK, tg=li)
    # ... away from it (and back) when it comes within skirmish_m (skirmish_targeted: one that attacks this shooter
    # and closes on it); one caught in melee breaks off for escape_s.
    melee_foe = foe & ~missile[:, None, :] & ~fighting[:, None, :]
    if p.skirmish_targeted:
        me = torch.arange(st.N, device=x.device)[None, :, None]
        on_me = (u["order_kind"][:, None, :] == O.ATTACK) & (u["order_target"][:, None, :] == me)
        clos_j = -(u["vx"][:, None, :] * dx + u["vz"][:, None, :] * dz) / d.clamp(min=1e-6)   # j's speed towards i
        melee_foe = melee_foe & on_me & (clos_j >= p.guard_closing)
    md, mi = torch.where(melee_foe, d, torch.full_like(d, BIG)).min(2)
    opp_i = torch.where(opp_ok, opp, d_foe.min(2).indices)     # in melee: the enemy it fights (else the nearest)
    from_i = torch.where(fighting, opp_i, mi)
    ax, az = x - x.gather(1, from_i), z - z.gather(1, from_i)
    an = torch.sqrt(ax * ax + az * az).clamp(min=1e-6)
    wx, wz = ax / an - fx, az / an - fz                        # away from it, and back
    if p.flee_straight:                                        # (free: straight away from it)
        wx, wz = torch.where(fighting, wx, ax / an), torch.where(fighting, wz, az / an)
    wn = torch.sqrt(wx * wx + wz * wz).clamp(min=1e-6)
    wx, wz = torch.where(wn > 1e-3, wx / wn, -fx), torch.where(wn > 1e-3, wz / wn, -fz)
    if p.skirmish_targeted:
        # a hop under way goes on to its point (out of melee, not just out of a fight: contact_s 0)
        left = torch.sqrt((u["ox"] - x) ** 2 + (u["oz"] - z) ** 2)
        hopping = (shooter & ~fighting & (u["order_kind"] == O.WITHDRAW) & (u["contact_s"] <= 0)
                   & (left > 2.0) & (left <= p.hop_m + 1.0))
        put(hopping, O.WITHDRAW, u["ox"], u["oz"], r=True)
        put(shooter & ~fighting & ~hopping & (md <= p.targeted_m), O.WITHDRAW, x + wx * p.hop_m, z + wz * p.hop_m,
            r=True)
        put(shooter & fighting & (u["contact_s"] < p.escape_s), O.WITHDRAW, x + wx * p.withdraw_m,
            z + wz * p.withdraw_m, r=True)
    else:
        back = shooter & ((~fighting & (md <= p.skirmish_m)) | (fighting & (u["contact_s"] < p.escape_s)))
        put(back, O.WITHDRAW, x + wx * p.withdraw_m, z + wz * p.withdraw_m, r=True)

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
    join = the_lord & ((bd <= p.lord_join_m) | (line_goes & (tgt_d <= p.lord_charge_m)) | cur_ok)
    put(join, O.ATTACK, tg=torch.where(bd <= p.lord_join_m, bi, tgt), r=True)
    hurt = the_lord & fighting & (u["hp"] < p.lord_retreat_hp) & has_line
    put(hurt, O.WITHDRAW, post_x - fx * p.lord_back_m, post_z - fz * p.lord_back_m, r=True)

    if p.guard_m > 0:
        # The guard (Line): the lord and the nearest free melee unit cut off an enemy closing on own missile units.
        dn = d.clamp(min=1e-6)
        clos = -(u["vx"][:, None, :] * dx + u["vz"][:, None, :] * dz) / dn      # [b, i, j]: j's speed towards i
        raider = standing & ~missile & ~fighting                                 # free melee units (lords too)
        thr = foe & (shooter & ~fighting)[:, :, None] & raider[:, None, :] & (d <= p.guard_m) & (clos >= p.guard_closing)
        may = foe & thr.any(1)[:, None, :]                                       # [b, i, j]: i may take j
        gd, gi = torch.where(may, d, torch.full_like(d, BIG)).min(2)
        lord_go = the_lord & ~fighting & (u["hp"] >= p.lord_retreat_hp) & side_fought & (gd <= p.guard_lord_m)
        free_line = line & ~fighting & side_fought
        dl = torch.where(may & free_line[:, :, None], d, torch.full_like(d, BIG))  # [b, i, j]
        nearest_i = dl.min(1).indices                                            # [b, j]: the nearest to each
        pick = torch.zeros_like(may).scatter_(1, nearest_i[:, None, :], True) & (dl <= p.guard_unit_m)
        ud, ui = torch.where(pick, d, torch.full_like(d, BIG)).min(2)
        unit_go = free_line & (ud <= p.guard_unit_m)
        put(lord_go, O.ATTACK, tg=gi, r=True)
        put(unit_go, O.ATTACK, tg=ui, r=True)

    ok = standing & has
    kind = torch.where(ok | (kind == O.HOLD), kind, torch.full_like(kind, O.HOLD))
    target = torch.where(kind == O.ATTACK, target, torch.full_like(target, -1))
    return O.Orders(kind=kind, x=tx, z=tz, target=target, run=run & ((kind == O.ATTACK) | (kind == O.MOVE)
                                                                       | (kind == O.WITHDRAW)))


SCRIPTS = {"nearest": nearest, "hold_shoot": hold_shoot, "hold": hold, "ai_like": ai_like}
