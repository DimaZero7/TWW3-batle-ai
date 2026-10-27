"""Branches on the approach (drill U1; the user's decision 27.10): which is better in THIS skirmish - to meet the enemy's
charge standing braced, or to charge it? The network found only 'attack' (an attack order turns the unit by itself):
the branches price both, and a bonus (train.py --branch-bonus) pays the network when its behaviour on the approach
was the better one.

The moment: the enemy's edge first comes within its charge distance (the database's charge_dist, 30 m for infantry)
+ LEAD_S seconds of its run of ours - the last moment our unit can still choose: a facing order takes 3.9 s to re-form
(config/nn/sim.json turn.face_delay_s), then it turns; later, standing means standing as it is. The skirmish is copied
there and played on twice by scripts (the enemy under `nearest` as before, the leash on in both):
    stand    HOLD facing the enemy (the bearing to it at the moment; the simulator's brace and stand-first-strike rules
             work for a unit standing under HOLD with the enemy within the database's bracing_attack_angle, 80 deg)
    attack   ATTACK the enemy at a run (the charge)
each until HORIZON_S after its contact, our rout or death, the enemy's death or leaving the map, or CAP_S after the
moment. The branch's trade: (enemy gold lost - ours) / budget from the moment (reward.measure).

The network's behaviour at the contact (the main battle, the first step of the enemy's contact): 'attack' if its order
in force is ATTACK; 'stand' if HOLD, not moving (0.3 m/s) and the enemy within bracing_attack_angle of its front;
else 'other' (no bonus).
"""
import torch

from tools.nn.sim import geometry
from tools.nn.sim import orders as O
from tools.nn.sim import state as S
from tools.nn.train import opponents, reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import view as V

LEAD_S = 4.0
HORIZON_S = 60.0
CAP_S = 120.0
STAND, ATTACK, OTHER = 0, 1, 2


def take(st, idx):
    """A copy of the battles idx of st (a new State)."""
    return S.State({k: v[idx].clone() for k, v in st.u.items()}, st.t[idx].clone(), st.attacker[idx].clone(),
                   st.done[idx].clone(), st.winner[idx].clone(), st.lord_dead_s[idx].clone(), st.bounds,
                   [st.keys[int(i)] for i in idx.tolist()] if st.keys else [])


def cat(states):
    return S.State({k: torch.cat([s.u[k] for s in states]) for k in states[0].u}, torch.cat([s.t for s in states]),
                   torch.cat([s.attacker for s in states]), torch.cat([s.done for s in states]),
                   torch.cat([s.winner for s in states]), torch.cat([s.lord_dead_s for s in states]), states[0].bounds,
                   sum((s.keys for s in states), []))


def edge_gap(st, spacing):
    """[B] the edge-to-edge distance of our unit (slot 0) and the enemy (slot H)."""
    now = V.raw(st, spacing)
    H = st.N // 2
    g = geometry.rect_gap(now[..., V.I["x"]], now[..., V.I["z"]], now[..., V.I["b"]], now[..., V.I["front"]],
                          now[..., V.I["depth"]])
    return g[:, 0, H]


def moment(st, spacing, lead_s=LEAD_S):
    """[B] the enemy is within its charge distance + lead_s of its run of our unit, both standing, no contact yet."""
    u = st.u
    H = st.N // 2
    reach = u["charge_dist"][:, H] + u["run"][:, H] * lead_s
    up = ~u["r"][:, 0] & ~u["r"][:, H] & (u["men"][:, 0] > 0) & (u["men"][:, H] > 0)
    return up & ~u["m"][:, 0] & (edge_gap(st, spacing) <= reach) & ~st.done


def behaviour(st, bracing_deg):
    """[B] STAND / ATTACK / OTHER of our unit now (see the module)."""
    u = st.u
    H = st.N // 2
    kind = u["order_kind"][:, 0]
    speed = torch.sqrt(u["vx"][:, 0] ** 2 + u["vz"][:, 0] ** 2)
    dx, dz = u["x"][:, H] - u["x"][:, 0], u["z"][:, H] - u["z"][:, 0]
    rel = torch.remainder(torch.rad2deg(torch.atan2(dx, dz)) - u["b"][:, 0] + 180, 360) - 180
    stand = (kind == O.HOLD) & (speed <= 0.3) & (rel.abs() <= bracing_deg)
    return torch.where(kind == O.ATTACK, ATTACK, torch.where(stand, STAND, OTHER))


def play(snap, area, P, step, horizon_s=HORIZON_S, cap_s=CAP_S, rows=None, report=None):
    """Both branches of the snapshots snap (a State of M battles at their moments), area {k: [M, N]}: (trade_stand [M],
    trade_attack [M]). rows: pad the batch to this many rows per branch (a fixed shape for the compiled step).
    report: a dict to fill with the stand branch's check - stood braced at its contact (behaviour STAND) [M]."""
    M = snap.B
    dev = snap.device
    pad = max(rows or M, M)
    if pad > M:
        idx = torch.cat([torch.arange(M, device=dev), torch.zeros(pad - M, dtype=torch.long, device=dev)])
        snap = take(snap, idx)
        area = {k: v[idx] for k, v in area.items()}
    st = cat([take(snap, torch.arange(pad, device=dev)), take(snap, torch.arange(pad, device=dev))])
    real = torch.cat([torch.arange(pad, device=dev) < M] * 2)
    st.done = ~real
    ar = {k: torch.cat([v, v]) for k, v in area.items()}
    own = torch.zeros((2 * pad, st.N), dtype=torch.bool, device=dev)
    own[:, 0] = True
    is_attack = torch.cat([torch.zeros(pad, dtype=torch.bool, device=dev), torch.ones(pad, dtype=torch.bool, device=dev)])
    H = st.N // 2
    u = st.u
    face = torch.remainder(torch.rad2deg(torch.atan2(u["x"][:, H] - u["x"][:, 0], u["z"][:, H] - u["z"][:, 0])), 360.0)
    mine = O.hold(st.B, st.N, dev)
    mine.face[:, 0] = torch.where(is_attack, torch.full_like(face, -1.0), face)
    mine.kind[:, 0] = torch.where(is_attack, O.ATTACK, O.HOLD)
    mine.target[:, 0] = torch.where(is_attack, H, -1)
    mine.run[:, 0] = is_attack
    leash = Dc.Leash(st.B, st.N, dev)
    m0 = reward.measure(st)
    t0 = st.t.clone()
    tc = torch.full_like(t0, -1.0)
    beh = torch.full((st.B,), OTHER, dtype=torch.long, device=dev)
    brace = bracing_deg(P)
    while not bool(st.done.all()):
        o = O.merge(opponents.nearest(st), leash.apply(st, mine, ar, own), own)
        was = st.done.clone()
        step(st, o, P)
        u = st.u
        hit = u["hit_sec"][:, H * st.N + 0] > 0
        first = (tc < 0) & hit & ~was
        beh = torch.where(first, behaviour(st, brace), beh)
        tc = torch.where(first, st.t, tc)
        over = ((tc >= 0) & (st.t - tc >= horizon_s)) | (st.t - t0 >= cap_s) | u["r"][:, 0] | (u["men"][:, 0] <= 0) \
            | (u["men"][:, H] <= 0) | u["gone"][:, H]
        st.done = was | over
    m1 = reward.measure(st)
    trade = (m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])
    if report is not None:
        report["stand_braced"] = beh[:M] == STAND
        report["stand_contact"] = tc[:M] >= 0
    return trade[:M], trade[pad:pad + M]


def bonus(beh, t_stand, t_attack, weight):
    """[M] weight x |the branches' trade difference| where the behaviour is the better branch, else 0."""
    best = torch.where(t_stand >= t_attack, STAND, ATTACK)
    return torch.where(beh == best, weight * (t_stand - t_attack).abs(), torch.zeros_like(t_stand)), best


def weight_at(w0, update, decay_updates):
    """The bonus's weight after `update` updates: w0, falling linearly to 0 over decay_updates (0: never)."""
    if decay_updates <= 0:
        return w0
    return w0 * max(0.0, 1.0 - update / decay_updates)


def bracing_deg(P):
    return float(P.battle["bracing_attack_angle"])

