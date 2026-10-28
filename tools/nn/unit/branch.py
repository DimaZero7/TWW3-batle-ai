"""Branches on the approach (drills U1 and U3; the user's decisions 27.10): which way of meeting the enemy is better in
THIS skirmish? The skirmish is copied at the moment and played on by scripts, once per variant; a bonus (train.py
--branch-bonus) pays the network when its behaviour at the contact was the best variant, by its margin over the second
best (a small margin: almost nothing), fading over the updates.

The moment: an enemy's edge first comes within its charge distance (the database's charge_dist, 30 m for infantry) +
LEAD_S seconds of its run of our unit - the last moment our unit can still choose: a facing order takes 3.9 s to
re-form (config/nn/sim.json turn.face_delay_s), then it turns; later, standing means standing as it is. With two enemies
(U3) the first one's approach: the choice whom to face or charge is made then.

A variant (the enemies under `nearest` as before, the leash on): ("stand", j) HOLD facing enemy j (the bearing to it at
the moment; the simulator's brace and stand-first-strike rules work for a unit standing under HOLD with the enemy within
the database's bracing_attack_angle, 80 deg), ("attack", j) ATTACK enemy j at a run. Each until HORIZON_S after our
first contact with any enemy, our rout or death, every enemy dead or gone, or CAP_S after the moment. Its trade:
(enemy gold lost - ours) / budget from the moment (reward.measure).

The network's behaviour at its first contact (the main battle): ("attack", j) if its order in force attacks j; ("stand",
j) if HOLD, not moving (0.3 m/s) and j - of the enemies the one nearest its front's axis - within bracing_attack_angle
of its front; else OTHER (no bonus).

Kits: KIT1 (U1: one enemy - stand, attack: STAND 0, ATTACK 1), KIT_MULTI (U3: attack A, attack B, stand facing A, stand
facing B, and 'switch' - attack A, then, after our first contact, attack B once B is within its charge distance + LEAD_S
of its run (a leave from A's melee: the simulator lets a formation in melee neither turn to a facing - 2 deg/s - nor
fight two ways, so 'stand facing B' after A is not a variant); A the first enemy slot). The network 'did the switch'
when it attacked A at its first contact and its order turned to attacking B once B was within that moment (track()).
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
STAND, ATTACK, OTHER = 0, 1, -1
BRACE_DEG = 80.0     # the database's bracing_attack_angle (track()'s 'faces B'; Kit.behaviour takes it from params)


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


def enemies(st):
    """[B, N] our unit's (slot 0) enemies present at the start (side 2's slots with men)."""
    return (st.u["side"] == 2) & (st.u["men0"] > 0)


def edge_gaps(st, spacing):
    """[B, N] the edge-to-edge distance of our unit (slot 0) to every slot."""
    now = V.raw(st, spacing)
    return geometry.rect_gap(now[..., V.I["x"]], now[..., V.I["z"]], now[..., V.I["b"]], now[..., V.I["front"]],
                             now[..., V.I["depth"]])[:, 0]


def edge_gap(st, spacing):
    """[B] to the first enemy slot (H)."""
    return edge_gaps(st, spacing)[:, st.N // 2]


def contact(st):
    """[B] an enemy's contact is on our unit now (state hit_sec)."""
    N = st.N
    hs = st.u["hit_sec"].view(st.B, N, N)[:, :, 0]               # [B, j]: j's contact on us
    return ((hs > 0) & enemies(st)).any(1)


def moment(st, spacing, lead_s=LEAD_S):
    """[B] some standing enemy within its charge distance + lead_s of its run of our unit, ours standing, no contact."""
    u = st.u
    foe = enemies(st) & ~u["r"] & (u["men"] > 0) & ~u["gone"]
    reach = u["charge_dist"] + u["run"] * lead_s
    near = (foe & (edge_gaps(st, spacing) <= reach)).any(1)
    return near & ~u["r"][:, 0] & (u["men"][:, 0] > 0) & ~u["m"][:, 0] & ~contact(st) & ~st.done


def _rel(u, j):
    """[B] the bearing of slot j [B] seen from our front, degrees -180..180."""
    x, z = u["x"].gather(1, j[:, None])[:, 0], u["z"].gather(1, j[:, None])[:, 0]
    return torch.remainder(torch.rad2deg(torch.atan2(x - u["x"][:, 0], z - u["z"][:, 0])) - u["b"][:, 0] + 180, 360) - 180


class Kit:
    """The variants [(kind, slot offset from H)], their scripts, the behaviour's classes."""

    def __init__(self, variants):
        self.variants = variants

    def names(self):
        if len(self.variants) <= 2:
            return [k for k, _ in self.variants]
        special = {"switch": "attack_A_then_B", "switch_face": "attack_A_then_face_B"}
        return [special.get(k, f"{k}_{'AB'[j]}") for k, j in self.variants]

    def switch_moment(self, st, spacing, lead_s=LEAD_S):
        """[B] B (slot H + 1) within its charge distance + lead_s of its run of our unit - the 'switch' variant's moment
        (the branches' own moment rule, for the second enemy)."""
        u = st.u
        Hb = st.N // 2 + 1
        reach = u["charge_dist"][:, Hb] + u["run"][:, Hb] * lead_s
        return (edge_gaps(st, spacing)[:, Hb] <= reach) & (u["men"][:, Hb] > 0) & ~u["gone"][:, Hb]

    def track(self, st, beh, spacing):
        """[B] the behaviour after the first contact: 'attack A at the contact' becomes 'attack A, then B' when the
        unit's order turns to B (an attack on it) once B is within its switch moment - the variant's own rule; the
        network decides once a second, its order lands 0.36 s later, so the window is open from B's moment on (until the
        branch's end), not one exact step. Without a switch variant: beh unchanged."""
        names = [k for k, _ in self.variants]
        if "switch" not in names and "switch_face" not in names:
            return beh
        u = st.u
        H = st.N // 2
        a_idx = self.variants.index(("attack", 0))
        moment = (beh == a_idx) & self.switch_moment(st, spacing)
        if "switch" in names:
            to_b = (u["order_kind"][:, 0] == O.ATTACK) & (u["order_target"][:, 0] == H + 1)
            beh = torch.where(moment & to_b, names.index("switch"), beh)
        if "switch_face" in names and "oface" in u:
            # ... or 'attack A, then face B': its order no attack any more and its facing (in force) within the
            # bracing angle of the way to B
            way_b = torch.remainder(torch.rad2deg(torch.atan2(u["x"][:, H + 1] - u["x"][:, 0],
                                                               u["z"][:, H + 1] - u["z"][:, 0])), 360.0)
            off = (torch.remainder(u["oface"][:, 0] - way_b + 180, 360) - 180).abs()
            face_b = (u["order_kind"][:, 0] != O.ATTACK) & (u["oface"][:, 0] >= 0) & (off <= BRACE_DEG)
            beh = torch.where(moment & face_b, names.index("switch_face"), beh)
        return beh

    def behaviour(self, st, bracing_deg):
        """[B] the index of the variant our unit's behaviour now is, OTHER none."""
        u = st.u
        H = st.N // 2
        kind = u["order_kind"][:, 0]
        tgt = u["order_target"][:, 0]
        speed = torch.sqrt(u["vx"][:, 0] ** 2 + u["vz"][:, 0] ** 2)
        foe = enemies(st) & (u["men"] > 0)
        rels = torch.stack([_rel(u, torch.full_like(tgt, H + j)) for _, j in self.variants], 1)       # [B, V]
        out = torch.full_like(kind, OTHER)
        # standing: facing the enemy nearest its front's axis (among the variants' enemies alive)
        alive = torch.stack([foe[:, H + j] for _, j in self.variants], 1)
        axis = torch.where(alive, rels.abs(), torch.full_like(rels, 1e9))
        still = (kind == O.HOLD) & (speed <= 0.3)
        for v, (k, j) in enumerate(self.variants):
            if k in ("switch", "switch_face"):
                continue                        # (known later: track())
            if k == "attack":
                hit = (kind == O.ATTACK) & (tgt == H + j)
            else:
                same = [w for w, (k2, j2) in enumerate(self.variants) if j2 == j]
                nearest = axis.argmin(1)[:, None] == torch.tensor(same, device=kind.device).view(1, -1)
                hit = still & nearest.any(1) & (rels[:, v].abs() <= bracing_deg) & alive[:, v]
            out = torch.where(hit & (out == OTHER), v, out)
        return out

    def play(self, snap, area, P, step, horizon_s=HORIZON_S, cap_s=CAP_S, rows=None, report=None):
        """[M, V] the variants' trades from the snapshots snap (M battles at their moments), area {k: [M, N]}. rows: pad
        the batch to this many rows per variant (a fixed shape for the compiled step). report: filled with the stand
        variants' check - braced at the contact (the behaviour of that variant) [M, V]."""
        M = snap.B
        dev = snap.device
        Vn = len(self.variants)
        pad = max(rows or M, M)
        idx = torch.cat([torch.arange(M, device=dev), torch.zeros(pad - M, dtype=torch.long, device=dev)])
        one = take(snap, idx)
        ar1 = {k: v[idx] for k, v in area.items()}
        st = cat([take(one, torch.arange(pad, device=dev)) for _ in range(Vn)])
        real = torch.cat([torch.arange(pad, device=dev) < M] * Vn)
        st.done = ~real
        ar = {k: torch.cat([v] * Vn) for k, v in ar1.items()}
        own = torch.zeros((st.B, st.N), dtype=torch.bool, device=dev)
        own[:, 0] = True
        H = st.N // 2
        u = st.u
        mine = O.hold(st.B, st.N, dev)
        var = torch.arange(Vn, device=dev).repeat_interleave(pad)
        for v, (k, j) in enumerate(self.variants):
            rows_v = var == v
            face = torch.remainder(torch.rad2deg(torch.atan2(u["x"][:, H + j] - u["x"][:, 0],
                                                             u["z"][:, H + j] - u["z"][:, 0])), 360.0)
            if k in ("attack", "switch", "switch_face"):
                mine.kind[rows_v, 0], mine.target[rows_v, 0], mine.run[rows_v, 0] = O.ATTACK, H + j, True
            else:
                mine.face[rows_v, 0] = face[rows_v]
        sw_v = [v for v, (k, _) in enumerate(self.variants) if k in ("switch", "switch_face")]
        sw_rows = torch.zeros(st.B, dtype=torch.bool, device=dev)
        sf_rows = torch.zeros(st.B, dtype=torch.bool, device=dev)
        for v in sw_v:
            sw_rows = sw_rows | (var == v)
            if self.variants[v][0] == "switch_face":
                sf_rows = sf_rows | (var == v)
        spacing = P.sim["formation"]["spacing_m"]
        leash = Dc.Leash(st.B, st.N, dev)
        m0 = reward.measure(st)
        t0 = st.t.clone()
        tc = torch.full_like(t0, -1.0)
        beh = torch.full((st.B,), OTHER, dtype=torch.long, device=dev)
        brace = bracing_deg(P)
        foe = enemies(st)
        while not bool(st.done.all()):
            if sw_v:
                # 'attack A, then B': after our first contact, at B's moment the attack turns to B (a leave from A's
                # melee by the simulator's rule: an attack on a target out of contact)
                # ('attack A, then face B': a hold facing B - in melee a facing order is a melee exit, 24 s without
                # blows at A, the turn after turn.face_melee_delay_s: contact.face_breakoff)
                now = sw_rows & (tc >= 0) & (mine.kind[:, 0] == O.ATTACK) & (mine.target[:, 0] == H) \
                    & self.switch_moment(st, spacing) & ~st.done
                uu = st.u
                way_b = torch.remainder(torch.rad2deg(torch.atan2(uu["x"][:, H + 1] - uu["x"][:, 0],
                                                                   uu["z"][:, H + 1] - uu["z"][:, 0])), 360.0)
                fnow = now & sf_rows
                mine.target[:, 0] = torch.where(now & ~sf_rows, H + 1, torch.where(fnow, -1, mine.target[:, 0]))
                mine.kind[:, 0] = torch.where(fnow, O.HOLD, mine.kind[:, 0])
                mine.run[:, 0] = mine.run[:, 0] & ~fnow
                mine.face[:, 0] = torch.where(fnow, way_b, mine.face[:, 0])
            o = O.merge(opponents.nearest(st), leash.apply(st, mine, ar, own), own)
            was = st.done.clone()
            step(st, o, P)
            u = st.u
            first = (tc < 0) & contact(st) & ~was
            beh = torch.where(first, self.behaviour(st, brace), beh)
            tc = torch.where(first, st.t, tc)
            gone = ~(foe & (u["men"] > 0) & ~u["gone"]).any(1)
            over = ((tc >= 0) & (st.t - tc >= horizon_s)) | (st.t - t0 >= cap_s) | u["r"][:, 0] | (u["men"][:, 0] <= 0) \
                | gone
            st.done = was | over
        m1 = reward.measure(st)
        trade = (m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])
        trades = torch.stack([trade[v * pad:v * pad + M] for v in range(Vn)], 1)
        if report is not None:
            report["braced"] = torch.stack([beh[v * pad:v * pad + M] == v for v in range(Vn)], 1)
            report["contact"] = torch.stack([tc[v * pad:v * pad + M] >= 0 for v in range(Vn)], 1)
        return trades


KIT1 = Kit([("stand", 0), ("attack", 0)])                                   # STAND 0, ATTACK 1
KIT_MULTI = Kit([("attack", 0), ("attack", 1), ("stand", 0), ("stand", 1), ("switch", 0)])
# the exam's: with 'attack A, then face B' (possible since the facing in melee: contact.face_breakoff)
KIT_MULTI_EXAM = Kit([("attack", 0), ("attack", 1), ("stand", 0), ("stand", 1), ("switch", 0), ("switch_face", 0)])


def behaviour(st, bracing_deg):
    """U1: [B] STAND / ATTACK / OTHER of our unit now."""
    return KIT1.behaviour(st, bracing_deg)


def play(snap, area, P, step, horizon_s=HORIZON_S, cap_s=CAP_S, rows=None, report=None):
    """U1: (trade_stand [M], trade_attack [M]); report: stand_braced, stand_contact [M]."""
    rep = {}
    t = KIT1.play(snap, area, P, step, horizon_s, cap_s, rows, rep)
    if report is not None:
        report["stand_braced"], report["stand_contact"] = rep["braced"][:, STAND], rep["contact"][:, STAND]
    return t[:, STAND], t[:, ATTACK]


def best_of(trades):
    """[M] the best variant and [M] its margin over the second best (0 with one variant)."""
    top = trades.topk(min(2, trades.shape[1]), dim=1)
    best = top.indices[:, 0]
    margin = (top.values[:, 0] - top.values[:, 1]) if trades.shape[1] > 1 else torch.zeros_like(top.values[:, 0])
    return best, margin


def bonus_of(beh, trades, weight):
    """[M] weight x the best variant's margin where the behaviour is the best, else 0; and [M] the best."""
    best, margin = best_of(trades)
    return torch.where(beh == best, weight * margin, torch.zeros_like(margin)), best


def bonus(beh, t_stand, t_attack, weight):
    """U1: weight x |the two branches' trade difference| where the behaviour is the better one, else 0; and the best."""
    best = torch.where(t_stand >= t_attack, STAND, ATTACK)
    return torch.where(beh == best, weight * (t_stand - t_attack).abs(), torch.zeros_like(t_stand)), best


def weight_at(w0, update, decay_updates):
    """The bonus's weight after `update` updates: w0, falling linearly to 0 over decay_updates (0: never)."""
    if decay_updates <= 0:
        return w0
    return w0 * max(0.0, 1.0 - update / decay_updates)


def bracing_deg(P):
    return float(P.battle["bracing_attack_angle"])
