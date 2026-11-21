"""The enemy of drill O0-1 'advance on the holder' in the simulator (tmp/op1/design_merged.md FINAL: "the simulator's
defender script for the drill: stands until our front is within its sally gap (draw from the observed 60-100 m), then
all attack"). Scripts with the interface of sim/scripts.py's enemy scripts (policy(st) -> Orders for every unit; the
caller keeps its side's), kept here, not in sim/ (sim/'s files are the exams' key).

    Sally(gap_m)      [B] m: each battle's enemy (side 2) holds until the least edge gap between any of its standing
                      units (its lord out) and any standing unit of the other side (its lord out) is gap_m or less,
                      then every unit of it attacks the nearest standing enemy at a run (sim/replay.nearest_attack) for
                      the rest of the battle; latched per battle (fired [B], fired_s [B] the second it fired)
    WalkIn(goal, until_s, then, pose, v_run)   the come-twin's enemy: until until_s each unit MOVEs towards its goal
                      (x, z [B, N] - a point beyond the cause scene's pose on its line of march), at a run while it is
                      behind its schedule (the distance left to its pose along its march over v_run [B, N] x the time
                      left), else at a walk - so it passes the pose (pose x, z [B, N]) at the decision tick, moving; then
                      the script `then` ('nearest' | 'human' - Human.attack as the attacker)
    Enemy(specs, device)   the enemy side of a batch of drill specs (spec['o1']['enemy'] 'sally' | 'walkin' | 'hold'),
                      its lord holding - duel.Script's interface (start, orders(st, j))
"""
import torch

from sim import geometry
from sim import orders as O
from sim import replay
from sim import scripts as SS


class Sally:
    def __init__(self, gap_m, side=2):
        self.gap = gap_m
        self.side = side
        self.fired = torch.zeros_like(gap_m, dtype=torch.bool)
        self.fired_s = torch.full_like(gap_m, float("nan"))
        self.spacing = None

    def __call__(self, st):
        u = st.u
        if self.spacing is None:
            from sim.params import load
            self.spacing = load().sim["formation"]["spacing_m"]
        gone = u["r"] | (u["men"] <= 0) | u["gone"] | (u["side"] <= 0)
        mine = (u["side"] == self.side) & ~u["lord"] & ~gone
        other = (u["side"] != self.side) & ~u["lord"] & ~gone
        front, depth = geometry.dims(u, self.spacing)
        gp = geometry.rect_gap(u["x"], u["z"], u["b"], front, depth)                    # [B, N, N]
        big = torch.full_like(gp, float("inf"))
        g = torch.where(mine[:, :, None] & other[:, None, :], gp, big).flatten(1).amin(1)
        new = ~self.fired & (g <= self.gap)
        t = st.t.float()
        self.fired_s = torch.where(new, t, self.fired_s)
        self.fired = self.fired | new
        o = replay.nearest_attack(st)
        return O.merge(O.hold(st.B, st.N, st.device), o, self.fired[:, None].expand(st.B, st.N))


class WalkIn:
    def __init__(self, gx, gz, until_s, then="nearest", pose=None, v_run=None):
        self.gx, self.gz, self.until, self.then = gx, gz, float(until_s), then
        self.pose, self.v = pose, v_run

    def __call__(self, st):
        mv = O.hold(st.B, st.N, st.device)
        mv.kind[:] = O.MOVE
        mv.x, mv.z = self.gx.clone(), self.gz.clone()
        mv.run = torch.ones_like(mv.run)
        if self.pose is not None:
            u = st.u
            px, pz = self.pose
            dx, dz = self.gx - px, self.gz - pz
            n = torch.sqrt(dx * dx + dz * dz).clamp(min=1e-6)
            left = ((px - u["x"]) * dx + (pz - u["z"]) * dz) / n                    # (m to the pose along the march)
            t_left = (self.until - st.t.float())[:, None].clamp(min=0.0)
            mv.run = left >= self.v * t_left - 2.0                      # (2 m: the schedule's slack)
        if self.then == "human":
            att = st.attacker
            st.attacker = torch.full_like(att, 2)          # (Human.attack: the side that attacks - the enemy here)
            try:
                after = SS.human(st)
            finally:
                st.attacker = att
        else:
            after = SS.nearest(st)
        late = (st.t.float() >= self.until)[:, None].expand(st.B, st.N)
        return O.merge(mv, after, late)


class Enemy:
    """(module doc) Each battle's enemy by its spec: 'sally' (Sally at spec o1.sally_m), 'walkin' (WalkIn to the spec's
    goals until o1.t_dec, then o1.comer), 'hold' (sim/scripts hold: it never comes)."""

    def __init__(self, specs, device, per_side):
        B, N = len(specs), 2 * per_side
        self.side = 2
        kind = [s["o1"]["enemy"] for s in specs]
        self.m_sally = torch.tensor([k == "sally" for k in kind], device=device)
        self.m_walk = torch.tensor([k == "walkin" for k in kind], device=device)
        gap = torch.tensor([float(s["o1"].get("sally_m") or 0.0) for s in specs], device=device)
        self.sally = Sally(gap)
        from net.tactic import scenes as TS
        z = lambda: torch.zeros((B, N), device=device)                       # noqa: E731
        gx, gz, px, pz, v = z(), z(), z(), z(), z()
        for b, s in enumerate(specs):
            if s["o1"]["enemy"] != "walkin":
                continue
            for j, (g, p, k) in enumerate(zip(s["o1"]["goals"], s["o1"]["pose"], s["theirs"])):
                i = per_side + 1 + j
                gx[b, i], gz[b, i], px[b, i], pz[b, i] = float(g[0]), float(g[1]), float(p[0]), float(p[1])
                v[b, i] = float(TS._passports()[k]["speed"]["run"])
        until = max([float(s["o1"].get("t_dec", 0.0)) for s in specs] + [0.0])
        thens = {s["o1"].get("comer", "nearest") for s in specs if s["o1"]["enemy"] == "walkin"}
        self.walks = []
        for th in sorted(thens):
            m = torch.tensor([s["o1"]["enemy"] == "walkin" and s["o1"].get("comer", "nearest") == th for s in specs],
                             device=device)
            self.walks.append((m, WalkIn(gx, gz, until, th, (px, pz), v)))

    def start(self, *a):
        pass

    def orders(self, st, j):
        B, N = st.B, st.N
        o = O.hold(B, N, st.device)
        if bool(self.m_sally.any()):
            o = O.merge(o, self.sally(st), self.m_sally[:, None].expand(B, N))
        for m, w in self.walks:
            if bool(m.any()):
                o = O.merge(o, w(st), m[:, None].expand(B, N))
        lord = st.u["lord"] & (st.u["side"] == self.side)
        return O.merge(o, O.hold(B, N, st.device), lord)
