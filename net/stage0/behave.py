"""The scripted behaviours the drills are ranked with (net/config/stage0.json behaviours): each a rule of one level that
works the same in its drill and in a general battle of the stage.

    Behaviour            the chain as it is (the teacher: the formulas of all levels)
    Place(mode)          the strategist's placement (S0-1): after the formula, its whole formation shifted to the place
                         of the zone the mode likes - 'crest' (the ground crest_ahead_m ahead lowest under the line),
                         'high' (the line's ground highest), 'low' (lowest), 'front' (the zone's front middle)
    Flank(mode, delay)   the tactician's translator (T0-2): a free unit of mine (not in melee, not routing) within
                         flank_reach_m of an enemy another of mine has touched for delay s runs to a point by that
                         enemy - 'flank' (flank_gap_m out from its formation's edge, on my unit's side), 'rear'
                         (rear_gap_m behind it), 'front' (front_gap_m before it, beside the one touching: all in the
                         face) - then attacks it (within arrive_m of the point or after move_max_s); one per enemy;
                         'near' as 'flank' without waiting for a touch: the enemy within flank0_m of another of mine
    Reserve(mode, delay) the operative's reserve (O0-3): the named reserve (a drill's; in a general battle the rearmost
                         reserve_share units after the strategist, moved reserve_back_m behind the front unit) -
                         'chain' (the chain leads it), 'early' (attacks the nearest enemy from the start), 'never'
                         (holds the whole battle), 'break' (holds until my first rout + delay s, then attacks the enemy
                         nearest the routed unit's place)

second(chain, st) -> None or (orders, mask [B, N]) for this second: the orders of my units under the rule (merged over the
chain's); place(prep, out) - the strategist's plan changed in place; reserve_of(specs, places) - the named reserve.
"""
import torch

from sim import geometry, orders as O
from net.stage0 import cfg


def _bc():
    return cfg()["behaviours"]


class Behaviour:
    name = "teacher"
    level = None

    def place(self, prep, out):
        pass

    def wants_reserve(self):
        return False

    def start(self, chain, st):
        pass

    def second(self, chain, st):
        return None


def _out(u):
    return u["r"] | (u["men"] <= 0) | u["gone"]


def _dist(u):
    dx = u["x"][:, :, None] - u["x"][:, None, :]
    dz = u["z"][:, :, None] - u["z"][:, None, :]
    return torch.sqrt(dx * dx + dz * dz)


class Place(Behaviour):
    level = "strategist"

    def __init__(self, mode):
        assert mode in ("crest", "high", "low", "front"), mode
        self.mode, self.name = mode, mode

    def place(self, prep, out):
        from sim import terrain
        from net.strategist import zones as ZS
        plan = out["plan"]
        on = plan["on"]
        if not bool(on.any()):
            return
        zone = prep.mine_zone
        dev = on.device
        B, G = on.shape
        d = cfg()["drills"]["c01"]
        step = float(d["cell_m"])
        gs, gf = ZS.local(zone, plan["cx"], plan["cz"])                       # [B, G]
        w = on.float()
        k = w.sum(1, keepdim=True).clamp(min=1)
        cs, cf = (gs * w).sum(1, keepdim=True) / k, (gf * w).sum(1, keepdim=True) / k
        os_, of = gs - cs, gf - cf                                            # each group's offset from the middle
        hw, hd = zone["front"][:, None] / 2 - 25.0, zone["depth"][:, None] / 2 - 25.0
        smax = float(zone["front"].max()) / 2
        fmax = float(zone["depth"].max()) / 2
        S = torch.arange(-smax, smax + 1e-3, step, device=dev)
        F = torch.arange(-fmax, fmax + 1e-3, step, device=dev)
        ps = S[None, :, None].expand(len(F), len(S), 1).reshape(1, -1)        # [1, C]
        pf = F[:, None, None].expand(len(F), len(S), 1).reshape(1, -1)
        gs2 = ps[..., None] + os_[:, None, :]                                 # [B, C, G]
        gf2 = pf[..., None] + of[:, None, :]
        inside = ((gs2.abs() <= hw[..., None]) & (gf2.abs() <= hd[..., None])) | ~on[:, None, :]
        valid = inside.all(2)                                                 # [B, C]
        st = prep.st0
        u = st.u

        def h_at(s, f):
            x, z = ZS.world(zone, s.reshape(B, -1), f.reshape(B, -1))
            return terrain.height(u, x, z).reshape(s.shape)
        gs2b, gf2b = gs2.expand(B, -1, -1), gf2.expand(B, -1, -1)
        h0 = h_at(gs2b, gf2b)
        if self.mode == "crest":
            a1, a2 = (float(v) for v in d["crest_ahead_m"])
            sc = h0 - (h_at(gs2b, gf2b + a1) + h_at(gs2b, gf2b + a2)) / 2 + 0.01 * h0
        elif self.mode == "high":
            sc = h0
        elif self.mode == "low":
            sc = -h0
        else:
            sc = gf2b - 0.01 * gs2b.abs()
        score = (sc * w[:, None, :]).sum(2) / k
        score = torch.where(valid, score, torch.full_like(score, -1e9))
        best = score.argmax(1)
        ok = valid.any(1)
        ns, nf = ps[0][best], pf[0][best]
        ds = torch.where(ok, ns - cs[:, 0], torch.zeros_like(ns))
        df = torch.where(ok, nf - cf[:, 0], torch.zeros_like(nf))
        x, z = ZS.world(zone, gs + ds[:, None], gf + df[:, None])
        plan["cx"] = torch.where(on, x, plan["cx"])
        plan["cz"] = torch.where(on, z, plan["cz"])


def _hold_orders(B, N, dev):
    return O.hold(B, N, dev)


class Flank(Behaviour):
    level = "tactician"

    def __init__(self, mode, delay=5.0, name=None):
        assert mode in ("flank", "rear", "front", "near"), mode
        self.mode, self.delay = mode, float(delay)
        self.name = name or f"{mode}{int(delay)}"

    def start(self, chain, st):
        B, N = st.B, st.N
        dev = st.device
        self.task = torch.full((B, N), -1, dtype=torch.long, device=dev)
        self.sign = torch.ones((B, N), device=dev)
        self.phase = torch.zeros((B, N), dtype=torch.bool, device=dev)
        self.t0 = torch.zeros((B, N), device=dev)
        self.pin_t = torch.full((B, N), -1.0, device=dev)
        self.spacing = chain.setup.P.sim["formation"]["spacing_m"]

    def second(self, chain, st):
        c = _bc()
        u = st.u
        B, N = st.B, st.N
        t = st.t.float()[:, None].expand(B, N)
        side = u["side"]
        out = _out(u)
        mine = (side == chain.side) & ~u["lord"] & ~out
        seen = u["vis"].bool() if "vis" in u else torch.ones_like(out)
        foe = (side > 0) & (side != chain.side) & ~u["lord"] & ~out & seen
        hs = u["hit_sec"].view(B, N, N)
        pinned = ((hs > 0) & foe[:, :, None] & mine[:, None, :]).any(2)       # [B, enemy]
        self.pin_t = torch.where(pinned, torch.where(self.pin_t < 0, t, self.pin_t), torch.full_like(self.pin_t, -1.0))
        d = _dist(u)                                                          # [B, i, j]
        if self.mode == "near":
            near = ((d <= float(c["flank0_m"])) & foe[:, :, None] & mine[:, None, :]).any(2)
            ready = foe & near
        else:
            ready = foe & pinned & (t - self.pin_t >= self.delay)
        # tasks end: the target or the flanker out
        tg = self.task.clamp(min=0)
        done = (self.task >= 0) & (~mine | ~foe.gather(1, tg))
        self.task = torch.where(done, torch.full_like(self.task, -1), self.task)
        has = torch.zeros((B, N), dtype=torch.bool, device=st.device)        # [B, enemy] already has a flanker
        for j in range(N):
            tj = self.task[:, j]
            has[torch.arange(B, device=st.device)[tj >= 0], tj[tj >= 0]] = True
        free = mine & ~u["m"] & (self.task < 0)
        reach = float(c["flank_reach_m"])
        right_x, right_z = torch.cos(torch.deg2rad(u["b"])), -torch.sin(torch.deg2rad(u["b"]))
        for i in range(N):
            want = ready[:, i] & ~has[:, i]
            if not bool(want.any()):
                continue
            di = torch.where(free, d[:, i, :], torch.full_like(d[:, i, :], 1e9))
            j = di.argmin(1)
            ok = want & (di.min(1).values <= reach)
            if not bool(ok.any()):
                continue
            bi = torch.arange(B, device=st.device)[ok]
            jj = j[ok]
            self.task[bi, jj] = i
            self.phase[bi, jj] = False
            self.t0[bi, jj] = t[bi, jj]
            sx = (u["x"][bi, jj] - u["x"][bi, i]) * right_x[bi, i] + (u["z"][bi, jj] - u["z"][bi, i]) * right_z[bi, i]
            self.sign[bi, jj] = torch.where(sx >= 0, 1.0, -1.0)
            free[bi, jj] = False
        act = self.task >= 0
        if not bool(act.any()):
            return None
        tg = self.task.clamp(min=0)
        front, depth = geometry.dims(u, self.spacing)
        ex, ez, eb = u["x"].gather(1, tg), u["z"].gather(1, tg), torch.deg2rad(u["b"].gather(1, tg))
        ef, ed = front.gather(1, tg), depth.gather(1, tg)
        fx, fz = torch.sin(eb), torch.cos(eb)
        rx, rz = torch.cos(eb), -torch.sin(eb)
        mode = "flank" if self.mode == "near" else self.mode
        if mode == "flank":
            off = ef / 2 + float(c["flank_gap_m"])
            px, pz = ex + self.sign * rx * off, ez + self.sign * rz * off
        elif mode == "rear":
            off = ed / 2 + float(c["rear_gap_m"])
            px, pz = ex - fx * off, ez - fz * off
        else:
            off = ed / 2 + float(c["front_gap_m"])
            px, pz = ex + fx * off + self.sign * rx * ef / 4, ez + fz * off + self.sign * rz * ef / 4
        dp = torch.sqrt((u["x"] - px) ** 2 + (u["z"] - pz) ** 2)
        arrive = (dp <= float(c["arrive_m"])) | u["m"] | (t - self.t0 >= float(c["move_max_s"]))
        self.phase = self.phase | (act & arrive)
        o = _hold_orders(B, N, st.device)
        att = act & self.phase
        mv = act & ~self.phase
        o.kind = torch.where(att, torch.full_like(o.kind, O.ATTACK), torch.where(mv, torch.full_like(o.kind, O.MOVE),
                                                                                  o.kind))
        o.x, o.z = torch.where(mv, px, o.x), torch.where(mv, pz, o.z)
        o.target = torch.where(att, tg, o.target)
        o.run = act.clone()
        return o, act


class Reserve(Behaviour):
    level = "operative"

    def __init__(self, mode, delay=0.0, name=None):
        assert mode in ("chain", "early", "never", "break"), mode
        self.mode, self.delay = mode, float(delay)
        self.name = name or (f"res_{mode}" + (f"{int(delay)}" if mode == "break" and delay else ""))

    def wants_reserve(self):
        return True

    def start(self, chain, st):
        B, N = st.B, st.N
        self.t_trig = torch.full((B,), -1.0, device=st.device)
        self.px = torch.zeros(B, device=st.device)
        self.pz = torch.zeros(B, device=st.device)

    def second(self, chain, st):
        res = getattr(chain, "reserve", None)
        if res is None or not bool(res.any()) or self.mode == "chain":
            return None
        u = st.u
        B, N = st.B, st.N
        side = u["side"]
        out = _out(u)
        t = st.t.float()
        mine = (side == chain.side) & ~u["lord"]
        line = mine & ~res
        seen = u["vis"].bool() if "vis" in u else torch.ones_like(out)
        foe = (side > 0) & (side != chain.side) & ~u["lord"] & ~out & seen
        act = res & ~out
        o = _hold_orders(B, N, st.device)
        if self.mode == "never":
            return o, act
        d = _dist(u)
        if self.mode == "early":
            dd = torch.where(foe[:, None, :], d, torch.full_like(d, 1e9))
            tgt = dd.argmin(2)
            go = act & (dd.min(2).values < 1e8)
            o.kind = torch.where(go, torch.full_like(o.kind, O.ATTACK), o.kind)
            o.target = torch.where(go, tgt, o.target)
            o.run = go & (dd.min(2).values < 60.0)
            return o, act
        routed = (out & line).any(1)
        new = routed & (self.t_trig < 0)
        if bool(new.any()):
            w = (out & line).float()
            k = w.sum(1).clamp(min=1)
            self.px = torch.where(new, (u["x"] * w).sum(1) / k, self.px)
            self.pz = torch.where(new, (u["z"] * w).sum(1) / k, self.pz)
            self.t_trig = torch.where(new, t, self.t_trig)
        go_b = (self.t_trig >= 0) & (t - self.t_trig >= self.delay)
        de = torch.sqrt((u["x"] - self.px[:, None]) ** 2 + (u["z"] - self.pz[:, None]) ** 2)
        de = torch.where(foe, de, torch.full_like(de, 1e9))
        tgt = de.argmin(1)                                                    # [B] the enemy nearest the breach
        has_t = de.min(1).values < 1e8
        go = act & (go_b & has_t)[:, None]
        o.kind = torch.where(go, torch.full_like(o.kind, O.ATTACK), o.kind)
        o.target = torch.where(go, tgt[:, None].expand(B, N), o.target)
        o.run = go
        return o, act


def by_name(name):
    """The behaviour of a name (rank.py's lists)."""
    if name in ("teacher", "chain"):
        return Behaviour()
    if name in ("crest", "high", "low", "front"):
        return Place(name)
    if name.startswith("res_"):
        m = name[4:]
        if m.startswith("break"):
            return Reserve("break", float(m[5:] or 0.0), name=name)
        return Reserve(m, name=name)
    for mode in ("flank", "rear", "front", "near"):
        if name.startswith(mode) and name[len(mode):].isdigit():
            return Flank(mode, float(name[len(mode):]), name=name)
    if name == "infront":
        return Flank("front", 5.0, name=name)
    raise KeyError(name)


SETS = {"c01": ("teacher", "crest", "high", "low", "front"),
        "t02": ("teacher", "infront", "flank5", "near0", "rear5", "flank30"),
        "o03": ("res_chain", "res_early", "res_never", "res_break", "res_break40")}
