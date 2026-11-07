"""The levels under the operative (plan.md "Нижний уровень"): the tacticians of all groups and the frozen unit network.

    Tacticians   R = B x G rows, row b x G + g the group g of battle b (tactic/parcel.Parcel per row; an off row has no
                 slots). Every decide_s (5 s): the tiles kept between decisions (a unit that falls out takes away the
                 tile nearest its last place, a rallied one brings it back - tactic/env.TileKeeper's rule, kept per
                 slot so a row can start anew alone), then the frames: the tactician's formula (tactic/teacher.drive,
                 the default) or a TacticNet (the run4 network: the end-to-end run). A new parcel: the tactician
                 starts anew (new tiles, no frames); a move of the same members and form: the same tiles (holes too)
                 on the new centre and turn, the frames in the zone's axes go with it
    Units        the unit network's second (as tactic/env.play): the frames -> the units' areas (a shooter's fire
                 sector forward along the zone), the order with the stage by code, the latency, the leash
"""
import torch

from tools.nn.operative import cfg
from tools.nn.sim import orders as O
from tools.nn.tactic import cfg as tcfg
from tools.nn.tactic import decide as DC
from tools.nn.tactic import inputs as TIN
from tools.nn.tactic import parcel as PC
from tools.nn.tactic import raw as TR
from tools.nn.tactic import rows as RW
from tools.nn.tactic import teacher as TC
from tools.nn.tactic import uzone as U
from tools.nn.tactic import zone as ZN
from tools.nn.operative.groups import grid
from tools.nn.train import cadence
from tools.nn.unit import decode as Dc
from tools.nn.unit import net as Nt
from tools.nn.unit import train as UT
from tools.nn.unit import view as V

UNIT_CFG = {"cell_m": 8.0, "bounds": None}
FR_KEYS = ("s", "f", "hw", "hd")


class Tacticians:
    def __init__(self, B, N, G, device, net=None, greedy=False):
        R = B * G
        self.B, self.N, self.G, self.R = B, N, G, R
        self.dev = device
        self.net, self.greedy = net, greedy
        z = lambda dt=torch.float32: torch.zeros(R, dtype=dt, device=device)          # noqa: E731
        self.bi = torch.arange(B, device=device).repeat_interleave(G)
        self.gi = torch.arange(G, device=device).repeat(B)
        self.active = z(torch.bool)
        self.mine = torch.zeros((R, N), dtype=torch.bool, device=device)
        self.target = torch.zeros((R, N), dtype=torch.bool, device=device)
        self.cx, self.cz, self.heading = z(), z(), z()
        self.line = z(torch.bool)
        self.rows, self.cols = z(torch.long) + 1, z(torch.long) + 1
        self.tmask = torch.zeros((R, ZN.MAX_TILES), dtype=torch.bool, device=device)
        self.order = torch.zeros((R, len(DC.VERBS) + len(DC.PACES)), device=device)
        self.fr = {k: torch.zeros((R, N), device=device) for k in FR_KEYS}
        self.fr_on = torch.zeros((R, N), dtype=torch.bool, device=device)
        self.out = torch.zeros((R, N), dtype=torch.bool, device=device)
        self.last = torch.zeros((R, N, 2), device=device)
        self.tile = torch.zeros((R, N), dtype=torch.long, device=device)
        self.fell = 0.0
        self.ticks = 0

    def parcel(self, idx=None):
        idx = torch.arange(self.R, device=self.dev) if idx is None else idx
        h = self.heading[idx] * TR.DEG
        return PC.Parcel(self.bi[idx], self.mine[idx], self.target[idx], self.cx[idx], self.cz[idx], torch.sin(h),
                         torch.cos(h), self.line[idx], self.gi[idx])

    def tiles(self, idx):
        return ZN.Tiles(self.cx[idx], self.cz[idx], self.heading[idx], self.rows[idx], self.cols[idx], self.tmask[idx])

    def apply(self, groups, task, moved, raw):
        """The operative's decision: groups (groups.Groups after), task [B, G] (a new parcel), moved [B, G] (the same
        tiles may stay). Returns the parcels sent (tactic Parcel of the rows with a task)."""
        groups = groups.padded(self.G)                # (fewer groups than places: the strategist's 6)
        task = torch.cat([task, torch.zeros((task.shape[0], self.G - task.shape[1]), dtype=torch.bool,
                                            device=task.device)], 1)
        moved = torch.cat([moved, torch.zeros((moved.shape[0], self.G - moved.shape[1]), dtype=torch.bool,
                                              device=moved.device)], 1)
        alive = (raw[..., TR.I["alive"]] > 0)
        t = task.reshape(-1)
        mv = moved.reshape(-1) & t
        fresh = t & ~mv
        mem = groups.members().reshape(self.R, self.N) & alive[self.bi]
        self.active = groups.on.reshape(-1)
        self.mine = torch.where(t[:, None], mem, self.mine & self.active[:, None])
        self.target = torch.where(t[:, None], groups.target.reshape(self.R, self.N), self.target)
        for k in ("cx", "cz", "heading"):
            setattr(self, k, torch.where(t, getattr(groups, k).reshape(-1), getattr(self, k)))
        self.line = torch.where(t, groups.line.reshape(-1), self.line)
        rows, cols = grid(groups.n0.reshape(-1), groups.line.reshape(-1))
        self.rows = torch.where(fresh, rows, self.rows)
        self.cols = torch.where(fresh, cols, self.cols)
        full = torch.arange(ZN.MAX_TILES, device=self.dev)[None] < (rows * cols)[:, None]
        self.tmask = torch.where(fresh[:, None], full, self.tmask)
        o = DC.order_vec(self.R, "hold", "any", self.dev)
        vb = groups.verb.reshape(-1)
        pc = groups.pace.reshape(-1)
        o = torch.zeros_like(o)
        o[torch.arange(self.R, device=self.dev), vb] = 1.0
        o[torch.arange(self.R, device=self.dev), len(DC.VERBS) + pc] = 1.0
        self.order = torch.where(t[:, None], o, self.order)
        self.fr_on = self.fr_on & ~fresh[:, None]
        self.out = self.out & ~fresh[:, None]
        idx = t.nonzero()[:, 0]
        return self.parcel(idx), idx

    def _keep(self, raw, idx):
        """The tiles between decisions of the rows idx (module doc); writes tmask back."""
        I = TR.I
        tiles = self.tiles(idx)
        S = raw[self.bi[idx]]
        mine = self.mine[idx]
        able = (S[..., I["alive"]] > 0) & ~(S[..., I["r"]] > 0) & mine
        x, z = S[..., I["x"]], S[..., I["z"]]
        last = self.last[idx]
        last = torch.where(able[..., None], torch.stack([x, z], -1), last)
        out = self.out[idx]
        tile = self.tile[idx]
        for j in range(self.N):
            new = mine[:, j] & ~able[:, j] & ~out[:, j]
            if bool(new.any()):
                before = tiles.mask.clone()
                tiles.drop(new, last[:, j, 0], last[:, j, 1])
                gone = before & ~tiles.mask
                tile[:, j] = torch.where(new, gone.float().argmax(1), tile[:, j])
                out[:, j] = out[:, j] | new
            back = mine[:, j] & able[:, j] & out[:, j]
            if bool(back.any()):
                m = tiles.mask.clone()
                k = torch.arange(idx.shape[0], device=self.dev)
                m[k[back], tile[back, j]] = True
                tiles.mask = m
                out[:, j] = out[:, j] & ~back
        self.last[idx], self.out[idx], self.tile[idx] = last, out, tile
        self.tmask[idx] = tiles.mask
        return tiles

    def tick(self, raw, card, bounds=None):
        """The tacticians' decision (every 5 s): the frames of the active rows (per slot)."""
        idx = (self.active & self.mine.any(1)).nonzero()[:, 0]
        if idx.numel() == 0:
            return
        tiles = self._keep(raw, idx)
        p = self.parcel(idx)
        my = DC.my_slots(p)                                                   # [r, Gm]
        r = idx.shape[0]
        k = torch.arange(r, device=self.dev)[:, None]
        s0 = my.clamp(min=0)
        prev = {n: self.fr[n][idx][k, s0] for n in FR_KEYS}
        prev["on"] = self.fr_on[idx][k, s0] & (my >= 0)
        has_prev = bool(prev["on"].any())
        K = min(int(tcfg()["rows"]), self.N)
        if self.net is None:
            rr = RW.build(raw, card, p, tiles, rows=K)
            ctx = DC.ctx_of(rr, p, tiles, prev if has_prev else None)
            out = TC.drive(ctx.mine_raw, ctx.able, tiles)
        else:
            areas = U.world(prev, tiles) if has_prev else None
            inp = TIN.build(raw, card, p, tiles, areas, bounds=bounds, rows=K)
            out, _, ctx = self.net(inp, self.order[idx], p, tiles, prev=prev if has_prev else None, greedy=self.greedy)
        self.fell += float(out["fell"].sum())
        self.ticks += r
        new = out["frames"]
        on = new["on"] & (my >= 0)
        for n in FR_KEYS:
            v = torch.where(on, new[n], prev[n])
            cur = self.fr[n][idx]
            cur[k.expand_as(s0)[my >= 0], s0[my >= 0]] = v[my >= 0]
            self.fr[n][idx] = cur
        cur = self.fr_on[idx]
        cur[k.expand_as(s0)[my >= 0], s0[my >= 0]] = (on | prev["on"])[my >= 0]
        self.fr_on[idx] = cur

    def areas(self):
        """{x, z, b, front, depth, sb} [R, N] the frames in the world and on [R, N] (active rows, the members)."""
        tiles = ZN.Tiles(self.cx, self.cz, self.heading, self.rows, self.cols, self.tmask)
        w = U.world({**self.fr, "on": self.fr_on}, tiles)
        w["on"] = self.fr_on & self.mine & self.active[:, None]
        w["sb"] = self.heading[:, None].expand_as(w["x"])
        return w


class Units:
    """The frozen unit network's second for my units (as tactic/env.play)."""

    def __init__(self, setup, unit_net, unit_greedy=False):
        st, P, dev = setup.st, setup.P, setup.device
        B, N = st.B, st.N
        self.setup, self.net, self.greedy = setup, unit_net, unit_greedy
        self.spacing = P.sim["formation"]["spacing_m"]
        self.leash = Dc.Leash(B, N, dev)
        self.wlim = Dc.width_limits(st, P)
        self.cad = cadence.Cadence()
        self.ubuf = torch.zeros((V.T + 1, B, N, len(V.RAW)), device=dev)
        self.in_force = O.hold(B, N, dev)
        self.intended = O.hold(B, N, dev)
        own = setup.own
        u = st.u
        # before the first frames: a box of 48 m where each unit stands (hold in place)
        self.area = {"x": u["x"].clone(), "z": u["z"].clone(), "b": u["b"].clone(),
                     "front": torch.full((B, N), 48.0, device=dev), "depth": torch.full((B, N), 48.0, device=dev),
                     "sb": u["b"].clone(), "sw": torch.zeros((B, N), device=dev), "on": own.clone()}
        self.arrived = torch.zeros((B, N), dtype=torch.bool, device=dev)
        self.pace = torch.full((B, N), Nt.PACE_ANY, dtype=torch.long, device=dev)
        self.bi, self.ii = own.nonzero(as_tuple=True)
        self.n = 0

    def write(self, tac):
        """The tacticians' frames -> the units' areas (only the units with a frame; the others keep theirs)."""
        w = tac.areas()
        sector = float(tcfg()["sector_deg"])
        B, N = self.setup.st.B, self.setup.st.N
        G = tac.G
        on = w["on"].view(B, G, N)
        any_on = on.any(1)
        gsel = on.float().argmax(1)                                           # [B, N] the row of each unit's frame
        moved = torch.zeros((B, N), dtype=torch.bool, device=self.setup.device)
        for key in ("x", "z", "b", "front", "depth", "sb"):
            v = w[key].view(B, G, N).gather(1, gsel[:, None])[:, 0]
            if key in ("x", "z", "front", "depth"):
                moved = moved | (any_on & ((v - self.area[key]).abs() > 1e-3))
            self.area[key] = torch.where(any_on, v, self.area[key])
        self.area["sw"] = torch.where(any_on & self.setup.shooter, torch.full_like(self.area["sw"], sector),
                                      self.area["sw"])
        self.arrived = self.arrived & ~moved

    def second(self, st):
        """The units' decision of this second -> intended orders; returns them (the latency: cadence)."""
        P = self.setup.P
        raw_u = V.raw(st, self.spacing, leash=self.leash.on, missile=P.sim["missile"])
        self.ubuf = torch.cat([self.ubuf[1:], raw_u[None]])
        u = st.u
        own = self.setup.own
        ar = {k: self.area[k] for k in ("x", "z", "b", "front", "depth")}
        inside = Dc.signed_dist(u["x"], u["z"], ar) >= 0
        self.arrived = self.arrived | (inside & own)
        B, N = st.B, st.N
        bi, ii = self.bi, self.ii
        live = ~st.done[bi] & UT.viewer_ok(st, bi, ii)
        bl, il = bi[live], ii[live]
        intended = O.hold(B, N, st.device)
        intended.kind[:] = O.KEEP
        if bl.numel():
            idx = torch.full_like(bl, V.T)
            v = UT.views(self.ubuf, idx, bl, il, self.setup.card, self.area, UNIT_CFG, wlim=self.wlim)
            ord_u = Nt.order_of(raw_u[bl, il], {k: a[bl, il] for k, a in self.area.items() if k != "on"},
                                self.pace[bl, il], self.arrived[bl, il])
            act = Nt.sample(self.net(v, ord_u, v["cell_m"]), greedy=self.greedy)
            intended = Dc.to_orders(act, v, raw_u, v["cell_m"], B, N, wlim=self.wlim)
            self.n += int(bl.numel())
        self.intended = intended
        return self.cad.delays(P.dt, B, st.device)

    def orders(self, st, j, lands, script):
        now_in = lands <= j
        mine = O.Orders(*(torch.where(now_in[:, None], getattr(self.intended, f), getattr(self.in_force, f))
                          for f in O.FIELDS))
        return O.merge(script(st), self.leash.apply(st, mine, self.area, self.setup.own), self.setup.own)

    def end_second(self):
        self.in_force = self.intended
