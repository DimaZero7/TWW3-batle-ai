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
import math

import torch

from net.operative import cfg
from sim import orders as O
from net.tactic import cfg as tcfg
from net.tactic import decide as DC
from net.tactic import inputs as TIN
from net.tactic import parcel as PC
from net.tactic import raw as TR
from net.tactic import rows as RW
from net.tactic import teacher as TC
from net.tactic import uzone as U
from net.tactic import zone as ZN
from net.operative.groups import grid
from net.common import cadence
from net.unit import decode as Dc
from net.unit import net as Nt
from net.unit import train as UT
from net.unit import view as V

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
        self.seed = torch.full((R,), -1, dtype=torch.long, device=device)        # the target's seed slot (-1 none)
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
        self.seed = torch.where(t, groups.seed.reshape(-1), self.seed)
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

    def __init__(self, setup, unit_net, unit_greedy=False, targets=False, still=False, aim=False):
        st, P, dev = setup.st, setup.P, setup.device
        B, N = st.B, st.N
        self.setup, self.net, self.greedy = setup, unit_net, unit_greedy
        # the group's target to its units (07.11, sim/evidence/diag_m1/night.md): each unit's main target - the order's
        # has_target and the 'my_target' flag of that enemy's row (drill U8: the unit attacks it while allowed) - the
        # target unit of its group nearest to it; off: no target named (as before)
        self.targets = targets
        self.prio = torch.full((B, N), -1, dtype=torch.long, device=dev)
        # 'stand still in the frame' (07.11, the user's 'no fuss'; sim/evidence/diag_m1/night.md): a move of a unit already in
        # its area to a point in that area is a stand (turning to the move's facing) - a quiet battle walked 200-250 m
        # a unit in 120 s inside frames that never moved (the unit network's 'where' wanders over its cells)
        self.still = still
        # 'a shooter shoots its target' (08.11, sim/evidence/diag_m1/night.md 18): the game's shooter firing at will never
        # turns to its target (sim/config/sim.json missile.stand_fire_why: only an attack order turns it), so a gun that
        # walked to its place facing the walk stood loaded 240 s with its target behind it (the bridge base: a mortar
        # 0 shots of 88). A ranged unit standing in its area out of melee, a main target (prio) within its range ->
        # ATTACK on it (it turns and fires there); without a main target, the target it took itself out of its arc ->
        # ATTACK on that. A withdraw or a move out of the area of the unit network stays.
        self.aim = aim
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
        if self.targets:
            self.prio = self.prio_of(tac)

    def prio_of(self, tac):
        """[B, N] each own unit's main target: the alive, seen unit of its group's target nearest to it (-1 none)."""
        st = self.setup.st
        u = st.u
        B, N = st.B, st.N
        G = tac.G
        mem = (tac.mine & tac.active[:, None]).view(B, G, N)                  # [B, G, N] my units of each group
        tgt = tac.target.view(B, G, N) & tac.active.view(B, G)[..., None]
        ok = (u["men"] > 0) & ~u["gone"] & u["vis"].bool() & ~u["r"]
        tgt = tgt & ok[:, None, :]
        d = (u["x"][:, :, None] - u["x"][:, None, :]) ** 2 + (u["z"][:, :, None] - u["z"][:, None, :]) ** 2   # [B, i, j]
        # the unit's group's target (each unit in one group at most)
        unit_t = torch.einsum("bgn,bgm->bnm", mem.float(), tgt.float()) > 0                                # [B, i, j]
        dd = torch.where(unit_t, d, torch.full_like(d, float("inf")))
        best = dd.argmin(2)
        best = torch.where(torch.isfinite(dd.min(2).values), best, torch.full_like(best, -1))
        # the group's seed first while it stands and is seen (the operative named it: a focus - human_rules 2.6)
        seed = tac.seed.view(B, G)
        s_ok = (seed >= 0) & ok.gather(1, seed.clamp(min=0)) & tac.active.view(B, G)
        unit_seed = torch.einsum("bgn,bg->bn", mem.float(), torch.where(s_ok, seed + 1, torch.zeros_like(seed)).float())
        us = unit_seed.round().long() - 1                                                                   # [B, N]
        return torch.where(us >= 0, us, best)

    def stand_still(self, o, st, inside):
        """A MOVE inside the unit's own area to a point inside it -> HOLD with the move's facing (module: still)."""
        ar = {k: self.area[k] for k in ("x", "z", "b", "front", "depth")}
        to_in = Dc.signed_dist(o.x, o.z, ar) >= 0
        quiet = (o.kind == O.MOVE) & inside & to_in & self.setup.own
        if not bool(quiet.any()):
            return o
        kind = torch.where(quiet, torch.full_like(o.kind, O.HOLD), o.kind)
        return O.Orders(kind, o.x, o.z, o.target, o.run & ~quiet, o.ability, o.cast_target, o.face, o.width)

    def stay_rule(self, o, st, inside):
        """The rule 'shooters, guns and the lord stay in their area' (08.11, the bridge's game battles: a catapult 138 of
        138 s out of its area under 91 attack orders on a far target, 100 s in melee, 0 shots; sim/evidence/diag_m1/night.md 29)
        on the intended orders o, for my units out of melee and not routing:
        * a shooter's ATTACK on a target beyond its range: out of its area - a MOVE to the area's centre (at the order's
          pace), in it - a HOLD (the unit network's attack walks it at the target);
        * a gun (an artillery unit: guns of its own) goes only to its area and stands: out of it any MOVE / ATTACK is a
          MOVE to the area's centre, in it a MOVE out of it or an ATTACK beyond range is a HOLD;
        * the lord without a main target (the army's formula names one only when the enemy's front wavers): an ATTACK
          on an enemy farther than lord_guard_m - in his area a HOLD, out of it a MOVE to the area's centre.
        A WITHDRAW stays (the unit network's step back from melee)."""
        u = st.u
        own = self.setup.own
        ar = {k: self.area[k] for k in ("x", "z", "b", "front", "depth")}
        free = own & (u["men"] > 0) & ~u["r"] & (u["contact_s"] <= 0)
        tc = o.target.clamp(min=0)
        d = torch.sqrt((u["x"].gather(1, tc) - u["x"]) ** 2 + (u["z"].gather(1, tc) - u["z"]) ** 2)
        att = (o.kind == O.ATTACK) & (o.target >= 0)
        out_pt = (o.kind == O.MOVE) & (Dc.signed_dist(o.x, o.z, ar) < 0)
        ranged = u["range"] > 0
        gun = ranged & (u["eng0"] > 0) if "eng0" in u else torch.zeros_like(ranged)
        far = att & (d > u["range"])
        guard = float(cfg().get("lord_guard_m") or 0.0)
        lord = u["lord"] & ~ranged & (self.prio < 0) & (guard > 0)        # (lord_guard_m 0: the lord's part off)
        direct = ranged & u["direct"].bool() & ~gun if "direct" in u and cfg().get("direct_line", True)             else torch.zeros_like(ranged)
        # (08.11, the bridge's games: a direct-fire sniper ordered to attack a target in range walked through our line
        # to an open line of fire, into melee - out of its area it only goes to the area)
        to_area = free & ~inside & ((ranged & far) | (direct & att) | (gun & (att | (o.kind == O.MOVE)))
                                    | (lord & att & (d > guard)))
        stand = free & inside & ~to_area & ((ranged & far) | (gun & out_pt) | (lord & att & (d > guard)))
        if not bool((to_area | stand).any()):
            return o
        kind = torch.where(to_area, torch.full_like(o.kind, O.MOVE), torch.where(stand, torch.full_like(o.kind, O.HOLD),
                                                                                  o.kind))
        x = torch.where(to_area, self.area["x"], torch.where(stand, u["x"], o.x))
        z = torch.where(to_area, self.area["z"], torch.where(stand, u["z"], o.z))
        tg = torch.where(to_area | stand, torch.full_like(o.target, -1), o.target)
        self.n_stay = getattr(self, "n_stay", 0) + int((to_area | stand).sum())
        return O.Orders(kind, x, z, tg, o.run & ~stand, o.ability, o.cast_target, o.face, o.width)

    def aim_rule(self, o, st, inside):
        """The rule 'a shooter shoots its target' (module: aim) on the intended orders o."""
        u = st.u
        own = self.setup.own
        ar = {k: self.area[k] for k in ("x", "z", "b", "front", "depth")}
        stays = ((o.kind == O.HOLD) | (o.kind == O.KEEP) | (o.kind == O.ATTACK)
                 | ((o.kind == O.MOVE) & (Dc.signed_dist(o.x, o.z, ar) >= 0)))
        ok_u = own & inside & (u["range"] > 0) & (u["men"] > 0) & ~u["r"] & (u["contact_s"] <= 0) & stays
        if not bool(ok_u.any()):
            return o
        # (a target in melee is not ordered: our shooters hit their own there - can_damage_allies; fire at will decides)
        alive = (u["men"] > 0) & ~u["gone"] & u["vis"].bool() & ~u["r"] & (u["contact_s"] <= 0)

        def dist_to(t):
            tc = t.clamp(min=0)
            return torch.sqrt((u["x"].gather(1, tc) - u["x"]) ** 2 + (u["z"].gather(1, tc) - u["z"]) ** 2)
        p = self.prio if self.targets else torch.full_like(self.prio, -1)
        p_ok = (p >= 0) & alive.gather(1, p.clamp(min=0)) & (dist_to(p) <= u["range"])
        a = u["aim_tgt"]
        tc = a.clamp(min=0)
        brg = torch.rad2deg(torch.atan2(u["x"].gather(1, tc) - u["x"], u["z"].gather(1, tc) - u["z"]))
        off = (torch.remainder(brg - u["b"] + 180.0, 360.0) - 180.0).abs()
        a_ok = (a >= 0) & alive.gather(1, tc) & (off > u["arc"]) & (dist_to(a) <= u["range"])
        p_ok = p_ok & ~self.bad_fast(st, p)
        a_ok = a_ok & ~self.bad_fast(st, a)
        t = torch.where(p_ok, p, torch.where(a_ok, a, torch.full_like(a, -1)))
        go = ok_u & (t >= 0)
        # a direct-fire shooter: only with its line of fire open (no own unit in the band to the target as wide as its
        # front), else it stands - an attack order makes the game walk it to an open line, through our line (08.11)
        direct = (u["direct"].bool() & (u["eng0"] <= 0)) if "direct" in u and cfg().get("direct_line", True)             else torch.zeros_like(go)
        own_att = own & inside & (u["range"] > 0) & direct & (u["contact_s"] <= 0) & (o.kind == O.ATTACK) & (o.target >= 0)
        tt = torch.where(go, t, torch.where(own_att, o.target, torch.full_like(t, -1)))
        chk = (go | own_att) & direct
        closed = self.line_closed(st, tt) & chk
        go = go & ~closed
        stand = own_att & closed & ~go
        if not bool((go | stand).any()):
            return o
        kind = torch.where(go, torch.full_like(o.kind, O.ATTACK), torch.where(stand, torch.full_like(o.kind, O.HOLD),
                                                                               o.kind))
        x = torch.where(stand, u["x"], o.x)
        z = torch.where(stand, u["z"], o.z)
        tg = torch.where(go, t, torch.where(stand, torch.full_like(o.target, -1), o.target))
        return O.Orders(kind, x, z, tg, o.run & ~go & ~stand, o.ability, o.cast_target, o.face, o.width)

    def sticky_rule(self, o, st, inside):
        """The rule 'a shooter keeps its target' (08.11, the game's probe: every new order resets a shooter's aim - 3-8.5 s
        to the first shot; the chain changed the long rifles' order 17 times in 43 s and they never fired; plan.md
        'Н-прямой-огонь'): for my ranged units out of melee, not routing -
        * a target in force (its attack order's, else the one it fires at) alive, seen and in range, and no enemy melee
          unit within back_m: an attack on another target, a hold or the same attack again is no new order (KEEP);
        * an attack of a shooter is at a walk (the run flag does not change a standing shooter's order).
        A move out of its area and a withdraw stay."""
        u = st.u
        own = self.setup.own
        ranged = own & (u["range"] > 0) & (u["men"] > 0) & ~u["r"] & (u["contact_s"] <= 0)
        if not bool(ranged.any()):
            return o
        att_in = (u["order_kind"] == O.ATTACK) & (u["order_target"] >= 0)
        t0 = torch.where(att_in, u["order_target"], u["aim_tgt"])
        tc = t0.clamp(min=0)
        seen = (u["men"] > 0) & ~u["gone"] & u["vis"].bool() & ~u["r"]
        d0 = torch.sqrt((u["x"].gather(1, tc) - u["x"]) ** 2 + (u["z"].gather(1, tc) - u["z"]) ** 2)
        valid = (t0 >= 0) & seen.gather(1, tc) & (d0 <= u["range"]) & ~self.bad_fast(st, t0)
        back = float(cfg()["teacher"]["army"].get("back_m") or 70.0)
        dx = u["x"][:, None, :] - u["x"][:, :, None]
        dz = u["z"][:, None, :] - u["z"][:, :, None]
        foe = (u["side"][:, :, None] != u["side"][:, None, :]) & (u["side"][:, None, :] > 0) & seen[:, None, :]             & (u["range"] <= 0)[:, None, :]
        near = (foe & (dx * dx + dz * dz <= back * back)).any(2)
        quiet = (o.kind == O.HOLD) | (o.kind == O.KEEP) | (o.kind == O.ATTACK)
        keep = ranged & valid & ~near & quiet & (att_in | (o.kind != O.ATTACK) | (o.target != t0))
        keep = keep & ~(((o.kind == O.HOLD) | (o.kind == O.KEEP)) & ~att_in)   # (holding and firing at will: as it is)
        same = ranged & att_in & (o.kind == O.ATTACK) & (o.target == u["order_target"])
        keep = keep | same
        kind = torch.where(keep, torch.full_like(o.kind, O.KEEP), o.kind)
        run = torch.where(ranged & (kind == O.ATTACK), torch.zeros_like(o.run), o.run)
        self.n_keep = getattr(self, "n_keep", 0) + int(keep.sum())
        return O.Orders(kind, o.x, o.z, o.target, run, o.ability, o.cast_target, o.face, o.width)

    def bad_fast(self, st, t):
        """[B, N] bool: t (-1: none -> False) is a fast unit (run >= fast_run_ms) that crosses or leaves the shooter -
        moving over fast_move_ms at more than fast_side_deg off the way to the shooter - or stands out of the shooter's
        fire arc (08.11, the probe losmove, plan.md 'Н-прямой-огонь' 3: a shooter ordered to attack cavalry or giant
        rats going sideways turns and walks after it, not shooting)."""
        u = st.u
        c = cfg()
        tc = t.clamp(min=0)
        fast = u["run"].gather(1, tc) >= float(c.get("fast_run_ms", 6.0))
        vx, vz = u["vx"].gather(1, tc), u["vz"].gather(1, tc)
        sp = torch.sqrt(vx * vx + vz * vz)
        wx, wz = u["x"] - u["x"].gather(1, tc), u["z"] - u["z"].gather(1, tc)            # from the target to me
        wn = torch.sqrt(wx * wx + wz * wz).clamp(min=1e-6)
        cosv = (vx * wx + vz * wz) / (sp.clamp(min=1e-6) * wn)
        side = (sp > float(c.get("fast_move_ms", 1.0))) & (cosv < math.cos(math.radians(float(c.get("fast_side_deg", 45.0)))))
        brg = torch.rad2deg(torch.atan2(-wx, -wz))
        off = (torch.remainder(brg - u["b"] + 180.0, 360.0) - 180.0).abs() > u["arc"]
        return (t >= 0) & fast & (side | off)

    def fast_rule(self, o, st):
        """No attack of my shooter on a bad fast target (bad_fast): such an order (and one in force that the next is
        KEEP) is a HOLD - the shooter fires at will at what is in its arc."""
        u = st.u
        own = self.setup.own & (u["range"] > 0) & (u["men"] > 0) & ~u["r"]
        new_bad = own & (o.kind == O.ATTACK) & self.bad_fast(st, o.target)
        old_bad = own & (o.kind == O.KEEP) & (u["order_kind"] == O.ATTACK) & self.bad_fast(st, u["order_target"])
        bad = new_bad | old_bad
        if not bool(bad.any()):
            return o
        self.n_fast = getattr(self, "n_fast", 0) + int(bad.sum())
        return O.Orders(torch.where(bad, torch.full_like(o.kind, O.HOLD), o.kind), torch.where(bad, u["x"], o.x),
                        torch.where(bad, u["z"], o.z), torch.where(bad, torch.full_like(o.target, -1), o.target),
                        o.run & ~bad, o.ability, o.cast_target, o.face, o.width)

    def line_closed(self, st, t):
        """[B, N] bool: an own unit stands across the line from each unit to its target t (-1: none -> False): its centre
        between 10 m past the shooter and 10 m short of the target, its front covering the line's middle."""
        from sim import geometry
        u = st.u
        front, _ = geometry.dims(u, self.spacing)
        tc = t.clamp(min=0)
        vx, vz = u["x"].gather(1, tc) - u["x"], u["z"].gather(1, tc) - u["z"]
        L = torch.sqrt(vx * vx + vz * vz).clamp(min=1e-6)
        ex, ez = vx / L, vz / L
        dx = u["x"][:, None, :] - u["x"][:, :, None]                       # [b, i, j] from shooter i to unit j
        dz = u["z"][:, None, :] - u["z"][:, :, None]
        along = dx * ex[:, :, None] + dz * ez[:, :, None]
        across = (dx * ez[:, :, None] - dz * ex[:, :, None]).abs()
        side = u["side"]
        mine = (side[:, :, None] == side[:, None, :]) & (u["men"] > 0)[:, None, :] & ~u["r"][:, None, :]
        eye = torch.eye(st.N, dtype=torch.bool, device=u["x"].device)[None]
        # (a friend whose front covers the line's middle: the simulator's line of fire - missile.friend_lines, 3 rays
        # from the shooter's front, clear when 2 are - is closed then; the band of both fronts closed it too often:
        # the snipers' fire fell to 0.5x on M1 / M3)
        band = (along > 10.0) & (along < L[:, :, None] - 10.0) & (across < front[:, None, :] / 2)
        return (band & mine & ~eye).any(2) & (t >= 0)

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
            prio = self.prio if self.targets else None
            v = UT.views(self.ubuf, idx, bl, il, self.setup.card, self.area, UNIT_CFG, wlim=self.wlim, prio=prio)
            ord_u = Nt.order_of(raw_u[bl, il], {k: a[bl, il] for k, a in self.area.items() if k != "on"},
                                self.pace[bl, il], self.arrived[bl, il],
                                has_target=None if prio is None else v["prio"] >= 0)
            act = Nt.sample(self.net(v, ord_u, v["cell_m"]), greedy=self.greedy)
            intended = Dc.to_orders(act, v, raw_u, v["cell_m"], B, N, wlim=self.wlim)
            if self.still:
                intended = self.stand_still(intended, st, inside)
            if self.aim:
                intended = self.stay_rule(intended, st, inside)
                intended = self.aim_rule(intended, st, inside)
                intended = self.fast_rule(intended, st)
                intended = self.sticky_rule(intended, st, inside)
                intended = self.fast_rule(intended, st)
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
