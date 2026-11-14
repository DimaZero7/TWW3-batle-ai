"""The tactic level's orders in force and its masks (tmp/tactic_slice/design_flank_v2.md sections 2-4,
stage0_results.md): an order = skill + main target + side + frame + age; the frame lives in lower.Tacticians beside
the rest of the order. Code, no weights; the same for every player (K, T, ...: tactic/pin.py) and, later, the network.

    Facts(rr, ctx, order)   what the level reads at a tick, from its input only (rows.build's rows - place, facing,
                            front, depth, speed, the flags m and r, mine / enemy by 'who', the lord and the run from the
                            passport) and the orders in force; never spd, contact_s, hit_sec (not the same on the bridge)
    Walk(fx)                the strikers' cap within one tick in slot order: at most orders.strikers an enemy and one a
                            side (sides: 'axis' within half the enemy's front of the line along its facing - no count -,
                            else 'left' / 'right'; a striker keeps the side written at its order); an order given earlier
                            in the walk counts at once
    frame_walk(...)         the frames of a tick: the units that keep first (uzone.Seq.keep; the fixed ones - a unit in
                            melee, a striker - are never let go), then each unit that must change, in slot order, takes a
                            frame of frame_m x frame_m at the open cell whose nudged frame is nearest it
    masks(view, out)        the code's masks over a player's output: (1) hygiene - no new order to a unit in melee;
                            (3, 4) skills hold / attack (counter for a reference player), an attack only on a valid open
                            target; returns the orders as given and the counts of what was masked
    own(...)                the orders in force as the 16 numbers a unit of mine (stage 5's input; not used yet)
    View                    what a player gets at a tick (lower.Tacticians builds it)
A player: a callable view -> {skill, target (the battle's slot, -1 none), side [r, G] long; frames, fell, kept, must,
forced (frame_walk's / decide.choose's); drop [r, G] bool optional: frames to take off} in my-slot order (decide.my_slots).
"""
from dataclasses import dataclass

import torch

from net.tactic import cfg
from net.tactic import decide as DC
from net.tactic import raw as TR
from net.tactic import rows as RW
from net.tactic import uzone as U
from net.unit import scripts as SC
from net.unit import zmap as Z

SIDES = ("axis", "left", "right")
AXIS, LEFT, RIGHT = range(3)
LORD_COL = RW.RI["p36"]            # the passport's caste one-hot starts at 36 (net/cards/passport.features): 'lord'
RUN_COL = RW.RI["p5"]              # the run speed / 10 m/s
INF = float("inf")


def c_orders():
    return cfg()["orders"]


class Facts:
    """The level's facts at a tick (module doc). Rows k of rr [R, K]; my units g in slot order [R, G] (ctx.rows: their
    row). order {skill, tgt, side} [R, G] the orders in force (tgt: the battle's slot)."""

    def __init__(self, rr, ctx, order):
        from sim import geometry
        c = cfg()
        x = rr["rows"]
        RI = RW.RI
        pos, size, spd = float(c["pos_m"]), float(c["size_m"]), float(c["speed_ms"])
        self.rr, self.ctx, self.order = rr, ctx, order
        self.mask = rr["mask"]
        self.slot = rr["slot"]
        who = rr["who"]
        self.s, self.f = x[..., RI["right"]] * pos, x[..., RI["forward"]] * pos
        self.b = torch.rad2deg(torch.atan2(x[..., RI["front_sin"]], x[..., RI["front_cos"]]))   # (the zone's axes)
        self.front, self.depth = x[..., RI["front"]] * size, x[..., RI["depth"]] * size
        self.vs, self.vf = x[..., RI["v_right"]] * spd, x[..., RI["v_forward"]] * spd
        self.speed = torch.sqrt(self.vs ** 2 + self.vf ** 2)
        self.m = (x[..., RI["m"]] > 0.5) & self.mask
        self.r = (x[..., RI["r"]] > 0.5) & self.mask
        self.lord = x[..., LORD_COL] > 0.5
        self.run = x[..., RUN_COL] * 10.0
        self.mine = (who == RW.OWN_GROUP) & self.mask
        self.enemy = ((who == RW.ENEMY_TARGET) | (who == RW.ENEMY_OTHER)) & self.mask
        self.valid = self.enemy & ~self.r & ~self.lord                          # [R, K] valid(e)
        self.gap = geometry.rect_gap(self.s, self.f, self.b, self.front, self.depth)       # [R, K, K]
        self.dist = torch.sqrt((self.s[:, :, None] - self.s[:, None, :]) ** 2 + (self.f[:, :, None] - self.f[:, None, :]) ** 2)
        # my units g -> their rows
        self.my = ctx.my
        self.able = ctx.able
        self.g_row = ctx.rows
        R, G = self.my.shape
        self.R, self.G, self.K = R, G, x.shape[1]
        k = torch.arange(R, device=x.device)[:, None]
        self.g_m = self.m[k, self.g_row] & self.able
        # the orders' targets -> rows (-1: no row - hidden, dead, no target)
        tgt = order["tgt"]
        hit = (self.slot[:, None, :] == tgt[:, :, None]) & self.mask[:, None, :] & (tgt[:, :, None] >= 0)    # [R, G, K]
        self.t_row = torch.where(hit.any(2), hit.float().argmax(2), torch.full_like(tgt, -1))
        self.t_valid = (self.t_row >= 0) & self.valid.gather(1, self.t_row.clamp(min=0))
        self.attack = order["skill"] == SC.ATTACK
        self.striking = self.able & self.attack & self.t_valid                   # its attack has a valid target
        # my units against the rows: gap, distance, side
        gi = self.g_row[:, :, None].expand(R, G, self.K)
        self.g_gap = self.gap.gather(1, gi)                                       # [R, G, K]
        self.g_dist = self.dist.gather(1, gi)
        self.g_side = self.side_of(self.s.gather(1, self.g_row), self.f.gather(1, self.g_row))
        # free(i): mine, standing, not in melee, no standing enemy within STICK_M, its order hold or an attack whose
        # target is not valid
        near = (self.enemy & ~self.r)[:, None, :] & (self.g_gap <= SC.STICK_M)
        idle = (order["skill"] == SC.HOLD) | (order["skill"] == SC.COUNTER)
        self.free = self.able & ~self.g_m & ~near.any(2) & (idle | (self.attack & ~self.t_valid))

    def side_of(self, gs, gf):
        """[R, G, K] the side of each row's enemy my unit at (gs, gf) [R, G] stands on: its offset across the line through
        the enemy's centre along its facing - AXIS within half its front, else LEFT / RIGHT."""
        th = torch.deg2rad(self.b)[:, None, :]
        off = (gs[:, :, None] - self.s[:, None, :]) * torch.cos(th) - (gf[:, :, None] - self.f[:, None, :]) * torch.sin(th)
        side = torch.where(off < 0, torch.full_like(off, LEFT), torch.full_like(off, RIGHT)).long()
        return torch.where(off.abs() <= self.front[:, None, :] / 2, torch.full_like(side, AXIS), side)

    def row_of_slot(self, t):
        """[R, G] the row of the battle's slots t [R, G] (-1: none)."""
        hit = (self.slot[:, None, :] == t[:, :, None]) & self.mask[:, None, :] & (t[:, :, None] >= 0)
        return torch.where(hit.any(2), hit.float().argmax(2), torch.full_like(t, -1))

    def slot_of_row(self, k):
        """[R, G] the battle's slot of rows k [R, G] (-1 stays -1)."""
        return torch.where(k >= 0, self.slot.gather(1, k.clamp(min=0)), torch.full_like(k, -1))


class Walk:
    """The strikers' cap within one tick (module doc): count [R, K], held [R, K, 3] from the orders in force of my
    units - every unit of mine holding an attack order on the enemy, routing or not (config orders.cap_all; off: the
    standing ones only); give() / release() as the walk goes."""

    def __init__(self, fx, skill=None, t_row=None, side=None):
        self.fx = fx
        self.cap = int(c_orders()["strikers"])
        R, K = fx.R, fx.K
        dev = fx.s.device
        self.count = torch.zeros((R, K), device=dev)
        self.held = torch.zeros((R, K, 3), dtype=torch.bool, device=dev)
        self.k = torch.arange(R, device=dev)
        self.skill = fx.order["skill"].clone() if skill is None else skill
        self.t_row = fx.t_row.clone() if t_row is None else t_row
        self.side = fx.order["side"].clone() if side is None else side
        self.holder = (fx.my >= 0) if bool(c_orders().get("cap_all", True)) else fx.able
        for g in range(fx.G):
            self._add(g, self.holder[:, g] & (self.skill[:, g] == SC.ATTACK) & (self.t_row[:, g] >= 0), 1.0)

    def _add(self, g, on, sign):
        k = self.t_row[:, g].clamp(min=0)
        self.count[self.k, k] += sign * on.float()
        sd = self.side[:, g].clamp(min=0)
        hs = on & (self.side[:, g] > AXIS)
        self.held[self.k[hs], k[hs], sd[hs]] = sign > 0

    def open(self, g):
        """[R, K] the enemies open for my unit g now: valid, under the cap, its side free (or the axis)."""
        sd = self.fx.g_side[:, g]                                                  # [R, K]
        held = self.held.gather(2, sd.clamp(min=0)[..., None])[..., 0]
        return self.fx.valid & (self.count < self.cap) & ((sd == AXIS) | ~held)

    def release(self, g, on):
        """My unit g's attack order ends where on [R] (its count and side freed)."""
        self._add(g, on & self.holder[:, g] & (self.skill[:, g] == SC.ATTACK) & (self.t_row[:, g] >= 0), -1.0)
        self.skill[:, g] = torch.where(on, torch.full_like(self.skill[:, g], SC.HOLD), self.skill[:, g])
        self.t_row[:, g] = torch.where(on, torch.full_like(self.t_row[:, g], -1), self.t_row[:, g])

    def give(self, g, k, on):
        """My unit g attacks row k [R] where on [R] (its old attack released first); its side written."""
        self.release(g, on)
        self.skill[:, g] = torch.where(on, torch.full_like(self.skill[:, g], SC.ATTACK), self.skill[:, g])
        self.t_row[:, g] = torch.where(on, k, self.t_row[:, g])
        sd = self.fx.g_side[:, g].gather(1, k.clamp(min=0)[:, None])[:, 0]
        self.side[:, g] = torch.where(on, sd, self.side[:, g])
        self._add(g, on & self.holder[:, g], 1.0)


def frame_walk(mine_raw, able, tiles, prev, must, fixed=None, size=None):
    """The frames of a tick (module doc): mine_raw [R, G, RAW] my units' raw rows, able [R, G], tiles, prev the frames in
    force {s, f, hw, hd, on} [R, G] or None, must [R, G] the units that must change, fixed [R, G] never let go. Returns
    {frames, fell, kept, must, forced (changing: must or let go), let_go}."""
    R, G = able.shape
    dev = able.device
    seq = U.Seq(tiles, G)
    on = torch.zeros_like(able) if prev is None else prev["on"]
    want = able & ~must & on
    kept = seq.keep(want.clone(), prev, able, fixed)
    changing = (able & must) | (want & ~kept)
    left = DC.left_after(changing)
    us, uf = Z.local(mine_raw[..., TR.I["x"]], mine_raw[..., TR.I["z"]], tiles.cx[:, None], tiles.cz[:, None],
                     tiles.heading[:, None])
    m = torch.full((R,), float(size or c_orders()["frame_m"]), device=dev)
    fell = torch.zeros(R, device=dev)
    for j in range(G):
        a = changing[:, j]
        if not bool(a.any()):
            continue
        ok, s, f, wm, dm, fb = seq.place_or_least(j, m, m, left[:, j], a)
        d = ((s - us[:, j].view(-1, 1, 1)) ** 2 + (f - uf[:, j].view(-1, 1, 1)) ** 2).flatten(1)
        c = d.masked_fill(~ok.flatten(1), 1e12).argmin(1)
        seq.commit(j, c, s, f, wm, dm, a)
        fell = fell + fb.float()
    return {"frames": seq.frames(), "fell": fell, "kept": kept, "must": must, "forced": changing, "let_go": want & ~kept}


@dataclass
class View:
    """A tick of the level for a player (module doc)."""
    rr: dict                # rows.build's
    ctx: object             # decide.Ctx (prev: the frames in force)
    tiles: object
    fx: Facts
    order: dict             # {skill, tgt, side, born} [r, G] the orders in force (my-slot order)
    must: torch.Tensor      # [r, G] must change (the frame): the causes less the exempt
    fixed: torch.Tensor     # [r, G] never let go, never forced: in melee (hygiene), striking
    melee: torch.Tensor     # [r, G] my unit in melee (its row's m)
    cause: dict             # {first, hole, zone} [r, G] (before the exemptions)
    now: float              # battle seconds
    idx: torch.Tensor       # [r] the Tacticians' rows
    hygiene: bool = True

    def frames(self, must=None):
        """frame_walk with this tick's frames in force and masks (must: another set of units forced)."""
        return frame_walk(self.ctx.mine_raw, self.ctx.able, self.tiles, self.ctx.prev, self.must if must is None else must,
                          self.fixed)


def masks(view, out, allow=(SC.HOLD, SC.ATTACK)):
    """(skill, target, side, change [r, G], counts {melee_changes, melee_blocked, rejected}) - the player's out after the
    code's masks (module doc), walked in slot order so an earlier order counts at once. change: the order (skill,
    target or frame) differs from the one in force."""
    fx = view.fx
    o = view.order
    skill, tgt, side = out["skill"].clone(), out["target"].clone(), out["side"].clone()
    want = (skill != o["skill"]) | (tgt != o["tgt"])
    melee = view.melee
    blocked = want & melee & view.hygiene
    t_row = fx.row_of_slot(tgt)
    w = Walk(fx)
    rejected = torch.zeros_like(want)
    for g in range(fx.G):
        on = want[:, g] & ~blocked[:, g] & fx.able[:, g]
        if not bool(on.any()):
            continue
        att = on & (skill[:, g] == SC.ATTACK)
        k = t_row[:, g]
        ok = att & (k >= 0) & w.open(g).gather(1, k.clamp(min=0)[:, None])[:, 0]
        bad = (att & ~ok) | (on & ~torch.isin(skill[:, g], torch.tensor(allow, device=skill.device)))
        rejected[:, g] = bad
        w.give(g, k, ok & ~bad)
        w.release(g, on & ~att & ~bad)
        side[:, g] = torch.where(ok & ~bad, w.side[:, g], side[:, g])
    # (a unit that does not choose - routing, dead - may only have its order taken away: hold, no target)
    gone = want & ~fx.able & ~((skill == SC.HOLD) & (tgt < 0))
    undo = blocked | rejected | gone
    skill = torch.where(undo, o["skill"], skill)
    tgt = torch.where(undo, o["tgt"], tgt)
    side = torch.where(undo, o["side"], side)
    given = (skill != o["skill"]) | (tgt != o["tgt"])
    fr = out["forced"] if "forced" in out else torch.zeros_like(given)
    change = (given | fr) & fx.able
    cnt = {"melee_changes": (change & melee).float(), "melee_blocked": blocked.float(), "rejected": rejected.float()}
    return skill, tgt, side, change, cnt


def own(order, frames, fx, now):
    """[R, G, 16] the orders in force as numbers a unit of mine (design_flank_v2.md section 7: frame on, s, f, hw, hd /
    pos_m; skill one-hot 4; has a target; the target's place s, f / pos_m (0 without a row); side one-hot 3; age / 60
    s). Not an input yet (stage 5)."""
    pos = float(cfg()["pos_m"])
    on = frames["on"].float()
    fr = torch.stack([on, frames["s"] * on / pos, frames["f"] * on / pos, frames["hw"] * on / pos,
                      frames["hd"] * on / pos], -1)
    sk = torch.nn.functional.one_hot(order["skill"].clamp(0, len(SC.SKILLS) - 1), len(SC.SKILLS)).float()
    has = (order["tgt"] >= 0).float()[..., None]
    k = fx.t_row.clamp(min=0)
    seen = (fx.t_row >= 0).float()
    tp = torch.stack([fx.s.gather(1, k) * seen / pos, fx.f.gather(1, k) * seen / pos], -1)
    sd = torch.nn.functional.one_hot(order["side"].clamp(min=0), 3).float() * (order["side"] >= 0).float()[..., None]
    age = ((now - order["born"]) / 60.0).clamp(min=0)[..., None]
    return torch.cat([fr, sk, has, tp, sd, age], -1)
