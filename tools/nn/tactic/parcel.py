"""The operative's parcel to a tactician (design/tactic "Посылка от оперативника", the user's decisions 30-31.10): one
a group, renewed by every new step of the operative.

    my slots     [R, N] bool, 1-10 of my side's slots (the working size of the drills 2-4)
    target       [R, N] bool, the enemy's slots; may be all zeros ("hold the point" without a target)
    centre       x, z of the zone's middle
    turn         sin, cos: where "up" is - the group's map stands the same way
    form         "line" / "square" (2 positions 0 / 1; the tiles: zone.grid)
    group        the group's number, constant while the operative keeps the group (for memory)
vector(): S + S + 7 numbers. There is no width or depth: the code works them out from n and the form.

The parcel is checked on the way in (check): a bad parcel is the operative's (or the generator's) error, the tactician
does not mend it - my slots alive and my side's, each unit in one group only, the zone inside the field, the target
connected by the squares' rule and its frame inside the near window's 30 x 30 cells; the new zone not farther than half
the near window's side from the group's centre. random() - a simple generator of correct parcels for the tests (the
school's generator comes later).

The squares' rule (target_boxes, connected): round each target unit a square SQUARE x SQUARE m (80) in the zone's
axes from its centre (a formation wider than SQUARE: its extent + PAD m each side); two units are one group when their
squares touch or overlap, and in a chain. The frame: the boxes' bounding rectangle.
"""
from dataclasses import dataclass

import torch

from tools.nn.tactic import cfg
from tools.nn.unit import zmap as Z


class ParcelError(ValueError):
    """A parcel that breaks the format: the error of the level above (the operative / the generator)."""


@dataclass
class Parcel:
    bi: torch.Tensor        # [R] long: the battle of each group
    mine: torch.Tensor      # [R, N] bool
    target: torch.Tensor    # [R, N] bool
    cx: torch.Tensor        # [R]
    cz: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor
    line: torch.Tensor      # [R] bool: the form "line" (else "square")
    group: torch.Tensor     # [R] long

    def heading(self):
        """[R] the zone's turn as a bearing (degrees clockwise from +z, as the simulator's)."""
        return torch.remainder(torch.rad2deg(torch.atan2(self.sin, self.cos)), 360.0)

    def vector(self):
        """[R, S + S + 7] the parcel as numbers: my slots, the target, centre x, z, sin, cos, line, square, group."""
        f = self.mine.float()
        return torch.cat([f, self.target.float(), torch.stack([self.cx, self.cz, self.sin, self.cos,
                                                               self.line.float(), (~self.line).float(),
                                                               self.group.float()], 1)], 1)

    def take(self, k):
        """The parcel of groups k (an index / slice / mask)."""
        return Parcel(*(getattr(self, n)[k] for n in self.__dataclass_fields__))

    @staticmethod
    def make(bi, mine, target, cx, cz, heading_deg, line=False, group=0, n=None, device="cpu"):
        """One group's parcel from plain numbers: mine / target slot lists (n: the battle's slots) or [N] bool masks;
        heading in degrees."""
        def mask(v):
            if torch.is_tensor(v) and v.dtype == torch.bool:
                return v.view(1, -1).to(device)
            m = torch.zeros((1, n), dtype=torch.bool, device=device)
            if len(v):
                m[0, torch.as_tensor(list(v), device=device)] = True
            return m
        if n is None:
            n = next(v.numel() for v in (mine, target) if torch.is_tensor(v) and v.dtype == torch.bool)
        h = torch.tensor([float(heading_deg)], device=device) * (torch.pi / 180)
        t = lambda v, dt=torch.float32: torch.tensor([v], dtype=dt, device=device)      # noqa: E731
        return Parcel(t(int(bi), torch.long), mask(mine), mask(target), t(float(cx)), t(float(cz)),
                      torch.sin(h), torch.cos(h), t(bool(line), torch.bool), t(int(group), torch.long))

    @staticmethod
    def cat(parcels):
        return Parcel(*(torch.cat([getattr(p, n) for p in parcels]) for n in Parcel.__dataclass_fields__))


def frame_axes(x, z, b, front, depth, cx, cz, heading):
    """(s, f, hs, hf) [R, ...]: centres in the zone's frame and the half extents of the formations (turned rectangles)
    along its axes."""
    s, f = Z.local(x, z, cx[:, None], cz[:, None], heading[:, None])
    a = (b - heading[:, None]) * (torch.pi / 180)
    ca, sa = torch.cos(a).abs(), torch.sin(a).abs()
    hs = front / 2 * ca + depth / 2 * sa
    hf = front / 2 * sa + depth / 2 * ca
    return s, f, hs, hf


def target_boxes(raw, parcel, sel=None):
    """(s, f, hs, hf, on) [R, N]: each target unit's box of the squares' rule in the zone's axes (module doc); on: the
    target units counted (sel [R, N]; None: the parcel's target, alive and seen by my side)."""
    from tools.nn.tactic import raw as TR
    c = cfg()
    S = raw[parcel.bi]
    on = parcel.target & (S[..., TR.I["alive"]] > 0) & (S[..., TR.I["vis"]] > 0) if sel is None else sel
    s, f, hs, hf = frame_axes(S[..., TR.I["x"]], S[..., TR.I["z"]], S[..., TR.I["b"]], S[..., TR.I["front"]],
                              S[..., TR.I["depth"]], parcel.cx, parcel.cz, parcel.heading())
    half, pad = float(c["square_m"]) / 2, float(c["square_pad_m"])
    hs = torch.where(2 * hs > 2 * half, hs + pad, torch.full_like(hs, half))
    hf = torch.where(2 * hf > 2 * half, hf + pad, torch.full_like(hf, half))
    return s, f, hs, hf, on


def connected(s, f, hs, hf, on, tol=1e-6):
    """[R] bool: the boxes on (at least one; none: True) form one group - touching or overlapping, in a chain."""
    R, N = on.shape
    touch = ((s[:, :, None] - s[:, None, :]).abs() <= hs[:, :, None] + hs[:, None, :] + tol) & \
            ((f[:, :, None] - f[:, None, :]).abs() <= hf[:, :, None] + hf[:, None, :] + tol)
    touch = touch & on[:, :, None] & on[:, None, :]
    first = on.float().argmax(1)
    reach = torch.zeros_like(on)
    reach[torch.arange(R, device=on.device), first] = True
    reach = reach & on
    for _ in range(N):
        nxt = reach | (touch & reach[:, :, None]).any(1)
        if torch.equal(nxt, reach):
            break
        reach = nxt
    return (reach == on).all(1) | ~on.any(1)


def frame_size(s, f, hs, hf, on):
    """([R], [R]) the width and depth of the target's frame (the boxes' bounding rectangle in the zone's axes; 0 none)."""
    big = 1e9
    lo_s = torch.where(on, s - hs, torch.full_like(s, big)).amin(1)
    hi_s = torch.where(on, s + hs, torch.full_like(s, -big)).amax(1)
    lo_f = torch.where(on, f - hf, torch.full_like(f, big)).amin(1)
    hi_f = torch.where(on, f + hf, torch.full_like(f, -big)).amax(1)
    any_ = on.any(1)
    return torch.where(any_, hi_s - lo_s, torch.zeros_like(lo_s)), torch.where(any_, hi_f - lo_f, torch.zeros_like(lo_f))


def check(raw, parcel, field=None, groups_of_battle=True, near_rule=True):
    """The parcel's errors: a list per group of the rules it breaks (empty: correct). raw [B, N, len(tactic.raw.RAW)];
    field: {cx, cz, b, front, depth} [B] the playable field of each battle as a turned rectangle (None: not checked);
    groups_of_battle: also 'a unit in one group only' across the parcel's groups of a battle; near_rule: the new zone not
    farther than half the near window's side from the group's actual centre."""
    from tools.nn.tactic import raw as TR
    from tools.nn.tactic import zone as ZN
    c = cfg()
    S = raw[parcel.bi]
    alive = S[..., TR.I["alive"]] > 0
    side = S[..., TR.I["side"]]
    n = parcel.mine.sum(1)
    my_side = torch.where(parcel.mine, side, torch.zeros_like(side)).amax(1)
    errs = [[] for _ in range(parcel.bi.shape[0])]
    for r in range(len(errs)):
        if not 1 <= int(n[r]) <= int(c["max_group"]):
            errs[r].append(f"my slots: {int(n[r])} (1-{c['max_group']})")
    bad_mine = parcel.mine & (~alive | (side != my_side[:, None]))
    bad_target = parcel.target & ((side == my_side[:, None]) | (side <= 0))
    tiles = ZN.Tiles.of(parcel)
    s, f, hs, hf, on = target_boxes(raw, parcel)
    ok_conn = connected(s, f, hs, hf, on)
    w, d = frame_size(s, f, hs, hf, on)
    size = tiles.near_size()
    for r in range(len(errs)):
        if bool(bad_mine[r].any()):
            errs[r].append(f"my slots not alive or not one side: {bad_mine[r].nonzero()[:, 0].tolist()}")
        if bool(bad_target[r].any()):
            errs[r].append(f"target slots not the enemy's: {bad_target[r].nonzero()[:, 0].tolist()}")
        if not bool(ok_conn[r]):
            errs[r].append("target not connected (squares' rule)")
        if float(w[r]) > float(size[r]) + 1e-3 or float(d[r]) > float(size[r]) + 1e-3:
            errs[r].append(f"target not compact: frame {float(w[r]):.0f} x {float(d[r]):.0f} m > window "
                           f"{float(size[r]):.0f} m")
    if groups_of_battle:
        for b in parcel.bi.unique().tolist():
            k = (parcel.bi == b).nonzero()[:, 0]
            twice = parcel.mine[k].sum(0) > 1
            if bool(twice.any()):
                for r in k.tolist():
                    errs[r].append(f"a unit in two groups: {twice.nonzero()[:, 0].tolist()}")
    if field is not None:
        fx, fz = field["cx"][parcel.bi], field["cz"][parcel.bi]
        hw = torch.stack([tiles.front, tiles.depth], 1) / 2
        for r in range(len(errs)):
            for sx, sf in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                x, z = Z.world(torch.tensor(sx * float(hw[r, 0])), torch.tensor(sf * float(hw[r, 1])),
                               parcel.cx[r], parcel.cz[r], parcel.heading()[r])
                ls, lf = Z.local(x, z, fx[r], fz[r], field["b"][parcel.bi][r])
                if float(ls.abs()) > float(field["front"][parcel.bi][r]) / 2 + 1e-3 or \
                        float(lf.abs()) > float(field["depth"][parcel.bi][r]) / 2 + 1e-3:
                    errs[r].append("zone outside the field")
                    break
    if near_rule:
        x, z = S[..., TR.I["x"]], S[..., TR.I["z"]]
        m = (parcel.mine & alive).float()
        k = m.sum(1).clamp(min=1)
        gx, gz = (x * m).sum(1) / k, (z * m).sum(1) / k
        far = torch.sqrt((gx - parcel.cx) ** 2 + (gz - parcel.cz) ** 2) > size / 2
        for r in range(len(errs)):
            if bool(far[r]) and bool(m[r].any()):
                errs[r].append("the zone farther than half the window from the group")
    return errs


def ensure(raw, parcel, field=None, **kw):
    """check(); raise ParcelError on the first group with errors."""
    for r, e in enumerate(check(raw, parcel, field, **kw)):
        if e:
            raise ParcelError(f"group {r} (battle {int(parcel.bi[r])}): " + "; ".join(e))


def random(raw, gen, b=None, side=None, sizes=(1, 2, 2, 3, 3, 4, 4, 5, 6, 8, 10), no_target=0.15, tries=50,
           group=0):
    """A correct random parcel (one group) of a battle's raw state [B, N, RAW] (torch.Generator gen): a side's seed unit
    and its nearest alive units (n from sizes), the zone centred on them, turned to the target (or to the nearest enemy,
    or anyhow), a form at random; the target a visible enemy and the enemies touching it by the squares' rule, as long
    as the frame fits; None when no correct one is found."""
    from tools.nn.tactic import raw as TR
    from tools.nn.tactic import zone as ZN
    B, N = raw.shape[:2]
    dev = raw.device
    for _ in range(tries):
        bb = int(torch.randint(B, (1,), generator=gen)) if b is None else int(b)
        R = raw[bb]
        alive = R[:, TR.I["alive"]] > 0
        sd = R[:, TR.I["side"]]
        sides = [s for s in (1, 2) if bool((alive & (sd == s)).any())] if side is None else [side]
        if not sides:
            continue
        s = sides[int(torch.randint(len(sides), (1,), generator=gen))]
        own = (alive & (sd == s) & ~(R[:, TR.I["r"]] > 0)).nonzero()[:, 0]
        if own.numel() == 0:
            continue
        seed = own[int(torch.randint(own.numel(), (1,), generator=gen))]
        n = min(int(sizes[int(torch.randint(len(sizes), (1,), generator=gen))]), own.numel())
        x, z = R[:, TR.I["x"]], R[:, TR.I["z"]]
        d = (x[own] - x[seed]) ** 2 + (z[own] - z[seed]) ** 2
        mine_i = own[d.argsort()[:n]]
        gx, gz = float(x[mine_i].mean()), float(z[mine_i].mean())
        foes = (alive & (sd != s) & (sd > 0) & (R[:, TR.I["vis"]] > 0)).nonzero()[:, 0]
        target_i = []
        heading = float(torch.rand(1, generator=gen)) * 360.0
        line = bool(torch.rand(1, generator=gen) < 0.5)
        if foes.numel():
            fd = (x[foes] - gx) ** 2 + (z[foes] - gz) ** 2
            near = foes[fd.argmin()]
            heading = float(torch.rad2deg(torch.atan2(x[near] - gx, z[near] - gz))) % 360.0
            if float(torch.rand(1, generator=gen)) >= no_target:
                t0 = foes[int(torch.randint(foes.numel(), (1,), generator=gen))]
                target_i = [int(t0)]
        p = Parcel.make(bb, mine_i.tolist(), [], gx, gz, heading, line, group, n=N, device=dev)
        if target_i:
            # grow the target by the squares' rule from its seed, while the frame fits
            p.target[0, target_i[0]] = True
            cand = torch.zeros((1, N), dtype=torch.bool, device=dev)
            cand[0, foes] = True
            sz = float(ZN.Tiles.of(p).near_size()[0])
            for _k in range(N):
                s_, f_, hs, hf, on = target_boxes(raw, p, sel=cand)
                t = p.target[0]
                touch = ((s_[0, :, None] - s_[0, None, :]).abs() <= hs[0, :, None] + hs[0, None, :]) & \
                        ((f_[0, :, None] - f_[0, None, :]).abs() <= hf[0, :, None] + hf[0, None, :])
                grow = cand[0] & ~t & (touch & t[None, :]).any(1)
                if not bool(grow.any()):
                    break
                j = int(grow.nonzero()[0, 0])
                t2 = t.clone()
                t2[j] = True
                w, dd = frame_size(s_, f_, hs, hf, t2[None])
                if float(w[0]) > sz or float(dd[0]) > sz:
                    cand[0, j] = False
                    continue
                p.target[0] = t2
        if not any(check(raw, p, groups_of_battle=False)[0]):
            return p
    return None
