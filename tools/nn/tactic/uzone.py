"""'Зона отряда' (design/tactic; the user's decisions 01.11): no weights, code. A frame from a size and a cell of the
near window's 30 x 30, the zone's turn; the units choose in slot order. All in the ZONE's axes (s right, f forward): the
frames all stand the same way, so every test is on axis-aligned rectangles.

    Seq         one decision of R groups: the units' frames so far, the holes (tiles dropped), and for a size chosen
                place() -> open [R, C, C] (the 'where' head's allowed cells) and the frame each cell would give

A cell is closed when the frame of that size at its centre laps over the edge of the group's zone or over a frame / hole
by more than one cell, or when the nudge leaves it no room, or when it would leave too few tiles for the units still to
choose (room(): a tile counts while a square of the least size fits at one of its corners; a frame in the middle of a
2 x 2 would otherwise leave no place at all - the page does not name this rule, it is what keeps the page's last resort
an error of the level above and not a choice of the network).

The nudge makes the cell's coarse centre exact: a lap of at most a cell is pushed out, a gap under a cell to an edge or a
neighbour is closed flush (2 frames of 80 m in a 2 x 2: the cell +-37.33 m -> +-40 m).

The last resort (place_or_least): no cell for the size chosen -> the least size, the 'where' head then picks among that
size's open cells; no cell for it either -> ParcelError, the error of the level above.
"""
import torch

from tools.nn.tactic import cfg
from tools.nn.tactic.parcel import ParcelError
from tools.nn.unit import zmap as Z

TOL = 1e-3


def steps(device="cpu"):
    """[S] the size's steps, m (config size_steps_m: the same for width and depth)."""
    return torch.tensor(cfg()["size_steps_m"], dtype=torch.float32, device=device)


def cell_grid(cell, cells=None):
    """(s, f) [R, C, C] the cells' centres in the zone's axes."""
    C = int(cells or cfg()["cells"])
    k = torch.arange(C, device=cell.device, dtype=torch.float32)
    s, f = Z.cell_centre(k[:, None], k[None, :], cell.view(-1, 1, 1), C)
    R = cell.shape[0]
    return s.expand(R, C, C), f.expand(R, C, C)


def _snap(p, h, z, cell):
    """p [R, ...] (half size h, the zone +-z; all broadcast): onto the nearer edge when the gap to it (or the lap over it)
    is under a cell."""
    lo, hi = p - h + z, z - p - h
    near_lo, near_hi = lo < cell - TOL, hi < cell - TOL
    move = torch.where(near_lo & (~near_hi | (lo <= hi)), -lo, torch.where(near_hi, hi, torch.zeros_like(p)))
    return p + move


def _on(o, m, like):
    return o["on"][:, m].view((-1,) + (1,) * (like.dim() - 1))


def _gaps(s, f, hw, hd, o, m):
    """(ds, df) [R, ...]: the gap between the frames and obstacle m along each axis (negative: they overlap on it)."""
    sh = (-1,) + (1,) * (s.dim() - 1)
    return ((s - o["s"][:, m].view(sh)).abs() - (hw + o["hw"][:, m].view(sh)),
            (f - o["f"][:, m].view(sh)).abs() - (hd + o["hd"][:, m].view(sh)))


def _clear(s, f, hw, hd, o, zs, zf):
    """[R, ...] bool: the frames at (s, f) lie in the zone and on none of the obstacles."""
    ok = (s - hw >= -zs - TOL) & (s + hw <= zs + TOL) & (f - hd >= -zf - TOL) & (f + hd <= zf + TOL)
    for m in range(o["s"].shape[1]):
        ds, df = _gaps(s, f, hw, hd, o, m)
        ok = ok & ~(_on(o, m, s) & (ds < -TOL) & (df < -TOL))
    return ok


def place(s0, f0, hw, hd, zs, zf, cell, o):
    """(open [R, ...], s, f [R, ...]): the candidates (s0, f0) [R, C, C] of half sizes hw, hd [R, 1, 1], the zone's half
    sizes zs, zf and the cell [R, 1, 1], the obstacles o {s, f, hw, hd, on} [R, M]: open by the module's rule without
    room(), and the nudged centres."""
    # closed: the lap of the cell's own centre
    ok = (s0 - hw >= -zs - cell - TOL) & (s0 + hw <= zs + cell + TOL) & \
         (f0 - hd >= -zf - cell - TOL) & (f0 + hd <= zf + cell + TOL)
    M = o["s"].shape[1]
    sh = (-1,) + (1,) * (s0.dim() - 1)
    for m in range(M):
        ds, df = _gaps(s0, f0, hw, hd, o, m)
        ok = ok & ~(_on(o, m, s0) & (ds < -TOL) & (df < -TOL) & (torch.maximum(ds, df) < -cell - TOL))
    # the nudge: onto the edges, off the obstacles, flush to a neighbour
    s, f = _snap(s0, hw, zs, cell), _snap(f0, hd, zf, cell)
    for _ in range(2):
        for m in range(M):
            on = _on(o, m, s)
            ds, df = _gaps(s, f, hw, hd, o, m)
            lap = on & (ds < -TOL) & (df < -TOL)
            along_s = ds >= df
            s = torch.where(lap & along_s, s + torch.where(s >= o["s"][:, m].view(sh), -ds, ds), s)
            f = torch.where(lap & ~along_s, f + torch.where(f >= o["f"][:, m].view(sh), -df, df), f)
    s, f = torch.minimum(torch.maximum(s, -zs + hw), zs - hw), torch.minimum(torch.maximum(f, -zf + hd), zf - hd)
    for m in range(M):
        on = _on(o, m, s)
        ds, df = _gaps(s, f, hw, hd, o, m)
        go_s = on & (ds > TOL) & (ds < cell - TOL) & (df < -TOL)
        go_f = on & (df > TOL) & (df < cell - TOL) & (ds < -TOL)
        if bool(go_s.any()):
            t = s + torch.where(s >= o["s"][:, m].view(sh), -ds, ds)
            s = torch.where(go_s & _clear(t, f, hw, hd, o, zs, zf), t, s)
        if bool(go_f.any()):
            t = f + torch.where(f >= o["f"][:, m].view(sh), -df, df)
            f = torch.where(go_f & _clear(s, t, hw, hd, o, zs, zf), t, f)
    t = _snap(s, hw, zs, cell)
    s = torch.where(_clear(t, f, hw, hd, o, zs, zf), t, s)
    t = _snap(f, hd, zf, cell)
    f = torch.where(_clear(s, t, hw, hd, o, zs, zf), t, f)
    return ok & _clear(s, f, hw, hd, o, zs, zf), s, f


class Seq:
    """One decision's choices of R groups in slot order. tiles: zone.Tiles of the groups, G: the units' places."""

    def __init__(self, tiles, G):
        c = cfg()
        self.tiles, self.G = tiles, G
        self.R = tiles.cx.shape[0]
        self.C = int(c["cells"])
        self.dev = tiles.cx.device
        self.cell = tiles.near_size() / self.C
        self.zs, self.zf = tiles.front / 2, tiles.depth / 2
        self.s0, self.f0 = cell_grid(self.cell, self.C)
        zero = lambda: torch.zeros((self.R, G), device=self.dev)       # noqa: E731
        self.s, self.f, self.hw, self.hd = zero(), zero(), zero(), zero()
        self.on = torch.zeros((self.R, G), dtype=torch.bool, device=self.dev)
        T = int((tiles.rows * tiles.cols).max())
        self.T = T
        half = tiles.tile_m / 2
        full = torch.arange(T, device=self.dev)[None] < (tiles.rows * tiles.cols)[:, None]
        self.left = tiles.mask[:, :T] & full
        # the holes: tiles gone (they are no territory any more)
        self.hole = {"s": tiles.s[:, :T], "f": tiles.f[:, :T], "hw": torch.full((self.R, T), half, device=self.dev),
                     "hd": torch.full((self.R, T), half, device=self.dev), "on": full & ~self.left}
        # the corner squares of each tile (room()): 4 a tile
        low = float(c["size_steps_m"][0]) / 2
        d = half - low
        self.q_half = low
        sgn = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], device=self.dev)
        self.qs = tiles.s[:, :T, None] + d * sgn[:, 0]             # [R, T, 4]
        self.qf = tiles.f[:, :T, None] + d * sgn[:, 1]

    def obstacles(self, upto):
        """{s, f, hw, hd, on} [R, M]: the holes and the frames of units before `upto`, only columns on somewhere."""
        parts = [self.hole]
        if upto > 0:
            parts.append({"s": self.s[:, :upto], "f": self.f[:, :upto], "hw": self.hw[:, :upto],
                          "hd": self.hd[:, :upto], "on": self.on[:, :upto]})
        o = {k: torch.cat([p[k] for p in parts], 1) for k in parts[0]}
        keep = o["on"].any(0)
        return {k: v[:, keep] for k, v in o.items()}

    def _free_base(self, upto):
        """[R, T, 4] bool: the corner squares clear of the frames before `upto` (tiles gone: none)."""
        base = self.left[:, :, None].expand(self.R, self.T, 4)
        for m in range(upto):
            ov = ((self.qs - self.s[:, m, None, None]).abs() < self.q_half + self.hw[:, m, None, None] - TOL) & \
                 ((self.qf - self.f[:, m, None, None]).abs() < self.q_half + self.hd[:, m, None, None] - TOL)
            base = base & ~(ov & self.on[:, m, None, None])
        return base

    def room(self, s, f, hw, hd, need, upto):
        """[R, C, C] bool: a frame at (s, f) [R, C, C] leaves at least need [R] tiles for the units still to choose."""
        out = torch.ones_like(s, dtype=torch.bool)
        ask = need > 0
        if not bool(ask.any()):
            return out
        base = self._free_base(upto)[ask]
        qs, qf = self.qs[ask][:, None, None], self.qf[ask][:, None, None]         # [r, 1, 1, T, 4]
        ov = ((s[ask][..., None, None] - qs).abs() < hw[ask][..., None, None] + self.q_half - TOL) & \
             ((f[ask][..., None, None] - qf).abs() < hd[ask][..., None, None] + self.q_half - TOL)
        free = (base[:, None, None] & ~ov).any(-1).sum(-1)
        out[ask] = free >= need[ask].view(-1, 1, 1)
        return out

    def place(self, j, width_m, depth_m, need):
        """(open, s, f) [R, C, C] for unit place j and a size (m, [R]); need [R]: the units still to choose after it."""
        hw, hd = (width_m / 2).view(-1, 1, 1), (depth_m / 2).view(-1, 1, 1)
        zs, zf, cell = self.zs.view(-1, 1, 1), self.zf.view(-1, 1, 1), self.cell.view(-1, 1, 1)
        ok, s, f = place(self.s0, self.f0, hw, hd, zs, zf, cell, self.obstacles(j))
        return ok & self.room(s, f, hw, hd, need, j), s, f

    def place_or_least(self, j, width_m, depth_m, need, able):
        """place(), and the last resort: (open, s, f, width_m, depth_m, fell back [R]). able [R]: the groups whose unit j
        chooses at all (the others' answers are never used)."""
        ok, s, f = self.place(j, width_m, depth_m, need)
        none = able & ~ok.flatten(1).any(1)
        if bool(none.any()):
            low = steps(self.dev)[0]
            width_m, depth_m = torch.where(none, low, width_m), torch.where(none, low, depth_m)
            ok2, s2, f2 = self.place(j, width_m, depth_m, need)
            k = none.view(-1, 1, 1)
            ok, s, f = torch.where(k, ok2, ok), torch.where(k, s2, s), torch.where(k, f2, f)
            lost = able & ~ok.flatten(1).any(1)
            if bool(lost.any()):
                r = int(lost.nonzero()[0, 0])
                raise ParcelError(f"group {r}: no room for unit place {j} even at the least size (the group's zone is "
                                  "too small: the error of the level above)")
        return ok, s, f, width_m, depth_m, none

    def commit(self, j, cell_i, s, f, width_m, depth_m, able):
        """Unit j's frame from the cell chosen: cell_i [R] (row x C + col), s, f [R, C, C] of place(); able [R]."""
        k = torch.arange(self.R, device=self.dev)
        self.s[:, j] = s.flatten(1)[k, cell_i]
        self.f[:, j] = f.flatten(1)[k, cell_i]
        self.hw[:, j], self.hd[:, j] = width_m / 2, depth_m / 2
        self.on[:, j] = able

    def frames(self):
        """{s, f, hw, hd, on} [R, G]."""
        return {"s": self.s, "f": self.f, "hw": self.hw, "hd": self.hd, "on": self.on}


def world(frames, tiles):
    """{x, z, b, front, depth, on} [R, G]: frames in the battle's coordinates (windows.area_layer's and the unit's area)."""
    h = tiles.heading[:, None]
    x, z = Z.world(frames["s"], frames["f"], tiles.cx[:, None], tiles.cz[:, None], h)
    return {"x": x, "z": z, "b": h.expand_as(x), "front": 2 * frames["hw"], "depth": 2 * frames["hd"],
            "on": frames["on"]}


def bad_frames(frames, tiles):
    """([R] overlapping, [R] outside the group's zone) bool: the promise of 'Зона отряда' checked on the results."""
    on = frames["on"]
    G = on.shape[1]
    ds = (frames["s"][:, :, None] - frames["s"][:, None, :]).abs() - frames["hw"][:, :, None] - frames["hw"][:, None, :]
    df = (frames["f"][:, :, None] - frames["f"][:, None, :]).abs() - frames["hd"][:, :, None] - frames["hd"][:, None, :]
    pair = on[:, :, None] & on[:, None, :] & ~torch.eye(G, dtype=torch.bool, device=on.device)[None]
    over = (pair & (ds < -TOL * 2) & (df < -TOL * 2)).flatten(1).any(1)
    zs, zf = tiles.front[:, None] / 2, tiles.depth[:, None] / 2
    out = on & ((frames["s"].abs() + frames["hw"] > zs + TOL * 2) | (frames["f"].abs() + frames["hd"] > zf + TOL * 2))
    return over, out.any(1)


def changed(new, old, change_m=None):
    """[R, G] bool: a noticeable change of a unit's frame (both on): another size, or the centre moved change_m or more."""
    m = float(cfg()["reward"]["change_m"] if change_m is None else change_m)
    moved = (new["s"] - old["s"]) ** 2 + (new["f"] - old["f"]) ** 2 >= m * m
    size = ((new["hw"] - old["hw"]).abs() > TOL) | ((new["hd"] - old["hd"]).abs() > TOL)
    return new["on"] & old["on"] & (moved | size)
