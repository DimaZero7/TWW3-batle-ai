"""The group's zone as tiles (design/tactic "Зона группы — плитки", the user's decisions 30.10): a tile is one unit's
largest personal area, TILE x TILE m (80); the zone is the full rectangle of a grid of tiles, in the zone's axes - its
centre and turn from the operative's parcel ("up" = the turn).

    grid(n, line)            (rows, cols): "line" - 1 x n; "square" - the rows x columns nearest a square, rounded up:
                             cols = ceil(sqrt n), rows = ceil(n / cols) (1 -> 1x1, 2 -> 1x2, 3, 4 -> 2x2, 5, 6 -> 2x3,
                             7, 8, 9 -> 3x3, 10 -> 3x4); rows run forward (row 0 is the farthest forward), cols across
    Tiles                    the tiles of R groups: centres in the zone's frame and in the world, which are left
    Keeper                   the tiles kept between decisions: a fresh grid for a new parcel; a unit that fell out
                             (dead or routing) takes away the tile nearest its last position, once (the hole stays where
                             it was; no tile grows back without a new unit)

The personal areas inside the zone never overlap - the translator's rule, not the network's (there are at least n tiles
by construction); areas_overlap() checks a set of them.
"""
import math

import torch

from tools.nn.tactic import cfg
from tools.nn.unit import zmap as Z

MAX_TILES = 12          # 10 units -> 3 x 4


def grid(n, line=False):
    """(rows, cols) of the zone of n units (module doc)."""
    n = max(int(n), 1)
    if line:
        return 1, n
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols


class Tiles:
    """The tiles of R groups. cx, cz, heading [R] the zone's centre and turn (degrees, the simulator's bearings), rows,
    cols [R] long, tile_m; mask [R, MAX_TILES] the tiles left (in row-major order: tile t = row t // cols, col t % cols;
    the slots past rows x cols always False)."""

    def __init__(self, cx, cz, heading, rows, cols, mask=None, tile_m=None):
        self.cx, self.cz, self.heading = cx, cz, heading
        self.rows, self.cols = rows.long(), cols.long()
        self.tile_m = float(tile_m or cfg()["tile_m"])
        R = cx.shape[0]
        t = torch.arange(MAX_TILES, device=cx.device)
        full = t[None] < (self.rows * self.cols)[:, None]
        self.mask = full if mask is None else (mask & full)
        c = t[None] % self.cols[:, None].clamp(min=1)
        r = t[None] // self.cols[:, None].clamp(min=1)
        # the tile's centre in the zone's frame (s right, f forward): row 0 the farthest forward, col 0 the leftmost
        self.s = (c.float() + 0.5 - self.cols[:, None].float() / 2) * self.tile_m
        self.f = (self.rows[:, None].float() / 2 - r.float() - 0.5) * self.tile_m
        self.s = torch.where(full, self.s, torch.zeros_like(self.s))
        self.f = torch.where(full, self.f, torch.zeros_like(self.f))
        self.R = R

    @classmethod
    def of(cls, parcel, n=None):
        """The full grid of a parcel's zone (n [R]: units; None: the parcel's my-slots count)."""
        n = parcel.mine.sum(1) if n is None else n
        rc = [grid(int(k), bool(l)) for k, l in zip(n.tolist(), parcel.line.tolist())]
        dev = parcel.cx.device
        rows = torch.tensor([a for a, _ in rc], device=dev)
        cols = torch.tensor([b for _, b in rc], device=dev)
        return cls(parcel.cx, parcel.cz, parcel.heading(), rows, cols)

    @property
    def front(self):
        """[R] the zone's width across its turn, m."""
        return self.cols.float() * self.tile_m

    @property
    def depth(self):
        """[R] the zone's depth along its turn, m."""
        return self.rows.float() * self.tile_m

    def near_size(self, margin=None):
        """[R] the side of the near window: the zone's larger side + one tile each side."""
        m = float(cfg()["margin_m"] if margin is None else margin)
        return torch.maximum(self.front, self.depth) + 2 * m

    def world(self):
        """(x, z) [R, MAX_TILES] of the tiles' centres."""
        return Z.world(self.s, self.f, self.cx[:, None], self.cz[:, None], self.heading[:, None])

    def drop(self, which, x, z):
        """Take away, for the groups which [R] bool, the tile left nearest the point (x, z) [R] (in place)."""
        tx, tz = self.world()
        d = (tx - x[:, None]) ** 2 + (tz - z[:, None]) ** 2
        d = torch.where(self.mask, d, torch.full_like(d, float("inf")))
        k = d.argmin(1)
        hit = torch.zeros_like(self.mask)
        hit[torch.arange(self.R, device=d.device), k] = True
        self.mask = self.mask & ~(hit & which[:, None] & self.mask.any(1, keepdim=True))

    def rects(self):
        """{x, z, b, front, depth} [R, MAX_TILES] of the tiles left as rectangles (front 0 for the ones gone)."""
        x, z = self.world()
        side = torch.where(self.mask, torch.full_like(x, self.tile_m), torch.zeros_like(x))
        return {"x": x, "z": z, "b": self.heading[:, None].expand_as(x), "front": side, "depth": side.clone()}


class Keeper:
    """The tiles of one group between decisions (module doc). update(parcel, raw) [one group: R = 1 rows of the parcel]
    returns its Tiles."""

    def __init__(self):
        self.key = None
        self.tiles = None
        self.out = set()
        self.last = {}

    @staticmethod
    def _key(parcel):
        return (tuple(parcel.mine[0].nonzero()[:, 0].tolist()), round(float(parcel.cx[0]), 3),
                round(float(parcel.cz[0]), 3), round(float(parcel.heading()[0]), 3), bool(parcel.line[0]),
                int(parcel.group[0]))

    def update(self, parcel, raw):
        """raw [B, N, len(tactic.raw.RAW)] the battle's raw state now (parcel.bi picks the battle)."""
        from tools.nn.tactic import raw as TR
        key = self._key(parcel)
        if key != self.key:
            self.key, self.tiles, self.out = key, Tiles.of(parcel), set()
        b = int(parcel.bi[0])
        for i in key[0]:
            row = raw[b, i]
            able = bool(row[TR.I["alive"]] > 0) and not bool(row[TR.I["r"]] > 0)
            if able:
                self.last[i] = (float(row[TR.I["x"]]), float(row[TR.I["z"]]))
            elif i not in self.out:
                self.out.add(i)
                x, z = self.last.get(i, (float(row[TR.I["x"]]), float(row[TR.I["z"]])))
                dev = parcel.cx.device
                self.tiles.drop(torch.ones(1, dtype=torch.bool, device=dev), torch.tensor([x], device=dev),
                                torch.tensor([z], device=dev))
        return self.tiles


def areas_overlap(areas, tol=1e-3):
    """[R] bool: some two of a group's personal areas ({x, z, b, front, depth, on} [R, M]) overlap by more than tol m
    (the translator's rule: they never do)."""
    from tools.nn.sim import geometry
    g = geometry.rect_gap(areas["x"], areas["z"], areas["b"], areas["front"], areas["depth"])
    on = areas["on"].bool()
    M = on.shape[1]
    pair = on[:, :, None] & on[:, None, :] & ~torch.eye(M, dtype=torch.bool, device=on.device)[None]
    return ((g < -tol) & pair).flatten(1).any(1)
