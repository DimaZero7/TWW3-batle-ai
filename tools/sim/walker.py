"""The engine's walking, roughly, for the simulation (task 27, docs/ru/architecture/tasks/sim-obstacles.md).

In the game no unit walks or stands on an obstacle: the engine finds a way round it
and a block squeezes past its edge. Measured in the game (27.09.2026, 5 grounds with
obstacles, 3 runs; tools: this module's history in the task card):

* a walking unit faces where it goes (bearing vs the direction of motion: median
  3-4 deg, 90% within 37 deg while turning);
* it turns at up to about 20 deg/s (90% of turning samples within 15-20 deg/s);
* it walks at 1.3 m/s (median over 2 s samples, 1.5 m/s is the roster's pace) and a
  manoeuvre ends about 8 s after the block reaches its place (it settles): 53 steps
  of the approach, 4-50 m, took 9.7 s + distance / 1.37 m/s (mean error 9%); with
  1.3 m/s and 8 s the mean error is 8%, 81% of steps within 15%.

The walker moves a block's FRONT-RANK CENTRE (as placements and the logistics
orders give it) along a path on the map's 3 m grid: straight when the way is clear,
otherwise round the obstacle keeping half the block's front clear of it (less in a
narrow gap). A block faces its way and turns to the ordered facing at the place. An
order onto blocked ground stops at its nearest free edge. Soldiers that would stand
on blocked cells squeeze to the nearest free ground (Squeezer): the block deforms as
the engine's does. Units do not push each other (crowding is measured, not solved).
"""
import heapq
import math

import numpy as np

DT_S = 0.5
SPEED_MPS = 1.3     # the engine's walking, measured (the roster's pace is 1.5)
SETTLE_S = 8.0      # a block settles at its place before the manoeuvre is over, measured
TURN_DPS = 20.0     # the engine's turning, measured
ARRIVE_M = 0.25
FACE_MIN_M = 5.0    # closer than this to the place a block keeps its facing (no spin on the spot)
MAX_CLEARANCE_CELLS = 6
SQUEEZE_COST_M = 10.0  # a way one cell tighter than half the front costs this many metres more
SEARCH_MARGIN_CELLS = 80


def _wrap(deg):
    return ((deg + 540.0) % 360.0) - 180.0


def _bearing(dx, dz):
    return math.degrees(math.atan2(dx, dz)) % 360.0


class Terrain:
    """Blocked cells of a MapGrid-like grid (clear, inside, min_x, min_z, step), grown for clearance."""

    def __init__(self, grid):
        self.grid = grid
        inside = getattr(grid, "inside", None)
        self.blocked = ~grid.clear if inside is None else ~(grid.clear & inside)
        self.step, self.min_x, self.min_z = float(grid.step), float(grid.min_x), float(grid.min_z)
        self._grown = {0: self.blocked}

    def grown(self, k):
        """Blocked cells grown by k cells (8 neighbours)."""
        if k not in self._grown:
            g = self.grown(k - 1).copy()
            b = self.grown(k - 1)
            g[1:, :] |= b[:-1, :]
            g[:-1, :] |= b[1:, :]
            g[:, 1:] |= b[:, :-1]
            g[:, :-1] |= b[:, 1:]
            g[1:, 1:] |= b[:-1, :-1]
            g[:-1, :-1] |= b[1:, 1:]
            g[1:, :-1] |= b[:-1, 1:]
            g[:-1, 1:] |= b[1:, :-1]
            self._grown[k] = g
        return self._grown[k]

    def cell(self, x, z):
        return int((z - self.min_z) // self.step), int((x - self.min_x) // self.step)

    def centre(self, i, j):
        return self.min_x + (j + 0.5) * self.step, self.min_z + (i + 0.5) * self.step

    def is_blocked(self, i, j, k=0):
        b = self.grown(k)
        return not (0 <= i < b.shape[0] and 0 <= j < b.shape[1]) or bool(b[i, j])

    def nearest_free(self, i, j, k=0, towards=None, max_r=40):
        """The free cell nearest to (i, j); ties go to the one nearer `towards` (a cell)."""
        if not self.is_blocked(i, j, k):
            return i, j
        for r in range(1, max_r + 1):
            ring = []
            for di in range(-r, r + 1):
                for dj in (-r, r) if abs(di) != r else range(-r, r + 1):
                    if not self.is_blocked(i + di, j + dj, k):
                        ring.append((di * di + dj * dj, 0 if towards is None else
                                     (i + di - towards[0]) ** 2 + (j + dj - towards[1]) ** 2, i + di, j + dj))
            if ring:
                ring.sort()
                return ring[0][2], ring[0][3]
        return None

    def line_free(self, a, b, k=0):
        """Is the segment a-b (world) clear of cells grown by k?"""
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(dist / (self.step / 3)))
        for s in range(n + 1):
            x, z = a[0] + (b[0] - a[0]) * s / n, a[1] + (b[1] - a[1]) * s / n
            if self.is_blocked(*self.cell(x, z), k):
                return False
        return True

    def path(self, a, b, k):
        """Waypoints (world) from a to b keeping k cells clear; the last one is b, or its nearest
        free cell when b is blocked. None when there is no way at this clearance."""
        ia, ib = self.cell(*a), self.cell(*b)
        end = self.nearest_free(*ib, k, towards=ia)
        start = self.nearest_free(*ia, k, towards=ib)
        if end is None or start is None:
            return None
        goal = b if end == ib else self.centre(*end)
        if self.line_free(a, goal, k):
            return [goal]
        cells = self._astar(start, end, k)
        if cells is None:
            return None
        pts = [self.centre(i, j) for i, j in cells]
        pts[-1] = goal
        # Pull the string: from each point go to the farthest one in plain sight.
        out, at = [], a
        i = 0
        while i < len(pts):
            j = len(pts) - 1
            while j > i and not self.line_free(at, pts[j], k):
                j -= 1
            out.append(pts[j])
            at, i = pts[j], j + 1
        return out

    def _astar(self, start, end, k):
        b = self.grown(k)
        i0 = max(0, min(start[0], end[0]) - SEARCH_MARGIN_CELLS)
        i1 = min(b.shape[0] - 1, max(start[0], end[0]) + SEARCH_MARGIN_CELLS)
        j0 = max(0, min(start[1], end[1]) - SEARCH_MARGIN_CELLS)
        j1 = min(b.shape[1] - 1, max(start[1], end[1]) + SEARCH_MARGIN_CELLS)
        h = lambda c: math.hypot(c[0] - end[0], c[1] - end[1])
        seen = {start: None}
        cost = {start: 0.0}
        todo = [(h(start), start)]
        steps = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        while todo:
            _, c = heapq.heappop(todo)
            if c == end:
                out = []
                while c is not None:
                    out.append(c)
                    c = seen[c]
                return out[::-1]
            for di, dj, w in steps:
                n = (c[0] + di, c[1] + dj)
                if not (i0 <= n[0] <= i1 and j0 <= n[1] <= j1) or b[n]:
                    continue
                nc = cost[c] + w
                if nc < cost.get(n, math.inf):
                    cost[n], seen[n] = nc, c
                    heapq.heappush(todo, (nc + h(n), n))
        return None


class Squeezer:
    """Soldiers on blocked cells move to the nearest free ground: the block deforms at an edge."""

    def __init__(self, grid):
        self.terrain = grid if isinstance(grid, Terrain) else Terrain(grid)

    def squeeze(self, pts):
        """[x, z, ...] -> (points, how many moved)."""
        t = self.terrain
        out, moved = list(pts), 0
        for s in range(0, len(out) - 1, 2):
            i, j = t.cell(out[s], out[s + 1])
            if t.is_blocked(i, j):
                free = t.nearest_free(i, j)
                if free is not None:
                    # Keep the soldier's place inside the cell, moved into the free one.
                    fx, fz = t.centre(*free)
                    cx, cz = t.centre(i, j)
                    out[s], out[s + 1] = fx + (out[s] - cx) * 0.5, fz + (out[s + 1] - cz) * 0.5
                    moved += 1
        return out, moved


class Walker:
    """Blocks walking by orders, a step of DT_S at a time; grid=None: open ground."""

    def __init__(self, grid=None, speed=SPEED_MPS, turn_dps=TURN_DPS, dt=DT_S, settle_s=SETTLE_S):
        self.terrain = None if grid is None else (grid if isinstance(grid, Terrain) else Terrain(grid))
        self.speed, self.turn, self.dt, self.settle = speed, turn_dps, dt, settle_s
        self.units, self.t = {}, 0.0

    def add(self, uid, x, z, bearing, front_m, depth_m):
        """A block by its front-rank centre."""
        self.units[uid] = {"x": float(x), "z": float(z), "bearing": float(bearing) % 360, "front_m": front_m,
                           "depth_m": depth_m, "way": [], "facing": None, "shape": None, "pause": 0.0,
                           "arrived": True, "moving": False, "settle": 0.0}

    def order(self, uid, x, z, bearing=None, front_m=None, depth_m=None, pause_s=0.0):
        """Walk the front-rank centre to (x, z); face `bearing` there; take the new width there
        (the engine changes a width only at the destination); stand `pause_s` first (reforming)."""
        u = self.units[uid]
        u["way"] = self._way(u, (float(x), float(z)))
        # Settles at the place only after a walk (not after a reform on the spot).
        u["settle"] = self.settle if math.hypot(float(x) - u["x"], float(z) - u["z"]) > 1.0 else 0.0
        u["facing"] = None if bearing is None else float(bearing) % 360
        u["shape"] = (front_m, depth_m) if front_m is not None else None
        u["pause"] = pause_s
        u["arrived"] = False

    def _way(self, u, goal):
        if self.terrain is None:
            return [goal]
        # Half the block's front clear of the obstacle, but like the engine the shorter way wins:
        # a block squeezes through a gap rather than walk far round (gap_march_wide, 27.09.2026).
        want = min(MAX_CLEARANCE_CELLS, max(1, math.ceil(u["front_m"] / 2 / self.terrain.step)))
        at, best = (u["x"], u["z"]), None
        # Half the front clear, then the tightest (one cell), then touching: two or three searches at most.
        for k in sorted({want, 1, 0}, reverse=True):
            way = self.terrain.path(at, goal, k)
            if not way:
                continue
            length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip([at] + way[:-1], way))
            cost = length + (want - k) * SQUEEZE_COST_M
            if best is None or cost < best[0]:
                best = (cost, way)
            if len(way) == 1 or best[1] is way and k < want:
                break          # straight, or a tighter way found: tighter still is not tried
        return best[1] if best else [goal]

    def state(self, uid):
        return dict(self.units[uid])

    def _turn(self, u, towards):
        d = _wrap(towards - u["bearing"])
        lim = self.turn * self.dt
        u["bearing"] = (u["bearing"] + max(-lim, min(lim, d))) % 360
        return abs(d) <= lim

    def step(self):
        for u in self.units.values():
            u["moving"] = False
            if u["arrived"]:
                continue
            if u["pause"] > 0:
                u["pause"] -= self.dt
                continue
            if u["way"]:
                left = self.speed * self.dt
                rest = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                           for a, b in zip([(u["x"], u["z"])] + u["way"][:-1], u["way"]))
                while left > 0 and u["way"]:
                    tx, tz = u["way"][0]
                    d = math.hypot(tx - u["x"], tz - u["z"])
                    if d > 1e-9 and rest > FACE_MIN_M:
                        self._turn(u, _bearing(tx - u["x"], tz - u["z"]))
                    if d <= left:
                        u["x"], u["z"] = tx, tz
                        u["way"].pop(0)
                        left -= d
                    else:
                        u["x"] += (tx - u["x"]) / d * left
                        u["z"] += (tz - u["z"]) / d * left
                        left = 0
                u["moving"] = True
                continue
            # At the place: the new width, then the ordered facing.
            if u["shape"]:
                u["front_m"], u["depth_m"] = u["shape"]
                u["shape"] = None
            if u["facing"] is not None and not self._turn(u, u["facing"]):
                u["moving"] = True
            elif u["settle"] > 0:
                u["settle"] -= self.dt
            else:
                u["arrived"] = True
        self.t += self.dt

    def sample(self):
        return {uid: (round(self.t, 2), u["x"], u["z"], u["bearing"], u["front_m"], u["depth_m"])
                for uid, u in self.units.items()}

    def done(self):
        return all(u["arrived"] for u in self.units.values())

    def run(self, max_s=900.0, every=2):
        """Walk until everybody is in place: {uid: [(t_s, x, z, bearing, front_m, depth_m), ...]},
        a sample every `every` steps and the last one."""
        tracks = {uid: [s] for uid, s in self.sample().items()}
        n = 0
        while not self.done() and self.t < max_s:
            self.step()
            n += 1
            if n % every == 0 or self.done():
                for uid, s in self.sample().items():
                    tracks[uid].append(s)
        return tracks
