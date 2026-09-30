"""A side's frame: coordinates relative to the side, so the network plays both sides alike.

The world is the game's: x and z in metres, a unit's bearing b in degrees with the facing
(sin b, cos b) (b = 90 faces +x). A side's frame is fixed at the start of battle:
origin — the centre of the map; forward — from the side's army towards the enemy's army;
lateral — to the right of forward. Turning the world by 180 degrees and swapping the sides gives
the same numbers. Works on numpy arrays and on torch tensors alike (functions take either).
"""
import math

import numpy as np


def xp(a):
    """The array library of a (numpy or torch)."""
    if isinstance(a, np.ndarray) or not hasattr(a, "device"):
        return np
    import torch
    return torch


def unit(vx, vz):
    """(vx, vz) scaled to length 1; a zero vector gives (1, 0)."""
    m = xp(vx)
    n = m.sqrt(vx * vx + vz * vz)
    ok = n > 1e-6
    safe = m.where(ok, n, n * 0 + 1)
    return m.where(ok, vx / safe, vx * 0 + 1), m.where(ok, vz / safe, vz * 0)


def army_axis(x, z, own, enemy):
    """Forward per battle [B]: from the centroid of `own` units to that of `enemy` units (x, z [B, N], masks [B, N])."""
    def centroid(mask):
        w = mask * 1.0
        c = w.sum(-1)
        c = c + (c == 0) * 1.0
        return (x * w).sum(-1) / c, (z * w).sum(-1) / c
    ox, oz = centroid(own)
    ex, ez = centroid(enemy)
    return unit(ex - ox, ez - oz)


class Frame:
    """origin (cx, cz) and forward (ux, uz), each [B]; right is (uz, -ux)."""

    def __init__(self, cx, cz, ux, uz):
        self.cx, self.cz, self.ux, self.uz = cx, cz, ux, uz

    def _b(self, v, like):
        # [B] -> broadcast against like [B, ...]
        return v.reshape(v.shape + (1,) * (like.ndim - 1))

    def point(self, x, z):
        """World points [B, ...] -> (forward, lateral) in metres."""
        dx, dz = x - self._b(self.cx, x), z - self._b(self.cz, z)
        ux, uz = self._b(self.ux, x), self._b(self.uz, x)
        return dx * ux + dz * uz, dx * uz - dz * ux

    def vector(self, vx, vz):
        """World vectors -> (forward, lateral), no shift."""
        ux, uz = self._b(self.ux, vx), self._b(self.uz, vx)
        return vx * ux + vz * uz, vx * uz - vz * ux

    def world_vector(self, f, l):
        ux, uz = self._b(self.ux, f), self._b(self.uz, f)
        return f * ux + l * uz, f * uz - l * ux

    def world(self, f, l):
        """(forward, lateral) -> world (x, z)."""
        vx, vz = self.world_vector(f, l)
        return vx + self._b(self.cx, f), vz + self._b(self.cz, f)

    def facing(self, bearing_deg):
        """Bearing (degrees) -> (cos, sin) of the facing relative to forward (sin > 0: turned right)."""
        m = xp(bearing_deg)
        r = bearing_deg * (math.pi / 180)
        return self.vector(m.sin(r), m.cos(r))


def edge_distances(x, z, frame, bounds, cap=1000.0):
    """Distance (m) from each point to the map edge along +forward, -forward, +lateral, -lateral.

    bounds: (xmin, xmax, zmin, zmax), each [B]. A point outside the map gives 0. Capped at `cap`.
    """
    m = xp(x)
    xmin, xmax, zmin, zmax = (frame._b(v, x) for v in bounds)
    out = []
    for f, l in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        dx, dz = frame.world_vector(x * 0 + f, x * 0 + l)
        big = x * 0 + cap
        tx = m.where(dx > 1e-9, (xmax - x) / m.where(dx > 1e-9, dx, big),
                     m.where(dx < -1e-9, (xmin - x) / m.where(dx < -1e-9, dx, big), big))
        tz = m.where(dz > 1e-9, (zmax - z) / m.where(dz > 1e-9, dz, big),
                     m.where(dz < -1e-9, (zmin - z) / m.where(dz < -1e-9, dz, big), big))
        out.append(m.clip(m.minimum(tx, tz), 0, cap))
    return out
