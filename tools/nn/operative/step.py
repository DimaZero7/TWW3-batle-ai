"""The zone's size, the cells' bans and the step towards the goal (plan.md "Проверка посылки", "Шаг к цели"): code, no
weights - the rules of tools/nn/tactic/parcel.py check() kept by construction.

    size(n, line)       (front, depth) [R] m of the tiles' full grid (zone.grid x tile_m)
    near(n, line)       [R] the near window's side: the larger side + one tile each side (zone.Tiles.near_size)
    cells(axes)         (x, z) [B, C*C] the field map's cells' centres in the world
    field_ok(...)       [R, C*C] bool: a zone of that size and turn centred on the cell lies inside the field
    clamp(...)          (cx, cz) [R]: the zone the parcel gets for the goal chosen - on the segment from the group's
                        actual centre to the goal, no farther than half the near window less step_margin_m, then pushed
                        inside the field (the tactician's rule: a zone farther than half the window - the level above
                        gives an intermediate step; here the code gives it, the network chooses the goal)
"""
import torch

from tools.nn.operative import cfg
from tools.nn.operative.groups import grid
from tools.nn.tactic import cfg as tcfg
from tools.nn.unit import zmap as Z

TOL = 1e-3


def size(n, line):
    rows, cols = grid(n, line)
    t = float(tcfg()["tile_m"])
    return cols.float() * t, rows.float() * t


def near(n, line):
    front, depth = size(n, line)
    return torch.maximum(front, depth) + 2 * float(tcfg()["margin_m"])


def cells(axes):
    """(x, z) [B, C*C]: row-major (row 0 the farthest forward, col 0 the leftmost), as zmap's cells."""
    c = cfg()
    C, cm = int(c["cells"]), float(c["cell_m"])
    dev = axes.cx.device
    k = torch.arange(C, device=dev, dtype=torch.float32)
    s, f = Z.cell_centre(k[:, None], k[None, :], cm, C)
    s, f = torch.broadcast_tensors(s, f)
    return axes.world(s.reshape(1, -1).expand(axes.cx.shape[0], -1), f.reshape(1, -1).expand(axes.cx.shape[0], -1))


def cell_of(axes, x, z):
    """[B] long: the field map's cell of world points x, z [B] (clamped onto the map)."""
    c = cfg()
    C, cm = int(c["cells"]), float(c["cell_m"])
    s, f = axes.local(x, z)
    row, col = Z.cell_of(s, f, cm, C)
    return row.clamp(0, C - 1) * C + col.clamp(0, C - 1)


def extents(front, depth, heading, fb):
    """(es, ef) [...]: the half extents along the field's axes (bearing fb) of a rectangle front x depth turned heading."""
    a = (heading - fb) * (torch.pi / 180)
    ca, sa = torch.cos(a).abs(), torch.sin(a).abs()
    return front / 2 * ca + depth / 2 * sa, front / 2 * sa + depth / 2 * ca


def field_ok(x, z, front, depth, heading, field):
    """[R, M] bool: the zone front x depth [R] turned heading [R] centred at x, z [R, M] inside the field {x, z, b,
    front, depth} [R]."""
    ls, lf = Z.local(x, z, field["x"][:, None], field["z"][:, None], field["b"][:, None])
    es, ef = extents(front, depth, heading, field["b"])
    return (ls.abs() + es[:, None] <= field["front"][:, None] / 2 + TOL) & \
           (lf.abs() + ef[:, None] <= field["depth"][:, None] / 2 + TOL)


def inside(px, pz, front, depth, heading, field):
    """(x, z) [R]: the centre pushed inside the field along the field's axes (the least shift)."""
    ls, lf = Z.local(px, pz, field["x"], field["z"], field["b"])
    es, ef = extents(front, depth, heading, field["b"])
    hs = (field["front"] / 2 - es).clamp(min=0)
    hf = (field["depth"] / 2 - ef).clamp(min=0)
    ls = torch.maximum(torch.minimum(ls, hs), -hs)
    lf = torch.maximum(torch.minimum(lf, hf), -hf)
    return Z.world(ls, lf, field["x"], field["z"], field["b"])


def clamp(gx, gz, qx, qz, near_m, front, depth, heading, field):
    """(cx, cz) [R] (module doc). gx, gz the group's actual centre, qx, qz the goal, near_m the near window's side."""
    lim = (near_m / 2 - float(cfg()["step_margin_m"])).clamp(min=0)
    px, pz = qx, qz
    for _ in range(4):
        dx, dz = px - gx, pz - gz
        d = torch.sqrt(dx * dx + dz * dz)
        t = torch.where(d > lim, lim / d.clamp(min=1e-6), torch.ones_like(d))
        px, pz = gx + dx * t, gz + dz * t
        px, pz = inside(px, pz, front, depth, heading, field)
    return px, pz
