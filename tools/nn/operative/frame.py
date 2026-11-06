"""The operative's frame (plan.md "Оси поля"): the whole field's axes, fixed for the battle.

    field_of(raw, bounds)   {x, z, b, front, depth, on} [B]: the playable field as a turned rectangle in the battle's
                            frame - the captured map's frame (terrain.frame_local), a flat battle the square +-bounds at
                            the origin (on False)
    Axes                    the centre: the field's centre; "up": the field's own axis (b + k x 90) nearest the
                            direction from my side's centre to the seen enemies' centre at the battle's first decision
                            (none seen: towards the field's centre; there: the field's b) - so the field never turns
                            in the map (a square turned by 45 deg does not fit) and either side sees itself "below"
    place(raw, axes)        [B, N, 3]: a unit's place - right, forward of the field's centre / place_m (1024 m) and
                            the ground under it less the ground under the field's centre / 20 m (+-4)

One formula on the simulator's raw state and on the bridge's (tools/nn/companion/operative_input.py): the field and the
direction come from the raw state alone, so a battle seen in the game's map coordinates gives the same numbers.
"""
import torch

from tools.nn.operative import cfg
from tools.nn.tactic import cfg as tcfg
from tools.nn.tactic import raw as TR
from tools.nn.unit import zmap as Z


def field_of(raw, bounds=None):
    from tools.nn.sim import terrain
    B = raw.shape[0]
    dev = raw.device
    g = TR.ground_of(raw)
    terr = g["terr"][:, 0]
    on = terr >= 0
    half = float(bounds) if bounds is not None else float(cfg()["place_m"])
    zero = torch.zeros(B, device=dev)
    out = {"x": zero.clone(), "z": zero.clone(), "b": zero.clone(), "front": torch.full((B,), 2 * half, device=dev),
           "depth": torch.full((B,), 2 * half, device=dev), "on": on}
    if bool(on.any()):
        cx, cz, b, front, depth = terrain.frame_local(terr, [g[k][:, 0] for k in terrain.PLACE])
        for k, v in (("x", cx), ("z", cz), ("b", b), ("front", front), ("depth", depth)):
            out[k] = torch.where(on, v.float(), out[k])
    return out


def centre(raw, sel):
    """([B], [B], [B] bool) the mean of the selected slots' centres sel [B, N] (and whether any)."""
    I = TR.I
    w = sel.float()
    k = w.sum(1)
    return (raw[..., I["x"]] * w).sum(1) / k.clamp(min=1), (raw[..., I["z"]] * w).sum(1) / k.clamp(min=1), k > 0


def sides(raw, side):
    """(mine, enemy seen) [B, N] bool of the operative's side side [B] (alive slots)."""
    I = TR.I
    alive = (raw[..., I["alive"]] > 0) & (raw[..., I["side"]] > 0)
    s = raw[..., I["side"]]
    mine = alive & (s == side[:, None].float())
    enemy = alive & (s != side[:, None].float()) & (raw[..., I["vis"]] > 0)
    return mine, enemy


class Axes:
    """The field's axes of B battles: cx, cz, heading [B] (degrees, the simulator's bearings) and the field."""

    def __init__(self, cx, cz, heading, field):
        self.cx, self.cz, self.heading, self.field = cx, cz, heading, field

    @classmethod
    def of(cls, raw, side, bounds=None):
        field = field_of(raw, bounds)
        mine, enemy = sides(raw, side)
        mx, mz, has_m = centre(raw, mine)
        ex, ez, has_e = centre(raw, enemy)
        tx = torch.where(has_e, ex, field["x"])
        tz = torch.where(has_e, ez, field["z"])
        way = torch.rad2deg(torch.atan2(tx - mx, tz - mz))
        ok = has_m & (((tx - mx) ** 2 + (tz - mz) ** 2) > 1e-6)
        k = torch.round(torch.remainder(way - field["b"], 360.0) / 90.0)
        heading = torch.where(ok, torch.remainder(field["b"] + 90.0 * k, 360.0), torch.remainder(field["b"], 360.0))
        return cls(field["x"], field["z"], heading, field)

    def take(self, k):
        return Axes(self.cx[k], self.cz[k], self.heading[k], {n: v[k] for n, v in self.field.items()})

    def local(self, x, z):
        """(s, f) of world points x, z [B, ...] in the axes."""
        sh = (-1,) + (1,) * (x.dim() - 1)
        return Z.local(x, z, self.cx.view(sh), self.cz.view(sh), self.heading.view(sh))

    def world(self, s, f):
        sh = (-1,) + (1,) * (s.dim() - 1)
        return Z.world(s, f, self.cx.view(sh), self.cz.view(sh), self.heading.view(sh))


def ground_at_g(g, x, z):
    """[B, ...] the ground under points x, z [B, ...] of each battle's map g (tactic.raw.ground_of; 0 flat)."""
    from tools.nn.sim import terrain
    return terrain.height(g, x.reshape(x.shape[0], -1), z.reshape(z.shape[0], -1)).view_as(x)


def ground_at(raw, x, z):
    return ground_at_g(TR.ground_of(raw), x, z)


def place(raw, axes):
    """[B, N, 3] (module doc)."""
    I = TR.I
    c, t = cfg(), tcfg()
    s, f = axes.local(raw[..., I["x"]], raw[..., I["z"]])
    h0 = ground_at(raw, axes.cx[:, None], axes.cz[:, None])
    cap = float(t["height_cap"])
    dh = ((raw[..., I["gh"]] - h0) / float(t["height_m"])).clamp(-cap, cap)
    k = float(c["place_m"])
    return torch.stack([s / k, f / k, dh], -1)
