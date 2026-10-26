"""Who stands in our shooters' lines of fire (S28; a measure's definition, not the simulator's mechanics - the
simulator decides which shots friends take, missile.friend_los; this only says which friend stood there).

A line of fire: from a direct-fire shooter's centre to its target's centre. A friend k stands in it when its centre is
ahead of the shooter and short of the target (0 < along < the line's length) and its formation's band across the line
- the projection of its rectangle (geometry.dims: front x depth, facing b) on the line's normal - overlaps the
shooter's own band (half its front): |across| <= k's half projection + the shooter's half front. Units are boxes as
the simulator's (the same dims, the same spacing), the friend not the shooter itself.

    lines(u, b, s, shooter, target)    [R, N, N] i's line closed by friend k (rows of battles b, sides s)
    blocking(u, b, s, ...)            [R, N] k stands in a line of a shooter of its side that lost fire this step
                                      (the simulator's fc_skip > 0 at its aim): Runtime.note_fire sums it over steps
"""
import torch

from tools.nn.sim import geometry

SPACING_M = 1.5          # a man's place without its own (config/nn/sim.json formation.spacing_m: dims' fallback only)
DEG = 3.141592653589793 / 180.0


def _dims(u, b):
    keys = ("men", "men0", "width", "sp_h", "sp_v", "radius")
    sub = {k: u[k][b] for k in keys if k in u}
    return geometry.dims(sub, SPACING_M)


def lines(u, b, s, shooter, target):
    """[R, N (i), N (k)] bool: own unit k (side s, standing on the map, not i) stands in shooter i's line of fire at its
    target slot target [R, N] (-1: none) - module doc. shooter [R, N] bool: the units whose lines count."""
    x, z = u["x"][b].float(), u["z"][b].float()
    side = u["side"][b]
    alive = (side > 0) & (u["men"][b] > 0) & ~u["gone"][b]
    own = alive & (side == s[:, None])
    front, depth = _dims(u, b)
    t = target.clamp(min=0)
    dx, dz = x.gather(1, t) - x, z.gather(1, t) - z
    L = (dx * dx + dz * dz).sqrt().clamp(min=1e-3)
    ex, ez = dx / L, dz / L                                                   # [R, i] the line's direction
    rx, rz = x[:, None, :] - x[:, :, None], z[:, None, :] - z[:, :, None]      # [R, i, k] k from i
    along = rx * ex[..., None] + rz * ez[..., None]
    across = (rx * ez[..., None] - rz * ex[..., None]).abs()
    bk = u["b"][b].float() * DEG
    fx, fz = torch.sin(bk), torch.cos(bk)                                     # k's facing (x, z)
    # the normal (ez, -ex) against k's facing and its front axis (cos b, -sin b)
    nf = (ez[..., None] * fx[:, None, :] - ex[..., None] * fz[:, None, :]).abs()
    nl = (ez[..., None] * torch.cos(bk)[:, None, :] + ex[..., None] * torch.sin(bk)[:, None, :]).abs()
    half_k = front[:, None, :] / 2 * nl + depth[:, None, :] / 2 * nf
    reach = half_k + front[:, :, None] / 2
    eye = torch.eye(x.shape[1], dtype=torch.bool, device=x.device)[None]
    on = shooter & (target >= 0)
    return (on[..., None] & own[:, None, :] & ~eye & (along > 0) & (along < L[..., None]) & (across <= reach))


def blocking(u, b, s):
    """[R, N] bool: unit k stands in the line of fire of a direct-fire shooter of its side that lost fire to friends
    this step (fc_skip > 0, aimed: aim_tgt >= 0) - module doc. Zeros without the simulator's bookkeeping."""
    if "fc_skip" not in u:
        return torch.zeros_like(u["men"][b], dtype=torch.bool)
    side = u["side"][b]
    lost = (u["fc_skip"][b] > 0) & u["direct"][b] & (side == s[:, None])
    return lines(u, b, s, lost, u["aim_tgt"][b]).any(1)


def screen_lines(u, b, s, foe_standing):
    """[R, N] bool: own unit k stands in the line of fire of a standing own direct-fire shooter (range, ammunition, not
    routing, not in melee) at its aim (aim_tgt), else at the standing enemy nearest to it within its range - the
    lines a screen must keep clear (mission.py, --general-screen-clear)."""
    x, z = u["x"][b].float(), u["z"][b].float()
    side = u["side"][b]
    alive = (side > 0) & (u["men"][b] > 0) & ~u["gone"][b]
    sh = (alive & ~u["r"][b] & ~u["m"][b] & (side == s[:, None]) & (u["range"][b] > 0) & (u["a"][b] > 0)
          & u["direct"][b])
    D = ((x[:, :, None] - x[:, None, :]) ** 2 + (z[:, :, None] - z[:, None, :]) ** 2).sqrt()
    dE, jE = torch.where(foe_standing[:, None, :], D, torch.full_like(D, 1e9)).min(-1)
    aim = u["aim_tgt"][b] if "aim_tgt" in u else torch.full_like(jE, -1)
    tgt = torch.where(aim >= 0, aim, torch.where(dE <= u["range"][b].float(), jE, torch.full_like(jE, -1)))
    return lines(u, b, s, sh, tgt).any(1)
