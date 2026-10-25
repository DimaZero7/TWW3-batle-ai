"""The general's map tiles in the skeleton (docs/en/training/general.md, simplification 1) and the deployment zone
(simplification 5).

Tiles: the unit network's map sectors (tools/nn/model/sectors.py): 16 x 16 sectors of 100 m in the side's frame, 8
numbers each (own strength and units, the enemy's seen now and earlier, routs, the share inside the map) - the same
numbers the unit network reads, from the side's own observation in a re-plan (from_obs = sectors.features). At the
deployment, before the battle's first observation, from the simulator's state (from_state: the same arithmetic, every
unit seen - the game shows both armies in the deployment; test: from_state == from_obs at a battle's start).

The zone (ZoneRule): where tools/nn/armies/place.py deploys a side - the arena's frame of tools/nn/scenario.py (side 1
at x = -gap / 2 facing +x, side 2 at +gap / 2 facing -x), forward from ZONE_FRONT_M - deployment_m to ZONE_FRONT_M,
lateral within +-deployment_m / 2. A tile is in the zone when its centre is (zone_tiles). A battle whose units do not
all stand in their zone (a kite battle, a drill: other placements) keeps its scripted placement (valid).
"""
import math

import torch

from tools.nn.general import deploy as gdeploy
from tools.nn.model import sectors as msectors
from tools.nn.model.frame import Frame, army_axis

FEATURES = msectors.FEATURES


def from_obs(cfg, obs_t, cells_in):
    """[R, S, 8] of an observation (rows of observation.observe's dict with tokens, own, attend, pos); cells_in
    [R, S, fine^2] (sectors.inside of the rows' geometry). cfg: the unit network's ModelConfig (its sectors)."""
    return msectors.features(cfg, obs_t, cells_in)


def core(cfg, pos_m, own, attend, seen, vis, cost, hp, rout, cells_in):
    """sectors.features' arithmetic on plain tensors [R, N] (pos_m [R, N, 2]: forward, lateral in m)."""
    n, s = cfg.sectors, msectors.step(cfg)
    ij = torch.floor((pos_m + cfg.grid_half) / s).long()
    on = (ij >= 0).all(-1) & (ij < n).all(-1)
    idx = ij[..., 0].clamp(0, n - 1) * n + ij[..., 1].clamp(0, n - 1)
    mine = own & attend & on
    enemy = ~own & attend & on & seen
    strength = cost * hp
    one = msectors.one_hot(cfg, idx, pos_m.dtype)
    f = pos_m.dtype
    cols = torch.stack([strength * mine, mine.to(f), strength * (enemy & vis), (enemy & vis).to(f),
                        cost * (enemy & ~vis), (mine & rout).to(f), (enemy & vis & rout).to(f)], -1)
    per = torch.einsum("bns,bnf->bsf", one, cols)
    top = math.log1p(40.0)
    S0 = msectors.STRENGTH
    return torch.stack([torch.log1p(per[..., 0] / S0) / top, per[..., 1] / msectors.UNITS,
                        torch.log1p(per[..., 2] / S0) / top, per[..., 3] / msectors.UNITS,
                        torch.log1p(per[..., 4] / S0) / top, (per[..., 5] > 0).to(f),
                        (per[..., 6] > 0).to(f), cells_in.to(f).mean(-1)], -1)


# GeneralConfig.tile_feats "v2": the sector numbers that are always zero in the simulator (it hides nobody, and a
# plan's tiles hardly ever hold a rout: the atlas of g17r, their spread 0.000) give their columns to these, on the same
# scale as the strengths (log1p(strength / STRENGTH) / log1p(40)): own shooters' strength, the enemy's shooters' (seen
# now), the enemy's fast units' (seen now; groups.roles "fast")
EXTRA = {4: "own_shooters", 5: "enemy_shooters", 6: "enemy_fast"}


def extra(cfg, pos_m, own, alive, vis, cost, hp, shooter, fast):
    """[R, S, 3] the v2 numbers (EXTRA) of plain tensors [R, N] (pos_m [R, N, 2]: forward, lateral in m)."""
    n, s = cfg.sectors, msectors.step(cfg)
    ij = torch.floor((pos_m + cfg.grid_half) / s).long()
    on = (ij >= 0).all(-1) & (ij < n).all(-1)
    idx = ij[..., 0].clamp(0, n - 1) * n + ij[..., 1].clamp(0, n - 1)
    mine = own & alive & on
    enemy = ~own & alive & on & vis
    strength = (cost * hp).float()
    one = msectors.one_hot(cfg, idx, pos_m.dtype)
    cols = torch.stack([strength * (mine & shooter), strength * (enemy & shooter), strength * (enemy & fast)], -1)
    per = torch.einsum("bns,bnf->bsf", one, cols.to(pos_m.dtype))
    return torch.log1p(per / msectors.STRENGTH) / math.log1p(40.0)


def with_extra(cfg, tiles, u, own, alive, vis, num, lab, frame):
    """tiles [R, S, 8] with EXTRA's columns of the rows' state u [R, N] (x, z, cost, hp), own / alive / vis [R, N],
    the slots' numbers num and labels lab (units.py: the missile range and ammunition, the role) and the rows' frame."""
    from tools.nn.general import army_input as ai
    from tools.nn.general import groups as ggroups
    f, l = frame.point(u["x"].float(), u["z"].float())
    pos = torch.nan_to_num(torch.stack([f, l], -1))
    shooter = ai.armed(num)                      # (with ammunition left: an empty shooter is no shooter)
    fast = ggroups.roles(num, lab) == ggroups.ROLES.index("fast")
    ex = extra(cfg, pos, own, alive, vis, u["cost"].float(), u["hp"].float(), shooter, fast)
    out = tiles.clone()
    out[..., list(EXTRA)] = ex.to(out.dtype)
    return out


def from_state(cfg, u, cost, side, frame, cells_in, seen=None):
    """[R, S, 8] of the battles' state for rows of side `side` [R] (1 or 2): u the state's fields of the rows' battles
    [R, N] (x, z, side, men, hp, r, gone), cost [R, N] (multiplayer gold), frame: the rows' Frame, cells_in [R, S, k2].
    Every unit seen and visible (the deployment) but where seen [R, N] says not (an enemy hidden from the start:
    tools/nn/sim/stealth.py; None: all); the dead and those gone left out (as attend)."""
    present = u["side"] > 0
    alive = present & (u["men"] > 0) & ~u["gone"]
    own = present & (u["side"] == side[:, None])
    f, l = frame.point(u["x"].float(), u["z"].float())
    pos = torch.stack([f, l], -1)
    # the observation's attend: present, not a dead own unit, not an enemy seen dead (all are seen here)
    attend = present & alive
    # the token's routing columns are the morale state's (ms 6 routing, 7 shattered)
    ms = u["ms"].float()
    rout = (ms >= 5.5) & present
    seen = present if seen is None else (seen & present)
    return core(cfg, pos, own, attend, seen, seen, cost.float(), u["hp"].float(), rout, cells_in)


def frame_at(u, side, bounds):
    """The side's frame as observation.start makes it (the map's centre, forward from its army to the enemy's): rows
    [R] of side `side`, u fields [R, N], bounds [R, 4]."""
    x, z = torch.nan_to_num(u["x"].float()), torch.nan_to_num(u["z"].float())
    present = u["side"] > 0
    own, enemy = present & (u["side"] == side[:, None]), present & (u["side"] != side[:, None])
    ux, uz = army_axis(x, z, own, enemy)
    return Frame((bounds[:, 0] + bounds[:, 1]) / 2, (bounds[:, 2] + bounds[:, 3]) / 2, ux, uz)


def centres(cfg, device=None):
    """[S, 2] (forward, lateral) m of every tile's centre in the side's frame (sectors.centres)."""
    return msectors.centres(cfg, device)


def centres_world(cfg, frame):
    """[R, S, 2] (x, z) of every tile's centre for the rows' frames."""
    c = centres(cfg, frame.cx.device)
    f = c[None, :, 0].expand(frame.cx.shape[0], -1)
    l = c[None, :, 1].expand(frame.cx.shape[0], -1)
    x, z = frame.world(f, l)
    return torch.stack([x, z], -1)


def coarse_index(sectors, grid, device=None):
    """[S] the coarse tile (GeneralConfig.target_grid: grid x grid blocks of the sectors, row = forward, as the
    sectors' index) of every sector. The grid must split the sectors evenly."""
    if sectors % grid:
        raise ValueError(f"general target_grid {grid} does not split {sectors} sectors evenly")
    b = sectors // grid
    i = torch.arange(sectors, device=device)
    return ((i[:, None] // b) * grid + (i[None, :] // b)).reshape(-1)


def coarse_centres(cfg, grid, device=None):
    """[C, 2] (forward, lateral) m of every coarse tile's centre in the side's frame (the mean of its sectors')."""
    c = centres(cfg, device)
    idx = coarse_index(cfg.sectors, grid, c.device)
    out = torch.zeros(grid * grid, 2, device=c.device).index_add_(0, idx, c)
    return out / float((cfg.sectors // grid) ** 2)


def coarse_world(cfg, frame, grid):
    """[R, C, 2] (x, z) of every coarse tile's centre for the rows' frames."""
    c = coarse_centres(cfg, grid, frame.cx.device)
    f = c[None, :, 0].expand(frame.cx.shape[0], -1)
    l = c[None, :, 1].expand(frame.cx.shape[0], -1)
    x, z = frame.world(f, l)
    return torch.stack([x, z], -1)


ZoneRule = gdeploy.ZoneRule
