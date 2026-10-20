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


def from_state(cfg, u, cost, side, frame, cells_in):
    """[R, S, 8] of the battles' state for rows of side `side` [R] (1 or 2): u the state's fields of the rows' battles
    [R, N] (x, z, side, men, hp, r, gone), cost [R, N] (multiplayer gold), frame: the rows' Frame, cells_in [R, S, k2].
    Every unit seen and visible (the deployment); the dead and those gone left out (as attend)."""
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
    return core(cfg, pos, own, attend, present, present, cost.float(), u["hp"].float(), rout, cells_in)


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


ZoneRule = gdeploy.ZoneRule
