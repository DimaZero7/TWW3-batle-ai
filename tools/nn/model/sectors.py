"""The map as sector tokens (v2, ModelConfig.sectors > 0): what the units see of the ground around them.

The square +-cfg.grid_half m around the map's centre in the side's frame (fixed for the battle, as the grid head's,
heads.py) is cut into sectors x sectors sectors (16 x 16 of 100 m) and each sector into fine x fine cells (4 x 4 of
25 m): the move point is one of the (sectors x fine)^2 cells (chain.py). Sector i * sectors + j is the i-th forward,
the j-th lateral; cell (sector, k) is the fine cell k = a * fine + b of it (a forward, b lateral).

A sector's features (FEATURES), from the side's own observation only (the unit tokens): own strength (multiplayer
cost x share of health left, summed), own units, the strength and number of the enemies visible now, the strength of
the enemies seen earlier and not now (at their last seen place, cost only: their health is not known), whether an own
or a visible enemy unit there routs, and the share of its fine cells inside the map. The token: a learned embedding
of the sector's place + a small network of its features.

The geometry (Obs "geo" [B, 8]: the frame's origin and forward, the map's bounds) comes with the observation, so the
cells inside the map are known to the network (cells outside are never chosen) and stored with a decision (8 numbers,
not the 4096 cells).

The units look at the sectors through one attention layer (SectorAttention: queries from the unit tokens, keys and
values from the sector tokens, a learned bias per head for the distance from the unit to the sector's centre); the
sectors do not look at each other (256 x 256 per decision would cost more than all the units' attention).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tools.nn.model import observation as ob
from tools.nn.model import passport

FEATURES = ("own_strength", "own_units", "enemy_strength", "enemy_units", "enemy_earlier", "own_rout", "enemy_rout",
            "inside")
STRENGTH = 500.0     # gold: the strength log1p(sum / STRENGTH) / log1p(40) (one ~500-gold unit 0.19, 20 000 gold 1)
UNITS = 4.0          # units / UNITS


def _cost_column():
    """The passport column of the multiplayer cost (passport._log(cost, 5000))."""
    base = dict(men=1, hp_total=1, hp_per_man=1, mass=1, armour=0, leadership=0, multiplayer_cost=0, caste="", size="",
                speed={"walk": 0, "run": 0, "charge": 0}, shield={"missile_block_chance": 0},
                melee={"attack": 0, "defence": 0, "charge_bonus": 0, "damage": 0, "ap_damage": 0, "bonus_v_large": 0,
                       "bonus_v_infantry": 0, "attack_interval_s": 0, "splash_max_attacks": 1})
    a, b = np.array(passport.features(base)), np.array(passport.features(dict(base, multiplayer_cost=5000)))
    (i,) = np.nonzero(a != b)[0]
    return ob.INDEX[f"passport_{i}"]


COST = _cost_column()
COST_TOP = 5000.0


def step(cfg):
    """A sector's side, m."""
    return 2 * cfg.grid_half / cfg.sectors


def centres(cfg, device=None):
    """(forward, lateral) m of every sector's centre: [S, 2]."""
    n, s = cfg.sectors, step(cfg)
    c = -cfg.grid_half + (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * s
    f, l = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([f.reshape(-1), l.reshape(-1)], -1)


def cells(cfg, device=None):
    """(forward, lateral) m of every fine cell's centre, sector-major: [S, fine^2, 2]."""
    k = cfg.fine
    sub = step(cfg) / k
    o = -step(cfg) / 2 + (torch.arange(k, device=device, dtype=torch.float32) + 0.5) * sub
    a, b = torch.meshgrid(o, o, indexing="ij")
    off = torch.stack([a.reshape(-1), b.reshape(-1)], -1)                    # [k^2, 2]
    return centres(cfg, device)[:, None, :] + off[None]


def point_index(cfg, sector, fine):
    """(sector, fine cell) -> the cell's index in the (sectors x fine)^2 grid (row-major: forward, lateral)."""
    n, k = cfg.sectors, cfg.fine
    si, sj = sector // n, sector % n
    a, b = fine // k, fine % k
    return (si * k + a) * (n * k) + sj * k + b


def split(cfg, point):
    """The grid cell's index -> (sector, fine cell): point_index's inverse."""
    n, k = cfg.sectors, cfg.fine
    row, col = point // (n * k), point % (n * k)
    return (row // k) * n + col // k, (row % k) * k + col % k


def geo(frame, bounds):
    """The geometry stored with a decision: [B, 8] = frame cx, cz, ux, uz, bounds xmin, xmax, zmin, zmax."""
    f = [torch.as_tensor(getattr(frame, k), device=bounds.device).float().reshape(-1) for k in ("cx", "cz", "ux", "uz")]
    return torch.cat([torch.stack(f, -1), bounds.float()], -1)


def inside(cfg, g):
    """[B, S, fine^2] the fine cells whose centre is inside the map (g: geo [B, 8])."""
    c = cells(cfg, g.device)                                                  # [S, k2, 2]
    f, l = c[..., 0][None], c[..., 1][None]                                   # [1, S, k2]
    cx, cz, ux, uz = (g[:, i].reshape(-1, 1, 1) for i in range(4))
    x = f * ux + l * uz + cx
    z = f * uz - l * ux + cz
    b = g[:, 4:].reshape(-1, 4, 1, 1)
    return (x >= b[:, 0]) & (x <= b[:, 1]) & (z >= b[:, 2]) & (z <= b[:, 3])


def features(cfg, obs_t, cells_in):
    """[B, S, len(FEATURES)] (module doc). cells_in: inside() [B, S, k2]."""
    tok = obs_t["tokens"]
    S = cfg.sectors * cfg.sectors
    pos = obs_t["pos"].float() * ob.POS                                       # [B, N, 2] current or last seen
    n, s = cfg.sectors, step(cfg)
    ij = torch.floor((pos + cfg.grid_half) / s).long()
    on = (ij >= 0).all(-1) & (ij < n).all(-1)
    idx = ij[..., 0].clamp(0, n - 1) * n + ij[..., 1].clamp(0, n - 1)        # [B, N]
    seen = tok[..., ob.INDEX["seen"]] > 0.5
    vis = tok[..., ob.INDEX["visible"]] > 0.5
    own = obs_t["own"] & obs_t["attend"] & on
    enemy = ~obs_t["own"] & obs_t["attend"] & on & seen
    cost = torch.expm1(tok[..., COST] * math.log1p(COST_TOP))
    strength = cost * tok[..., ob.INDEX["hp"]]
    rout = (tok[..., ob.INDEX["state_routing"]] + tok[..., ob.INDEX["state_shattered"]]) > 0.5
    one = (idx[..., None] == torch.arange(S, device=idx.device)).to(tok.dtype)           # [B, N, S]
    cols = torch.stack([strength * own, own.to(tok.dtype), strength * (enemy & vis), (enemy & vis).to(tok.dtype),
                        cost * (enemy & ~vis), (own & rout).to(tok.dtype), (enemy & vis & rout).to(tok.dtype)], -1)
    per = torch.einsum("bns,bnf->bsf", one, cols)                                         # [B, S, 7]
    top = math.log1p(40.0)
    out = torch.stack([torch.log1p(per[..., 0] / STRENGTH) / top, per[..., 1] / UNITS,
                       torch.log1p(per[..., 2] / STRENGTH) / top, per[..., 3] / UNITS,
                       torch.log1p(per[..., 4] / STRENGTH) / top, (per[..., 5] > 0).to(tok.dtype),
                       (per[..., 6] > 0).to(tok.dtype), cells_in.to(tok.dtype).mean(-1)], -1)
    return out


class SectorEncoder(nn.Module):
    """Sector features -> tokens [B, S, sector_d]: a learned embedding of the place + a small network of the features."""

    def __init__(self, cfg):
        super().__init__()
        ds = cfg.sector_d
        self.place = nn.Parameter(torch.randn(cfg.sectors * cfg.sectors, ds) * 0.02)
        self.net = nn.Sequential(nn.Linear(len(FEATURES), ds), nn.GELU(), nn.Linear(ds, ds))
        self.norm = nn.LayerNorm(ds)

    def forward(self, f):
        return self.norm(self.place + self.net(f))


def distance_buckets(cfg, pos, known, bins):
    """[B, 1 + N, S] bucket of the distance from each token to each sector's centre (as encoder.distance_buckets:
    log-spaced 0 to ~1500 m); bucket `bins` for the context token and tokens without a known position."""
    c = centres(cfg, pos.device).to(pos.dtype)
    d = torch.cdist(pos * ob.POS, c[None].expand(pos.shape[0], -1, -1))
    b = (torch.log1p(d / 10) / math.log1p(150) * (bins - 1)).clamp(0, bins - 1).long()
    b = torch.where(known[..., None], b, torch.full_like(b, bins))
    return F.pad(b, (0, 0, 1, 0), value=bins)


class SectorAttention(nn.Module):
    """The units (and the context token) look at the sectors: x [B, L, d] += attention(x -> sector tokens)."""

    def __init__(self, cfg):
        super().__init__()
        d, ds = cfg.d, cfg.sector_d
        self.heads = cfg.sector_heads
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, ds)
        self.kv = nn.Linear(ds, 2 * ds)
        self.out = nn.Linear(ds, d)
        self.dist = nn.Embedding(cfg.dist_bins + 1, cfg.sector_heads)
        self.bins = cfg.dist_bins
        self.cfg = cfg

    def buckets(self, obs_t):
        """[B, 1 + N, S] the distance buckets (distance_buckets) of the observation."""
        known = obs_t["attend"] & (obs_t["tokens"][..., ob.INDEX["seen"]] > 0.5)
        return distance_buckets(self.cfg, obs_t["pos"].float(), known, self.bins)

    def forward(self, x, s, buckets):
        B, L, _ = x.shape
        bias = self.dist(buckets).permute(0, 3, 1, 2)                           # [B, H, L, S]
        S, ds, H = s.shape[1], s.shape[2], self.heads
        q = self.q(self.norm(x)).reshape(B, L, H, ds // H).transpose(1, 2)
        k, v = self.kv(s).reshape(B, S, 2, H, ds // H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return x + self.out(a.transpose(1, 2).reshape(B, L, ds))
