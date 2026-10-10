"""Pair features of two units (v2, ModelConfig.pairs): what the network knows of a pair beyond their distance.

The attention between the units had only a learned bias per head for the distance between two units (encoder.py,
16 buckets). Who reaches whom, who closes in on whom and from which side came only through the tokens, and the
attention's q.k had to work them out itself. Now, for every pair (i looks at j), from the side's own observation (the
unit tokens and their positions: the same in the simulator and the companion, nothing new to send):

* reach: the distance / the missile range of i, and of j (0 for a unit without missiles; at most REACH_CAP), and
  whether j is in i's range and i in j's (0 / 1) - who can shoot whom;
* closing: the speed at which the two close in, m/s / VEL: minus the projection of their relative velocity on the
  line between them (> 0: they come nearer), within +-CLOSE_CAP; both must be seen now (an unseen enemy's velocity is
  not known);
* bearing: where j stands as i faces (cos, sin of the angle from i's front to j: cos < 0 behind it, sin > 0 to its
  right) and where i stands as j faces - flank and rear, both ways; both seen now.

Each comes twice: for a pair of the same side and for a pair of opposite sides (an enemy closing in is not a friend
closing in; one linear layer cannot tell them apart otherwise): FEATURES x 2 = SIZE numbers per pair. The context
token and pairs with an unknown position give zeros.

The network uses them in two places, each a linear layer started at zero (so a network trained without them computes
exactly what it did: the loaded weights are kept and the new ones are 0, policy.Actor.load_state_dict):
* PairBias: + a number per attention head to the distance bias of every units' attention block (who to look at);
* the target pointer (chain.py): + a number to the logit of attacking j (who to attack).
"""
import numpy as np
import torch
from torch import nn

from tools.nn.model import observation as ob
from tools.nn.model import passport

FEATURES = ("reach_i", "reach_j", "in_range_i", "in_range_j", "closing", "bearing_cos_i", "bearing_sin_i",
            "bearing_cos_j", "bearing_sin_j")
SIZE = 2 * len(FEATURES)       # x (same side, opposite sides)
REACH_CAP = 2.0                # distance / range, at most
CLOSE_CAP = 2.0                # closing speed / VEL, within +-
RANGE_SCALE = 500.0            # the passport's range_m / 500 (passport.features)


def _range_column():
    """The passport column of the missile range (range_m / RANGE_SCALE)."""
    base = dict(men=1, hp_total=1, hp_per_man=1, mass=1, armour=0, leadership=0, multiplayer_cost=0, caste="", size="",
                speed={"walk": 0, "run": 0, "charge": 0}, shield={"missile_block_chance": 0},
                melee={"attack": 0, "defence": 0, "charge_bonus": 0, "damage": 0, "ap_damage": 0, "bonus_v_large": 0,
                       "bonus_v_infantry": 0, "attack_interval_s": 0, "splash_max_attacks": 1})
    a = np.array(passport.features(dict(base, missile={"range_m": 0})))
    b = np.array(passport.features(dict(base, missile={"range_m": RANGE_SCALE})))
    (i,) = np.nonzero(a != b)[0]
    assert abs(b[i] - 1.0) < 1e-9
    return ob.INDEX[f"passport_{i}"]


RANGE = _range_column()


def features(obs_t):
    """[B, 1 + N, 1 + N, SIZE] the pair features (module doc); the context token's row and column are 0."""
    tok = obs_t["tokens"]
    dt = tok.dtype
    pos = obs_t["pos"].to(dt) * ob.POS                                            # [B, N, 2] m (forward, lateral)
    known = obs_t["attend"] & (tok[..., ob.INDEX["seen"]] > 0.5)
    now = known & (tok[..., ob.INDEX["visible"]] > 0.5)
    rel = pos[:, None, :, :] - pos[:, :, None, :]                                 # [B, i, j, 2]: from i to j
    d = rel.square().sum(-1).clamp(min=1e-6).sqrt()
    u = rel / d[..., None]
    rng = tok[..., RANGE] * RANGE_SCALE                                           # [B, N] m (0: no missiles)
    shoots = rng > 0
    r_safe = torch.where(shoots, rng, torch.ones_like(rng))
    reach_i = torch.where(shoots[:, :, None], (d / r_safe[:, :, None]).clamp(max=REACH_CAP), torch.zeros_like(d))
    reach_j = torch.where(shoots[:, None, :], (d / r_safe[:, None, :]).clamp(max=REACH_CAP), torch.zeros_like(d))
    in_i = (shoots[:, :, None] & (d <= rng[:, :, None])).to(dt)
    in_j = (shoots[:, None, :] & (d <= rng[:, None, :])).to(dt)
    vel = torch.stack([tok[..., ob.INDEX["vel_fwd"]], tok[..., ob.INDEX["vel_lat"]]], -1)        # / VEL
    rel_v = vel[:, None, :, :] - vel[:, :, None, :]                               # j's velocity minus i's
    closing = (-(rel_v * u).sum(-1)).clamp(-CLOSE_CAP, CLOSE_CAP)
    face = torch.stack([tok[..., ob.INDEX["face_cos"]], tok[..., ob.INDEX["face_sin"]]], -1)     # (fwd, lat)
    fi, fj = face[:, :, None, :], face[:, None, :, :]
    cos_i = (fi * u).sum(-1)                                                      # j as i faces
    sin_i = fi[..., 0] * u[..., 1] - fi[..., 1] * u[..., 0]                       # > 0: j to i's right
    cos_j = -(fj * u).sum(-1)                                                     # i as j faces (j -> i is -u)
    sin_j = -(fj[..., 0] * u[..., 1] - fj[..., 1] * u[..., 0])
    both = known[:, :, None] & known[:, None, :]
    seen_now = now[:, :, None] & now[:, None, :]
    apart = d > 1.0                                                                # (no bearing for one place)
    zero = torch.zeros_like(d)
    feats = torch.stack([reach_i, reach_j, in_i, in_j]
                        + [torch.where(apart, c, zero) for c in (closing, cos_i, sin_i, cos_j, sin_j)], -1)
    ok = torch.stack([both] * 4 + [seen_now] * 5, -1)
    feats = torch.where(ok, feats, torch.zeros_like(feats))
    own = obs_t["own"]
    same = (own[:, :, None] == own[:, None, :])[..., None].to(dt)
    out = torch.cat([feats * same, feats * (1 - same)], -1)
    return torch.nn.functional.pad(out, (0, 0, 1, 0, 1, 0))                       # the context token


class PairBias(nn.Module):
    """The pair features -> + a number per attention head (started at zero: nothing changes until trained)."""

    def __init__(self, heads):
        super().__init__()
        self.lin = nn.Linear(SIZE, heads, bias=False)
        nn.init.zeros_(self.lin.weight)

    def forward(self, f):
        """f [B, L, L, SIZE] -> [B, heads, L, L]."""
        return self.lin(f).permute(0, 3, 1, 2)
