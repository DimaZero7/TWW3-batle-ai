"""The centralised critic: how good the battle is for a side, seeing the whole state.

Training only (MAPPO, Yu et al., 2021): it gets the full view (observation.observe(full=True):
every unit, exact morale, hidden enemies, the enemy's character) and is never shipped with the mod.
Its own weights, no memory: the full state already holds what the side would have to remember.

Two outputs: the side's value (its return: win, health, standing, lords, costs) and, per unit token,
the value of that unit's own return (tools/nn/train/reward.py unit_step: what happens to the unit
itself), for per-unit credit in PPO (tools/nn/train/ppo.py). The per-unit head starts at zero, so
a checkpoint without it loads with it empty (load()).
"""
import torch
from torch import nn

from tools.nn.model import observation as ob
from tools.nn.model.encoder import Block, TokenEncoder, attention_bias


class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.critic_d
        self.cfg = cfg
        self.encoder = TokenEncoder(d, n_ctx=ob.CONTEXT_FULL)
        self.dist = nn.Embedding(cfg.dist_bins + 1, cfg.critic_heads)
        self.blocks = nn.ModuleList(Block(d, cfg.critic_heads, cfg.ff) for _ in range(cfg.critic_layers))
        self.norm = nn.LayerNorm(d)
        self.value = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.unit_value = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.unit_value[2].weight)
        nn.init.zeros_(self.unit_value[2].bias)

    def load(self, state):
        """load_state_dict that accepts a checkpoint written before the per-unit head (it stays at zero)."""
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad = [k for k in missing if not k.startswith("unit_value.")] + list(unexpected)
        if bad:
            raise RuntimeError(f"critic state does not match: {bad[:5]}")
        return self

    def forward(self, obs_t, per_unit=False):
        """Full-view observation (torch dict) -> value [B]; per_unit: (value [B], per-unit value [B, N])."""
        x = self.encoder(obs_t["tokens"], obs_t["ctx"])
        bias = attention_bias(obs_t, self.dist, self.cfg.dist_bins)
        for block in self.blocks:
            x = block(x, bias)
        x = self.norm(x)
        units = x[:, 1:]

        def pool(mask):
            w = mask.float()[..., None]
            return (units * w).sum(1) / w.sum(1).clamp(min=1)
        own = obs_t["own"] & obs_t["attend"]
        enemy = ~obs_t["own"] & obs_t["attend"]
        value = self.value(torch.cat([x[:, 0], pool(own), pool(enemy)], -1))[..., 0]
        if not per_unit:
            return value
        ctx = x[:, :1].expand_as(units)
        return value, self.unit_value(torch.cat([units, ctx], -1))[..., 0]
