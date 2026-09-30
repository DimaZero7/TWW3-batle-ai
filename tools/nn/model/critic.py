"""The centralised critic: how good the battle is for a side, seeing the whole state.

Training only (MAPPO, Yu et al., 2021): it gets the full view (observation.observe(full=True):
every unit, exact morale, hidden enemies, the enemy's character) and is never shipped with the mod.
Its own weights, no memory: the full state already holds what the side would have to remember.
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

    def forward(self, obs_t):
        """Full-view observation (torch dict) -> value [B]."""
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
        return self.value(torch.cat([x[:, 0], pool(own), pool(enemy)], -1))[..., 0]
