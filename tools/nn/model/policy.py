"""The actor: what the side's units do. It is what ships with the mod (the critic does not).

observation -> TokenEncoder -> attention blocks -> GRU per token -> last attention block -> heads.
Conditioning: the side's context (character, role, time) is in every token and is a token itself;
the LoRA adapter of (faction, role) is chosen per battle (off by default).
"""
import numpy as np
import torch
from torch import nn

from tools.nn.model import factions
from tools.nn.model import heads as hd
from tools.nn.model.encoder import Block, TokenEncoder, attention_bias
from tools.nn.model.memory import TokenMemory

OBS_KEYS = ("tokens", "ctx", "own", "attend", "ctrl", "target_ok", "pos", "adapter")


def to_torch(obs, device=None):
    """An Obs (numpy or torch arrays) -> dict of tensors on the device."""
    out = {}
    for k in OBS_KEYS:
        v = getattr(obs, k)
        out[k] = (torch.as_tensor(np.asarray(v)) if isinstance(v, np.ndarray) else v).to(device)
    return out


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        adapters = factions.ADAPTERS if cfg.lora_rank else 0
        self.encoder = TokenEncoder(cfg.d)
        self.dist = nn.Embedding(cfg.dist_bins + 1, cfg.heads)
        self.blocks = nn.ModuleList(Block(cfg.d, cfg.heads, cfg.ff, cfg.lora_rank, adapters, cfg.lora_alpha)
                                    for _ in range(cfg.layers))
        self.memory = TokenMemory(cfg.d) if cfg.memory else None
        self.heads = hd.Heads(cfg)

    def initial(self, obs_t):
        """Empty memory for a batch: [B, 1 + N, d]."""
        B, N = obs_t["own"].shape
        return torch.zeros(B, 1 + N, self.cfg.d, device=obs_t["tokens"].device)

    def forward(self, obs_t, h=None, use_adapter=True):
        """-> (logits, new memory). h: memory of the last decision (None: start of battle)."""
        adapter = obs_t["adapter"] if use_adapter else None
        x = self.encoder(obs_t["tokens"], obs_t["ctx"])
        bias = attention_bias(obs_t, self.dist, self.cfg.dist_bins)
        h = self.initial(obs_t) if h is None else h
        last = len(self.blocks) - 1
        for i, block in enumerate(self.blocks):
            if i == last and self.memory is not None:
                keep = torch.nn.functional.pad(obs_t["attend"], (1, 0), value=True).float()
                x, h = self.memory(x, h, keep)
            x = block(x, bias, adapter)
        return self.heads(x, obs_t), h


def parameters(module):
    return sum(p.numel() for p in module.parameters())
