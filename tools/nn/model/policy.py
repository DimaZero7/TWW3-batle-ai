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

    def _split(self):
        """Attention blocks before the memory (the rest come after it)."""
        return len(self.blocks) - 1 if self.memory is not None else len(self.blocks)

    def encode(self, obs_t, use_adapter=True):
        """Tokens -> the blocks before the memory: (x [B, 1 + N, d], attention bias, adapter)."""
        adapter = obs_t["adapter"] if use_adapter else None
        x = self.encoder(obs_t["tokens"], obs_t["ctx"])
        bias = attention_bias(obs_t, self.dist, self.cfg.dist_bins)
        for block in self.blocks[:self._split()]:
            x = block(x, bias, adapter)
        return x, bias, adapter

    def finish(self, x, bias, adapter, obs_t):
        """The blocks after the memory and the heads -> logits."""
        for block in self.blocks[self._split():]:
            x = block(x, bias, adapter)
        return self.heads(x, obs_t)

    @staticmethod
    def keep(obs_t):
        return torch.nn.functional.pad(obs_t["attend"], (1, 0), value=True).float()

    def forward(self, obs_t, h=None, use_adapter=True):
        """-> (logits, new memory). h: memory of the last decision (None: start of battle)."""
        x, bias, adapter = self.encode(obs_t, use_adapter)
        h = self.initial(obs_t) if h is None else h
        if self.memory is not None:
            x, h = self.memory(x, h, self.keep(obs_t))
        return self.finish(x, bias, adapter, obs_t), h

    def sequence(self, obs_seq, h0, reset, use_adapter=True):
        """Decisions in a row, for training the memory through time.

        obs_seq: dict of [T, B, ...]; h0 [B, 1 + N, d]: the memory before the first; reset [T, B]:
        a new battle begins at step t (the memory starts empty). -> (logits of [T, B, ...], last memory).
        The parts without memory run on all T x B at once; only the GRU steps one by one."""
        T, B = obs_seq["own"].shape[:2]
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs_seq.items()}
        x, bias, adapter = self.encode(flat, use_adapter)
        h = self.initial(flat)[:B] if h0 is None else h0
        if self.memory is not None:
            x = x.reshape(T, B, *x.shape[1:])
            keep = self.keep(flat).reshape(T, B, -1)
            out = []
            for t in range(T):
                h = h * (~reset[t]).float()[:, None, None]
                xt, h = self.memory(x[t], h, keep[t])
                out.append(xt)
            x = torch.stack(out).reshape(T * B, *x.shape[2:])
        logits = self.finish(x, bias, adapter, flat)
        return {k: v.reshape(T, B, *v.shape[1:]) for k, v in logits.items()}, h


def parameters(module):
    return sum(p.numel() for p in module.parameters())
