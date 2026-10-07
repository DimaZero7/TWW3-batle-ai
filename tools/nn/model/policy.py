"""The actor: what the side's units do. It is what ships with the mod (the critic does not).

observation -> TokenEncoder (+ AbilityEncoder) -> attention blocks -> GRU per token -> last attention block
-> heads.
Conditioning: the side's context (character, role, time) is in every token and is a token itself.
"""
import numpy as np
import torch
from torch import nn

from tools.nn.model import heads as hd
from tools.nn.model.encoder import AbilityEncoder, Block, TokenEncoder, attention_bias
from tools.nn.model.memory import TokenMemory

OBS_KEYS = ("tokens", "ctx", "own", "attend", "ctrl", "target_ok", "pos", "abil", "abil_ok")
# Parameters an actor saved before abilities lacks: they start fresh when it is loaded.
ABILITY_PARAMS = ("abilities.", "heads.ability_")
# ... and one of the bin head loaded into a grid actor (cfg.grid): its grid head starts fresh.
FRESH_PARAMS = ABILITY_PARAMS + ("heads.cell_",)


def to_torch(obs, device=None):
    """An Obs (numpy or torch arrays) -> dict of tensors on the device."""
    out = {}
    for k in OBS_KEYS:
        v = getattr(obs, k)
        if v is not None:
            out[k] = (torch.as_tensor(np.asarray(v)) if isinstance(v, np.ndarray) else v).to(device)
    return out


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = TokenEncoder(cfg.d)
        self.abilities = AbilityEncoder(cfg.d)
        self.dist = nn.Embedding(cfg.dist_bins + 1, cfg.heads)
        self.blocks = nn.ModuleList(Block(cfg.d, cfg.heads, cfg.ff) for _ in range(cfg.layers))
        self.memory = TokenMemory(cfg.d) if cfg.memory else None
        self.heads = hd.Heads(cfg)

    def initial(self, obs_t):
        """Empty memory for a batch: [B, 1 + N, d]."""
        B, N = obs_t["own"].shape
        return torch.zeros(B, 1 + N, self.cfg.d, device=obs_t["tokens"].device)

    def _split(self):
        """Attention blocks before the memory (the rest come after it)."""
        return len(self.blocks) - 1 if self.memory is not None else len(self.blocks)

    def encode(self, obs_t):
        """Tokens -> the blocks before the memory: (x [B, 1 + N, d], attention bias)."""
        x = self.encoder(obs_t["tokens"], obs_t["ctx"])
        if obs_t.get("abil") is not None:
            x = x + torch.nn.functional.pad(self.abilities(obs_t["abil"]), (0, 0, 1, 0))
        bias = attention_bias(obs_t, self.dist, self.cfg.dist_bins)
        for block in self.blocks[:self._split()]:
            x = block(x, bias)
        return x, bias

    def finish(self, x, bias, obs_t):
        """The blocks after the memory and the heads -> logits."""
        for block in self.blocks[self._split():]:
            x = block(x, bias)
        return self.heads(x, obs_t)

    @staticmethod
    def keep(obs_t):
        return torch.nn.functional.pad(obs_t["attend"], (1, 0), value=True).float()

    def forward(self, obs_t, h=None):
        """-> (logits, new memory). h: memory of the last decision (None: start of battle)."""
        x, bias = self.encode(obs_t)
        h = self.initial(obs_t) if h is None else h
        if self.memory is not None:
            x, h = self.memory(x, h, self.keep(obs_t))
        return self.finish(x, bias, obs_t), h

    def sequence(self, obs_seq, h0, reset):
        """Decisions in a row, for training the memory through time.

        obs_seq: dict of [T, B, ...]; h0 [B, 1 + N, d]: the memory before the first; reset [T, B]:
        a new battle begins at step t (the memory starts empty). -> (logits of [T, B, ...], last memory).
        The parts without memory run on all T x B at once; only the GRU steps one by one."""
        T, B = obs_seq["own"].shape[:2]
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs_seq.items()}
        x, bias = self.encode(flat)
        h = self.initial(flat)[:B] if h0 is None else h0
        if self.memory is not None:
            x, h = self.memory.scan(x.reshape(T, B, *x.shape[1:]), h, self.keep(flat).reshape(T, B, -1), reset)
            x = x.reshape(T * B, *x.shape[2:])
        logits = self.finish(x, bias, flat)
        return {k: v.reshape(T, B, *v.shape[1:]) for k, v in logits.items()}, h


    def load_state_dict(self, state_dict, strict=True, assign=False):
        """As nn.Module's, but an actor saved before abilities loads: its ability parts start fresh
        (the encoder's adds nothing until trained; the head chooses at random among ready abilities)."""
        missing, unexpected = super().load_state_dict(state_dict, strict=False, assign=assign)
        bad = [k for k in missing if not k.startswith(FRESH_PARAMS)] + list(unexpected)
        if strict and bad:
            raise RuntimeError(f"actor state does not match: {bad[:5]}")
        return missing, unexpected


def parameters(module):
    return sum(p.numel() for p in module.parameters())
