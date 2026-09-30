"""Unit tokens and attention: every unit looks at the others.

* TokenEncoder: the same weights for every unit, own and enemy (the unit is known by its passport,
  not by its name); the side's context (character, role, time) is added to every token and is
  also a token of its own (index 0).
* Block: attention + feed-forward. Masks: padding, own dead units and known-dead enemies are
  never looked at. Bias: a learned number per head for the distance between two units (buckets),
  so "who is near" is easy; tokens without a known position use a bucket of their own.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from tools.nn.model import observation as ob
from tools.nn.model.lora import LoRALinear


class TokenEncoder(nn.Module):
    def __init__(self, d, n_token=ob.TOKEN, n_ctx=ob.CONTEXT):
        super().__init__()
        self.unit = nn.Sequential(nn.Linear(n_token, d), nn.GELU(), nn.Linear(d, d))
        self.ctx = nn.Sequential(nn.Linear(n_ctx, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens, ctx):
        """tokens [B, N, F], ctx [B, C] -> [B, 1 + N, d] (the context token first)."""
        c = self.ctx(ctx)
        u = self.unit(tokens) + c[:, None]
        return self.norm(torch.cat([c[:, None], u], 1))


def distance_buckets(pos, known, bins):
    """[B, L, L] bucket of the distance between tokens; bucket `bins` = unknown (and the context token).

    pos [B, N, 2] in POS units, known [B, N]. Buckets are log-spaced from 0 to ~1500 m.
    """
    d = torch.cdist(pos, pos) * ob.POS
    b = (torch.log1p(d / 10) / math.log1p(150) * (bins - 1)).clamp(0, bins - 1).long()
    ok = known[:, :, None] & known[:, None, :]
    b = torch.where(ok, b, torch.full_like(b, bins))
    return F.pad(b, (1, 0, 1, 0), value=bins)          # the context token


def attention_bias(obs_t, bias_table, bins):
    """Float mask [B, H, L, L]: learned distance bias, -inf for keys that may not be looked at."""
    known = obs_t["attend"] & (obs_t["tokens"][..., ob.INDEX["seen"]] > 0.5)
    buckets = distance_buckets(obs_t["pos"], known, bins)
    bias = bias_table(buckets).permute(0, 3, 1, 2)                     # [B, H, L, L]
    keys = F.pad(obs_t["attend"], (1, 0), value=True)                  # [B, L]
    return bias.masked_fill(~keys[:, None, None, :], float("-inf"))


class Block(nn.Module):
    def __init__(self, d, heads, ff, rank=0, adapters=0, alpha=8.0):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = LoRALinear(d, 3 * d, rank, adapters, alpha)
        self.out = LoRALinear(d, d, rank, adapters, alpha)
        self.f1 = LoRALinear(d, ff * d, rank, adapters, alpha)
        self.f2 = LoRALinear(ff * d, d, rank, adapters, alpha)

    def forward(self, x, bias, adapter=None):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x), adapter).reshape(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.out(a.transpose(1, 2).reshape(B, L, d), adapter)
        return x + self.f2(F.gelu(self.f1(self.n2(x), adapter)), adapter)
