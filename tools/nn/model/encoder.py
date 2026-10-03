"""Unit tokens and attention: every unit looks at the others.

* TokenEncoder: the same weights for every unit, own and enemy (the unit is known by its passport,
  not by its name); the side's context (character, role, time) is added to every token and is
  also a token of its own (index 0). Inputs added since a checkpoint was saved (passport features
  appended at the token's end) load with zero weights (pad_inputs): it computes what it did.
* AbilityEncoder: each ability slot (its state and passport, tools/nn/model/abilities.py) through
  the same small network; the sum over the unit's owned slots is added to the unit's token, so the
  order of the slots does not matter. Its last layer starts at zero: an actor trained before
  abilities sees exactly what it saw.
* Block: attention + feed-forward. Masks: padding, own dead units and known-dead enemies are
  never looked at. Bias: a learned number per head for the distance between two units (buckets),
  so "who is near" is easy; tokens without a known position use a bucket of their own.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from tools.nn.model import abilities as ab
from tools.nn.model import observation as ob


def pad_inputs(state_dict, key, n_in):
    """A first layer saved with fewer inputs (inputs added since at the END: unit passport features,
    tools/nn/model/passport.py; ability features, tools/nn/model/abilities.py) gets zero weights for
    the new ones, so the old network computes exactly what it did. In place."""
    w = state_dict.get(key)
    if w is not None and w.dim() == 2 and w.shape[1] < n_in:
        state_dict[key] = torch.cat([w, w.new_zeros(w.shape[0], n_in - w.shape[1])], 1)


class TokenEncoder(nn.Module):
    def __init__(self, d, n_token=ob.TOKEN, n_ctx=ob.CONTEXT):
        super().__init__()
        self.unit = nn.Sequential(nn.Linear(n_token, d), nn.GELU(), nn.Linear(d, d))
        self.ctx = nn.Sequential(nn.Linear(n_ctx, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """A checkpoint of an older context loads and computes what it did: one without PROGRESS
        (observation.py) gets zero weights for it; one older still (a t / 3600 column, no TIMERS) has
        that column's weights dropped and zero weights for TIMERS and PROGRESS (the time column at 0).
        The critic's enemy character, after them, keeps its weights."""
        key = prefix + "ctx.0.weight"
        w = state_dict.get(key)
        n, timers, progress = self.ctx[0].in_features, len(ob.TIMERS), len(ob.PROGRESS)
        at = ob.CONTEXT_BASE + timers                                    # where PROGRESS begins
        if w is not None and w.shape[1] == n - timers - progress + 1:
            w = torch.cat([w[:, :ob.OLD_TIME], w[:, ob.OLD_TIME + 1:ob.CONTEXT_BASE + 1],
                           w.new_zeros(w.shape[0], timers), w[:, ob.CONTEXT_BASE + 1:]], 1)
        if w is not None and w.shape[1] == n - progress:
            state_dict[key] = torch.cat([w[:, :at], w.new_zeros(w.shape[0], progress), w[:, at:]], 1)
        pad_inputs(state_dict, prefix + "unit.0.weight", self.unit[0].in_features)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, tokens, ctx):
        """tokens [B, N, F], ctx [B, C] -> [B, 1 + N, d] (the context token first)."""
        c = self.ctx(ctx)
        u = self.unit(tokens) + c[:, None]
        return self.norm(torch.cat([c[:, None], u], 1))


class AbilityEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ab.SIZE, d), nn.GELU(), nn.Linear(d, d))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        pad_inputs(state_dict, prefix + "net.0.weight", self.net[0].in_features)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, abil):
        """abil [B, N, SLOTS, SIZE] -> [B, N, d]: the sum over the owned slots. Only the owned slots
        go through the network (lords: a few of the B x N x SLOTS), so the cost stays small. Compiled
        (the decisions of training, tools/nn/train/rollout.py fast()), every slot goes through it and
        those not owned are dropped after it - the same sum: picking the owned ones (nonzero) reads
        their number back from the GPU and breaks the graph, and a decision's batch is small."""
        if torch.compiler.is_compiling():
            owned = (abil[..., ab.INDEX["owned"]] > 0.5)[..., None]
            # (the slots not owned enter as zeros: a stray NaN there would poison the weights' gradient)
            e = self.net(torch.where(owned, abil, torch.zeros_like(abil)))             # [B, N, SLOTS, d]
            return torch.where(owned, e, torch.zeros_like(e)).sum(2)
        B, N, K, _ = abil.shape
        b, n, k = (abil[..., ab.INDEX["owned"]] > 0.5).nonzero(as_tuple=True)
        e = self.net(abil[b, n, k])                                                   # [M, d]
        out = e.new_zeros(B * N, e.shape[-1]).index_add(0, b * N + n, e)
        return out.reshape(B, N, -1)


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
    bias = bias.masked_fill(~keys[:, None, None, :], float("-inf"))
    # Made once in memory the attention kernel takes as it is: rows aligned to 16 numbers (a view
    # of a padded tensor). Otherwise scaled_dot_product_attention copies it into such memory in
    # every layer, forward and backward (~15 % of a training update).
    L = bias.shape[-1]
    return F.pad(bias, (0, -L % 16))[..., :L]


class Block(nn.Module):
    LINEAR = ("qkv", "out", "f1", "f2")

    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.f1 = nn.Linear(d, ff * d)
        self.f2 = nn.Linear(ff * d, d)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """A checkpoint of the LoRA wrapper's time (until 03.10: qkv.base.weight, never trained adapters)
        loads as the plain layers it was (in place)."""
        for name in self.LINEAR:
            for p in ("weight", "bias"):
                old = f"{prefix}{name}.base.{p}"
                if old in state_dict:
                    state_dict[f"{prefix}{name}.{p}"] = state_dict.pop(old)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x, bias):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.out(a.transpose(1, 2).reshape(B, L, d))
        return x + self.f2(F.gelu(self.f1(self.n2(x))))
