"""Unit tokens and attention: every unit looks at the others.

* TokenEncoder: the same weights for every unit, own and enemy (the unit is known by its passport,
  not by its name); the side's context (character, role, time) is added to every token and is
  also a token of its own (index 0). Inputs added since a checkpoint was saved (passport features
  appended at the token's end) load with zero weights (pad_inputs): it computes what it did.
* AbilityEncoder: each ability slot (its state and passport, tools/nn/model/abilities.py full) through
  the same small network; the sum over the unit's owned slots is added to the unit's token, so the
  order of the slots does not matter. Its last layer starts at zero: an actor trained before
  abilities sees exactly what it saw. It reads the slots a network saw before 13.10.2026 (abilities.legacy: the
  simulator's first 3 of the unit's abilities); AbilityExtra (the same, its last layer at zero too) reads every owned
  slot: an actor saved before it computes what it did, and learns the abilities it never saw (the 4th and 5th of a
  lord: Wounded, Scurry Away!) and the new columns through it.
* Block: attention + feed-forward. Masks: padding, own dead units and known-dead enemies are
  never looked at. Bias: a learned number per head for the distance between two units (buckets),
  so "who is near" is easy; tokens without a known position use a bucket of their own. v2 adds the pair
  features' bias to it (pairs.py: reach, closing speed, bearing).
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
        winds = len(ob.WINDS)
        # (n: the context without WINDS - the older shims below are relative to it; WINDS go in last, 16.10.2026)
        n, timers, progress = self.ctx[0].in_features - winds, len(ob.TIMERS), len(ob.PROGRESS)
        at = ob.CONTEXT_BASE + timers                                    # where PROGRESS begins
        if w is not None and w.shape[1] == n - timers - progress + 1:
            w = torch.cat([w[:, :ob.OLD_TIME], w[:, ob.OLD_TIME + 1:ob.CONTEXT_BASE + 1],
                           w.new_zeros(w.shape[0], timers), w[:, ob.CONTEXT_BASE + 1:]], 1)
        if w is not None and w.shape[1] == n - progress:
            w = torch.cat([w[:, :at], w.new_zeros(w.shape[0], progress), w[:, at:]], 1)
        if w is not None and w.shape[1] == n:
            # a context without WINDS: zero weights for them (before the critic's enemy character)
            wat = ob.CONTEXT - winds
            w = torch.cat([w[:, :wat], w.new_zeros(w.shape[0], winds), w[:, wat:]], 1)
        if w is not None:
            state_dict[key] = w
        # a token from before the new units (14 effect pairs, no WEAPON; observation.old_columns): its columns to their
        # new places, zero weights for the new effect pairs and WEAPON - the old network computes what it did
        uw = state_dict.get(prefix + "unit.0.weight")
        cols = ob.old_columns(uw.shape[1]) if uw is not None and uw.dim() == 2 else None
        if cols is not None and self.unit[0].in_features == ob.TOKEN:
            new = uw.new_zeros(uw.shape[0], ob.TOKEN)
            new[:, cols] = uw
            state_dict[prefix + "unit.0.weight"] = new
        pad_inputs(state_dict, prefix + "unit.0.weight", self.unit[0].in_features)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, tokens, ctx):
        """tokens [B, N, F], ctx [B, C] -> [B, 1 + N, d] (the context token first)."""
        c = self.ctx(ctx)
        u = self.unit(tokens) + c[:, None]
        return self.norm(torch.cat([c[:, None], u], 1))


def _rows(abil, idx, mask):
    """(the full features [M, SIZE] of the slots where mask [B, N, K], (b, n)) - eager only (nonzero)."""
    b, n, k = mask.nonzero(as_tuple=True)
    return ab.full(abil[b, n, k], idx[b, n, k]), b, n


def _sum(e, b, n, B, N):
    """[M, d] rows of (b, n) -> [B, N, d] summed per unit."""
    return e.new_zeros(B * N, e.shape[-1]).index_add(0, b * N + n, e).reshape(B, N, -1)


def _dense(net, abil, idx, mask, slots):
    """sum over the slots (a static list) of net(the slot's full features) where mask [B, N, K] - compiled."""
    a, i, keep = abil[:, :, slots], idx[:, :, slots], mask[:, :, slots, None]
    x = ab.full(a, i)
    # (the slots not kept enter as zeros: a stray NaN there would poison the weights' gradient)
    e = net(torch.where(keep, x, torch.zeros_like(x)))
    return torch.where(keep, e, torch.zeros_like(e)).sum(2)


class AbilityEncoder(nn.Module):
    """The slots a network saw before 13.10.2026 (module doc)."""
    # compiled: where those slots can be (the unit's first OLD_SLOTS abilities: its first actives, its first passives)
    COMPILED = tuple(range(ab.OLD_SLOTS)) + tuple(range(ab.ACTIVE, ab.ACTIVE + ab.OLD_SLOTS))

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ab.SIZE, d), nn.GELU(), nn.Linear(d, d))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        pad_inputs(state_dict, prefix + "net.0.weight", self.net[0].in_features)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, abil, idx):
        """abil [B, N, SLOTS, STATE], idx [B, N, SLOTS] -> [B, N, d]: the sum over the legacy owned slots. Only those
        go through the network (lords: a few of the B x N x SLOTS), so the cost stays small. Compiled (the decisions
        of training, tools/nn/train/rollout.py fast()), the 6 places they can be go through it and the rest are
        dropped after it - the same sum: picking the owned ones (nonzero) reads their number back from the GPU and
        breaks the graph."""
        keep = ab.legacy(abil[..., ab.STATE_INDEX["owned"]] > 0.5)
        if torch.compiler.is_compiling():
            return _dense(self.net, abil, idx, keep, list(self.COMPILED))
        B, N = abil.shape[:2]
        x, b, n = _rows(abil, idx, keep)
        return _sum(self.net(x), b, n, B, N)


class AbilityExtra(nn.Module):
    """Every owned slot (module doc): the actives through the network one by one (compiled: the simulator's timer
    slots and the spells', abilities.COMPILED - a simulator setup has no other, abilities.sim_fits), the passives -
    whose state is owned and on now only - by a table of the network's answer per (ability, on now), counted per
    unit."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ab.SIZE, d), nn.GELU(), nn.Linear(d, d))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        pad_inputs(state_dict, prefix + "net.0.weight", self.net[0].in_features)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _passives(self, abil, idx):
        """[B, N, d]: the passive slots' sum (class doc)."""
        tab = ab.torch_table(abil.device)
        A1 = tab.shape[0]                                            # abilities + the empty row
        rows = torch.arange(A1, device=abil.device)
        on = torch.arange(2, device=abil.device, dtype=abil.dtype)
        state = torch.zeros(A1, 2, ab.STATE, device=abil.device, dtype=abil.dtype)
        state[..., ab.STATE_INDEX["owned"]] = 1.0
        state[..., ab.STATE_INDEX["on_now"]] = on
        table = self.net(ab.full(state, rows[:, None].expand(A1, 2), tab))     # [A1, 2, d]
        p = abil[:, :, ab.ACTIVE:]                                   # [B, N, P, STATE]
        owned = p[..., ab.STATE_INDEX["owned"]] > 0.5
        code = idx[:, :, ab.ACTIVE:].clamp(min=0) * 2 + (p[..., ab.STATE_INDEX["on_now"]] > 0.5).long()
        counts = abil.new_zeros(*code.shape[:2], 2 * A1).scatter_add_(2, code, owned.to(abil.dtype))
        return counts @ table.reshape(2 * A1, -1)

    def forward(self, abil, idx):
        """abil [B, N, SLOTS, STATE], idx [B, N, SLOTS] -> [B, N, d]: the sum over every owned slot."""
        owned = abil[..., ab.STATE_INDEX["owned"]] > 0.5
        act = owned[:, :, :ab.ACTIVE]
        if torch.compiler.is_compiling():
            out = _dense(self.net, abil, idx, owned, list(ab.COMPILED))     # (the timer slots and the spells')
        else:
            B, N = abil.shape[:2]
            x, b, n = _rows(abil[:, :, :ab.ACTIVE], idx[:, :, :ab.ACTIVE], act)
            out = _sum(self.net(x), b, n, B, N)
        return out + self._passives(abil, idx)


def distance_buckets(pos, known, bins):
    """[B, L, L] bucket of the distance between tokens; bucket `bins` = unknown (and the context token).

    pos [B, N, 2] in POS units, known [B, N]. Buckets are log-spaced from 0 to ~1500 m.
    """
    d = torch.cdist(pos, pos) * ob.POS
    b = (torch.log1p(d / 10) / math.log1p(150) * (bins - 1)).clamp(0, bins - 1).long()
    ok = known[:, :, None] & known[:, None, :]
    b = torch.where(ok, b, torch.full_like(b, bins))
    return F.pad(b, (1, 0, 1, 0), value=bins)          # the context token


def attention_bias(obs_t, bias_table, bins, extra=None):
    """Float mask [B, H, L, L]: learned distance bias (+ extra [B, H, L, L]: v2's pair bias, pairs.py), -inf for keys
    that may not be looked at."""
    known = obs_t["attend"] & (obs_t["tokens"][..., ob.INDEX["seen"]] > 0.5)
    buckets = distance_buckets(obs_t["pos"], known, bins)
    bias = bias_table(buckets).permute(0, 3, 1, 2)                     # [B, H, L, L]
    if extra is not None:
        bias = bias + extra.to(bias.dtype)
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
