"""The actor: what the side's units do. It is what ships with the mod (the critic does not).

observation -> TokenEncoder (+ AbilityEncoder) -> attention blocks -> GRU per token -> last attention block
-> heads.
Conditioning: the side's context (character, role, time) is in every token and is a token itself.

v2 (ModelConfig.sectors > 0; Actor(cfg) then makes an ActorV2): the same base + the map's sector tokens the units look
at (sectors.py), the commitment's inputs (commit.py) and the chained heads (chain.py), which sample the parts in order:
act() / forward(action=None) sample, forward / sequence with an action give the logits under it (training).
"""
import numpy as np
import torch
from torch import nn

from tools.nn.model import heads as hd
from tools.nn.model.encoder import AbilityEncoder, Block, TokenEncoder, attention_bias
from tools.nn.model.memory import TokenMemory

OBS_KEYS = ("tokens", "ctx", "own", "attend", "ctrl", "target_ok", "pos", "abil", "abil_ok")
# v2's extra inputs (not from observation.observe: added at the decision, commit.inputs and sectors.geo)
V2_KEYS = ("free", "commit", "geo")
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
    def __new__(cls, cfg=None, *args, **kwargs):
        """Actor(cfg) of a v2 config (cfg.sectors > 0) is an ActorV2: every place that builds an actor from a
        checkpoint's config gets the right one."""
        if cls is Actor and cfg is not None and getattr(cfg, "sectors", 0):
            return super().__new__(ActorV2)
        return super().__new__(cls)

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

    def act(self, obs_t, h=None, greedy=False, temperature=1.0, abilities=True):
        """One decision -> (logits, sampled Action, new memory)."""
        logits, h = self(obs_t, h)
        return logits, hd.sample(logits, greedy, temperature, abilities), h

    def sequence(self, obs_seq, h0, reset, action=None):
        """Decisions in a row, for training the memory through time.

        obs_seq: dict of [T, B, ...]; h0 [B, 1 + N, d]: the memory before the first; reset [T, B]:
        a new battle begins at step t (the memory starts empty). -> (logits of [T, B, ...], last memory).
        The parts without memory run on all T x B at once; only the GRU steps one by one. action: the actions
        taken [T, B, ...] (v2's chained heads need them; here unused)."""
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


class ActorV2(Actor):
    """v2 (module doc): encoder -> blocks with the units -> sectors attention after cfg.sector_at of them -> GRU per
    token -> the eyes (cfg.eyes; their predictions added to the tokens) -> the last block -> the chained heads."""

    def __init__(self, cfg):
        from tools.nn.model import chain, sectors
        nn.Module.__init__(self)
        self.cfg = cfg
        self.encoder = TokenEncoder(cfg.d)
        self.abilities = AbilityEncoder(cfg.d)
        self.commit_in = nn.Linear(2, cfg.d)
        self.dist = nn.Embedding(cfg.dist_bins + 1, cfg.heads)
        self.blocks = nn.ModuleList(Block(cfg.d, cfg.heads, cfg.ff) for _ in range(cfg.layers))
        self.sector_enc = sectors.SectorEncoder(cfg)
        self.sector_att = sectors.SectorAttention(cfg)
        self.memory = TokenMemory(cfg.d) if cfg.memory else None
        self.heads = chain.ChainHeads(cfg)
        if cfg.eyes:
            from tools.nn.model.eyes import Eyes
            self.eyes = Eyes(cfg)
        assert cfg.sector_at <= self._split(), "the sectors' attention comes before the memory"

    def encode(self, obs_t):
        """-> (x [B, 1 + N, d] after the blocks before the memory, (attention bias, sector tokens, cells inside))."""
        from tools.nn.model import sectors
        x = self.encoder(obs_t["tokens"], obs_t["ctx"])
        pad = (lambda v: torch.nn.functional.pad(v, (0, 0, 1, 0)))
        if obs_t.get("abil") is not None:
            x = x + pad(self.abilities(obs_t["abil"]))
        if obs_t.get("commit") is not None:
            x = x + pad(self.commit_in(obs_t["commit"].to(x.dtype)))
        bias = attention_bias(obs_t, self.dist, self.cfg.dist_bins)
        g = obs_t.get("geo")
        S, k2 = self.cfg.sectors ** 2, self.cfg.fine ** 2
        cells_in = (sectors.inside(self.cfg, g) if g is not None
                    else torch.ones((x.shape[0], S, k2), dtype=torch.bool, device=x.device))
        f = sectors.features(self.cfg, obs_t, cells_in).to(x.dtype)
        buckets = self.sector_att.buckets(obs_t)
        s = None
        for i, block in enumerate(self.blocks[:self._split()]):
            if i == self.cfg.sector_at:
                x, s = self._look(x, f, buckets)
            x = block(x, bias)
        if self.cfg.sector_at == self._split():
            x, s = self._look(x, f, buckets)
        return x, (bias, s, cells_in)

    def _look(self, x, f, buckets):
        """The sector tokens and the units' look at them -> (x, sector tokens). In training (with gradients) its
        insides are recomputed in the backward pass instead of kept (activation checkpointing): the 256 sector
        tokens of every decision took ~0.8 GB more than wide at the update's peak, near the 16 GB card's edge."""
        def run(x, f):
            s = self.sector_enc(f)
            return self.sector_att(x, s, buckets), s
        if torch.is_grad_enabled() and not torch.compiler.is_compiling():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(run, x, f, use_reentrant=False)
        return run(x, f)

    def _finish(self, x, pack, obs_t, action, greedy, temperature, abilities):
        bias, s, cells_in = pack
        seen = None
        if self.cfg.eyes:                    # the eyes (eyes.py): predicted, then added to the tokens
            x, s, seen = self.eyes(x, s, obs_t)
        for block in self.blocks[self._split():]:
            x = block(x, bias)
        logits, a = self.heads(x, s, cells_in, obs_t, action, greedy, temperature, abilities)
        if seen is not None:
            logits.update(seen)
        return logits, a

    def act(self, obs_t, h=None, greedy=False, temperature=1.0, abilities=True, action=None):
        x, pack = self.encode(obs_t)
        h = self.initial(obs_t) if h is None else h
        if self.memory is not None:
            x, h = self.memory(x, h, self.keep(obs_t))
        logits, a = self._finish(x, pack, obs_t, action, greedy, temperature, abilities)
        return logits, a, h

    def forward(self, obs_t, h=None, action=None):
        """-> (logits, new memory): under `action` when given, else under a sampled one."""
        logits, _, h = self.act(obs_t, h, action=action)
        return logits, h

    def sequence(self, obs_seq, h0, reset, action=None):
        """As Actor.sequence; the logits under the actions taken (action [T, B, ...]; None: sampled)."""
        T, B = obs_seq["own"].shape[:2]
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs_seq.items()}
        x, pack = self.encode(flat)
        h = self.initial(flat)[:B] if h0 is None else h0
        if self.memory is not None:
            x, h = self.memory.scan(x.reshape(T, B, *x.shape[1:]), h, self.keep(flat).reshape(T, B, -1), reset)
            x = x.reshape(T * B, *x.shape[2:])
        if action is not None:
            action = hd.Action(**{f: getattr(action, f).reshape(T * B, *getattr(action, f).shape[2:])
                                  for f in action.parts()})
        logits, _ = self._finish(x, pack, flat, action, False, 1.0, True)
        return {k: v.reshape(T, B, *v.shape[1:]) for k, v in logits.items()}, h


def parameters(module):
    return sum(p.numel() for p in module.parameters())
