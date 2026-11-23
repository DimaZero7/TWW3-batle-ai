"""The decision record (tmp/engine/design_merged.md section 2, frozen first; after tmp/engine/fixtures/READY it changes
only additively).

    Batch       M rows x G decision units x C candidates (0 = KEEP): what a unit saw, the order in force, the legal
                candidates and don't-cares (only from masks.of), decided / delegated and the reason bits, the proposed /
                executed / teacher candidate (-1 none) and the conditional heads' facts (aux). Delegated is not KEEP: a
                delegated unit has no label and is in no denominator; KEEP (candidate 0) is a legal choice, a label
    Level       the protocol a level adapter fills (tick_s, play, rebuild, losses, ask, tick0, judge, save, load)
    loss(level, net, batch, w)   (the sum of the level's loss terms, {term: tensor | None}) - the one loss call of fit,
                probe and V
    from_netbuffer(d, idx)   the tactic's NetBuffer rows (net/tactic/imitate.py NetBuffer.data()) as a Batch: obs = the
                network's stored input (the per-battle passport card gathered per row), order = the command, legal = KEEP
                + the walk's open rows fed with the labeller's choices, dont_care = 0 + dc, status = head-free and not
                done, teacher = the label's row + 1 (KEEP 0) on decided units, aux = the side head's facts and the label
                drop's parts
    to_netbuffer(batch)   the inverse view (NetBuffer keys) for the legacy loss / rebuild calls
"""
from dataclasses import dataclass, field, fields
from typing import Protocol

import torch
from torch import Tensor

# status of a decision unit
DELEGATED, DECIDED = 0, 1
NONE = -1                                   # proposed / executed / teacher: no candidate


@dataclass
class Batch:
    b: Tensor                               # [M] battle id (long)
    sec: Tensor                             # [M] the tick's second
    tw: Tensor                              # [M] the trivial-thinning weight (1, or trivial_every for a kept one)
    net: Tensor                             # [M] bool: the network drove
    obs: dict = field(default_factory=dict)       # the visible observation = the network's stored input
    order: dict = field(default_factory=dict)     # the order in force + age (operative: + goal, birth)
    legal: Tensor = None                    # [M, G, C] bool
    dont_care: Tensor = None                # [M, G, C] bool
    status: Tensor = None                   # [M, G] int8: DECIDED / DELEGATED
    why: Tensor = None                      # [M, G] int32: reason bits (masks.R_*)
    proposed: Tensor = None                 # [M, G] long candidate index, -1 none
    executed: Tensor = None                 # [M, G]
    teacher: Tensor = None                  # [M, G]
    aux: dict = field(default_factory=dict)       # conditional heads (tactic: ap, x_k, sband, sclosed)

    @property
    def M(self):
        return int(self.b.shape[0])

    def map(self, fn):
        """A new Batch with fn applied to every tensor (nested dicts included)."""
        def f(x):
            if isinstance(x, dict):
                return {k: f(v) for k, v in x.items()}
            return fn(x) if torch.is_tensor(x) else x
        return Batch(**{fl.name: f(getattr(self, fl.name)) for fl in fields(self)})

    def to(self, dev):
        return self.map(lambda t: t.to(dev))

    def index(self, idx):
        return self.map(lambda t: t[idx])

    def as_dict(self):
        return {fl.name: getattr(self, fl.name) for fl in fields(self)}

    @staticmethod
    def from_dict(x):
        return Batch(**x)

    @staticmethod
    def cat(bs):
        """Rows of several Batches (same keys, same G and C) in order."""
        def c(xs):
            if isinstance(xs[0], dict):
                return {k: c([x[k] for x in xs]) for k in xs[0]}
            return torch.cat(xs, 0) if torch.is_tensor(xs[0]) else xs[0]
        return Batch(**{fl.name: c([getattr(b, fl.name) for b in bs]) for fl in fields(bs[0])})


class Level(Protocol):
    """A level adapter (design section 3): execution stays in its live runner."""
    tick_s: float

    def play(self, *a, **k): ...                 # the live runner with the recorder hook
    def rebuild(self, batch): ...                # the network's input of stored rows
    def losses(self, net, batch, w): ...         # {term: tensor | None}
    def ask(self, net, batch): ...               # the network's logits / picks on stored rows (V's teacher path)
    def tick0(self, *a, **k): ...                # the first decision tick of a battle
    def judge(self, *a, **k): ...                # the level's judge at its window
    def save(self, path, net, **more): ...
    def load(self, path, device="cpu"): ...


def loss(level, net, batch, w):
    """(sum of the level's terms in their order, {term: tensor | None}); w [M, G] the rows' weights."""
    terms = level.losses(net, batch, w)
    tot = None
    for v in terms.values():
        if v is not None:
            tot = v if tot is None else tot + v
    return tot, terms


# ------------------------------------------------------------------ the tactic's NetBuffer rows
# (net/tactic/imitate.py NET_KEYS / SIDE_KEYS / P_KEYS / T_KEYS / A_KEYS; kept literal here so the record does not
# import the legacy trainer)
P_KEYS = ("mine", "target", "cx", "cz", "sin", "cos", "line", "group")
T_KEYS = ("cx", "cz", "heading", "rows", "cols", "mask")
A_KEYS = ("x", "z", "b", "front", "depth", "on")
OBS_KEYS = ("raw", "cols", "own", "g_row", "g_on", "pair", "bounds") + tuple("p_" + k for k in P_KEYS) + tuple(
    "t_" + k for k in T_KEYS) + tuple("a_" + k for k in A_KEYS)
OBS_SIDE = ("s2",)
AUX_KEYS = ("ap", "x_k", "dc_thr", "drop", "kinds", "trivial", "near2") + ("drop_base",)   # (+ drop_base: additive, step 3)


def from_netbuffer(d, idx=None):
    """(module doc) d: NetBuffer.data() (its 'cards' [B, N, 69] per battle); idx: the rows (None: all)."""
    from net.drill import masks as MK
    if idx is None:
        idx = torch.arange(d["b"].shape[0])
    g = lambda k: d[k][idx]                                                        # noqa: E731
    b = g("b").long()
    obs = {k: g(k) for k in OBS_KEYS + OBS_SIDE if k in d}
    obs["card"] = d["cards"][b]
    head = g("head").bool()
    done = g("done_own").bool()[:, None] if "done_own" in d else None
    if done is not None:                                # (additive: the engine's own post-finish guard, R_DONE)
        head = head & ~done
    opn = g("open").bool()
    dc = g("dc").bool()
    one = torch.ones(opn.shape[:-1] + (1,), dtype=torch.bool)
    label = g("label").long()
    aux = {k: g(k) for k in AUX_KEYS if k in d}
    why = MK.tactic_why(head, aux)
    if done is not None:
        why = why | (done.expand_as(head).int() * MK.R_DONE)
    return Batch(b=b, sec=g("sec"), tw=g("tw"), net=g("net").bool(), obs=obs, order={"command": g("command")},
                 legal=torch.cat([one, opn], -1), dont_care=torch.cat([~one, dc], -1),
                 status=head.to(torch.int8), why=why,
                 proposed=_pick(d, "proposed", idx, head, label), executed=_pick(d, "executed", idx, head, label),
                 teacher=torch.where(head, label + 1, torch.full_like(label, NONE)), aux=aux)


def _pick(d, k, idx, head, label):
    """A decided unit's proposed / executed candidate as recorded (the driver's pick: collect's Seen; KEEP 0, a row
    + 1); NONE on an undecided unit and on rows recorded before the field existed."""
    if k not in d:
        return torch.full_like(label, NONE)
    x = d[k][idx].long()
    return torch.where(head, x, torch.full_like(x, NONE))


def label_of(batch):
    """[M, G] the pointer label in the legacy code: the target row (teacher - 1), -1 for KEEP or no label."""
    t = batch.teacher.long()
    return torch.where(t > 0, t - 1, torch.full_like(t, -1))


def to_netbuffer(batch):
    """NetBuffer-keyed tensors of a tactic Batch (the legacy loss's / rebuild's view; no copy)."""
    t = {k: v for k, v in batch.obs.items() if k != "card"}
    t["command"] = batch.order["command"]
    t["open"] = batch.legal[..., 1:]
    t["dc"] = batch.dont_care[..., 1:]
    t["label"] = label_of(batch)
    t["head"] = batch.status == DECIDED
    t.update({k: v for k, v in batch.aux.items()})
    t["b"], t["sec"], t["tw"], t["net"] = batch.b, batch.sec, batch.tw, batch.net
    return t
