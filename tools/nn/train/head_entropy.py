"""The spread of each v2 head apart (run.py --head-entropy; docs/en/training/training.md "Per-head spread").

The old bonus (ppo.PPOConfig.entropy, --entropy-target) is on the order kind only. The heads after it - the target,
the place's sector and cell, the commitment ("hold") and run - had no floor and collapsed to one choice (build/net_audit:
the chosen sector's median probability 0.9994, the cell's 1.0; the commitment 2 s or 16 s, never 4 s).

The scale: the SHARE of the maximum - a head's entropy over log(the choices allowed in that row), averaged over the
rows where the head is used. 0: always one choice; 1: every allowed choice alike. A share s means ~exp(s x log n) =
n^s choices in play: the sector (up to 256 on the map) at 0.4 ~9 sectors, the cell (16) at 0.5 ~4 cells, the hold
(4 durations) at 0.5 ~2. One scale for heads of 2 to 256 choices; nats would need another number per head and per
map (sectors outside the map are masked).

The rows (the parts that matter, as chain.log_prob): the kind - the units that decide (Obs "free"); the target - those
whose action was an attack; the sector and the cell - a move or a withdraw (their logits are under that kind's
condition); the hold - a new order (not keep); run - a move or an attack. Rows with fewer than 2 allowed choices
are left out (nothing to spread).

The floor: each head with a target has its own weight in the loss (- weight x share): x rate every update while its
share is below the target, / rate above it, within [low, high] (as the kind's floor, ppo.entropy_weight).
"""
import math

import torch
from torch.distributions import Bernoulli, Categorical

from tools.nn.model.heads import NEG
from tools.nn.sim.orders import ATTACK, KEEP, MOVE, WITHDRAW

HEADS = ("kind", "target", "sector", "cell", "hold", "run")
LOGITS = {"kind": "kind", "target": "target", "sector": "sector", "cell": "fine", "hold": "commit", "run": "run"}


def parse(spec):
    """'sector=0.4,cell=0.5,hold=0.5' -> {head: target share}; '' / None -> {}."""
    out = {}
    for part in filter(None, (p.strip() for p in (spec or "").split(","))):
        name, _, v = part.partition("=")
        name = name.strip()
        if name not in HEADS or not v:
            raise ValueError(f"--head-entropy {part!r}: head=share with a head of {HEADS}")
        share = float(v)
        if not 0.0 < share < 1.0:
            raise ValueError(f"--head-entropy {part!r}: the share of the maximum is between 0 and 1")
        out[name] = share
    return out


def rows(head, action, free, ctrl):
    """[R, N] the rows where the head is used (module doc)."""
    k = action.kind
    if head == "kind":
        return free
    if head == "target":
        return ctrl & (k == ATTACK)
    if head in ("sector", "cell"):
        return ctrl & ((k == MOVE) | (k == WITHDRAW))
    if head == "hold":
        return ctrl & (k != KEEP)
    return ctrl & ((k == MOVE) | (k == ATTACK))


def per_row(head, logits):
    """(entropy [R, N] in nats, its maximum [R, N] = log of the allowed choices) of the head's distribution."""
    x = torch.nan_to_num(logits[LOGITS[head]], nan=NEG)
    if head == "run":
        return Bernoulli(logits=x, validate_args=False).entropy(), torch.full_like(x, math.log(2.0))
    n = (x > NEG / 2).sum(-1)
    return Categorical(logits=x, validate_args=False).entropy(), n.clamp(min=1).to(x.dtype).log()


def measure(logits, action, free, ctrl, heads=HEADS, grad=()):
    """{head: (share of the maximum, entropy in nats, rows)} averaged over the rows where it is used and has 2 or more
    choices; the share keeps its graph for the heads in `grad` (their floor's bonus), the rest without gradients.
    Heads whose logits are missing (v1, no commitment) are skipped."""
    out = {}
    for h in heads:
        if LOGITS[h] not in logits or (h == "hold" and action.commit is None):
            continue
        with torch.set_grad_enabled(torch.is_grad_enabled() and h in grad):
            ent, top = per_row(h, logits)
            m = rows(h, action, free, ctrl) & (top > math.log(1.5))
            mf = m.to(ent.dtype)
            n = mf.sum()
            share = (ent / top.clamp(min=1e-6) * mf).sum() / n.clamp(min=1)
            nats = (ent.detach() * mf).sum() / n.clamp(min=1)
        out[h] = (share, nats, n)
    return out


def bonus(measured, weights):
    """sum over the heads with a weight of weight x share (subtracted from the loss)."""
    terms = [w * measured[h][0] for h, w in weights.items() if w and h in measured and measured[h][2] > 0]
    return sum(terms) if terms else None


def next_weights(weights, shares, targets, low, high, rate=1.25):
    """Every head's next weight (module doc): x rate below its target share, / rate at or above it, within
    [low, high]. A head not measured this update (no rows) keeps its weight."""
    out = {}
    for h, target in targets.items():
        w = weights.get(h, low)
        if h in shares and shares[h] is not None:
            w = w * rate if shares[h] < target else w / rate
        out[h] = min(high, max(low, w))
    return out
