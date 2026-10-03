"""The drills' teacher: an annealed imitation term where PPO does not find a drill's skill on its own
(docs/en/training/training.md "Drills", "Teacher"; run.py --drill-teach).

In the battles of a taught drill the network plays as always; the drill's `skilled` script does not
act, it only labels. At every decision the rollout (rollout.Battles, teach=) runs the script on the
same state and turns its orders for OUR units (the learner's rows) into the network's action space
(label): the order kind; the move point as the bin whose point (the unit's position + the bin's
offset, heads.point_world) is nearest the script's point; the attack target as the pointer to the
same slot (the network's unit index is the state's slot, decide.to_orders); run. PPO's update adds
weight(t) x the cross-entropy of the policy against the label (heads.log_prob of the label: kind,
point for move / withdraw, target for attack, run for move / attack; no ability) on those units
only, the weight per drill going linearly from --drill-teach's value to 0 over
--drill-teach-minutes of training (a hint, not a leash). Logged per drill: the term (ce) and the
agreement (the policy's most likely action is the label): over all labels, on the kind alone, and on
the active labels alone (any kind but hold: for kiting the run-back, which the many hold labels of
the approach would hide).

A label is used where the unit takes orders (obs ctrl), its battle runs, and an attack's target
may be attacked now (obs target_ok).
"""
import torch

from tools.nn.model import heads as hd
from tools.nn.model import observation as ob


def point_bins(cfg, obs_t, frame, x, z):
    """[R, N] the move-point bin whose point is nearest the world point (x, z) [R, N] of each unit
    (the bin's point: the unit's position + the bin's offset in the side's frame, before the map's
    clip; heads.point_world)."""
    f, l = frame.point(x, z)
    off = torch.stack([f, l], -1) - obs_t["pos"] * ob.POS                      # [R, N, 2]
    bins = hd.point_offsets(cfg, off.device).to(off.dtype)                     # [P, 2]
    return ((off[..., None, :] - bins) ** 2).sum(-1).argmin(-1)


def label(cfg, orders, obs_t, frame, rows, B):
    """The script's Orders [B, N] (state slots) -> (Action [R, N], valid [R, N]) of the network's rows
    (row = (side - 1) * B + battle; obs_t and frame: those rows'): kind; point bin (move, withdraw;
    0 else); target slot (attack; -1 else); run (move, attack). valid: the unit takes orders and an
    attack's target may be attacked now."""
    b = rows % B
    kind, target, run = orders.kind[b], orders.target[b], orders.run[b]
    moves = (kind == hd.MOVE) | (kind == hd.WITHDRAW)
    point = torch.where(moves, point_bins(cfg, obs_t, frame, orders.x[b], orders.z[b]), torch.zeros_like(kind))
    attack = kind == hd.ATTACK
    target = torch.where(attack, target, torch.full_like(target, -1))
    ok = obs_t["target_ok"].gather(-1, target.clamp(min=0))
    run = run & ((kind == hd.MOVE) | attack)
    valid = obs_t["ctrl"] & (~attack | ((target >= 0) & ok))
    return hd.Action(kind, point, target, run), valid


def agree(logits, a):
    """[.., N] bool: the policy's most likely action is the label a: the kind; and the point (move,
    withdraw), the target (attack), run (move, attack)."""
    ok = logits["kind"].argmax(-1) == a.kind
    ok = ok & ~(((a.kind == hd.MOVE) | (a.kind == hd.WITHDRAW)) & (logits["point"].argmax(-1) != a.point))
    ok = ok & ~((a.kind == hd.ATTACK) & (logits["target"].argmax(-1) != a.target))
    ok = ok & ~(((a.kind == hd.MOVE) | (a.kind == hd.ATTACK)) & ((logits["run"] > 0) != a.run))
    return ok


def cross_entropy(logits, a, valid):
    """[.., N] -log p(label) per unit (0 where not valid)."""
    return -hd.log_prob(logits, a, valid)[0]


def weights(start, minutes, trained_min):
    """{drill: the term's weight now}: linear from start's value to 0 over `minutes` of training."""
    share = 1.0 - min(1.0, max(0.0, trained_min / minutes)) if minutes > 0 else 0.0
    return {n: float(w) * share for n, w in start.items()}


def terms(logits, a, valid, drill, names, weight=None):
    """(loss, sums) of a minibatch's units (logits, a, valid [S, N]; drill [S]: index into names, -1
    none): loss = sum over drills of weight[name] x the mean cross-entropy over the drill's labelled
    units (a zero tensor without weights); sums {name: [ce sum, agree sum, kind agree sum, units,
    agree sum on the active labels (not hold), active units]} (floats, for the log)."""
    ce = cross_entropy(logits, a, valid)
    with torch.no_grad():
        hit = agree(logits, a)
        kind_hit = logits["kind"].argmax(-1) == a.kind
        active = a.kind != hd.HOLD
    loss = torch.zeros((), device=ce.device)
    sums = {}
    for i, name in enumerate(names):
        m = valid & (drill == i)[:, None]
        n = m.float().sum()
        mean = (ce * m).sum() / n.clamp(min=1)
        w = (weight or {}).get(name, 0.0)
        if w:
            loss = loss + w * mean
        sums[name] = [float((ce.detach() * m).sum()), float((hit & m).float().sum()),
                      float((kind_hit & m).float().sum()), float(n), float((hit & m & active).float().sum()),
                      float((m & active).float().sum())]
    return loss, sums


def summary(acc, weight=None):
    """{name: {weight, ce, agree, agree_kind, units, agree_active (None without active labels),
    active}} of terms()' sums added up over an update."""
    out = {}
    for name, (ce, hit, kind_hit, n, act_hit, act_n) in acc.items():
        out[name] = {"weight": round(float((weight or {}).get(name, 0.0)), 5),
                     "ce": round(ce / max(1.0, n), 4), "agree": round(hit / max(1.0, n), 4),
                     "agree_kind": round(kind_hit / max(1.0, n), 4), "units": int(n),
                     "agree_active": round(act_hit / act_n, 4) if act_n else None, "active": int(act_n)}
    return out
