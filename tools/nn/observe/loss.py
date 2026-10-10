"""The imitation term on the demonstrations (docs/en/training/training.md "Learning by observation"):

    L_obs = sum over the labelled units of c x (-log P(label | s)) / the number of labels with c > 0
    c = min(exp(A / beta), clip) where the label is selected (A > 0 by the network's critic: store.py), else 0

P(label) is the chained heads' probability of the order's parts the recording reads well (labels.py): the kind; the
target of an attack; the place (sector and cell) of a move; run when known. A running move's label is the set {move +
run, withdraw} (one order in the game): P = P(move, place, run) + P(withdraw, place) - the heads run twice, under each.
"""
import torch
import torch.nn.functional as F

from tools.nn.model import heads as hd
from tools.nn.model import sectors as sc
from tools.nn.sim.orders import ATTACK, HOLD, MOVE, WITHDRAW

FLOOR = -50.0      # a label's log-probability at least this (a masked choice gives -1e9: one bad label must not blow up)


def trunk(actor, obs, h0):
    """(x [T * B, 1 + N, d] after the memory, the encoder's pack, flat obs, the memory after the last decision): the
    first half of ActorV2.sequence (no heads). obs: dict of [T, B, ...]; h0 [B, 1 + N, d]; no new battle inside."""
    T, B = obs["own"].shape[:2]
    flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()}
    x, pack = actor.encode(flat)
    h = h0
    if actor.memory is not None:
        reset = torch.zeros((T, B), dtype=torch.bool, device=x.device)
        x, h = actor.memory.scan(x.reshape(T, B, *x.shape[1:]), h0, actor.keep(flat).reshape(T, B, -1), reset)
        x = x.reshape(T * B, *x.shape[2:])
    return x, pack, flat, h


def logits_under(actor, obs, h0, actions):
    """The actor's logits [T * B, ...] under each of `actions` (Action [T * B, N] each), the trunk and the memory run
    once (ActorV2.sequence's path). obs: dict of [T, B, ...]."""
    x, pack, flat, _ = trunk(actor, obs, h0)
    out = []
    for a in actions:
        logits, _ = actor._finish(x, pack, flat, a, False, 1.0, True)
        out.append(logits)
    return out, flat


def _lsm(logits, idx):
    """log softmax(logits)[idx] (idx clamped to >= 0)."""
    return torch.log_softmax(logits.float(), -1).gather(-1, idx.clamp(min=0)[..., None])[..., 0]


def label_logp(cfg, lg, kind, target, point, run, run_known):
    """[T * B, N] log P of the label's parts under logits lg (module doc)."""
    sector, fine = sc.split(cfg, point.clamp(min=0))
    lp = _lsm(lg["kind"], kind)
    zero = torch.zeros_like(lp)
    attack = kind == ATTACK
    place = (kind == MOVE) | (kind == WITHDRAW)
    lp = lp + torch.where(attack, _lsm(lg["target"], target), zero)
    lp = lp + torch.where(place, _lsm(lg["sector"], sector) + _lsm(lg["fine"], fine), zero)
    known = run_known & ((kind == MOVE) | attack)
    r = lg["run"].float()
    lp = lp + torch.where(known, torch.where(run, F.logsigmoid(r), F.logsigmoid(-r)), zero)
    return lp


def term(actor, demo):
    """(loss, stats) on a demonstration batch (store.Demos.sample). The stats (sums, summary() makes them means): the
    selected labels, their nll, their weights, the network's agreement with them (its most likely kind - either of the
    set's two for a running move - and target)."""
    cfg = actor.cfg
    kind, target, point = demo["kind"], demo["target"], demo["point"]
    run, run_known, alt, c = demo["run"], demo["run_known"], demo["alt"], demo["c"]
    T, B, N = kind.shape
    flat = lambda v: v.reshape(T * B, N)
    kind, target, point, run, run_known, alt, c = map(flat, (kind, target, point, run, run_known, alt, c))
    labelled = kind >= 0
    k1 = torch.where(labelled, kind, torch.full_like(kind, HOLD))
    tg = torch.where(labelled & (k1 == ATTACK), target, torch.full_like(target, -1))
    pt = point.clamp(min=0)
    commit = torch.zeros_like(k1)
    actions = [hd.Action(k1, pt, tg, run, None, commit)]
    two = bool(alt.any())
    if two:
        actions.append(hd.Action(torch.where(alt, torch.full_like(k1, WITHDRAW), k1), pt, tg, run, None, commit))
    logits, _ = logits_under(actor, demo["obs"], demo["h0"], actions)
    lp = label_logp(cfg, logits[0], k1, tg, pt, run, run_known)
    if two:
        lw = label_logp(cfg, logits[1], torch.where(alt, torch.full_like(k1, WITHDRAW), k1), tg, pt, run,
                        torch.zeros_like(run_known))
        lp = torch.where(alt, torch.logaddexp(lp, lw), lp)
    lp = lp.clamp(min=FLOOR)
    sel = labelled & (c > 0)
    n = sel.sum().clamp(min=1)
    loss = -(c * lp * sel).sum() / n
    with torch.no_grad():
        best = logits[0]["kind"].argmax(-1)
        ok_kind = (best == k1) | (alt & (best == WITHDRAW))
        att = sel & (k1 == ATTACK)
        ok_tg = logits[0]["target"].argmax(-1) == tg
        m = sel.float()
        stats = {"labels": float(m.sum()), "nll_sum": float(-(lp * m).sum()), "c_sum": float((c * m).sum()),
                 "kind_ok": float((ok_kind.float() * m).sum()), "attack_labels": float(att.float().sum()),
                 "target_ok": float((ok_tg & att).float().sum())}
    return loss, stats


def summary(sums):
    """The update's stats from the sums of its minibatches' term() stats: nll and mean weight a selected label, the
    agreement of the most likely kind (and target, over the attack labels)."""
    n = max(1.0, sums.get("labels", 0.0))
    return {"labels": sums.get("labels", 0.0), "nll": sums.get("nll_sum", 0.0) / n, "mean_c": sums.get("c_sum", 0.0) / n,
            "kind_acc": sums.get("kind_ok", 0.0) / n, "attack_labels": sums.get("attack_labels", 0.0),
            "target_acc": sums.get("target_ok", 0.0) / max(1.0, sums.get("attack_labels", 0.0))}
