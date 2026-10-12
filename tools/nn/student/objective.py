"""The student's loss and accuracies on a batch (dataset.Streams):

    L = sum over the free units' seconds of w x (-log P(label | s)) / their number

the label: KEEP, or the AI's new order part by part as the chained heads choose it (chain.py): the kind; the target
of an attack; the sector and the cell of a move; run when the recording knows it; the commitment. A running move is
the set {move + run, withdraw} (one order in the game): P = P(move, place, run, commitment) + P(withdraw, place,
commitment). w: new_weight for a new order, 1 for keep. The parts after the kind are those under the label's own
earlier parts (teacher forcing), as in the PPO update.

The accuracies (sums, summary() makes them shares), over the free units' seconds:
* "when": the student's chance of a new order (1 - P(keep)) where the AI gave one against where it kept (a constant
  guess: the two equal); and the most likely kind keep or not (precision, recall: with ~7 % new orders the most
  likely kind is nearly always keep - sampled, the student still gives orders);
* over the new orders: the kind (the most likely of hold / move / attack / withdraw; a running move's set: move or
  withdraw), the attack target, the move's sector, the sector or one of its 8 neighbours, the cell within the right
  sector, run (where known), the commitment.
"""
import torch

from tools.nn.model import heads as hd
from tools.nn.model import sectors as sc
from tools.nn.observe import loss as obs_loss
from tools.nn.sim.orders import ATTACK, KEEP, MOVE, WITHDRAW

FLOOR = obs_loss.FLOOR


def trunk(actor, obs, h0, reset):
    """(x [T * B, 1 + N, d] after the memory, the encoder's pack, flat obs, the memory after the last second) - the
    first half of ActorV2.sequence, the memory starting empty where reset [T, B]."""
    T, B = obs["own"].shape[:2]
    flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()}
    x, pack = actor.encode(flat)
    h = h0
    if actor.memory is not None:
        x, h = actor.memory.scan(x.reshape(T, B, *x.shape[1:]), h0, actor.keep(flat).reshape(T, B, -1), reset)
        x = x.reshape(T * B, *x.shape[2:])
    return x, pack, flat, h


def _lsm(logits, idx):
    return torch.log_softmax(logits.float(), -1).gather(-1, idx.clamp(min=0)[..., None])[..., 0]


def label_logp(cfg, lg, kind, target, point, run, run_known, commit):
    """[M, N] log P of the label (observe/loss.py label_logp + the commitment of a new order)."""
    lp = obs_loss.label_logp(cfg, lg, kind, target, point, run, run_known)
    new = kind != KEEP
    return lp + torch.where(new, _lsm(lg["commit"], commit), torch.zeros_like(lp))


def actions(b):
    """The label's Action(s) [T * B, N]: under the label (keep where none), and - when a running move's set is in the
    batch - under its withdraw."""
    T, B, N = b["kind"].shape
    f = lambda v: v.reshape(T * B, N)
    kind = f(b["kind"])
    k1 = torch.where(kind >= 0, kind, torch.full_like(kind, KEEP))
    tg = torch.where(k1 == ATTACK, f(b["target"]), torch.full_like(kind, -1))
    pt, cm = f(b["point"]).clamp(min=0), f(b["commit"]).clamp(min=0)
    run, alt = f(b["run"]), f(b["alt"]) & (k1 == MOVE)
    acts = [hd.Action(k1, pt, tg, run, None, cm)]
    if bool(alt.any()):
        acts.append(hd.Action(torch.where(alt, torch.full_like(k1, WITHDRAW), k1), pt, tg, run, None, cm))
    return acts, dict(kind=kind, k1=k1, target=tg, point=pt, commit=cm, run=run, alt=alt,
                      run_known=f(b["run_known"]))


def run(actor, b, h0, new_weight=1.0, stats=True):
    """(loss, stats sums, the memory after the batch, aux: the logits under the label, the trunk's x, pack, flat obs and the
    labels flat - for a further pass under other actions) on a batch. h0 [B, 1 + N, d]."""
    cfg = actor.cfg
    acts, L = actions(b)
    x, pack, flat, h = trunk(actor, b["obs"], h0, b["reset"])
    logits = [actor._finish(x, pack, flat, a, False, 1.0, True, False)[0] for a in acts]
    lp = label_logp(cfg, logits[0], L["k1"], L["target"], L["point"], L["run"], L["run_known"], L["commit"])
    if len(acts) > 1:
        wd = torch.full_like(L["k1"], WITHDRAW)
        lw = label_logp(cfg, logits[1], wd, L["target"], L["point"], L["run"], torch.zeros_like(L["run_known"]),
                        L["commit"])
        lp = torch.where(L["alt"], torch.logaddexp(lp, lw), lp)
    lp = lp.clamp(min=FLOOR)
    lab = L["kind"] >= 0
    new = lab & (L["k1"] != KEEP)
    w = torch.where(new, torch.full_like(lp, float(new_weight)), torch.ones_like(lp)) * lab
    loss = -(w * lp).sum() / w.sum().clamp(min=1)
    st = sums(cfg, logits[0], L, lp.detach()) if stats else {}
    return loss, st, h.detach(), {"logits": logits[0], "x": x, "pack": pack, "flat": flat, "L": L}


@torch.no_grad()
def sums(cfg, lg, L, lp):
    """The accuracies' sums (module doc)."""
    kind, k1, alt = L["kind"], L["k1"], L["alt"]
    lab = kind >= 0
    new = lab & (k1 != KEEP)
    keep = lab & (k1 == KEEP)
    best = lg["kind"].argmax(-1)
    order_logits = lg["kind"].float().clone()
    order_logits[..., KEEP] = -1e30
    best_order = order_logits.argmax(-1)
    att = new & (k1 == ATTACK)
    place = new & ((k1 == MOVE) | (k1 == WITHDRAW))
    sector, fine = sc.split(cfg, L["point"])
    ps, pf = lg["sector"].argmax(-1), lg["fine"].argmax(-1)
    n = cfg.sectors
    near = ((ps // n - sector // n).abs() <= 1) & ((ps % n - sector % n).abs() <= 1)
    runk = new & L["run_known"] & ((k1 == MOVE) | (k1 == ATTACK))
    s = lambda m: float(m.float().sum())
    pn = 1 - torch.softmax(lg["kind"].float(), -1)[..., KEEP]          # the chance of a new order
    return {"labels": s(lab), "new": s(new), "keep": s(keep),
            "nll_new": float(-(lp * new).sum()), "nll_keep": float(-(lp * keep).sum()),
            "pred_new": s(lab & (best != KEEP)), "pred_new_ok": s(new & (best != KEEP)),
            "keep_ok": s(keep & (best == KEEP)), "p_new_at_new": float((pn * new).sum()),
            "p_new_at_keep": float((pn * keep).sum()),
            "kind_ok": s(new & ((best_order == k1) | (alt & (best_order == WITHDRAW)))),
            "kind_top_ok": s(new & ((best == k1) | (alt & (best == WITHDRAW)))),
            "attack": s(att), "target_ok": s(att & (lg["target"].argmax(-1) == L["target"])),
            "place": s(place), "sector_ok": s(place & (ps == sector)), "sector_near": s(place & near),
            "fine_ok": s(place & (pf == fine)), "run_n": s(runk),
            "run_ok": s(runk & ((lg["run"] > 0) == L["run"])),
            "commit_ok": s(new & (lg["commit"].argmax(-1) == L["commit"]))}


def add(acc, st):
    for k, v in st.items():
        acc[k] = acc.get(k, 0.0) + float(v)
    return acc


def summary(acc):
    """The shares from the sums."""
    d = lambda a, b: acc.get(a, 0.0) / max(1.0, acc.get(b, 0.0))
    return {"labels": acc.get("labels", 0.0), "new": acc.get("new", 0.0), "new_share": d("new", "labels"),
            "nll": (acc.get("nll_new", 0.0) + acc.get("nll_keep", 0.0)) / max(1.0, acc.get("labels", 0.0)),
            "nll_new": d("nll_new", "new"), "nll_keep": d("nll_keep", "keep"),
            "when_precision": d("pred_new_ok", "pred_new"), "when_recall": d("pred_new_ok", "new"),
            "keep_acc": d("keep_ok", "keep"), "p_new_at_new": d("p_new_at_new", "new"),
            "p_new_at_keep": d("p_new_at_keep", "keep"),
            "kind_acc": d("kind_ok", "new"), "kind_top_acc": d("kind_top_ok", "new"),
            "target_acc": d("target_ok", "attack"), "sector_acc": d("sector_ok", "place"),
            "sector_near_acc": d("sector_near", "place"), "fine_acc": d("fine_ok", "place"),
            "run_acc": d("run_ok", "run_n"), "commit_acc": d("commit_ok", "new"),
            "attack": acc.get("attack", 0.0), "place": acc.get("place", 0.0)}


def text(sm):
    return (f"nll {sm['nll']:.3f} (new {sm['nll_new']:.2f}, keep {sm['nll_keep']:.3f}); when: P(new order) where "
            f"the AI gave one {sm['p_new_at_new']:.3f}, where it kept {sm['p_new_at_keep']:.3f} (argmax precision "
            f"{sm['when_precision']:.2f} recall {sm['when_recall']:.2f}, keep {sm['keep_acc']:.3f}); new orders "
            f"{sm['new']:.0f} ({sm['new_share']:.3f}): kind {sm['kind_acc']:.3f}, target {sm['target_acc']:.3f} "
            f"({sm['attack']:.0f}), sector {sm['sector_acc']:.3f} (+-1 {sm['sector_near_acc']:.3f}), cell "
            f"{sm['fine_acc']:.3f} ({sm['place']:.0f}), run {sm['run_acc']:.3f}, commitment {sm['commit_acc']:.3f}")


