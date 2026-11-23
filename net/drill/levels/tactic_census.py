"""The tactic level's census parts (moved from net/drill/census.py): the classes of the pointer and approach heads
from masks.of only, the legacy numbers (side_counts, flank_weighted, side_labels) and the weights (learn.weights with
the drop from the masks).

    batch_of(rows)        the mixture as a Batch view without the per-row cards
    pointer_labels(batch), side_counts(batch, w), flank_weighted(batch, meta, w, weights_fn), weights(rows, meta, item)
    classes(batch, w, side)   {head: {class: census.Cls}} - grad = weight > 0, eligible and >= 2 open candidates
    extra(batch, rows, meta, w, side)   the legacy numbers and the band's share
"""
import torch

from net.drill import census as CN
from net.drill import masks as MK
from net.drill import record as RC

APPROACHES = ("front", "left", "right")       # net/tactic/orders.py APPROACHES
share = CN.share


def batch_of(rows):
    """The mixture as a Batch view without the per-row cards (census needs no input)."""
    # (idx = slice(None): views of the mixture's tensors, no copy of a 1.3 GB buffer)
    return RC.from_netbuffer({**rows, "cards": torch.zeros((int(rows["b"].max()) + 1 if rows["b"].numel() else 1, 1))},
                             slice(None))


def pointer_labels(batch):
    """(masks, attack [M, G], keep [M, G]) - the pointer's labels on decided units (KEEP is a label)."""
    m = MK.of("tactic_pointer", batch)
    lab = RC.label_of(batch)
    dec = batch.status == RC.DECIDED
    return m, dec & (lab >= 0), dec & (lab < 0)


def side_counts(batch, w):
    """imitate.side_counts from masks.of('tactic_approach')."""
    m = MK.of("tactic_approach", batch, w)
    lab, ap = RC.label_of(batch), batch.aux["ap"].long()
    base = (lab >= 0) & (ap >= 0) & (w > 0)
    bd = (m.why & MK.R_SIDE_BAND) != 0
    cl = (m.why & MK.R_SIDE_CLOSED) != 0
    return {"side_labels": int(base.sum()), "at_threshold": int((bd & base).sum()),
            "closed_for_pointer": int((cl & base).sum()),
            "closed_share": round(int((cl & base).sum()) / max(int(base.sum()), 1), 4),
            "counting": int((m.eligible & base).sum())}


def flank_weighted(batch, meta, w, weights_fn):
    """imitate.flank_weight_share from the masks: of the flank-labelled decided units the share with weight > 0, all and
    on second_row starts; with the fix (w) and without (the weights re-made with T0-1's drop, tactic_drop lift=False)."""
    lab = RC.label_of(batch) >= 0
    ap = batch.aux["ap"]
    dec = batch.status == RC.DECIDED
    fl = dec & lab & ((ap == MK.FLANKS[0]) | (ap == MK.FLANKS[1]))
    sr = torch.tensor([m.get("start") == "second_row" for m in meta], dtype=torch.bool)[batch.b][:, None] & fl
    w0 = weights_fn(MK.of("tactic_pointer", batch, lift=False).item)
    out = {}
    for nm, ww in (("with_fix", w), ("without_fix", w0)):
        out[nm] = {"all": share(int((fl & (ww > 0)).sum()), int(fl.sum())),
                   "second_row": share(int((sr & (ww > 0)).sum()), int(sr.sum()))}
    return out


def weights(rows, meta, item=None):
    """learn.weights (mixture-wide masses: family, change / KEEP split, taught and trivial factors) with the drop from
    the masks (item = decided & ~drop) - the census's one way to the weights."""
    from net.tactic import learn as LN
    if item is None:
        item = MK.of("tactic_pointer", batch_of(rows)).item
    head = rows["head"].bool()
    if "done_own" in rows:                             # (post-finish rows: not decided, no weight)
        head = head & ~rows["done_own"].bool()[:, None]
    return LN.weights({**rows, "head": head, "drop": head & ~item}, meta)


def _open2(batch):
    """[M, G] at least two open candidates (legal and not don't-care) at the pointer - a label that can move it."""
    return (batch.legal & ~batch.dont_care).sum(-1) >= 2


def classes(batch, w, side):
    """(module doc)."""
    m, att, keep = pointer_labels(batch)
    dc = ~m.item & (batch.status == RC.DECIDED)
    out = {"pointer": {}}
    for nm, x in (("attack", att), ("keep", keep)):
        rs = {n: x & ((m.why & MK.BIT[n]) != 0) for n in ("DROPPED", "DC_THR", "OUT_OF_SET")}
        rs["zero_weight"] = x & ~dc & (w <= 0)
        wt = x & (w > 0) & m.eligible
        out["pointer"][nm] = CN.Cls(x, x & dc, wt, wt & _open2(batch), rs)
    if side:
        ma = MK.of("tactic_approach", batch)
        ap = batch.aux["ap"].long()
        out["approach"] = {}
        for i, nm in enumerate(APPROACHES):
            x = att & (ap == i)
            band = x & ((ma.why & MK.R_SIDE_BAND) != 0)
            rs = {"drop": x & dc, "band": band, "closed": x & ((ma.why & MK.R_SIDE_CLOSED) != 0),
                  "zero_weight": x & ~dc & (w <= 0)}
            wt = x & (w > 0) & ma.eligible
            out["approach"][nm] = CN.Cls(x, x & (dc | band), wt, wt, rs)   # (3 approaches always open: the side head)
    return out


def extra(batch, rows, meta, w, side):
    """(module doc) {side_labels, side_count, flank_weighted, band_share} of a side drill; {} else."""
    if not side:
        return {}
    out = {}
    sc = side_counts(batch, w)
    cls = classes(batch, w, side)["approach"]
    # (train.log line 9's side_labels: attack labels on decided units, not dropped, with an approach)
    out["side_labels"] = {nm: int(cls[nm].labels.sum()) - int(cls[nm].removed["drop"].sum()) for nm in APPROACHES}
    out["side_count"] = sc
    out["flank_weighted"] = flank_weighted(batch, meta, w, lambda item: weights(rows, meta, item)[0])
    out["band_share"] = round(sc["at_threshold"] / max(sc["side_labels"], 1), 4)
    out["band_n"] = sc["side_labels"]
    return out
