"""The label census (tmp/engine/design_merged.md section 4; brief_A2 Part 4), level-blind: what supervision reaches
each head. The level gives the classes (levels/<level>.census_classes, from masks.of only); the card gives the minima.

    Cls(labels, dont_care, weighted, grad, removed)   [M, G] bool masks of one class: its labels on decided units, the
        don't-care ones (dropped, banded - by design out of the loss), the weighted ones (w > 0 and eligible), the ones
        with a gradient (weighted and at least two open candidates: a label that moves the head), removed {reason: mask}
    check(classes, meta, b, card, smoke, extra)   per head and class: labels, don't-care (count and share), weighted,
        grad, kept = weighted / (labels - don't-care) (net/config/drill.json census, its _why); per start cell (labels -
        don't-care >= cell_min_n) the same. Halts (CensusHalt):
          - grad < the card's minimum (card.census[head][class], default cards.CENSUS_MIN) - also when the class has
            fewer labels than that left after the don't-care ones, or none at all;
          - kept < keep_min (n >= cell_min_n), the class or a cell;
          - the don't-care share over card.dont_care_max[head][class];
          - extra's band_share over card.band_share_max (extra['band_n'] >= cell_min_n).
        A smoke run (smoke > 0, a few battles) applies no minimum and says so ("minima": "smoke: not applied").
    old_masks(batch, parent_fn, child_fn, classes)   parent and child mask functions on the SAME frozen parent rows: per
        class the labels each keeps and the moved ones; moved(res, card) halts over card.masks_moved_max
"""
from collections import namedtuple

import torch


def _conf():
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "config" / "drill.json"
    return json.loads(p.read_text(encoding="utf-8"))["census"]


KEEP_MIN = float(_conf()["keep_min"])        # (net/config/drill.json census, its _why: weighted / (labels - dc))
CELL_MIN_N = int(_conf()["cell_min_n"])

Cls = namedtuple("Cls", "labels dont_care weighted grad removed")


class CensusHalt(RuntimeError):
    pass


def share(x, n):
    import math
    if n == 0:
        return {"share": None, "two_errors": None, "n": 0}
    p = x / n
    return {"share": round(p, 4), "two_errors": round(2 * math.sqrt(max(p * (1 - p), 1e-12) / n), 4), "n": int(n)}


def _row(c, sel=None):
    s = (lambda t: t) if sel is None else (lambda t: t & sel)                         # noqa: E731
    n, ndc, nw, ng = int(s(c.labels).sum()), int(s(c.dont_care).sum()), int(s(c.weighted).sum()), int(s(c.grad).sum())
    return {"labels": n, "dont_care": ndc, "dont_care_share": round(ndc / n, 4) if n else None, "weighted": nw,
            "grad": ng, "kept": share(nw, n - ndc), "removed": {k: int(s(v).sum()) for k, v in c.removed.items()}}


def check(classes, meta, b, card, smoke=0, extra=None, done=None, halt=True):
    """(module doc) b [M] the rows' battle ids (the cells' start from meta)."""
    from net.drill.cards import CENSUS_MIN
    mins = getattr(card, "census", {}) or {}
    dcm = getattr(card, "dont_care_max", {}) or {}
    starts = sorted({m.get("start") for m in meta if m.get("start") is not None})
    st_of = [m.get("start") for m in meta]
    out, fails = {"minima": "smoke: not applied" if smoke else "applied"}, []
    for head, cs in classes.items():
        out[head] = {}
        for nm, c in cs.items():
            r = _row(c)
            r["cells"] = {}
            for s in starts:
                sel = torch.tensor([v == s for v in st_of], dtype=torch.bool)[b][:, None]
                cc = _row(c, sel)
                if cc["labels"] - cc["dont_care"] >= CELL_MIN_N:
                    r["cells"][s] = cc
                    if cc["kept"]["share"] < KEEP_MIN:
                        fails.append(f"{head}/{nm} at start {s}: weighted / (labels - don't-care) "
                                     f"{cc['kept']['share']} < {KEEP_MIN} (n {cc['kept']['n']})")
            k = r["kept"]
            if k["n"] >= CELL_MIN_N and k["share"] < KEEP_MIN:
                fails.append(f"{head}/{nm}: weighted / (labels - don't-care) {k['share']} < {KEEP_MIN} (n {k['n']}); "
                             f"removed {r['removed']}")
            mn = int((mins.get(head) or {}).get(nm, CENSUS_MIN))
            r["grad_min"] = mn
            if not smoke:
                if r["grad"] < mn:
                    fails.append(f"{head}/{nm}: {r['grad']} labels reach the head with a gradient < the card's minimum "
                                 f"{mn} (labels {r['labels']}, don't-care {r['dont_care']}, weighted {r['weighted']})")
                mx = (dcm.get(head) or {}).get(nm)
                if mx is not None and r["dont_care_share"] is not None and r["dont_care_share"] > float(mx):
                    fails.append(f"{head}/{nm}: the don't-care share {r['dont_care_share']} > the card's "
                                 f"dont_care_max {mx}")
            out[head][nm] = r
    if extra:
        out.update({k: v for k, v in extra.items() if k != "band_n"})
        bm = getattr(card, "band_share_max", None)
        if bm is not None and "band_share" in extra and int(extra.get("band_n", 0)) >= CELL_MIN_N \
                and extra["band_share"] > float(bm):
            fails.append(f"the band's share {extra['band_share']:.4f} > the card's band_share_max {bm}")
    if done is not None:
        out["done_rows"] = {k: {"rows": v[0], "left_out": v[1], "share": round(v[1] / max(v[0], 1), 4)}
                            for k, v in done.items()}
    out["halts"] = fails
    if fails and halt:
        raise CensusHalt("; ".join(fails))
    return out


def old_masks(batch, parent_fn, child_fn, classes):
    """Parent and child mask functions (each batch -> eligible [M, G]) on the SAME frozen parent rows; classes {name:
    [M, G] bool}: per class the labels each keeps and the moved count (a sample kept with the parent, never two
    collections)."""
    p, c = parent_fn(batch), child_fn(batch)
    return {nm: {"labels": int(x.sum()), "parent": int((x & p).sum()), "child": int((x & c).sum()),
                 "moved": int((x & (p ^ c)).sum())} for nm, x in classes.items()}


def moved(res, card, halt=True):
    """old_masks' moved labels against card.masks_moved_max (more halts)."""
    n = sum(v["moved"] for v in res.values())
    mx = int(getattr(card, "masks_moved_max", 0) or 0)
    out = {"by_class": res, "moved": n, "max": mx, "halts": []}
    if n > mx:
        out["halts"].append(f"the child's masks move {n} old-drill labels > the card's masks_moved_max {mx}")
        if halt:
            raise CensusHalt(out["halts"][0])
    return out
