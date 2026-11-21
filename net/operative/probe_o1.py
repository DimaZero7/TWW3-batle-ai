"""Drill O0-1 - the label probe (METHOD.md recipe step 4; config o1.probe): can the level see the cause? A small MLP
on the level's input row (lines.INPUT) learns the teacher's change label (the shadow label: advance.teach asked on every
tick of the exam's play) on the taught set's states, and is read on states of seeds it never saw: the held set, and the
taught set's other half of seeds (seed // 2 odd).
    followed       the share of the teacher's change ticks the probe also changes on (>= 0.90 on held-out seeds)
    false_advance  the share of the teacher's KEEP ticks the probe changes on (<= 0.03)
    leakage        the same probe with the motion columns (lines.MOTION) zeroed, read on the COMPLETE decision-tick twin
                   pairs (a cause scene and its come twin of the same seed at start_s; role twins out) - its balanced
                   accuracy there within 0.5 +- CHANCE_BAND (two-sided): no other column gives the cause away. The full
                   probe's on the same pairs beside it (should be ~1). First-decision ticks (start_s) and repeated
                   opportunities (later ticks) reported apart.
THIS PROBE READS THE ENGINEERED ROW lines.INPUT (scaled operands of the teacher's own tests): it shows only that the
label can be recovered from those facts. The binding probe is stage B's, on the network's real input (the visible
rows through the frozen card encoder, the group / order / pointer inputs and masks). Line 7 here: reported, not gated.
Data: the exam's files <dir>/<set>_<player>.json (ticks), players TEACH and DEFAULT (the teacher's states and the
default's - on DEFAULT's the label stays 'change' in a cause scene at every tick: the shadow).
    python -m net.operative.probe_o1 --dir tmp/o1/exam [--out tmp/o1/exam/probe.json]
"""
import argparse
import json
from pathlib import Path

import torch

from net.operative import cfg
from net.operative import lines as LN
from net.operative.exam_o1 import TICK_COLS


def _load(D, nm, players):
    rows, metas = [], json.loads((D / f"{nm}_meta.json").read_text(encoding="utf-8"))
    for p in players:
        f = D / f"{nm}_{p}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            rows += [(p, r) for r in d["ticks"]]
    return rows, metas


def _xy(rows, metas, sel=None, ablate=False):
    ci = {k: i for i, k in enumerate(TICK_COLS)}
    x0 = ci[LN.INPUT[0]]
    X, Y, K = [], [], []
    mot = [LN.INPUT.index(m) for m in LN.MOTION]
    for p, r in rows:
        if sel is not None and not sel(metas[r[0]], r):
            continue
        x = list(r[x0:x0 + len(LN.INPUT)])
        if ablate:
            for j in mot:
                x[j] = 0.0
        X.append(x)
        Y.append(float(r[ci["label"]]))
        K.append((p, r[0], r[1]))
    if not X:
        return None, None, K
    return torch.tensor(X), torch.tensor(Y), K


def fit(X, Y, seed=0):
    c = cfg()["o1"]["probe"]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], int(c["hidden"])), torch.nn.ReLU(),
                              torch.nn.Linear(int(c["hidden"]), 1))
    opt = torch.optim.Adam(net.parameters(), lr=float(c["lr"]))
    pos = float(Y.sum())
    w = torch.tensor(max((len(Y) - pos) / max(pos, 1.0), 1.0))        # (the change ticks weighted up to balance)
    for _ in range(int(c["epochs"])):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X)[:, 0], Y, pos_weight=w)
        loss.backward()
        opt.step()
    return net


def score(net, X, Y):
    if X is None:
        return None
    with torch.no_grad():
        p = (net(X)[:, 0] > 0).float()
    ch, kp = Y > 0.5, Y < 0.5
    return {"followed": None if not bool(ch.any()) else round(float(p[ch].mean()), 4), "change_ticks": int(ch.sum()),
            "false_advance": None if not bool(kp.any()) else round(float(p[kp].mean()), 4), "keep_ticks": int(kp.sum())}


def twin_pairs(metas, t0):
    """The decision-tick rows (sec t0) of COMPLETE pairs only - a cause scene and its come twin of the same seed and
    held axis, both in the set (role twins out: they differ by att, not by motion; the review 3)."""
    have = {}
    for m in metas:
        if m.get("seed") is not None and (m["cause"] or m["twin"] == "come"):
            have.setdefault((m["seed"], m.get("held")), set()).add("cause" if m["cause"] else "come")
    full = {k for k, v in have.items() if v == {"cause", "come"}}
    return lambda m, r: r[1] == t0 and (m["cause"] or m["twin"] == "come") and (m.get("seed"), m.get("held")) in full


def balanced(net, X, Y):
    if X is None:
        return None
    with torch.no_grad():
        p = (net(X)[:, 0] > 0).float()
    ch, kp = Y > 0.5, Y < 0.5
    if not bool(ch.any()) or not bool(kp.any()):
        return None
    return round(0.5 * (float(p[ch].mean()) + float(1 - p[kp].mean())), 4)


CHANCE_BAND = 0.10        # (the ablated probe's balanced accuracy on the pairs: within 0.5 +- this = chance, two-sided)


def run(D, players=("TEACH", "DEFAULT")):
    D = Path(D)
    t0 = int(LN.conf()["start_s"])
    tr_rows, tr_meta = _load(D, "taught", players)
    out = {"what": "a probe of the ENGINEERED row lines.INPUT (scaled operands of the teacher's own tests): it shows the "
                   "label is recoverable from those facts, no more; the binding probe is stage B's, on the network's real "
                   "input. Line 7 here is REPORTED, NOT GATED.",
           "inputs": list(LN.INPUT), "motion": list(LN.MOTION)}
    # (the seeds split in pairs: seed // 2 even / odd - the twins alternate come / role with the seed's parity, so each
    # half holds both kinds of twin and a cause scene stays with its twins)
    even = lambda m, r: (m["seed"] // 2) % 2 == 0                                  # noqa: E731
    odd = lambda m, r: (m["seed"] // 2) % 2 == 1                                   # noqa: E731
    first = lambda m, r: r[1] == t0                                                # noqa: E731
    later = lambda m, r: r[1] > t0                                                 # noqa: E731
    h_rows, h_meta = _load(D, "held", players) if (D / "held_meta.json").exists() else ([], None)
    for ab in (False, True):
        key = "ablated" if ab else "full"
        X, Y, _ = _xy(tr_rows, tr_meta, even, ab)
        if X is None:
            return {"why": "no taught ticks"}
        net = fit(X, Y)
        res = {"train": score(net, X, Y)}
        for nm, rows, mt, sel in (("taught_odd", tr_rows, tr_meta, odd), ("held", h_rows, h_meta, lambda m, r: True)):
            if mt is None:
                continue
            res[nm] = {"all": score(net, *_xy(rows, mt, sel, ab)[:2]),
                       "first_decision": score(net, *_xy(rows, mt, lambda m, r: sel(m, r) and first(m, r), ab)[:2]),
                       "repeated": score(net, *_xy(rows, mt, lambda m, r: sel(m, r) and later(m, r), ab)[:2])}
            tp = twin_pairs(mt, t0)
            Xp, Yp, Kp = _xy([r for r in rows if r[0] == "DEFAULT"], mt, lambda m, r: sel(m, r) and tp(m, r), ab)
            res[nm]["twin_pairs"] = {"n_pairs": len(Kp) // 2, "balanced": balanced(net, Xp, Yp)}
        out[key] = res
    f, a = out["full"], out["ablated"]
    parts = [k for k in ("taught_odd", "held") if k in f]
    fol = {k: f[k]["all"]["followed"] for k in parts}
    fa = {k: f[k]["all"]["false_advance"] for k in parts}
    tw = {k: a[k]["twin_pairs"] for k in parts}
    ok = all(fol[k] is not None and fol[k] >= 0.90 and fa[k] is not None and fa[k] <= 0.03 for k in parts) and all(
        v["balanced"] is not None and abs(v["balanced"] - 0.5) <= CHANCE_BAND for v in tw.values())
    out["line_7"] = {"rule": "REPORTED, NOT GATED (the engineered row): the probe follows the teacher's change ticks >= 0.90 "
                             "and false advance <= 3 %% on BOTH held-out parts (taught odd seeds, held); with the motion "
                             "columns zeroed its balanced accuracy on the complete cause / come pairs at the decision tick "
                             "within 0.5 +- %.2f" % CHANCE_BAND,
                     "followed": fol, "false_advance": fa, "ablated_twin_pairs": tw,
                     "full_twin_pairs": {k: f[k]["twin_pairs"] for k in parts}, "would_pass": ok, "passed": "reported"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tmp/o1/exam")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    res = run(a.dir)
    Path(a.out or Path(a.dir) / "probe.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
