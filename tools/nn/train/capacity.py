"""Cheap capacity and forgetting signals of a training iteration (docs/en/training/training.md
"Capacity and forgetting"): when to widen the network. Nothing is played: it reads what a test5 run
already wrote.

Plain python + numpy (no torch): runs in the project's .venv.

    python -m tools.nn.train.capacity build/nn-train/test5/it2               # vs the previous iteration
    python -m tools.nn.train.capacity build/nn-train/test5/it2 --prev build/nn-train/test5/it1

forgetting(cur, prev)  per opponent: pair score and pair gold; per opponent and matchup (ours first):
                       win rate and gold trade ((destroyed - lost) / budget); per opponent, our faction
                       and role: win rate. Higher is better for all, in "points" (0.05 = 5 points).
                       With the per-battle lists of both evaluations (evaluate.play "battles") the
                       battles are matched (opponent, seed, side: the same armies) and the noise is the
                       paired difference's standard error; without them (an older evaluation) the two
                       evaluations' own errors (binomial for win rates; none for gold trade: a drop is
                       then only "watch"; an evaluation without matchups: its scene rows). Also per
                       opponent and role: win rate and gold trade. A drop is forgetting when it is MIN_DROP or more AND beyond
                       Z x its standard error (Z = 2.58: some 40 numbers are compared each time).
training(log)          the run's log.jsonl: the critic's explained variance (early third -> late
                       third), value and policy loss (the late half's slope: a plateau when it is
                       within 2 standard errors of 0), entropy, grad norm (ppo.update, when logged).
rating(points)         the fitted overall rating (skill.fit) at minute 0 and at the end, ± 95%.
verdict()              ok / watch / widen-candidate and the reasons:
                         widen-candidate  forgetting (old numbers drop beyond noise) while others
                                          rise beyond noise and the overall rating does not: the
                                          network trades old skill for new, a sign it is out of room;
                         watch            forgetting with a rising rating (a trade-off), forgetting
                                          with nothing rising (the training loses skill: settings
                                          first), a drop
                                          without a noise estimate, a plateau of the rating and the
                                          value loss, a falling critic, an entropy collapse, a
                                          rising grad norm;
                         ok               none of these.
Against the previous iteration a drop counts as forgetting only when both evaluations name the same
simulator version (the script baseline's cache name: tools/nn/train/evaluate.py sim_version, which
covers the simulator, the armies' generator and the scripts); else it is "drop?".
The previous iteration: --prev, else the folder of the run's --init (build/nn-train/test5/<label>/m10.pt
-> <label>), else the label with its number one less (it3 -> it2). Its final evaluation (after.json)
against ours; within the iteration: minute 0 (before.json) against the end, the same battles.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from tools.nn.train import matchups

ROOT = Path(__file__).resolve().parents[3]
MIN_DROP = 0.05
Z = 2.58
EV_LOW = 0.9            # the critic explains less than this: it falls behind
EV_FALL = 0.03          # ... or loses this much from the early third to the late one
ENTROPY_FALL = 0.5      # the late third's entropy below this share of the early third's: a collapse
GRAD_RISE = 1.5         # the late third's grad norm this many times the early third's


# --- the numbers of one evaluation ----------------------------------------------------------------

def _mean_se(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return None, None
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None


def cells(ev):
    """{name: {"value", "se", "per": {key: value} or None}} of one evaluation (evaluate.play's result):
    the per-battle (per-pair) values when it has the lists, else the summary's numbers."""
    out = {}
    for opp, o in (ev or {}).get("by_opponent", {}).items():
        per = (ev.get("battles") or {}).get(opp)
        if per and per.get("won"):
            _from_battles(out, opp, per)
        else:
            _from_summary(out, opp, o, (ev.get("by_scene") or {}).get(opp))
    return out


def _put(out, name, per):
    v, se = _mean_se(list(per.values()))
    if v is not None:
        out[name] = {"value": v, "se": se, "per": per}


def _from_battles(out, opp, b):
    n = len(b["won"])
    won = np.asarray(b["won"], dtype=float)
    own = [matchups.short(f) for f in b["own"]]
    enemy = [matchups.short(f) for f in b["enemy"]]
    key = list(zip(b["seed"], b["side"]))
    trade = np.asarray(b.get("trade") or [np.nan] * n, dtype=float)
    pairs = {}
    for i, p in enumerate(b.get("pair") or [-1] * n):
        if p >= 0:
            pairs.setdefault(p, []).append(i)
    ps, pg = {}, {}
    for idx in pairs.values():
        if len(idx) != 2:
            continue
        seed = b["seed"][idx[0]]
        ps[seed] = float(won[idx].sum() - 1)                       # +1 won both, 0 split, -1 lost both
        if "destroyed" in b:
            d = sum(b["destroyed"][i] for i in idx) - sum(b["lost"][i] for i in idx)
            pg[seed] = d / max(1e-9, (b["budget"][idx[0]] + b["budget"][idx[1]]) / 2)
    _put(out, f"{opp} pair score", ps)
    _put(out, f"{opp} pair gold", pg)
    for m in sorted(set(zip(own, enemy)), key=lambda p: (matchups._order(p[0]), matchups._order(p[1]))):
        sel = [i for i in range(n) if (own[i], enemy[i]) == m]
        _put(out, f"{opp} {m[0]}-{m[1]} win", {key[i]: won[i] for i in sel})
        _put(out, f"{opp} {m[0]}-{m[1]} trade", {key[i]: trade[i] for i in sel if np.isfinite(trade[i])})
    for f in sorted(set(own), key=matchups._order):
        for role, att in (("attack", True), ("defend", False)):
            sel = [i for i in range(n) if own[i] == f and bool(b["attacks"][i]) == att]
            _put(out, f"{opp} {f} {role} win", {key[i]: won[i] for i in sel})
    for role, att in (("attack", True), ("defend", False)):
        sel = [i for i in range(n) if bool(b["attacks"][i]) == att]
        _put(out, f"{opp} {role} win", {key[i]: won[i] for i in sel})
        _put(out, f"{opp} {role} trade", {key[i]: trade[i] for i in sel if np.isfinite(trade[i])})


def _binom(p, n):
    return float(np.sqrt(p * (1 - p) / n)) if p is not None and n and n > 1 else None


NAMES = {"Empire": "EMP", "Skaven": "SKV"}      # evaluate.faction_name -> matchups.short


def _scene_rows(rows):
    """An older evaluation's by_scene rows ("Empire v Skaven, attack": {games, wins}) -> {matchup: [wins,
    games]}, {(faction, role): [wins, games]}."""
    ms, fr = {}, {}
    for label, c in (rows or {}).items():
        m = re.match(r"(.+) v (.+), (attack|defend)$", label)
        if not m:
            continue
        a, e = (NAMES.get(x, x[:3].upper()) for x in m.group(1, 2))
        for d, k in ((ms, f"{a}-{e}"), (fr, (a, m.group(3)))):
            w = d.setdefault(k, [0, 0])
            w[0] += c["wins"]
            w[1] += c["games"]
    return ms, fr


def _from_summary(out, opp, o, scene_rows=None):
    p = o.get("pairs") or {}
    if p.get("pair_score") is not None and p.get("pairs"):
        var = p["won_both"] + p["lost_both"] - p["pair_score"] ** 2
        out[f"{opp} pair score"] = {"value": p["pair_score"], "se": float(np.sqrt(max(var, 0) / p["pairs"])), "per": None}
    g = (o.get("pair_gold") or {}).get("pair_gold")
    if g:
        out[f"{opp} pair gold"] = {"value": g["value"], "se": g["se"], "per": None}
    for m, c in (o.get("matchups") or {}).items():
        if c.get("games"):
            out[f"{opp} {m} win"] = {"value": c["win_rate"], "se": _binom(c["win_rate"], c["games"]), "per": None}
            if c.get("gold_trade") is not None:
                out[f"{opp} {m} trade"] = {"value": c["gold_trade"], "se": None, "per": None}
    for f, roles in (o.get("factions") or {}).items():
        for role, c in roles.items():
            if c.get("games"):
                out[f"{opp} {f} {role} win"] = {"value": c["win_rate"], "se": _binom(c["win_rate"], c["games"]),
                                                "per": None}
    if not o.get("matchups"):                       # an older evaluation: its scene rows
        ms, fr = _scene_rows(scene_rows)
        for name, (w, g) in [(f"{opp} {m} win", x) for m, x in ms.items()] +                 [(f"{opp} {f} {r} win", x) for (f, r), x in fr.items()]:
            if g:
                out[name] = {"value": w / g, "se": _binom(w / g, g), "per": None}
    for role, c in (o.get("roles") or {}).items():
        if c.get("games"):
            out[f"{opp} {role} win"] = {"value": c["win_rate"], "se": _binom(c["win_rate"], c["games"]), "per": None}
            if c.get("gold_trade") is not None:
                out[f"{opp} {role} trade"] = {"value": c["gold_trade"], "se": None, "per": None}


# --- comparisons ------------------------------------------------------------------------------------

def compare(cur, prev, min_drop=MIN_DROP, z=Z):
    """[{name, prev, cur, diff, se, paired, n, flag}] for every number of prev (cells()) also in cur;
    flag: "forgetting" (a drop of min_drop or more beyond z standard errors), "drop?" (min_drop or
    more, within the noise or without an estimate), "gain" (a rise beyond the noise), "" otherwise."""
    rows = []
    for name, p in prev.items():
        c = cur.get(name)
        if not c:
            continue
        paired = bool(p["per"] and c["per"])
        if paired:
            keys = [k for k in p["per"] if k in c["per"]]
            if len(keys) < 2:
                continue
            d = np.array([c["per"][k] - p["per"][k] for k in keys])
            diff, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
            n = len(keys)
        else:
            diff = c["value"] - p["value"]
            se = float(np.hypot(p["se"], c["se"])) if p["se"] is not None and c["se"] is not None else None
            n = None
        beyond = se is not None and abs(diff) > z * se
        flag = ("forgetting" if -diff >= min_drop and beyond else "drop?" if -diff >= min_drop
                else "gain" if diff > 0 and beyond else "")
        rows.append({"name": name, "prev": p["value"], "cur": c["value"], "diff": diff, "se": se, "paired": paired,
                     "n": n, "flag": flag})
    return rows


def _version(ev):
    """The simulator version of an evaluation (its script baseline's cache name ends with it), or None."""
    for o in (ev or {}).get("by_opponent", {}).values():
        c = (o.get("baseline") or {}).get("cache")
        if c:
            return Path(c).stem.rsplit("_", 1)[-1]
    return None


# --- the training log -------------------------------------------------------------------------------

def _slope(y):
    """(slope a step, its standard error) of a straight line through y."""
    y = np.asarray(y, dtype=float)
    if len(y) < 4:
        return None, None
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    b = float((x * (y - y.mean())).sum() / (x * x).sum())
    res = y - y.mean() - b * x
    return b, float(np.sqrt((res ** 2).sum() / (len(y) - 2) / (x * x).sum()))


def training(log):
    """{name: {early, late, ...}} of a run's log.jsonl lines ([dict]): early / late thirds' means of ev,
    value_loss, policy_loss, entropy, grad_norm, kl, clip; for the losses also the late half's slope a
    100 updates and its standard error, plateau when within 2 of them."""
    if len(log) < 6:
        return {"updates": len(log)}
    out = {"updates": len(log)}
    third = len(log) // 3
    for k in ("ev", "value_loss", "policy_loss", "entropy", "grad_norm", "kl", "clip"):
        y = np.array([x.get(k, np.nan) if x.get(k) is not None else np.nan for x in log], dtype=float)
        if not np.isfinite(y).any():
            continue
        e, l_ = y[:third], y[-third:]
        c = {"early": float(np.nanmean(e)) if np.isfinite(e).any() else None,
             "late": float(np.nanmean(l_)) if np.isfinite(l_).any() else None}
        if k in ("value_loss", "policy_loss"):
            half = y[len(y) // 2:]
            b, se = _slope(half[np.isfinite(half)])
            if b is not None:
                c.update(slope_100=100 * b, slope_se_100=100 * se, plateau=bool(abs(b) <= 2 * se))
        out[k] = c
    return out


def rating(ev):
    o = ((ev or {}).get("skill") or {}).get("overall")
    return (o["value"], o["se"]) if o else None


# --- the verdict --------------------------------------------------------------------------------------

def verdict(across, within, train, r0, r1):
    """(verdict, [reasons]) from the comparisons with the previous iteration (across) and within this
    one (within), the training log's signals (train: training()) and the overall rating at the start
    and the end (value, se) or None."""
    reasons, watch = [], []
    forgot = [r for rows in (across, within) for r in rows if r["flag"] == "forgetting"]
    unsure = [r for rows in (across, within) for r in rows if r["flag"] == "drop?"]
    rises = None
    if r0 and r1:
        d, se = r1[0] - r0[0], float(np.hypot(r0[1], r1[1]))
        rises = d > 1.96 * se
        flat = abs(d) <= 1.96 * se
    else:
        flat = False
    gains = [r for rows in (across, within) for r in rows if r["flag"] == "gain"]
    if forgot:
        worst = min(forgot, key=lambda r: r["diff"])
        what = f"{len(forgot)} old number(s) drop beyond noise (worst {worst['name']} {100 * worst['diff']:+.1f} points)"
        if rises:
            watch.append(what + " while the overall rating rises: a trade-off between matchups")
        elif gains:
            reasons.append(what + f" while {len(gains)} rise beyond noise and the overall rating does not: "
                                  "new skill at the cost of old (interference)")
        else:
            watch.append(what + ", nothing rises beyond noise: the training loses skill (settings or reward "
                                "first, not the size)")
    if unsure:
        watch.append(f"{len(unsure)} drop(s) of 5+ points within the noise or without a noise estimate")
    vl = train.get("value_loss") or {}
    if flat and vl.get("plateau"):
        watch.append("the overall rating and the value loss both plateau")
    ev = train.get("ev") or {}
    if ev.get("late") is not None and (ev["late"] < EV_LOW or (ev.get("early") or 0) - ev["late"] > EV_FALL):
        watch.append(f"the critic's explained variance {ev.get('early', 0):.3f} -> {ev['late']:.3f}")
    en = train.get("entropy") or {}
    if en.get("early") and en.get("late") is not None and en["late"] < ENTROPY_FALL * en["early"]:
        watch.append(f"entropy collapses {en['early']:.4f} -> {en['late']:.4f}")
    gn = train.get("grad_norm") or {}
    if gn.get("early") and gn.get("late") is not None and gn["late"] > GRAD_RISE * gn["early"]:
        watch.append(f"grad norm rises {gn['early']:.3f} -> {gn['late']:.3f}")
    if reasons:
        return "widen-candidate", reasons + watch
    if watch:
        return "watch", watch
    return "ok", ["no old number drops beyond noise; the critic, entropy and gradients steady"]


# --- files ----------------------------------------------------------------------------------------------

def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def final(folder):
    """The final evaluation of a test5 folder: after.json, else the last eval_m<minute>.json."""
    folder = Path(folder)
    ev = _load(folder / "after.json")
    if ev:
        return ev
    evs = sorted(folder.glob("eval_m*.json"), key=lambda p: float(p.stem[6:]))
    return _load(evs[-1]) if evs else None


def previous(folder, report=None):
    """The previous iteration's folder (the module's docstring) or None."""
    folder = Path(folder)
    init = Path((report or {}).get("init") or "")
    cand = []
    if init.parent.parent.name == "test5":
        cand.append(folder.parent / init.parent.name)
    m = re.match(r"(.*?)(\d+)$", folder.name)
    if m and int(m.group(2)) > 0:
        cand.append(folder.parent / f"{m.group(1)}{int(m.group(2)) - 1}")
    for c in cand:
        if c != folder and final(c):
            return c
    return None


def run_log(report, run_dir=None):
    """The training's log.jsonl lines: run_dir, else the report's run (a container path /repo/... is
    mapped to this checkout)."""
    run = run_dir or (report or {}).get("run")
    if not run:
        return []
    p = Path(run)
    if str(run).startswith("/repo/"):
        p = ROOT / str(run)[len("/repo/"):]
    try:
        return [json.loads(x) for x in (p / "log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, ValueError):
        return []


def report(folder, prev=None, run_dir=None, start=None, end=None, rep=None):
    """The capacity report of a test5 folder: {prev, verdict, reasons, across, within, training,
    rating, versions}. prev: the previous iteration's folder (or its report.json) or None (found as the
    module's docstring says); start / end: the evaluations at minute 0 and at the end (default: the
    folder's before.json and final()); rep: its report.json (default: read)."""
    folder = Path(folder)
    rep = rep if rep is not None else (_load(folder / "report.json") or {})
    start = start if start is not None else _load(folder / "before.json")
    end = end if end is not None else final(folder)
    if prev is not None and Path(prev).suffix == ".json":
        prev = Path(prev).parent
    prev = Path(prev) if prev is not None else previous(folder, rep)
    pev = final(prev) if prev else None
    c_end = cells(end)
    same_sim = bool(_version(pev)) and _version(pev) == _version(end)
    across = compare(c_end, cells(pev)) if pev else []
    if not same_sim:          # another simulator or armies (or unknown): a drop may be theirs, not forgetting
        for r in across:
            r["flag"] = {"forgetting": "drop?", "gain": ""}.get(r["flag"], r["flag"])
    within = compare(c_end, cells(start)) if start else []
    train = training(run_log(rep, run_dir))
    r0, r1 = rating(start), rating(end)
    v, reasons = verdict(across, within, train, r0, r1)
    versions = {"prev": _version(pev), "start": _version(start), "end": _version(end)}
    if pev and not same_sim:
        reasons.append(f"against {prev.name} only 'drop?': its simulator / armies "
                       + (f"differ (version {versions['prev']} -> {versions['end']})" if versions["prev"]
                          else "are unknown (an evaluation without the script baseline)"))
    return {"prev": str(prev) if prev else None, "verdict": v, "reasons": reasons, "across": across,
            "within": within, "training": train, "rating": {"start": r0, "end": r1}, "versions": versions}


def _row(r):
    se = "?" if r["se"] is None else f"{100 * Z * r['se']:.1f}"
    return (f"| {r['name']} | {100 * r['prev']:+.1f} → {100 * r['cur']:+.1f} | {100 * r['diff']:+.1f} ± {se}"
            f"{' (paired, n ' + str(r['n']) + ')' if r['paired'] else ''} | {r['flag']} |")


def lines(rep, top=5):
    """The capacity block as markdown lines."""
    out = ["capacity and forgetting (docs/en/training/training.md \"Capacity and forgetting\"; points: x100, "
           f"± {Z} SE):", ""]
    r0, r1 = rep["rating"]["start"], rep["rating"]["end"]
    if r0 and r1:
        out.append(f"overall rating, minute 0 → end: {r0[0]:+.2f} → {r1[0]:+.2f} (± {1.96 * np.hypot(r0[1], r1[1]):.2f})")
    for title, rows in ((f"vs the previous iteration ({Path(rep['prev']).name if rep['prev'] else 'none'}, "
                         "its final evaluation)", rep["across"]), ("within the iteration (minute 0 → end)", rep["within"])):
        if not rows:
            out.append(f"{title}: -")
            continue
        drops = sorted([r for r in rows if r["diff"] < 0], key=lambda r: r["diff"])
        flagged = [r for r in drops if r["flag"] in ("forgetting", "drop?")]
        shown = flagged or drops[:top]
        n = {f: sum(r["flag"] == f for r in rows) for f in ("forgetting", "drop?", "gain")}
        out += ["", f"{title}: {len(rows)} numbers, forgetting {n['forgetting']}, drop? {n['drop?']}, gains {n['gain']}"
                + ("" if flagged else f"; the largest drops:"), "",
                "| number | before → now | change ± noise | flag |", "|---|---|---|---|"] + [_row(r) for r in shown[:max(top, len(flagged))]]
    t = rep["training"]
    sig = []
    for k, fmt in (("ev", "{:.3f}"), ("value_loss", "{:.4f}"), ("policy_loss", "{:+.4f}"), ("entropy", "{:.4f}"),
                   ("grad_norm", "{:.3f}"), ("kl", "{:.4f}"), ("clip", "{:.4f}")):
        c = t.get(k)
        if not c:
            continue
        s = f"{k} {fmt.format(c['early'])} → {fmt.format(c['late'])}" if c.get("early") is not None else k
        if "slope_100" in c:
            s += f" (late slope {c['slope_100']:+.4f} ± {2 * c['slope_se_100']:.4f} / 100 upd{', plateau' if c['plateau'] else ''})"
        sig.append(s)
    out += ["", f"training log ({t.get('updates', 0)} updates, early third → late third): "
            + ("; ".join(sig) if sig else "-") + ("" if "grad_norm" in t else "; grad norm not logged (older run)")]
    out += ["", f"verdict: **{rep['verdict']}** — " + "; ".join(rep["reasons"]), ""]
    return out


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="a test5 folder (build/nn-train/test5/<label>)")
    ap.add_argument("--prev", type=Path, help="the previous iteration's folder or report.json")
    ap.add_argument("--run", type=Path, help="the training's run folder (default: the report's)")
    args = ap.parse_args(argv)
    rep = report(args.folder, args.prev, args.run)
    print("\n".join(lines(rep)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
