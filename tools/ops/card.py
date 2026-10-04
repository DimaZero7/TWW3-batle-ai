"""The run card of a test5 folder: the ~15 numbers a step is judged by, per evaluation point, with
the change over the run and anomalies (docs/en/training/workflow.md).

    python -m tools.ops.card build/nn-train/test5/n3_consolidate
    python -m tools.ops.card build/nn-train/test5/n4 --prev build/nn-train/test5/n3_consolidate
    python -m tools.ops.card build/nn-train/test5/n4 --json

Reads report.json (a finished run) or, while a run goes, before.json and the eval_m<minute>.json
written so far (the fair metrics then computed here with tools/nn/train/skill.py, numpy). Rows:
rating (logit, higher is better), pair gold per opponent (ours - the opponent's with the same
armies, / budget; positive: we trade better), the exchange vs ai_like, each drill's win rate (the
skilled script's beside), each drill's TRANSFER to normal battles (APPLIED share, higher is better,
and MISTAKE share, lower is better, network / ai_like), the teachers' shares, own lord dead, the
order kinds hold / move / attack, missile time in melee, timeouts, the evaluation's seconds, the KL
distance from the start. The last column is the change from the first point to the last: a mark
when it is beyond the noise (two standard errors where the numbers carry one; |change| >= 0.05
otherwise, marked *); anomalies (a collapse between points, the move share x1.5, lord deaths x1.5, timeouts)
are listed under the table; --prev adds the previous step's end point and the change from it.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

from tools.nn.train import skill

OPPONENTS = ("ai_like", "nearest", "hold_shoot")
MOVE_X, LORD_X, TIMEOUTS, PLAIN = 1.5, 1.5, 0.05, 0.05    # anomaly thresholds; the plain change threshold
NO_MARK = ("seconds", "kl", "kinds")                      # rows whose change is shown but never marked
Z = 2.0


# --- points -----------------------------------------------------------------------------------------

def point_from_eval(res):
    """The report's point of a raw evaluation (the subset of test5.metrics the card reads)."""
    out = {"skill": skill.summary(res)}
    for k in ("drills", "transfer", "distance", "teach_auto", "teach_normal"):
        if res.get(k):
            out[k] = res[k]
    out["seconds"] = res.get("seconds")
    for opp, o in res.get("by_opponent", {}).items():
        for role, x in o.get("roles", {}).items():
            if not x.get("games"):
                continue
            b = x.get("behaviour", {})
            out[f"{opp}/{role}"] = {"win": x.get("win_rate"), "lord_dead_own": x.get("lord_dead_own"), "games": x.get("games"),
                                    "timeouts": x.get("timeouts"), "gold_ratio": x.get("gold_ratio"),
                                    "missile_melee_share": b.get("missile_melee_share")}
        out[f"{opp}/all"] = {f"kind_{k}": v for k, v in o.get("kinds", {}).items()}
    return out


def points_of(folder):
    """[(minute label, point)] of a test5 folder: report.json's trend, else the evaluations on disk."""
    folder = Path(folder)
    rep = folder / "report.json"
    if rep.is_file():
        r = json.loads(rep.read_text(encoding="utf-8"))
        if r.get("trend"):
            pts = sorted(r["trend"].items(), key=lambda kv: float(kv[0]))
            for m, p in pts:                                 # the seconds live in the evaluations
                f = folder / ("before.json" if float(m) == 0 else f"eval_m{m}.json")
                p.setdefault("seconds", _seconds(f) or (_seconds(folder / "after.json") if m == pts[-1][0] else None))
            return pts, r
        return [("before", r["before"]), ("after", r["after"])], r
    pts = []
    for f in sorted(folder.glob("eval_m*.json"), key=lambda p: float(p.stem[6:])):
        pts.append((f.stem[6:], point_from_eval(json.loads(f.read_text(encoding="utf-8")))))
    before = folder / "before.json"
    if before.is_file():
        pts.insert(0, ("0", point_from_eval(json.loads(before.read_text(encoding="utf-8")))))
    if not pts:
        raise SystemExit(f"no report.json, before.json or eval_m*.json in {folder}")
    return pts, None


def _seconds(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("seconds")
    except (OSError, ValueError):
        return None


# --- rows -------------------------------------------------------------------------------------------

def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _mean(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def _binom(p, n):
    return math.sqrt(p * (1 - p) / n) if p is not None and n else None


def rows(point):
    """[(key, title, value, se, fmt)] of one point; se None when the number carries no noise estimate."""
    out = []
    s = point.get("skill") or {}
    r = _g(s, "rating", "overall")
    out.append(("rating", "rating, overall (logit, ± 95%)", _g(r, "value"), _g(r, "se"), "{:+.2f}"))
    for opp in OPPONENTS:
        g = _g(s, "pair_gold", opp, "pair_gold")
        out.append((f"gold_{opp}", f"pair gold vs {opp} (/ budget, ± 95%)", _g(g, "value"), _g(g, "se"), "{:+.3f}"))
    out.append(("exchange", "pair exchange vs ai_like", _g(s, "pair_gold", "ai_like", "exchange", "value"), None, "x{:.2f}"))
    for d, v in (point.get("drills") or {}).items():
        sk = _g(v, "scripts", "skilled", "win_rate")
        out.append((f"drill_{d}", f"drill {d}: win" + (f" (skilled {sk:.2f})" if sk is not None else ""),
                    v.get("win_rate"), _binom(v.get("win_rate"), v.get("games")), "{:.2f}"))
    for d, v in (point.get("transfer") or {}).items():
        net, ai = v.get("network") or {}, v.get("ai_like") or {}
        out.append((f"applied_{d}", f"transfer {d}: APPLIED net / ai_like (higher better)", net.get("share"), None,
                    "{:.2f}" + (f" / {ai['share']:.2f}" if ai.get("share") is not None else "")))
        out.append((f"mistake_{d}", f"transfer {d}: MISTAKE net / ai_like (lower better)", net.get("mistake"), None,
                    "{:.2f}" + (f" / {ai['mistake']:.2f}" if ai.get("mistake") is not None else "")))
    for key, title in (("teach_auto", "teacher share"), ("teach_normal", "teacher share")):
        for t in point.get(key) or []:
            out.append((f"{key}_{t['drill']}", f"{title} {t['drill']} (deficit {t.get('deficit', 0):.2f})", t.get("share"), None, "{:.2f}"))
    cells = [v for k, v in point.items() if k.count("/") == 1 and k.split("/")[1] in ("attack", "defend") and isinstance(v, dict)]
    lord = _mean([c.get("lord_dead_own") for c in cells])
    games = sum(c.get("games") or 0 for c in cells) or None
    out.append(("lord", "own lord dead (all cells)", lord, _binom(lord, games) if games else None, "{:.3f}"))
    a = point.get("ai_like/all") or {}
    out.append(("kinds", "kinds hold / move / attack, ai_like", a.get("kind_hold"), None,
                "{:.2f}" + "".join(f" / {a[k]:.2f}" for k in ("kind_move", "kind_attack") if a.get(k) is not None)))
    out.append(("move", "kind move, ai_like", a.get("kind_move"), None, "{:.2f}"))
    out.append(("missile", "missile time in melee (mean of cells)", _mean([c.get("missile_melee_share") for c in cells]), None, "{:.3f}"))
    out.append(("timeouts", "timeouts (max cell)", max([c.get("timeouts") or 0 for c in cells], default=None), None, "{:.3f}"))
    out.append(("seconds", "evaluation, s", point.get("seconds"), None, "{:.0f}"))
    out.append(("kl", "KL from the start (mean of the updates)", _g(point, "distance", "start_kl"), None, "{:.3f}"))
    return out


def fmt(value, se, f):
    if value is None:
        return "-"
    text = f.format(value)
    return text + (f" ± {Z * 0.98 * se:.{_digits(f)}f}" if se else "")


def _digits(f):
    m = re.search(r"\.(\d)f", f)
    return int(m.group(1)) if m else 2


def change(a, b, se_a, se_b):
    """(diff, beyond): the change b - a and whether it is beyond the noise (Z standard errors when
    both carry one, |diff| >= PLAIN otherwise)."""
    if a is None or b is None:
        return None, False
    d = b - a
    if se_a is not None and se_b is not None:
        return d, abs(d) > Z * math.hypot(se_a, se_b)
    return d, abs(d) >= PLAIN


def anomalies(pts):
    """Lines about what went wrong along the points: a rating fall beyond the noise between two points,
    the move share or lord deaths x1.5 of the first point, timeouts, a drill's win falling."""
    out = []
    table = [(m, {r[0]: r for r in rows(p)}) for m, p in pts]
    for (m0, r0), (m1, r1) in zip(table, table[1:]):
        d, beyond = change(r0["rating"][2], r1["rating"][2], r0["rating"][3], r1["rating"][3])
        if d is not None and d < 0 and beyond:
            out.append(f"rating fell beyond noise min {m0} -> {m1}: {r0['rating'][2]:+.2f} -> {r1['rating'][2]:+.2f}")
    first, last = table[0][1], table[-1][1]
    for key, title, x in (("move", "move share", MOVE_X), ("lord", "own lord dead", LORD_X)):
        a, b = first[key][2], last[key][2]
        if a is not None and b is not None and a > 0 and b >= x * a and b - a >= 0.05:
            out.append(f"{title} x{b / a:.1f}: {a:.2f} -> {b:.2f}")
    for m, r in table:
        t = r["timeouts"][2]
        if t is not None and t > TIMEOUTS:
            out.append(f"timeouts {t:.2f} at min {m}")
    for key in first:
        if key.startswith("drill_") and key in last:
            a, b = first[key][2], last[key][2]
            if a is not None and b is not None and a - b >= 0.15:
                out.append(f"{key[6:]} drill win fell {a:.2f} -> {b:.2f}")
    return out


def card(folder, prev=None):
    """The card's lines (markdown) and its data ({"points", "rows", "anomalies", "capacity"})."""
    pts, rep = points_of(folder)
    table = [rows(p) for _, p in pts]
    heads = [f"min {m}" for m, _ in pts]
    prev_rows = None
    if prev:
        ppts, _ = points_of(prev)
        prev_rows = {r[0]: r for r in rows(ppts[-1][1])}
        heads.append(f"prev end ({Path(prev).name} min {ppts[-1][0]})")
    lines = [f"run card: {Path(folder).name}" + (f" (init {rep['init']})" if rep and rep.get("init") else ""), "",
             "| number | " + " | ".join(heads) + " | change first -> last" + (" | vs prev end" if prev else "") + " |",
             "|---" * (len(heads) + 2 + bool(prev)) + "|"]
    data = []
    for i, (key, title, _, _, f) in enumerate(table[0]):
        cells = [next((r for r in t if r[0] == key), None) for t in table]
        vals = [fmt(c[2], c[3], c[4]) if c else "-" for c in cells]
        a, b = cells[0], cells[-1]
        d, beyond = change(a[2] if a else None, b[2] if b else None, a[3] if a else None, b[3] if b else None)
        beyond = beyond and key not in NO_MARK
        mark = "*" if (d is not None and beyond) else ""
        row = {"key": key, "title": title, "values": [c[2] if c else None for c in cells], "change": d, "beyond": beyond}
        extra = ""
        if prev_rows:
            p = prev_rows.get(key)
            vals.append(fmt(p[2], p[3], p[4]) if p else "-")
            dp, bp = change(p[2] if p else None, b[2] if b else None, p[3] if p else None, b[3] if b else None)
            bp = bp and key not in NO_MARK
            extra = " | " + (f"{dp:+.3f}{' *' if bp else ''}" if dp is not None else "-")
            row["vs_prev"] = dp
        lines.append(f"| {title} | " + " | ".join(vals) + f" | {(f'{d:+.3f}' + (' ' + mark if mark else '') if d is not None else '-')}{extra} |")
        data.append(row)
    anomaly = anomalies(pts)
    lines += ["", "anomalies: " + ("; ".join(anomaly) if anomaly else "none")]
    capacity = (rep or {}).get("capacity")
    if capacity:
        lines.append(f"capacity: {capacity.get('verdict')} - {'; '.join(capacity.get('reasons') or [])}")
    if rep and rep.get("train_s"):
        lines.append(f"training: {rep['train_s'] / 60:.0f} min, {rep.get('updates')} updates")
    return lines, {"points": [m for m, _ in pts], "rows": data, "anomalies": anomaly, "capacity": capacity}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="build/nn-train/test5/<label>")
    ap.add_argument("--prev", help="the previous step's folder: its end point beside, and the change from it")
    ap.add_argument("--json", action="store_true", help="the card's data as JSON instead of the table")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # the ± on a cp1251 console
    lines, data = card(args.folder, args.prev)
    if args.json:
        print(json.dumps(data, indent=1, ensure_ascii=False))
    else:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
