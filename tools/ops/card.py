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
distance from the start; where the exam's proxy was played (profile exam, tools/nn/train/exam_sim.py) its trade
(± 95 %), wins per 12, pairs 2-0/1-1/0-2 and own lord deaths in its battles; the training window since the last point
(test5 "window", run.py log): the win rate against the pinned version (--past-pin) and own lord deaths in the
training's battles; the friends in the way in the proxy's battles (behaviour.py: the share of the bullets' / the
arrows' fire taken by own units, the share of arrows forced to a 45 deg lob). The general's groups (a line under the
table, from the run's log build/nn-train/runs/test5_<label>/log.jsonl "general", the mean of its updates): our groups a
plan, the composition changes a battle (re-plans whose membership of our side differed from the last plan's) and their
share of the re-plans, the share of our groups with shooters and melee together, and with groups "learned" the ARI
to the rule, K and the share of plans that kept the last groups (GeneralConfig.groups_sticky). A step of test5 --no-script-metrics has no rating, pair gold or script-battle rows ("-"); the card
says so under the table. The last column is the change from the first point to the last: a mark
when it is beyond the noise (two standard errors where the numbers carry one; |change| >= 0.05
otherwise, marked *); anomalies (a collapse between points, the move share x1.5, lord deaths x1.5, timeouts)
are listed under the table; --prev adds the previous step's end point and the change from it. The
line "computed:" says which metric profiles each point computed (test5 --profile,
tools/nn/train/profiles.py; "full" for evaluations older than the profiles): a row shown "-" was not
computed there, not zero.
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
NO_MARK = ("seconds", "kl", "kinds", "exam_wins")         # rows whose change is shown but never marked
Z = 2.0
WIN_SHARE = 0.05          # run.py --opponents: the window's "win vs <name>" rows for the opponents of this share or more
FRIENDS = ("friend_skip_share_direct", "friend_skip_share_arc", "forced_lob_share_arc")   # behaviour.Tracker.friends


# --- points -----------------------------------------------------------------------------------------

def point_from_eval(res):
    """The report's point of a raw evaluation (the subset of test5.metrics the card reads)."""
    out = {"skill": skill.summary(res)}
    for k in ("drills", "transfer", "distance", "window", "teach_auto", "teach_normal", "profile", "no_script_metrics"):
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
                                    "missile_melee_share": b.get("missile_melee_share"),
                                    "fatigue_exhausted_share": b.get("fatigue_exhausted_share"),
                                    "run_far_share": b.get("run_far_share")}
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
    e = s.get("exam")
    if e and e.get("battles"):                   # the exam's proxy (exam_sim.py): only at the points that played it
        enemy = Path(e.get("enemy") or "itS27_4").stem
        out.append(("exam_trade", f"exam vs {enemy}: trade (sim, we defend, ± 95%)", _g(e, "trade", "value"),
                    _g(e, "trade", "se"), "{:+.3f}"))
        a, b, c = e["pairs"]
        out.append(("exam_wins", f"exam vs {enemy}: wins /12 (pairs 2-0/1-1/0-2)", e.get("wins12"), None,
                    "{:.1f}" + f" ({a}/{b}/{c})"))
    w = point.get("window") or {}
    pin = w.get("pinned/all") or {}
    if pin.get("games"):                         # the training window against the pinned version (test5 window, --past-pin)
        roles = "".join(f" {r[:3]} {w[f'pinned/{r}']['win_rate']:.2f}" for r in ("attack", "defend")
                        if (w.get(f"pinned/{r}") or {}).get("games"))
        out.append(("pinned_win", "window: win vs pinned (training, since the last point)", pin.get("win_rate"),
                    _binom(pin.get("win_rate"), pin.get("games")), "{:.3f}" + f" ({roles.strip()}, n {pin['games']})"))
    for name, c in (w.get("opponents") or {}).items():     # run.py --opponents: every opponent of share >= WIN_SHARE
        a = c.get("all") or {}
        if (c.get("share") or 0) < WIN_SHARE or not a.get("games"):
            continue
        roles = "".join(f" {r[:3]} {c[r]['win_rate']:.2f}" for r in ("attack", "defend") if (c.get(r) or {}).get("games"))
        out.append((f"win_{name}", f"window: win vs {name} (training, since the last point)", a.get("win_rate"),
                    _binom(a.get("win_rate"), a.get("games")), "{:.3f}" + f" ({roles.strip()}, n {a['games']})"))
    if w.get("lord_dead_own") is not None:
        out.append(("lord_train", "own lord dead, training window (all its battles)", w["lord_dead_own"],
                    _binom(w["lord_dead_own"], w.get("games")), "{:.3f}"))
    fb = (e or {}).get("behaviour") or {}
    if any(fb.get(k) is not None for k in FRIENDS):          # the friends in the way in the proxy's battles (behaviour.py)
        d, a, lob = (fb.get(k) for k in FRIENDS)
        out.append(("friend_skip", "friend fire skip: bullets / arrows, forced lob (exam's proxy)", d, None,
                    "{:.3f}" + "".join(" / " + ("-" if v is None else f"{v:.3f}") for v in (a, lob))))
    if e and e.get("lord_dead_own") is not None:
        out.append(("lord_exam", "own lord dead, exam's proxy", e["lord_dead_own"], _binom(e["lord_dead_own"], e.get("battles")),
                    "{:.3f}"))
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
    out.append(("fatigue", "own unit-time exhausted (mean of cells)",
                _mean([c.get("fatigue_exhausted_share") for c in cells]), None, "{:.3f}"))
    out.append(("run_far", "own moves running far from the fight (mean of cells)",
                _mean([c.get("run_far_share") for c in cells]), None, "{:.3f}"))
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
    for key, title, x in (("move", "move share", MOVE_X), ("lord", "own lord dead", LORD_X),
                          ("lord_train", "own lord dead (training window)", LORD_X),
                          ("lord_exam", "own lord dead (exam's proxy)", LORD_X)):
        # the first point that has the row (the training window and the exam's proxy are not at minute 0)
        a = next((r[key][2] for _, r in table if key in r and r[key][2] is not None), None)
        b = last[key][2] if key in last else None
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


def lord_line(tr):
    """["own lord dead by opponent (training): <name> att x (n) def y (n); ..."] of report.json training "lord_dead"
    (run.py: lord_dead/<opponent>/<role>); [] without it."""
    ld = tr.get("lord_dead") or {}
    if not ld:
        return []
    by = {}
    for k, c in ld.items():
        name, _, role = k.rpartition("/")
        by.setdefault(name, {})[role] = c
    return ["own lord dead by opponent (training, share (battles)): " + "; ".join(
        f"{n} " + " ".join(f"{r[:3]} {c[r]['share']:.2f} ({c[r]['games']})" for r in ("attack", "defend") if r in c)
        for n, c in by.items())]


def opponents_table(rep):
    """Lines of the training's battles by opponent (run.py --opponents: report.json "training" "opponents", the
    learner's wins by role) and each big checkpoint's matchups ("opponent_matchups"); [] without them."""
    tr = (rep or {}).get("training") or {}
    opp = tr.get("opponents")
    if not opp:
        return []
    shares = tr.get("opponent_shares") or {}
    rate = (lambda c: f"{c['win_rate']:.2f} ({c['games']})" if c.get("games") else "-")
    out = ["", "opponents in training (run.py --opponents; the learner's win rate, battles):", "",
           "| opponent | share | battles | win attack | win defend | win all |", "|---|---|---|---|---|---|"]
    for name, c in opp.items():
        sh = shares.get(name)
        out.append(f"| {name} | {sh:.2f} | {c['all']['games']} | {rate(c['attack'])} | {rate(c['defend'])} | "
                   f"{rate(c['all'])} |" if sh is not None else
                   f"| {name} | - | {c['all']['games']} | {rate(c['attack'])} | {rate(c['defend'])} | {rate(c['all'])} |")
    out += lord_line(tr)
    for name, cells in (tr.get("opponent_matchups") or {}).items():
        out.append(f"{name} by matchup (the learner's role/own-enemy): " + ", ".join(
            f"{k} {v['win_rate']:.2f} ({v['games']})" for k, v in cells.items()))
    return out


def computed(pts, rep=None):
    """The line naming the metric profiles of the points: one name when all agree, else per point."""
    names = [p.get("profile") or (rep or {}).get("profile") or "full" for _, p in pts]
    if len(set(names)) == 1:
        return f"computed: {names[0]} (tools/nn/train/profiles.py)"
    return "computed: " + "; ".join(f"min {m} {n}" for (m, _), n in zip(pts, names))


GROUP_KEYS = (("groups_own", "groups a plan", "{:.2f}"), ("group_changes_per_battle", "composition changes a battle", "{:.2f}"),
              ("group_change_share", "of re-plans", "{:.3f}"), ("mixed_share", "shoot+melee groups", "{:.3f}"),
              ("assign_ari", "ARI to rule", "{:.3f}"), ("assign_k", "K", "{:.2f}"), ("assign_kept", "plans kept", "{:.3f}"))


def general_groups(folder):
    """({key: mean over the run's updates} of GROUP_KEYS, updates) from the run's log (module doc); ({}, 0) without."""
    folder = Path(folder)
    log = folder.parent.parent / "runs" / f"test5_{folder.name}" / "log.jsonl"
    if not log.is_file():
        return {}, 0
    vals, n = {}, 0
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            g = json.loads(line).get("general")
        except ValueError:
            continue
        if not isinstance(g, dict):
            continue
        n += 1
        for k, _, _ in GROUP_KEYS:
            if isinstance(g.get(k), (int, float)):
                vals.setdefault(k, []).append(float(g[k]))
    return {k: sum(v) / len(v) for k, v in vals.items()}, n


def groups_line(folder):
    """["general groups (training log, mean of N updates): ..."] or [] (module doc)."""
    g, n = general_groups(folder)
    if not g:
        return []
    return [f"general groups (training log, mean of {n} updates): "
            + ", ".join(f"{title} {f.format(g[k])}" for k, title, f in GROUP_KEYS if k in g)]


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
    # every row of any point, in the order of the point with the most (a row only some points have - the exam's proxy -
    # shows "-" elsewhere)
    order = list(dict.fromkeys([r[0] for r in max(table, key=len)] + [r[0] for t in table for r in t]))
    first = {r[0]: r for t in reversed(table) for r in t}
    for i, (key, title, _, _, f) in enumerate(first[k] for k in order):
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
    lines += ["", computed(pts, rep)]
    if (rep or {}).get("no_script_metrics") or any(p.get("no_script_metrics") for _, p in pts):
        lines.append("no script metrics (test5 --no-script-metrics): the rating, the pair gold and the script battles' "
                     "rows are not computed (\"-\"); judge by the exam's proxy, the pinned window and own lord deaths")
    lines += ["anomalies: " + ("; ".join(anomaly) if anomaly else "none")]
    lines += groups_line(folder)
    capacity = (rep or {}).get("capacity")
    if capacity:
        lines.append(f"capacity: {capacity.get('verdict')} - {'; '.join(capacity.get('reasons') or [])}")
    if rep and rep.get("train_s"):
        lines.append(f"training: {rep['train_s'] / 60:.0f} min, {rep.get('updates')} updates")
    lines += opponents_table(rep)
    if not ((rep or {}).get("training") or {}).get("opponents"):          # (with --opponents: inside the table's lines)
        lines += lord_line((rep or {}).get("training") or {})
    return lines, {"points": [m for m, _ in pts], "rows": data, "anomalies": anomaly, "capacity": capacity,
                   "opponents": ((rep or {}).get("training") or {}).get("opponents"),
                   "computed": computed(pts, rep), "general_groups": general_groups(folder)[0],
                   "no_script_metrics": bool((rep or {}).get("no_script_metrics")
                                             or any(p.get("no_script_metrics") for _, p in pts))}


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
