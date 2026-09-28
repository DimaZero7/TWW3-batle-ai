"""The window in battle (battle theory, phase 2): formation-probe --enemy-ai native --fire.

    python -m tools.analysis.window_game <run dir> [--out DIR]

From events.jsonl: every approach decision with the window (apps.reach), then
the hold: do our archers shoot (arrows spent), is any of our units under
missile attack, men of both sides over time, and how far the enemy walked.
Prints a short report; with --out writes summary.json there.
"""
import argparse
import json
import math
import sys
from pathlib import Path


def _front_gap(sample):
    """Our front to theirs along X (the armies face each other along X in window_game)."""
    own = [u["motion"] for u in sample.get("units") or [] if (u["motion"].get("number_of_men_alive") or 0) > 1]
    enemy = [e["motion"] for e in sample.get("enemy") or [] if (e["motion"].get("number_of_men_alive") or 0) > 1]
    if not own or not enemy:
        return None
    return min(m["x"] for m in enemy) - max(m["x"] for m in own)


def reaction(rows):
    """How the game's AI answered our approach (research only): when it first
    moved (our front to theirs), how far everyone had gone 15 s later, which
    units charged (went more than 50 m towards us) and when, and per unit of
    theirs the men over the hold."""
    samples = [r for r in rows if r["event"] in ("approach_sample", "hold_sample") and r.get("enemy")]
    if not samples:
        return None
    start = {e["script_name"]: e["motion"] for e in samples[0]["enemy"]}
    first, out, charged = None, {}, {}
    for s in samples:
        moved = {e["script_name"]: start[e["script_name"]]["x"] - e["motion"]["x"] for e in s["enemy"]
                 if e["script_name"] in start}
        if first is None and any(abs(v) > 3 for v in moved.values()):
            first = s
            out["first_move_s"] = s["model_ms"] / 1000
            out["first_move_gap_m"] = _front_gap(s) and round(_front_gap(s), 1)
        if first is not None and "step_m" not in out and s["model_ms"] - first["model_ms"] >= 15000:
            out["step_m"] = {k: round(v, 1) for k, v in sorted(moved.items())}
        for k, v in moved.items():
            if v > 50 and k not in charged:
                charged[k] = round((s["model_ms"] - (first or s)["model_ms"]) / 1000, 1)
    out["charged_after_first_move_s"] = charged
    holds = [r for r in rows if r["event"] == "hold_sample" and r.get("enemy")]
    if holds:
        men = lambda s: {e["script_name"]: e["motion"].get("number_of_men_alive") for e in s["enemy"]}
        a, b = men(holds[0]), men(holds[-1])
        out["enemy_lost_by_unit"] = {k: a[k] - (b.get(k) or 0) for k in a if a[k] != b.get(k)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [json.loads(line) for line in (args.run / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    errors = [r.get("message", "")[:200] for r in rows if r["event"] in ("error", "probe_error")]
    decisions = [r for r in rows if r["event"] == "approach_decision"]
    holds = [r for r in rows if r["event"] == "hold_sample"]
    result = next((r for r in rows if r["event"] == "result"), {})
    summary = {"status": result.get("status"), "errors": errors, "decisions": [], "hold": None}
    for d in decisions:
        w = d.get("window") or {}
        span = w.get("window") or {}
        gap = d.get("gap_m")
        summary["decisions"].append({
            "decision": d["decision"], "reason": d.get("reason"), "gap_m": gap and round(gap, 1),
            "window": w.get("reason"), "reached": w.get("reached"), "shooters": w.get("shooters"),
            "ours_from_gap_m": round(gap - span["from"], 1) if gap and span.get("from") is not None else None,
            "theirs_from_gap_m": round(gap - w["safe_to"], 1) if gap and w.get("safe_to") is not None else None,
            "under_fire_now": w.get("under_fire_now")})
    if holds:
        first, last = holds[0], holds[-1]

        def men(sample, key):
            return sum((u["motion"].get("number_of_men_alive") or 0) for u in sample.get(key) or [])
        ammo0 = {u["script_name"]: u.get("ammo") for u in first["units"] if u.get("ammo")}
        ammo1 = {u["script_name"]: u.get("ammo") for u in last["units"] if u.get("ammo") is not None}
        firing = {n: ammo0[n] - (ammo1.get(n) or 0) for n in ammo0}
        under = sorted({u["script_name"] for s in holds for u in s["units"] if u.get("under_fire")})
        first_under = next((s["t_ms"] / 1000 for s in holds if any(u.get("under_fire") for u in s["units"])), None)
        moved = {}
        e0 = {e["script_name"]: e["motion"] for e in first.get("enemy") or []}
        for e in last.get("enemy") or []:
            a = e0.get(e["script_name"])
            if a:
                moved[e["script_name"]] = round(math.hypot(e["motion"]["x"] - a["x"], e["motion"]["z"] - a["z"]), 1)
        summary["hold"] = {
            "seconds": round((last["t_ms"] - first["t_ms"]) / 1000, 1),
            "own_men": [men(first, "units"), men(last, "units")],
            "enemy_men": [men(first, "enemy"), men(last, "enemy")],
            "arrows_spent": firing, "units_under_fire": under, "first_under_fire_s": first_under,
            "enemy_moved_m": moved}
    summary["reaction"] = reaction(rows)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(f"status: {summary['status']}; errors: {errors[:2]}")
    for d in summary["decisions"]:
        print(f"  {d['decision']:9} {d['reason']:16} gap {d['gap_m']} m; {d['window']}: ours from {d['ours_from_gap_m']}, "
              f"theirs from {d['theirs_from_gap_m']}; reach {d['reached']}/{d['shooters']}; under fire {d['under_fire_now']}")
    h = summary["hold"]
    if h:
        print(f"hold {h['seconds']} s: our men {h['own_men'][0]} -> {h['own_men'][1]}, "
              f"theirs {h['enemy_men'][0]} -> {h['enemy_men'][1]}")
        print(f"  arrows spent: {h['arrows_spent']}")
        print(f"  under fire: {h['units_under_fire']} (first at {h['first_under_fire_s']} s)")
        print(f"  enemy moved (m): {h['enemy_moved_m']}")
    r = summary.get("reaction")
    if r:
        print(f"reaction: first move at {r.get('first_move_s')} s, fronts {r.get('first_move_gap_m')} m apart; "
              f"15 s later moved (m): {r.get('step_m')}")
        print(f"  charged (s after the first move): {r['charged_after_first_move_s']}; "
              f"their losses by unit in the hold: {r.get('enemy_lost_by_unit')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
