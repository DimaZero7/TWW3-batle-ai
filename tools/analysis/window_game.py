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
    return 0


if __name__ == "__main__":
    sys.exit(main())
