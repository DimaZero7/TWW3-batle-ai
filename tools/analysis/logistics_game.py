"""Queues past an obstacle in battle: formation-probe runs with and without apps.logistics.

    python -m tools.analysis.logistics_game <run dir> [<run dir> ...] [--out DIR]

For the approach step that walks round an obstacle (the decision with a
detour) in every run: the army, whether logistics was on, how long the step
took, the longest stretch a unit was moving but made less than 2 m in 10 s
(stuck), the crowding peak (soldiers of two units closer than 1 m, measured in
battle) as soldiers and as a share of the army, the seconds with any crowding
and the orders given. Prints a Markdown table; with --out also writes
summary.json there (analysis data: kept local, not committed).
"""
import argparse
import json
import sys
from pathlib import Path

from tools.analysis.approach_game import stuck_ms


def detour_step(run):
    rows = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    manifest = run / "manifest.json"
    army = (json.loads(manifest.read_text(encoding="utf-8")).get("config") or {}).get("army") if manifest.exists() else None
    decisions = [r for r in rows if r["event"] == "approach_decision" and r["decision"] in ("approach", "align")]
    manoeuvres = [r for r in rows if r["event"] == "approach_manoeuvre"]
    samples = [r for r in rows if r["event"] == "approach_sample"]
    runs, current = [], []
    for s in samples:
        if current and s["t_ms"] < current[-1]["t_ms"]:
            runs.append(current)
            current = []
        current.append(s)
    if current:
        runs.append(current)
    for d, m, part in zip(decisions, manoeuvres, runs):
        if not (d.get("path") or {}).get("detour"):
            continue
        tracks = {}
        for s in part:
            for u in s["units"]:
                mo = u["motion"]
                tracks.setdefault(u["script_name"], []).append((s["t_ms"], mo["x"], mo["z"], mo.get("is_moving") is True))
        crowd = [(s["t_ms"], s["crowd"]["soldiers"]) for s in part if s.get("crowd")]
        seconds = 0.0
        for (t0, c0), (t1, _) in zip(crowd, crowd[1:]):
            if c0 > 0:
                seconds += (t1 - t0) / 1000
        army_men = sum(len(u.get("soldiers_dm") or []) // 2 for u in m["units"])
        peak = max((c for _, c in crowd), default=None)
        return {"run": run.name, "army": army, "logistics": m.get("logistics") is True,
                "advance_m": round(d["path"]["advance_m"]), "took_s": round(m["t_ms"] / 1000),
                "stuck_s": max((stuck_ms(t) for t in tracks.values()), default=0) / 1000,
                "crowd_peak": peak, "army_men": army_men,
                "crowd_peak_share": round(peak / army_men, 3) if peak is not None and army_men else None,
                "crowd_s": round(seconds), "orders": m.get("logistics_orders")}
    return {"run": run.name, "army": army, "detour": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [detour_step(r) for r in args.runs]
    print("| Прогон | Армия | Логистика | Шаг, м | Время, с | Застревание, с | Давка на пике | Доля армии | С давкой, с |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("detour") is False:
            print(f'| {r["run"]} | {r["army"]} | — | нет обхода | | | | | |')
            continue
        share = f'{r["crowd_peak_share"]:.0%}' if r["crowd_peak_share"] is not None else "—"
        print(f'| {r["run"]} | {r["army"]} | {"да" if r["logistics"] else "нет"} | {r["advance_m"]} | {r["took_s"]} | '
              f'{r["stuck_s"]:.0f} | {r["crowd_peak"]} | {share} | {r["crowd_s"]} |')
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
