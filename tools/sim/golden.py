"""Golden runs of the simulator: a short summary of the battle for every army,
kept in tests/golden/sim/ and compared by tests/golden/test_golden.py.

    python -m tools.sim.golden              # compare, print what changed
    python -m tools.sim.golden --update     # write new goldens (explain why in the commit)
    python -m tools.sim.golden <army> ...   # only these armies

A change in any module shows up here as a changed battle: the formation, the
places, every approach decision and its time, how long each manoeuvre walked
(tools/sim/walker.py), the window, the missile exchange. Numbers are
compared with a tolerance (TOLERANCE), so float noise between machines passes.
"""
import argparse
import json
import sys
from pathlib import Path

from tools import config as project
from tools.sim import formation as sim

GOLDEN = project.ROOT / "tests" / "golden" / "sim"
TOLERANCE = 0.05
# Armies that are not battles of our AI (research layouts) are left out.
SKIP = {"defender_layouts"}


def armies():
    return sorted(p.stem for p in (project.CONFIG_DIR / "armies").glob("*.json") if p.stem not in SKIP)


def _r(v, n=2):
    return round(v, n) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def summary(army_name, planner=None):
    """What the golden keeps of a simulated battle."""
    army, _ = sim.load_army(army_name)
    sides, _ = sim.simulate(army, planner)
    own = sides["own"]
    out = {"army": army_name, "status": own.get("status"), "strategy": own.get("strategy")}
    c = own.get("choice") or {}
    out["choice"] = {k: _r(c.get(k)) for k in ("wall_width", "archer_width", "rows", "wall_depth_m", "passage_m",
                                               "window_m") if c.get(k) is not None}
    out["fit"] = {k: _r(v) for k, v in (sides.get("start_fit") or {}).items() if k in ("how", "ok", "turn_deg")}
    shift = (sides.get("start_fit") or {}).get("shift")
    if shift:
        out["fit"]["shift"] = {k: _r(v) for k, v in shift.items()}
    out["anchor"] = [_r(v) for v in own.get("anchor") or []]
    out["bearing"] = _r(own.get("bearing"))
    out["placements"] = [{"id": p["id"], "role": p["role"], "x": _r(p["x"]), "z": _r(p["z"]),
                          "bearing": _r(p["bearing"]), "width": p.get("width")}
                         for p in sorted(own.get("placements") or [], key=lambda p: p["id"])]
    appr = sides.get("approach")
    if appr:
        rows = []
        for r in appr["log"]:
            w = r.get("window") or {}
            rows.append({"t_s": _r(r["t_s"]), "decision": r["decision"], "reason": r["reason"], "gap_m": _r(r["gap_m"]),
                         "stop_gap_m": _r(r["stop_gap_m"]), "advance_m": _r(r.get("advance_m")),
                         "window": w.get("reason"), "reached": w.get("reached"),
                         "path": {k: _r(v) for k, v in (r.get("path") or {}).items()
                                  if k in ("ok", "advance_m", "aside_m", "detour", "beyond_stop_line", "reason")}})
        out["approach"] = rows
        # How long each manoeuvre walked (tools/sim/walker.py: round obstacles, in time).
        out["walk_s"] = [_r(m["walk_s"]) for m in appr["moves"]]
    fire = sides.get("fire")
    if fire:
        out["fire"] = {"lost": fire["lost"],
                       "units": {i: u["men"] for i, u in sorted(fire["units"].items())}}
    mask = sides.get("mask")
    if mask:
        out["mask"] = {"fits": sum(1 for f in mask["fits"] if f["ok"]), "units": len(mask["fits"]),
                       "lane_free": mask["lane"]["free"]}
    return out


def differences(a, b, path=""):
    """Paths where two summaries differ beyond TOLERANCE."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: {'missing' if k not in b else 'new'}")
            else:
                out += differences(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} items -> {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += differences(x, y, f"{path}[{i}]")
        return out
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
        return [] if abs(a - b) <= TOLERANCE else [f"{path}: {a} -> {b}"]
    return [] if a == b else [f"{path}: {a!r} -> {b!r}"]


def path_of(army_name):
    return GOLDEN / f"{army_name}.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("armies", nargs="*")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    planner = sim.Planner()
    changed = 0
    for name in args.armies or armies():
        now = summary(name, planner)
        path = path_of(name)
        if args.update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(now, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
            print(f"{name}: written")
            continue
        if not path.exists():
            print(f"{name}: no golden (run with --update)")
            changed += 1
            continue
        diff = differences(json.loads(path.read_text(encoding="utf-8")), now)
        changed += bool(diff)
        print(f"{name}: {'same' if not diff else 'CHANGED'}")
        for d in diff[:20]:
            print("   ", d)
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
