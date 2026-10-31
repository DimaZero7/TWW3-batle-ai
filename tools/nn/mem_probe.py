"""The projectiles from the game's memory in a missile probe run (memreader Plus; docs/ru/game/memreader.md) against
what the game's script shows of the same battle: each lane's ammo (the projectiles it fired) and its units' places.

    python -m tools.nn.mem_probe [run folders...]      # default: the newest build/missile-probe/runs/* with mem rows

A run built with `tools.build missile-probe --mem` has, besides the probe's rows (tools/nn/missile_probe.py):
  mem_attach  {status}: memreader's version, each code site's check, hooked or why not
  mem_proj    every probe tick (model_ms): calls (the hook's call count: one a simulation tick), spawn / state / gone
              (apps.mem.projectiles: {id, c, ...} with the call number c), fly (the projectiles flying now), hook
              (memreader's counters: calls, other_thread_calls)

Measures (load + compare):
  hook       did it fire on the script thread (calls > 0, other_thread_calls), its errors
  axes       the spawn points against the shooters' script places: the median distance (m) to the nearest shooter
             (memory x, z = the script's x, z if small)
  per lane   projectiles from memory attributed to the lane (the nearest shooter within ATTRIB_M at the spawn) against
             the script's ammo drop; matched = the sum over 0.5 s bins of min(memory, script) with one bin of slack
             (the hook counts at the simulation tick, the script reads ammo every tick_ms); share = matched / max(both)
  ends       the share of spawned projectiles that ended (state 5 / 6 lying, 7 done, or gone) and the flight time at the
             end; where they came down against the target's centre
Writes build/mem/projectiles.json (not in Git).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from tools import config as project

RUNS = project.BUILD / "missile-probe" / "runs"
OUT = project.BUILD / "mem" / "projectiles.json"
ATTRIB_M = 40.0
BIN_S = 0.5
LYING, DONE = (5, 6), 7
# The objects in the vector by their first state (the game's run, dist 2): a launched projectile appears in
# state 0 with flight time 0 and goes to 7 when it comes down; where it came down the game puts a NEW object lying on
# the ground (first seen in 6, ~20 s); one that hit a man is a new object too (first seen in 2, then 3, 7: the arrow in
# the body, its shooter kept)
LAUNCH, BODY = 0, 2


def kind(s):
    return {LAUNCH: "launch", BODY: "body", 5: "lying", 6: "lying"}.get(s["st"], "other")
# the spawn / state / fly rows (apps.mem.projectiles)
SPAWN = ("id", "c", "st", "x", "y", "z", "vx", "vy", "vz", "t", "rec", "sx", "sy", "sz")
STATE = ("id", "c", "st", "x", "y", "z", "t")
FLY = ("id", "st", "x", "y", "z", "vx", "vy", "vz", "t")


def rows(run_dir):
    for line in (Path(run_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def has_mem(run_dir):
    p = Path(run_dir) / "events.jsonl"
    return p.is_file() and '"mem_attach"' in p.read_text(encoding="utf-8")


def as_dicts(items, names):
    return [dict(zip(names, it)) for it in (items or [])]


def load(run_dir):
    """{attach, ticks [(model_ms, calls)], spawn, state, gone, fly [(model_ms, [..])], hook (the last), errors, records,
    lanes {name: {spec, t [model s], s_x, s_z, t_x, t_z, ammo}}} of one run."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    out = {"run": run_dir.name, "attach": None, "ticks": [], "spawn": [], "state": [], "gone": [], "fly": [],
           "hook": None, "errors": [], "records": [], "lanes": {}, "ammo_hook": [], "script_ammo": {}}
    lanes = {l["name"]: {"spec": l, "t": [], "s_x": [], "s_z": [], "t_x": [], "t_z": [], "ammo": [], "t_hp": []}
             for l in cfg.get("lanes", [])}
    for r in rows(run_dir):
        ev = r.get("event")
        if ev == "mem_attach":
            out["attach"] = r.get("status")
        elif ev == "mem_proj":
            if r.get("calls") is not None:
                out["ticks"].append((float(r["model_ms"]), int(r["calls"])))
            out["spawn"] += as_dicts(r.get("spawn"), SPAWN)
            out["state"] += as_dicts(r.get("state"), STATE)
            out["gone"] += [dict(zip(("id", "c"), g)) for g in (r.get("gone") or [])]
            if r.get("fly"):
                out["fly"].append((float(r["model_ms"]), as_dicts(r["fly"], FLY)))
            out["errors"] += list(r.get("errors") or [])
            out["records"] += list(r.get("records") or [])
            if r.get("hook"):
                out["hook"] = r["hook"]
            if r.get("discover"):
                out["discover"] = r["discover"]
            if r.get("chain"):
                out.setdefault("chains", []).append(r["chain"])
            if r.get("ammo"):
                out["ammo_hook"].append((float(r["model_ms"]), [a[1] for a in r["ammo"] if a and a[0]]))
        elif ev == "probe_sample":
            out["script_ammo"][float(r["model_ms"])] = {(x.get("s") or {}).get("ammo") for x in r["lanes"]} | {
                (x.get("f") or {}).get("ammo") for x in r["lanes"]}
            for x in r["lanes"]:
                L = lanes.get(x["lane"])
                if L is None:
                    continue
                s, tg = x.get("s") or {}, x.get("tg") or {}
                L["t"].append(float(r["model_ms"]) / 1000)
                for k, src, key in (("s_x", s, "x"), ("s_z", s, "z"), ("t_x", tg, "x"), ("t_z", tg, "z"),
                                    ("ammo", s, "ammo"), ("t_hp", tg, "hp")):
                    v = src.get(key)
                    L[k].append(np.nan if v is None else float(v))
    out["lanes"] = {k: {kk: (np.asarray(vv, float) if kk != "spec" else vv) for kk, vv in v.items()}
                    for k, v in lanes.items() if v["t"]}
    return out


def call_time(ticks):
    """f(c) -> model s of the hook's call number c, linear between the probe ticks (calls counted at each)."""
    if len(ticks) < 2:
        return lambda c: np.nan
    ms = np.array([t[0] for t in ticks]) / 1000
    calls = np.array([t[1] for t in ticks], float)
    keep = np.concatenate([[True], np.diff(calls) > 0])
    ms, calls = ms[keep], calls[keep]
    return lambda c: float(np.interp(c, calls, ms))


def attribute(spawn, lanes, at):
    """Each spawn's lane (the nearest shooter centre at the spawn's time within ATTRIB_M; None) and that distance."""
    out = []
    for s in spawn:
        t = at(s["c"])
        best, dist = None, math.inf
        for name, L in lanes.items():
            k = int(np.clip(np.searchsorted(L["t"], t), 0, len(L["t"]) - 1))
            d = math.hypot(s["x"] - L["s_x"][k], s["z"] - L["s_z"][k])
            if d < dist:
                best, dist = name, d
        out.append((best if dist <= ATTRIB_M else None, dist, t))
    return out


def bins_match(a, b, slack=1):
    """matched count of two count series on the same bins (greedy, a bin of a may take from b's bins +- slack)."""
    a, b = np.asarray(a, float).copy(), np.asarray(b, float).copy()
    matched = 0.0
    for d in [0] + [x for k in range(1, slack + 1) for x in (-k, k)]:
        for i in range(len(a)):
            j = i + d
            if 0 <= j < len(b):
                m = min(a[i], b[j])
                matched += m
                a[i] -= m
                b[j] -= m
    return matched


def compare(run):
    """The run's numbers (module doc)."""
    at = call_time(run["ticks"])
    launches = [s for s in run["spawn"] if kind(s) == "launch"]
    att = attribute(launches, run["lanes"], at)
    hook = run["hook"] or {}
    out = {"run": run["run"], "attach_ok": bool((run["attach"] or {}).get("ok")),
           "attach": run["attach"], "hook": hook, "errors": run["errors"][:10], "n_errors": len(run["errors"]),
           "spawned": len(run["spawn"]), "records": run["records"],
           "kinds": dict(sorted(__import__("collections").Counter(kind(x) for x in run["spawn"]).items())),
           "discover": run.get("discover"), "chain_first": (run.get("chains") or [None])[0], "chain_last": (run.get("chains") or [None])[-1]}
    # the ammo_left hook: each value it read from memory against the script's ammo_left values of the same tick
    vals = [(ms, v) for ms, vs in run["ammo_hook"] for v in vs]
    if vals:
        same = sum(v in run["script_ammo"].get(ms, set()) for ms, v in vals)
        out["ammo_hook"] = {"reads": len(vals), "equal_to_script": same, "first": vals[:4]}
    if not launches:
        return out
    dists = np.array([a[1] for a in att])
    out["spawn_to_shooter_m"] = {"median": float(np.median(dists)), "p90": float(np.percentile(dists, 90)),
                                 "unattributed": int(sum(a[0] is None for a in att))}
    # ends
    last = {}
    for s in run["state"]:
        last[s["id"]] = s
    gone = {g["id"] for g in run["gone"]}
    ended = [s for s in launches if s["id"] in gone or (s["id"] in last and last[s["id"]]["st"] in LYING + (DONE,))]
    out["ended_share"] = len(ended) / len(launches)
    bodies = [s for s in run["spawn"] if kind(s) == "body"]
    out["states_seen"] = sorted({s["st"] for s in run["spawn"]} | {s["st"] for s in run["state"]})
    lanes = {}
    for name, L in run["lanes"].items():
        mine = [(s, a) for s, a in zip(launches, att) if a[0] == name]
        t = L["t"]
        ammo = L["ammo"]
        ok = np.isfinite(ammo)
        script = float(np.nanmax(ammo) - np.nanmin(ammo)) if ok.sum() > 1 else 0.0
        edges = np.concatenate([[t[0] - BIN_S], t])
        drop = np.concatenate([[0.0], np.clip(-np.diff(np.where(ok, ammo, np.nan)), 0, None)])
        drop = np.nan_to_num(drop)
        mem_t = np.array([a[2] for _, a in mine])
        mem_bins = np.histogram(mem_t, bins=edges)[0] if len(mem_t) else np.zeros(len(t))
        matched = bins_match(mem_bins, drop)
        row = {"memory": len(mine), "script": script, "matched": matched,
               "share": matched / max(1.0, max(script, float(len(mine))))}
        # where they came down: the last known point of each against the target's centre at that time
        ends = []
        for s, a in mine:
            e = last.get(s["id"])
            if e and e["st"] in LYING + (DONE,):
                k = int(np.clip(np.searchsorted(t, at(e["c"])), 0, len(t) - 1))
                ends.append((math.hypot(e["x"] - L["t_x"][k], e["z"] - L["t_z"][k]), e["t"]))
        if ends:
            d = np.array([x[0] for x in ends])
            ft = np.array([x[1] for x in ends])
            row.update(end_to_target_m_median=float(np.median(d)), within_15m=float(np.mean(d <= 15)),
                       flight_s_median=float(np.median(ft)))
        if mine:
            v = np.array([math.hypot(s["vx"], s["vz"]) for s, _ in mine])
            row["launch_speed_median"] = float(np.median(v))
        # arrows in bodies (new 'body' objects) within 20 m of this lane's target, against the target's HP lost
        nb = 0
        for b in bodies:
            k = int(np.clip(np.searchsorted(t, at(b["c"])), 0, len(t) - 1))
            nb += math.hypot(b["x"] - L["t_x"][k], b["z"] - L["t_z"][k]) <= 20
        hp = L["t_hp"][np.isfinite(L["t_hp"])]
        row["bodies_at_target"] = nb
        row["target_hp_lost"] = float(hp[0] - hp[-1]) if len(hp) > 1 else None
        lanes[name] = row
    out["lanes"] = lanes
    mem_total = sum(r["memory"] for r in lanes.values())
    script_total = sum(r["script"] for r in lanes.values())
    out["matched_share"] = sum(r["matched"] for r in lanes.values()) / max(1.0, max(mem_total, script_total))
    out["fly_ticks"] = len(run["fly"])
    out["fly_max"] = max((len(f) for _, f in run["fly"]), default=0)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    runs = args.runs or [p for p in sorted(RUNS.glob("*")) if has_mem(p)][-1:]
    res = [compare(load(r)) for r in runs]
    for r in res:
        print(f"{r['run']}: attach {r['attach_ok']}, hook {r['hook']}, ammo hook {r.get('ammo_hook')}, spawned {r['spawned']}, errors {r['n_errors']}")
        print("  chain first", r["chain_first"], "last", r["chain_last"], "discover", r["discover"])
        if not r["attach_ok"]:
            print("  attach:", json.dumps(r["attach"], ensure_ascii=False))
        if "lanes" in r:
            print(f"  spawn->shooter {r['spawn_to_shooter_m']}, ended {r['ended_share']:.3f}, states {r['states_seen']},"
                  f" matched {r['matched_share']:.3f}, fly ticks {r['fly_ticks']} max {r['fly_max']}")
            for name, row in r["lanes"].items():
                print("  ", name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
