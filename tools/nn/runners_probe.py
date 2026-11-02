"""The Death Runners' probe table (plan 'runners' of tools/nn/morale_probe.py, runners_lane; build/diag_runners): where a hidden unit
is seen with and without Concealment Bombs, whether the press reveals, the 'surprised' morale term (from the game's memory, a run
built with --mem), the weeping blade (the card's armour, its life, one armour for blows and missiles), the uses, and the melee's HP a
second both ways - in the game and in the simulator's twin on the same lanes.

    python -m tools.nn.morale_probe run --plan runners --mem --speed 20       # the battles
    python -m tools.nn.runners_probe [run folders...] [--sim] [--out f.json]  # the table (default: the newest three runners runs)

No torch without --sim.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools import config as project
from tools.nn import mem_probe, morale_probe as mp

OUT = project.BUILD / "diag_runners" / "table.json"
BOMBS = mp.BOMBS
BLADE = "wh2_main_unit_contact_weeping_blade"


def units():
    return json.loads(mp.UNITS_JSON.read_text(encoding="utf-8"))["units"]


def _hp_total(key):
    p = units()[key]
    return float(p["hp_total"])


def key_of(lane, role):
    return next(x["key"] for x in lane["spec"]["units"] if x["role"] == role)


def short_of(lane, role):
    return next(x["short"] for x in lane["spec"]["units"] if x["role"] == role)


def gaps(lane, a="R", b="U"):
    """[m] the gap between the two formations' facing edges along z (the lanes come straight at each other): centres less half
    the planned depths."""
    d = np.abs(mp.col(lane, a, "z") - mp.col(lane, b, "z"))
    return d - (mp.depth(short_of(lane, a)) + mp.depth(short_of(lane, b))) / 2


def hp_abs(lane, role):
    """[HP] the unit's health in the lane (the game's unary hitpoints x its passport's health; the twin's hp_abs)."""
    if lane.get("sim"):
        return mp.col(lane, role, "hp_abs")
    return mp.col(lane, role, "hp") * _hp_total(key_of(lane, role))


def slope(t, v, a, b):
    """HP a second lost over [a, b] (None: not recorded there)."""
    if a is None or b is None or b <= a:
        return None
    va, vb = mp.at(t, v, a), mp.at(t, v, b)
    return None if va is None or vb is None else round((va - vb) / (b - a), 1)


def contact_of(lane, role="U"):
    return (lane.get("events") or {}).get("contact", {}).get(role)


def seen_at(lane):
    """(t, gap) of the first sample the enemy sees R (the game: is_visible_to_alliance; the twin: vis) after it was unseen, before the
    contact; the unit's own is_hidden at the same time (hid_off); never unseen: None."""
    t = np.array(lane["t"])
    v = mp.flag(lane, "R", "v")
    hid = mp.flag(lane, "R", "hid")
    g = gaps(lane)
    out = {"unseen_at_start": bool(len(v) and not v[0])}
    i = next((i for i in range(1, len(v)) if v[i] and not v[i - 1]), None)
    if i is not None:
        out["seen_t"], out["seen_gap"] = round(float(t[i]), 2), round(float(g[i]), 1)
        out["seen_gap_before"] = round(float(g[i - 1]), 1)
    j = next((j for j in range(1, len(hid)) if hid[j - 1] and not hid[j]), None)
    if j is not None:
        out["hid_off_t"], out["hid_off_gap"] = round(float(t[j]), 2), round(float(g[j]), 1)
    return out


def spans(lane, role, key):
    t = np.array(lane["t"])
    on = mp.has_fx(lane, role, key)
    out = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            out.append((round(float(t[i]), 2), round(float(t[j]), 2)))
            i = j + 1
        else:
            i += 1
    return out


def armour_series(lane, role="U"):
    """[armour on the unit]: the game's card (arm), the twin: its armour with the blade as apply() lays it."""
    if not lane.get("sim"):
        return mp.col(lane, role, "arm")
    a = np.array(mp.col(lane, role, "arm0"))
    blade = mp.has_fx(lane, role, BLADE)
    return np.where(blade, np.floor(a * 0.5 + 0.5), a)


def morale_points(lane, role="U"):
    return mp.points(lane, role)


def mem_rows(run_dir, name):
    """[(t s, row)] of a unit in the run's mem_morale events (the morale from the game's memory; none: [])."""
    path = Path(run_dir) / "events.jsonl"
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if '"mem_morale"' not in line:
            continue
        r = json.loads(line)
        row = (r.get("units") or {}).get(name)
        if row:
            out.append((float(r.get("model_ms", r.get("t", 0))) / 1000.0, row))
    return out


def surprised(lane, runs_dir):
    """From memory: the target's 'surprised' timer and effect (mem_morale tim[0], the effect number EFFECT index in fx) - the
    first time each is on, how long, the points. None without memory rows."""
    if lane.get("sim"):
        return None
    name = next(x["name"] for x in lane["spec"]["units"] if x["role"] == "U")
    rows = mem_rows(Path(runs_dir) / lane["run"], name)
    if not rows:
        return None
    sid = mem_probe.EFFECT.index("surprised") if "surprised" in mem_probe.EFFECT else None
    t0 = rows[0][0]
    timer_on = [(t, r["tim"][0]) for t, r in rows if r.get("tim") and r["tim"][0] is not None and r["tim"][0] >= 0]
    fx_on = []
    for t, r in rows:
        fx = r.get("fx") or []
        for k in range(0, len(fx) - 1, 2):
            if fx[k] == sid:
                fx_on.append((t, fx[k + 1]))
    out = {"polls": len(rows), "timer_polls": len(timer_on), "effect_polls": len(fx_on), "effect_id": sid}
    if timer_on:
        out["timer_first_t"], out["timer_first_value"] = round(timer_on[0][0] - t0, 2), timer_on[0][1]
        out["timer_last_t"] = round(timer_on[-1][0] - t0, 2)
    if fx_on:
        out["effect_first_t"], out["effect_points"] = round(fx_on[0][0] - t0, 2), fx_on[0][1]
        out["effect_last_t"] = round(fx_on[-1][0] - t0, 2)
    return out


def measure(lane, runs_dir=None):
    spec = lane["spec"]
    kind = spec["kind"]
    t = np.array(lane["t"])
    c = contact_of(lane)
    out = {"run": lane["run"], "lane": spec["name"], "kind": kind, "contact_s": c}
    if kind in ("plain", "bomb", "bomb_walk"):
        out.update(seen_at(lane))
        if c is not None:
            out["R_speed_before"] = None
            pts = morale_points(lane)
            for dt in (-1, 1, 3, 5, 10, 20, 30):
                v = mp.at(t, pts, c + dt)
                out[f"U_pts_{dt:+d}"] = v
            out["R_hp_s_10"] = slope(t, hp_abs(lane, "R"), c + 1, c + 10)
            out["U_hp_s_10"] = slope(t, hp_abs(lane, "U"), c + 1, c + 10)
            lab = mp.labels(lane, "U")
            out["U_labels"] = sorted({x for x in lab if x})
        if not lane.get("sim") and runs_dir is not None:
            out["surprised"] = surprised(lane, runs_dir)
        out["bombs"] = spans(lane, "R", BOMBS)
    if kind in ("blade_gs", "blade_leave", "melee_sw", "arrow_blade"):
        arm = armour_series(lane)
        out["armour_before"] = mp.at(t, arm, (c - 1) if c else 0.5)
        if c is not None:
            out["armour_after"] = mp.at(t, arm, c + 5)
            on = mp.has_fx(lane, "U", BLADE)
            late = t >= c + 2
            mask = late & (t <= c + (14 if kind == "blade_leave" else 3 if kind == "arrow_blade" else 55))
            out["blade_share"] = round(float(on[mask].mean()), 3) if mask.any() else None
            first = next((float(x) for x, o in zip(t, on) if o), None)
            out["blade_first_after_contact"] = round(first - c, 2) if first is not None else None
    if kind in ("blade_gs", "melee_sw") and c is not None:
        for a, b in ((-1, 10), (10, 30), (30, 55), (-1, 55)):
            out[f"U_hp_s_{a}_{b}"] = slope(t, hp_abs(lane, "U"), c + a, c + b)
            out[f"R_hp_s_{a}_{b}"] = slope(t, hp_abs(lane, "R"), c + a, c + b)
    if kind in ("blade_leave", "arrow_blade") and c is not None:
        m_r, m_u = mp.flag(lane, "R", "m"), mp.flag(lane, "U", "m")
        on = mp.has_fx(lane, "U", BLADE)
        lastm = next((float(x) for x, o in zip(t[::-1], m_u[::-1]) if o), None)
        lastb = next((float(x) for x, o in zip(t[::-1], on[::-1]) if o), None)
        out["melee_end_s"] = round(lastm - c, 2) if lastm is not None else None
        out["blade_after_melee_s"] = round(lastb - lastm, 2) if lastm is not None and lastb is not None else None
        arm = armour_series(lane)
        out["armour_end"] = mp.at(t, arm, float(t[-1]) - 0.5)
        out["R_hp_s_in_melee"] = slope(t, hp_abs(lane, "R"), c + 1, c + (14 if kind == "blade_leave" else 3))
        out["U_hp_s_in_melee"] = slope(t, hp_abs(lane, "U"), c + 1, c + (14 if kind == "blade_leave" else 3))
    if kind == "uses":
        out["bombs"] = spans(lane, "R", BOMBS)
        out["hid_share"] = round(float(mp.flag(lane, "R", "hid").mean()), 3)
    if kind in ("arrow_ctrl", "arrow_blade"):
        steps = lane.get("steps") or {}
        spec_steps = spec["steps"]
        i_shoot = next(i for i, x in enumerate(spec_steps, 1) if x["do"] == "shoot")
        ts = steps.get(i_shoot)
        if ts is not None:
            out["fire_t"] = round(float(ts), 2)
            out["U_hp_s_fire"] = slope(t, hp_abs(lane, "U"), ts + 2, ts + 28)
            out["armour_fire"] = mp.at(t, armour_series(lane), ts + 10)
            on = mp.has_fx(lane, "U", BLADE)
            mask = (t >= ts + 2) & (t <= ts + 28)
            out["blade_share_fire"] = round(float(on[mask].mean()), 3) if mask.any() else None
    return out


def newest(n=3):
    """The newest n run folders of the plan (the three battles)."""
    out = []
    for d in reversed(mp.runs()):
        cfg = json.loads((d / "manifest.json").read_text(encoding="utf-8")).get("config") or {}
        if cfg.get("plan") == "runners":
            out.append(d)
        if len(out) == n:
            break
    return sorted(out)


def mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return round(float(np.mean(vals)), 2) if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="the twin alone on the plan's lanes (no game run)")
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    result = {}
    if args.plan_only:
        lanes = mp.plan_lanes("runners")
    else:
        dirs = args.runs or newest()
        lanes = [ln for d in dirs for ln in mp.load_run(d)]
        result["game"] = [measure(ln, mp.RUNS if not args.runs else args.runs[0].parent) for ln in lanes]
        for r in result["game"]:
            print("game", json.dumps(r, ensure_ascii=False, default=float))
    if args.sim or args.plan_only:
        result["sim"] = [measure(ln) for ln in mp.sim_lanes(lanes, device=args.device, copies=args.copies)]
        kinds = {}
        for r in result["sim"]:
            kinds.setdefault(r["kind"], []).append(r)
        result["sim_mean"] = {k: {q: (mean(v, q) if isinstance(v[0].get(q), (int, float)) or v[0].get(q) is None else v[0].get(q))
                                  for q in {q for r in v for q in r} - {"run", "lane", "kind"}} for k, v in kinds.items()}
        for k, r in result["sim_mean"].items():
            print("sim ", k, json.dumps(r, ensure_ascii=False, default=float, sort_keys=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
