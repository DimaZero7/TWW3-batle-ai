"""The probe of the tactician's first drill (design/tactic/index.html, 'Первое упражнение'; build/diag_tactic/probe): does the
simulator agree with the game on the two things the drill stands on? Lanes of tools/nn/charge_probe.py, flat map.

  tacpass  our infantry passes through our standing shooters, and back; 1 battle of 3 lanes. Each lane's unit makes four
           80 m legs on one line: a clean one at a run, through the block at a run, turning back through it at a walk,
           a clean one at a walk (the last goes on the same way). Lanes: 'centre' swordsmen 40 m wide through archers 60 m
           wide (the teacher's zone is 64 m), the centres meeting; 'side' the same through a deeper block (archers 30 m
           wide) 20 m to a side: the blocks overlap by 15 m and the centres stay 20 m apart; 'back' archers 40 m wide
           through swordsmen 70 m wide (the teacher's infantry zone is 80 m). The men every 0.1 s from memory (attacker),
           every 0.5 s by script (the block's), the formation's state (apps.mem.order).
  tacfire  arcing shooters (archers, trajectory dual_low_fixed) on skavenslaves with our swordsmen in front, 60 m wide over
           their 40 m; 1 battle of 4 lanes, 60 s. Two pairs at one range, with friends and without: 110 m front to front with
           the friends 20 m ahead; 85 m with them 60 m ahead (~17 m short of the enemy). Shots, the enemy's and the friends'
           HP by script, the launch angle from the projectiles in memory.
  tacwalk  the same archers while the line closes: the skavenslaves walk at them from 120 m front to front, straight, 110 s;
           1 battle of 3 lanes. 'w20' / 'w48' our swordsmen 20 / 48 m ahead of the archers: the slaves walk into them and
           fight (the drill's moment); 'w0' no friends, the slaves stop 12 m off (the control). The extras' men from memory:
           whose man an arrow is in (apps.mem.hits).

    python -m tools.nn.charge_probe run --plan tacpass --mem --speed 20
    python -m tools.nn.charge_probe run --plan tacfire --mem --speed 20
    python -m tools.nn.charge_probe run --plan tacwalk --mem --speed 20
    bash tools/nn/dock.sh tools.nn.tactic_probe [--pass RUN] [--fire RUN] [--walk RUN] [--device cpu] [--out build/diag_tactic/probe/x.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp

PLANS = ("tacpass", "tacfire", "tacwalk")
LEG_M = 80.0
BACK_Z = 420.0             # the enemy, parked out of the bows' reach, never ordered
START_Z = -140.0           # the attacker's centre at the go (each lane's own x)
BLOCK_Z = -20.0            # the block's centre: START_Z + LEG_M + LEG_M / 2, the middle of the legs through it
# (name, the order's point from the start along z, at a run, at_s)
LEGS = (("clean_run", LEG_M, True, 1.0), ("through_run", 2 * LEG_M, True, 45.0),
        ("back_walk", LEG_M, False, 100.0), ("clean_walk", 0.0, False, 180.0))
PASS_MAX_S = 250
# (kind, the unit that walks, its width, the block, its width, the block's offset across the line)
PASS = (("centre", "swords", 40, "archer", 60, 0.0), ("side", "swords", 40, "archer", 30, 20.0),
        ("back", "archer", 40, "swords", 70, 0.0))
FIRE_S = 60
FIRE_W, FRIEND_W, ENEMY_W = 40, 60, 40
# (kind, range front to front, the friends' back edge ahead of the shooters' front; None: no friends)
FIRE = (("c110", 110.0, None), ("f20", 110.0, 20.0), ("c85", 85.0, None), ("f60", 85.0, 60.0))
LANE_X = (-450.0, -150.0, 150.0, 450.0)
WALK_GAP, WALK_S = 120.0, 110
WALK = (("w20", 20.0), ("w48", 48.0), ("w0", None))


def pass_lanes():
    out = []
    for (kind, unit, width, block, block_w, off), x in zip(PASS, (-300.0, 0.0, 300.0)):
        steps = [{"at_s": at, "kind": "move", "dx": 0.0, "dz": dz, "run": run} for _, dz, run, at in LEGS]
        out.append(cp.lane(unit, "cspear", "script", "hold", gap_m=round(BACK_Z - START_Z), fight_s=0, answer=False, kind=kind,
                           a_w=width, max_s=PASS_MAX_S, men_all_s=PASS_MAX_S, men_all_tg=True, steps=steps, place=(x, BACK_Z),
                           a_at={"x": x, "z": START_Z, "b": 0.0}, t_at={"x": x, "z": BACK_Z, "b": 180.0},
                           extras=[{"short": block, "dx": off, "dz": BLOCK_Z - BACK_Z, "b": 0, "width": block_w}]))
    return out


def fire_lanes():
    out = []
    a_d = cp.depth(cp.UNITS["archer"][0], cp.UNITS["archer"][1], FIRE_W)
    t_d = cp.depth(cp.UNITS["slave"][0], cp.UNITS["slave"][1], ENEMY_W)
    f_d = cp.depth(cp.UNITS["swords"][0], cp.UNITS["swords"][1], FRIEND_W)
    for (kind, gap, friend), x in zip(FIRE, LANE_X):
        za = -60.0
        zt = za + a_d / 2 + gap + t_d / 2
        extra = {}
        if friend is not None:
            zf = za + a_d / 2 + friend + f_d / 2
            extra["extras"] = [{"short": "swords", "dx": 0.0, "dz": round(zf - zt, 1), "b": 0, "width": FRIEND_W}]
        out.append(cp.lane("archer", "slave", "shoot", "hold", gap_m=gap, fight_s=FIRE_S, max_s=FIRE_S, answer=False, kind=kind,
                           a_w=FIRE_W, t_w=ENEMY_W, place=(x, zt), a_at={"x": x, "z": za, "b": 0.0},
                           t_at={"x": x, "z": round(zt, 1), "b": 180.0}, **extra))
    return out


def walk_lanes():
    out = []
    a_d = cp.depth(cp.UNITS["archer"][0], cp.UNITS["archer"][1], FIRE_W)
    t_d = cp.depth(cp.UNITS["slave"][0], cp.UNITS["slave"][1], ENEMY_W)
    f_d = cp.depth(cp.UNITS["swords"][0], cp.UNITS["swords"][1], FRIEND_W)
    for (kind, friend), x in zip(WALK, (-300.0, 0.0, 300.0)):
        za = -60.0
        zt = za + a_d / 2 + WALK_GAP + t_d / 2
        extra = {"push_m": -(WALK_GAP - 12.0)}
        if friend is not None:
            zf = za + a_d / 2 + friend + f_d / 2
            extra = {"push_m": -(WALK_GAP + 40.0),
                     "extras": [{"short": "swords", "dx": 0.0, "dz": round(zf - zt, 1), "b": 0, "width": FRIEND_W}]}
        out.append(cp.lane("archer", "slave", "shoot", "push", gap_m=WALK_GAP, fight_s=WALK_S, max_s=WALK_S, answer=False,
                           kind=kind, a_w=FIRE_W, t_w=ENEMY_W, place=(x, zt), a_at={"x": x, "z": za, "b": 0.0},
                           t_at={"x": x, "z": round(zt, 1), "b": 180.0}, **extra))
    return out


def battles(plan):
    return [{"tacpass": pass_lanes, "tacfire": fire_lanes, "tacwalk": walk_lanes}[plan]()]


def config_extra(plan):
    """charge_probe.run_config: the rows every 0.25 s, the men from memory every 0.1 s (lanes 'near' all the time)."""
    if plan == "tacpass":
        return {"tick_ms": cp.TURN_TICK_MS, "men_ms": 500, "men_after_s": 0, "men_near_m": 800, "mem_form": True}
    if plan == "tacwalk":
        return {"tick_ms": cp.TICK_MS, "men_ms": 2000, "men_after_s": 0, "men_near_m": 300, "mem_proj": True, "mem_extras": True}
    return {"tick_ms": cp.TICK_MS, "men_ms": 2000, "men_after_s": 0, "men_near_m": 300, "mem_proj": True}


# ---------------------------------------------------------------- the recordings

def nominal(lanes, men=None):
    """The lanes with their first sample at the units' real centres: a unit's script place is one of its men (0.3-8 m off the
    centre, docs/ru/game/memreader.md) and the entry's point is the formation's FRONT centre. The men's median from memory
    when bound, else (the extras) the script's list, else the point less half the block's depth along its facing."""
    from tools.nn.mem_probe import COL
    men = men or {}

    def from_men(name):
        rows = men.get(name)
        if not rows:
            return None
        return float(np.median(rows[0][1][:, COL["x"]])), float(np.median(rows[0][1][:, COL["z"]]))
    out = []
    for ln in lanes:
        ln = dict(ln)
        sp = ln["spec"]
        a, t = sp["a_at"], sp["t_at"]
        ax, az = from_men(sp["attacker"]) or (a["x"], a["z"] - sp["a_depth"] / 2)
        tx, tz = from_men(sp["target"]) or (t["x"], t["z"] + sp["t_depth"] / 2)
        s = list(ln["samples"])
        t0, ra, rt, rl = s[0]
        s[0] = (t0, dict(ra, x=ax, z=az, b=a["b"]), dict(rt, x=tx, z=tz, b=t["b"]), rl)
        ln["samples"] = s
        if ln.get("extras"):
            ex = list(ln["extras"])
            rows = []
            for j, (r, e) in enumerate(zip(ex[0][1], sp["extras"])):
                place = from_men(e["name"])
                em = ln.get("extra_men")
                if place is None and em:
                    q = np.asarray(em[0][1][j], float).reshape(-1, 2) / 10
                    place = float(np.median(q[:, 0])), float(np.median(q[:, 1]))
                if place is None:
                    d = cp.depth(cp.UNITS[e["short"]][0], cp.UNITS[e["short"]][1], e["width"])
                    place = t["x"] + e["dx"], t["z"] + e["dz"] - d / 2
                rows.append(dict(r, x=place[0], z=place[1], b=e["b"]))
            ex[0] = (ex[0][0], rows)
            ln["extras"] = ex
        out.append(ln)
    return out


def mem_men(run_dir):
    """{unit: [(t on the lanes' clock, men [n, MAN])]} from the run's mem_men rows (their t: model time, 5.9 s ahead of it)."""
    from tools.nn.mem_probe import MAN
    w = len(MAN)
    out = {}
    t0 = None
    with open(Path(run_dir) / "events.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if t0 is None and '"event":"probe_sample"' in line:
                r = json.loads(line)
                t0 = (r["model_ms"] - r["lanes"][0]["t"]) / 1000
            if '"event":"mem_men"' not in line:
                continue
            r = json.loads(line)
            for name, u in r["units"].items():
                a = np.asarray(u.get("rows") or [], float)
                if len(a) >= w:
                    out.setdefault(name, []).append((r["t"] / 1000, a.reshape(-1, w)))
    return {name: [(t - t0, a) for t, a in rows] for name, rows in out.items()}


def centre_game(ln, men):
    """(t, x, z, the men's median speed, t0) of the attacker from memory: the script's place is one man, not the centre."""
    from tools.nn.mem_probe import COL
    rows = men.get(ln["spec"]["attacker"]) or []
    t0 = float(ln["samples"][0][0])
    t = np.array([r[0] for r in rows])
    x = np.array([np.median(r[1][:, COL["x"]]) for r in rows])
    z = np.array([np.median(r[1][:, COL["z"]]) for r in rows])
    v = np.array([np.median(np.hypot(r[1][:, COL["vx"]], r[1][:, COL["vz"]])) for r in rows])
    return t, x, z, v, t0


def centre_row(ln, who=1):
    s = [r for r in ln["samples"] if r[who] and r[who].get("x") is not None]
    t = np.array([r[0] for r in s])
    x, z = np.array([r[who]["x"] for r in s]), np.array([r[who]["z"] for r in s])
    dt = np.maximum(np.diff(t, append=t[-1] + 0.5), 1e-3)
    v = np.hypot(np.diff(x, append=x[-1]), np.diff(z, append=z[-1])) / dt
    return t, x, z, v, float(t[0])


def legs(ln, c):
    """Per leg: the seconds from the order to the end of the way (the last moment its centre is over 1.5 m short of where it
    ended), the way along the line, the seconds over the 60 m round the block (+- 30 m of its z), the speed over its middle
    30 m, a clean leg's speed away from its ends and the most it left its straight line."""
    t, x, z, _, _ = c
    sp = ln["spec"]
    block = sp["extras"][0]
    bz = sp["t_at"]["z"] + block["dz"]
    x0 = sp["a_at"]["x"]
    out = {}
    for i, (name, _, run, at) in enumerate(LEGS):
        t1 = LEGS[i + 1][3] if i + 1 < len(LEGS) else t[-1]
        k = (t >= at) & (t <= t1)
        if k.sum() < 5:
            continue
        tt, xx, zz = t[k], x[k], z[k]
        d = np.hypot(xx - xx[-1], zz - zz[-1])
        moved = np.nonzero(d > 1.5)[0]
        last = tt[moved[-1]] if len(moved) else tt[0]
        row = {"run": run, "leg_s": float(last - at), "way_m": float(abs(zz[-1] - zz[0])), "z_end": float(zz[-1]),
               "dx_max": float(np.abs(xx - x0).max())}
        if name in ("through_run", "back_walk"):
            near = np.nonzero(np.abs(zz - bz) <= 30.0)[0]
            if len(near) > 1:
                row["s_60m"] = float(tt[near[-1]] - tt[near[0]])
                row["v_60m"] = float(abs(zz[near[-1]] - zz[near[0]]) / max(tt[near[-1]] - tt[near[0]], 1e-3))
            mid = np.nonzero(np.abs(zz - bz) <= 15.0)[0]
            if len(mid) > 1:
                row["v_30m"] = float(abs(zz[mid[-1]] - zz[mid[0]]) / max(tt[mid[-1]] - tt[mid[0]], 1e-3))
        a = np.nonzero(np.abs(zz - zz[0]) >= 15.0)[0]
        e = np.nonzero(np.abs(zz - zz[-1]) >= 15.0)[0]
        if name.startswith("clean") and len(a) and len(e) and e[-1] > a[0]:
            row["v_clean"] = float(abs(zz[e[-1]] - zz[a[0]]) / (tt[e[-1]] - tt[a[0]]))
        out[name] = row
    return out


def inside_game(ln, men):
    """Per leg through the block: the men inside the block's frame (P1-P99 of its first places, by script, +- 1 m) against
    the men of the same unit outside it - their median speeds and the share of the unit's men-seconds spent inside; the
    speed of the unit's median while the two frames overlap along the line."""
    from tools.nn.mem_probe import COL
    rows = ln.get("extra_men") or []
    first = np.asarray(rows[0][1][0], float).reshape(-1, 2) / 10
    x0, x1 = np.percentile(first[:, 0], 1) - 1, np.percentile(first[:, 0], 99) + 1
    z0, z1 = np.percentile(first[:, 1], 1) - 1, np.percentile(first[:, 1], 99) + 1
    r = men.get(ln["spec"]["attacker"]) or []
    t = np.array([q[0] for q in r])
    out = {"block_frame": [float(x0), float(x1), float(z0), float(z1)]}
    for i, (name, _, run, at) in enumerate(LEGS):
        if name not in ("through_run", "back_walk"):
            continue
        t1 = LEGS[i + 1][3]
        v_in, v_out = [], []
        ts = []
        for tq, a in r:
            if not at <= tq <= t1:
                continue
            x, z = a[:, COL["x"]], a[:, COL["z"]]
            sp = np.hypot(a[:, COL["vx"]], a[:, COL["vz"]])
            k = (x >= x0) & (x <= x1) & (z >= z0) & (z <= z1)
            go = np.median(sp) > 0.7
            if go and k.any():
                v_in.append(sp[k])
                ts.append(tq)
            if go and (~k).any():
                v_out.append(sp[~k])
        if not v_in:
            out[name] = {"men_inside": 0}
            continue
        vi, vo = np.concatenate(v_in), np.concatenate(v_out)
        out[name] = {"v_in": float(np.median(vi)), "v_out": float(np.median(vo)), "share": float(np.median(vi) / np.median(vo)),
                     "v_in_mean": float(vi.mean()), "v_out_mean": float(vo.mean()), "share_mean": float(vi.mean() / vo.mean()),
                     "inside_s": float(ts[-1] - ts[0]), "men_seconds": float(len(vi) * 0.1)}
    return out


def order_game(ln, men):
    """Per leg, the men from memory: the front and the depth of the block (P5-P95 along x and z), the most during the leg and
    at its end; the share of men stopped (under a third of the median speed while the median goes) at its worst and mean,
    the seconds from the median's stop to the last men's stop (P95 speed under 0.3 m/s) and to the depth and the front back
    within 10 % of the go's."""
    from tools.nn.mem_probe import COL
    rows = men.get(ln["spec"]["attacker"]) or []
    t = np.array([r[0] for r in rows])
    w = np.array([np.percentile(r[1][:, COL["x"]], 95) - np.percentile(r[1][:, COL["x"]], 5) for r in rows])
    d = np.array([np.percentile(r[1][:, COL["z"]], 95) - np.percentile(r[1][:, COL["z"]], 5) for r in rows])
    sp = [np.hypot(r[1][:, COL["vx"]], r[1][:, COL["vz"]]) for r in rows]
    med = np.array([np.median(s) for s in sp])
    p95 = np.array([np.percentile(s, 95) for s in sp])
    slow = np.array([float((s < m / 3).mean()) if m > 0.7 else 0.0 for s, m in zip(sp, med)])
    out = {"w0": float(w[t <= 1.0].mean()), "d0": float(d[t <= 1.0].mean())}
    for i, (name, _, _, at) in enumerate(LEGS):
        t1 = LEGS[i + 1][3] if i + 1 < len(LEGS) else t[-1]
        k = (t >= at) & (t <= t1)
        if k.sum() < 5:
            continue
        tk = t[k]
        moving = np.nonzero(med[k] > 0.3)[0]
        stop = tk[moving[-1]] if len(moving) else tk[0]
        quiet = np.nonzero(p95[k] > 0.3)[0]
        last = tk[quiet[-1]] if len(quiet) else tk[0]
        messy = np.nonzero((d[k] > 1.1 * out["d0"]) | (w[k] > 1.1 * out["w0"]) | (w[k] < 0.9 * out["w0"]))[0]
        going = med[k] > 0.7
        out[name] = {"w_max": float(w[k].max()), "d_max": float(d[k].max()), "w_end": float(w[k][-1]), "d_end": float(d[k][-1]),
                     "slow_max": float(slow[k].max()), "slow_mean": float(slow[k][going].mean()) if going.any() else 0.0,
                     "stop_s": float(stop - at), "men_stop_after_s": float(last - stop),
                     "tidy_after_s": max(float(tk[messy[-1]] - stop), 0.0) if len(messy) else 0.0}
    return out


def block_game(ln):
    """The standing block's men by script: how far they went from their first places (matched by the list's order), at their
    worst and at the end, its centre's shift. A soldier row: flat x, z in decimetres (entries/charge_probe.lua soldiers)."""
    rows = ln.get("extra_men") or []
    arr, ts = [], []
    for t, e in rows:
        a = np.asarray(e[0], float)          # flat: x, z, x, z ...
        if a.size < 4:
            continue
        arr.append(a.reshape(-1, 2) / 10)
        ts.append(t)
    if len(arr) < 3:
        return None
    n = min(len(a) for a in arr)
    X = np.stack([a[:n] for a in arr])            # [polls, men, 2]
    dist = np.hypot(*(X - X[0]).transpose(2, 0, 1))
    c = X.mean(1)
    ts = np.array(ts)
    out = {"men": n, "moved_2m_max": float((dist > 2).mean(1).max()), "moved_2m_end": float((dist[-1] > 2).mean()),
           "moved_5m_max": float((dist > 5).mean(1).max()), "dist_p95_max": float(np.percentile(dist, 95, axis=1).max()),
           "centre_shift_max": float(np.hypot(*(c - c[0]).T).max()), "centre_shift_end": float(np.hypot(*(c[-1] - c[0])))}
    for i, (name, _, _, at) in enumerate(LEGS):
        t1 = LEGS[i + 1][3] if i + 1 < len(LEGS) else ts[-1]
        k = (ts >= at) & (ts <= t1)
        if k.any():
            out[name] = {"moved_2m": float((dist[k] > 2).mean(1).max()), "p95": float(np.percentile(dist[k], 95, axis=1).max())}
    return out


def angles(run_dir, spec_of):
    """{lane kind: launch angles, degrees} of the projectiles from memory (spawned in state 0, nearest shooter's place)."""
    out = {}
    with open(Path(run_dir) / "events.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if '"mem_proj"' not in line:
                continue
            r = json.loads(line)
            if r.get("event") != "mem_proj":
                continue
            for p in r.get("spawn") or []:
                p = dict(zip(("id", "c", "st", "x", "y", "z", "vx", "vy", "vz", "t", "rec", "sx", "sy", "sz"), p))
                if p.get("st") != 0 or p.get("vx") is None:
                    continue
                sx, sz = p.get("sx", p["x"]), p.get("sz", p["z"])
                best = min(spec_of, key=lambda n: (spec_of[n]["x"] - sx) ** 2 + (spec_of[n]["z"] - sz) ** 2)
                out.setdefault(best, []).append(float(np.degrees(np.arctan2(p["vy"], np.hypot(p["vx"], p["vz"])))))
    return out


def fire_measure(ln):
    """Shots per man per minute from the first shot to the lane's end, the enemy's and the friends' HP lost."""
    sp = ln["spec"]
    smp = [r for r in ln["samples"] if r[0] <= sp["max_s"] + 1e-6]
    t = np.array([r[0] for r in smp])
    ammo = np.array([np.nan if r[1].get("ammo") is None else float(r[1]["ammo"]) for r in smp])
    hp = np.array([float(r[2]["hp"]) for r in smp])
    men = float(smp[0][1]["men"])
    ok = np.isfinite(ammo)
    drop = np.diff(ammo[ok])
    tt = t[ok][1:]
    fired = np.nonzero(drop < 0)[0]
    out = {"kind": sp.get("kind"), "ammo0": float(ammo[ok][0]), "first_s": float(tt[fired[0]]) if len(fired) else None,
           "shots": float(ammo[ok][0] - ammo[ok][-1]), "hp_enemy": float(hp[0] - hp[-1]), "hp0": float(hp[0]),
           "men_enemy_lost": float(smp[0][2]["men"] - smp[-1][2]["men"])}
    if len(fired):
        span = float(tt[-1]) - float(tt[fired[0]]) + 0.5
        out["span_s"] = span
        out["shots_man_min"] = out["shots"] / men / span * 60
        if out["ammo0"] <= 100:       # (the simulator's ammo is a man's)
            out["shots_man_min"] = out["shots"] / span * 60
    if ln.get("extras"):
        ex = [r for r in ln["extras"] if r[0] <= sp["max_s"] + 1e-6]
        e0, e1 = ex[0][1][0], ex[-1][1][0]
        out["hp_friend"] = float(e0["hp"]) - float(e1["hp"])
        out["men_friend_lost"] = float(e0["men"]) - float(e1["men"])
    return out


WIN = (0, 15, 30, 45, 60, 75, 90, 110)
REACH_M = 4.0


def spawns(run_dir):
    """The projectiles' spawn rows of a run as dicts on the lanes' clock (a poll c is 0.1 s; c 1 is the go + ...: the row's own
    c against the polls' t), and {man id: unit} from mem_men's new."""
    F = ("id", "c", "st", "x", "y", "z", "vx", "vy", "vz", "t", "rec", "sx", "sy", "sz", "w")
    rows, owner, c_t = [], {}, {}
    t0 = None
    with open(Path(run_dir) / "events.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if t0 is None and '"event":"probe_sample"' in line:
                r = json.loads(line)
                t0 = (r["model_ms"] - r["lanes"][0]["t"]) / 1000
            if '"event":"mem_proj"' in line:
                r = json.loads(line)
                rows += [dict(zip(F, p)) for p in r.get("spawn") or []]
            elif '"event":"mem_men"' in line:
                r = json.loads(line)
                c_t[r["c"]] = r["t"] / 1000
                for n in r.get("new") or []:
                    owner[n[0]] = n[1]
    for p in rows:
        p["time"] = c_t.get(p["c"], np.nan) - t0
    return rows, owner


def walk_game(run):
    lanes = cp.load_run(run)
    rows, owner = spawns(run)
    men = mem_men(run)
    out = {}
    of = {ln["spec"]["kind"]: ln["spec"]["a_at"] for ln in lanes}
    names = {ln["spec"]["kind"]: ln["spec"] for ln in lanes}
    launch = [p for p in rows if p["st"] == 0 and p.get("vx") not in (None, False) and p.get("sx") not in (None, False)]
    for p in launch:
        p["lane"] = min(of, key=lambda n: (of[n]["x"] - p["sx"]) ** 2 + (of[n]["z"] - p["sz"]) ** 2)
        p["ang"] = float(np.degrees(np.arctan2(p["vy"], np.hypot(p["vx"], p["vz"]))))
    bodies = [p for p in rows if p["st"] == 2]
    for kind, sp in names.items():
        ln = next(x for x in lanes if x["spec"]["kind"] == kind)
        friend = (sp.get("extras") or [{}])[0].get("name")
        row = {}
        ls = [p for p in launch if p["lane"] == kind]
        by = []
        for lo, hi in zip(WIN, WIN[1:]):
            q = [p["ang"] for p in ls if lo <= p["time"] < hi]
            smp = [s for s in ln["samples"] if lo <= s[0] < hi]
            d = [s[2]["z"] - s[1]["z"] for s in smp]
            by.append({"from": lo, "shots": len(q), "angle": float(np.median(q)) if q else None,
                       "high": float(np.mean(np.array(q) > 30)) if q else None, "range_m": float(np.mean(d)) if d else None,
                       "enemy_melee": float(np.mean([bool(s[2].get("m")) for s in smp])) if smp else None})
        row["windows"] = by
        high = sorted(p["time"] for p in ls if p["ang"] > 30)
        row["high_from_s"] = high[int(0.02 * len(high))] if high else None
        row["shots_after_high"] = len([p for p in ls if high and p["time"] >= row["high_from_s"]])
        n_f, n_t = 0, 0
        for b in bodies:
            ids = (b.get("w") or [])[1::2] if isinstance(b.get("w"), list) else []
            who = {owner.get(i) for i in ids}
            if friend and friend in who:
                n_f += 1
            if sp["target"] in who:
                n_t += 1
        row["bodies_friend"], row["bodies_enemy"] = n_f, n_t
        row["contact_s"] = next((s[0] for s in ln["samples"] if s[2].get("m")), None)
        row.update(fire_measure(ln))
        if friend:
            row["friend_hits"] = friend_hits(men[friend], men[sp["target"]])
        out[kind] = row
    return out


def friend_hits(fm, tm):
    """A friend's man losing HP with no enemy man within REACH_M cannot be struck in melee (the men from memory, every 0.1 s):
    those are arrows, a lower bound; the bound / the share of his men out of the enemy's reach (arrows fall on all alike):
    the estimate. Reach: the lanes' drops stop at 4 m, then those of the arrows go on to 100 m."""
    from tools.nn.mem_probe import COL
    tt = np.array([q[0] for q in tm])
    last, drops, out_of_reach = {}, [], []
    for tq, a in fm:
        e = tm[int(np.argmin(np.abs(tt - tq)))][1]
        d = np.hypot(a[:, COL["x"]][:, None] - e[:, COL["x"]][None], a[:, COL["z"]][:, None] - e[:, COL["z"]][None]).min(1)
        if d.min() < 3:
            out_of_reach.append(float((d >= REACH_M).mean()))
        for r, near in zip(a, d):
            i, h = int(r[COL["id"]]), r[COL["hp"]]
            if i in last and h < last[i]:
                drops.append((tq, last[i] - h, float(near)))
            last[i] = h
    d = np.array(drops)
    far = d[:, 2] >= REACH_M
    share = float(np.mean(out_of_reach)) if out_of_reach else 1.0
    return {"hp_all": float(d[:, 1].sum()), "hits_far": int(far.sum()), "hp_far": float(d[far, 1].sum()), "first_s": float(d[:, 0].min()),
            "share_men_far": share, "hp_arrows_est": float(d[far, 1].sum() / max(share, 1e-3))}


def walk_sim(run, device, copies, no_ff=False):
    """The twin on the same lanes: per window the shots, the share of the fire forced to 45 deg and the fire the friends took
    away (battle.py's bookkeeping fc_lob, fc_skip against fc_cap), the HP lost; no_ff: friendly_fire 0 (the friends' HP then
    is the melee's alone)."""
    import copy
    import dataclasses

    from tools.nn.sim import battle
    from tools.nn.sim.params import load
    lanes = cp.load_run(run)
    params = load()
    params = dataclasses.replace(params, units={**cp._passports(), **params.units})
    if no_ff:
        sim = copy.deepcopy(params.sim)
        sim["missile"]["friendly_fire"] = {k: 0.0 for k in sim["missile"]["friendly_fire"]}
        sim["missile"]["friendly_fire_other"] = 0.0
        params = dataclasses.replace(params, sim=sim)
    orig, log = battle.run, []

    def run_(st, policy, P, until_s=None, record=None, **kw):
        def rec(s):
            if "fc_cap" in s.u:
                log.append((float(s.t.max()), s.u["fc_cap"].sum(1).cpu().numpy().copy(), s.u["fc_lob"].sum(1).cpu().numpy().copy(),
                            s.u["fc_skip"].sum(1).cpu().numpy().copy()))
            record(s)
        return orig(st, policy, P, until_s=until_s, record=rec, **kw)
    battle.run = run_
    try:
        sims = cp.sim_lanes(nominal(lanes, mem_men(run)), params=params, device=device, copies=copies)
    finally:
        battle.run = orig
    t = np.array([x[0] for x in log])
    cap, lob, skip = (np.stack([x[i] for x in log]) for i in (1, 2, 3))
    out = {}
    for b, s in enumerate(sims):
        kind = s["spec"]["kind"]
        row = out.setdefault(kind, {"runs": [], "win": []})
        row["runs"].append(fire_measure(s))
        tt = np.array([q[0] for q in s["samples"]])
        am = np.array([q[1].get("ammo", np.nan) for q in s["samples"]], float)
        cont = next((q[0] for q in s["samples"] if q[2].get("m")), None)
        by = []
        for lo, hi in zip(WIN, WIN[1:]):
            k = (t >= lo) & (t < hi)
            i0, i1 = np.searchsorted(tt, lo), min(np.searchsorted(tt, hi), len(tt) - 1)
            smp = [q for q in s["samples"] if lo <= q[0] < hi]
            by.append({"from": lo, "shots": float(am[i0] - am[i1]), "lob": float(lob[k, b].sum() / max(cap[k, b].sum(), 1e-6)),
                       "skip": float(skip[k, b].sum() / max(cap[k, b].sum(), 1e-6)),
                       "range_m": float(np.mean([q[2]["z"] - q[1]["z"] for q in smp])) if smp else None})
        row["win"].append(by)
        row.setdefault("contact", []).append(cont)
        kl = (lob[:, b] > 0.5 * cap[:, b]) & (cap[:, b] > 0)
        row.setdefault("lob_from", []).append(float(t[kl][0]) if kl.any() else np.nan)
    for kind, row in out.items():
        row["mean"] = mean_rows(row.pop("runs"))
        row["windows"] = [mean_rows([w[i] for w in row["win"]]) for i in range(len(WIN) - 1)]
        row.pop("win")
        row["lob_from_s"] = float(np.nanmean(row.pop("lob_from")))
        c = [x for x in row.pop("contact") if x is not None]
        row["contact_s"] = float(np.mean(c)) if c else None
    return out


def mean_rows(rows):
    keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {k: float(np.mean([r[k] for r in rows if isinstance(r.get(k), (int, float))])) for k in keys}


def twin(lanes, device, copies, men=None):
    import dataclasses

    from tools.nn.sim.params import load
    params = load()
    params = dataclasses.replace(params, units={**cp._passports(), **params.units})
    return cp.sim_lanes(nominal(lanes, men), params=params, device=device, copies=copies)


def report_pass(run, device, copies):
    lanes = cp.load_run(run)
    men = mem_men(run)
    sims = twin(lanes, device, copies, men)
    out = {}
    for ln in lanes:
        kind = ln["spec"]["kind"]
        sim_legs = [legs(s, centre_row(s)) for s in sims if s["spec"]["kind"] == kind]
        out[kind] = {"game": {"legs": legs(ln, centre_game(ln, men)), "order": order_game(ln, men), "block": block_game(ln),
                              "inside": inside_game(ln, men)},
                     "sim": {k: mean_rows([r[k] for r in sim_legs]) for k in sim_legs[0]}}
    return out


def report_fire(run, device, copies):
    lanes = cp.load_run(run)
    sims = twin(lanes, device, copies, mem_men(run))
    ang = angles(run, {ln["spec"]["kind"]: ln["spec"]["a_at"] for ln in lanes})
    out = {}
    for ln in lanes:
        kind = ln["spec"]["kind"]
        a = np.array(ang.get(kind) or [np.nan])
        out[kind] = {"game": dict(fire_measure(ln), angle_med=float(np.median(a)), angle_p10=float(np.percentile(a, 10)),
                                  angle_p90=float(np.percentile(a, 90)), arrows=len(a)),
                     "sim": mean_rows([fire_measure(s) for s in sims if s["spec"]["kind"] == kind])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass", dest="pass_run", type=Path)
    ap.add_argument("--fire", type=Path)
    ap.add_argument("--walk", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    out = {}
    if a.pass_run:
        out["pass"] = report_pass(a.pass_run, a.device, a.copies)
    if a.fire:
        out["fire"] = report_fire(a.fire, a.device, a.copies)
    if a.walk:
        out["walk"] = {"game": walk_game(a.walk), "sim": walk_sim(a.walk, a.device, a.copies),
                       "sim_no_ff": walk_sim(a.walk, a.device, a.copies, no_ff=True)}
    text = json.dumps(out, indent=1, default=float)
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
