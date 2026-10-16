"""The knock-down man by man in the game: the knock probe (magic battle 6, tools/nn/magic_probe.py; the soldiers'
places every 0.1 s) read man by man (numpy, no torch).

  python -m tools.nn.fighters.knockdata build/magic/runs/<battle 6 run> [--out build/fighters/knock.json]

Each target's men are followed from sample to sample (the nearest unclaimed man of the next sample within TRACK_M,
nearest pairs first); a man missing from the next sample has died. A strike: a sample where at least STRIKE_MEN men
jump (move faster than FAST_MPS); its centre the jumping men's mean place before the jump. Per man of the target at
the strike (his place: the mean over the second before it):
  thrown     he moved faster than FAST_MPS within the strike's window (STRIKE_S); the control lane (no cast) gives the
             rate of such moves without a strike
  throw      his displacement when he first stops (speed under STILL_MPS) after the jump; its share along the line from
             the strike's centre (+1 straight away from it)
  lying      from that stop until he moves again (speed over MOVE_MPS: up and walking back)
  died       he left the list while lying / while standing - the deaths a second of the men lying and of the men up,
             within the strike's reach (THROWN men's farthest distance), STRIKE_S after the strike
The melee lanes (swordsmen fought by clanrats from behind, with and without the strike): the clanrats' HP lost a
second after the strike against the control's at the same times - do knocked swordsmen still strike."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp
from tools.nn.fighters import shape

TRACK_M = 2.5
FAST_MPS = 4.0          # faster than any man walks or runs in formation (run 3.0 m/s, charge 3.8)
STILL_MPS = 0.5
MOVE_MPS = 0.8
STRIKE_MEN = 3
STRIKE_S = 12.0         # a bombardment's window (Searing Doom ~8 s of landings) and the last getting up


def track(men, who="tg"):
    """Tracks of the men of one unit: (times [T], pos [T, K, 2] NaN where gone, died [K] the time he left)."""
    col = {"a": 1, "tg": 2}[who]
    times = np.array([m[0] for m in men])
    frames = [shape.xy(m[col]) for m in men]
    K0 = len(frames[0])
    pos = [frames[0]]
    ids = np.arange(K0)
    T = len(frames)
    out = np.full((T, K0, 2), np.nan)
    out[0, :len(frames[0])] = frames[0]
    cur = frames[0].copy()
    alive = np.ones(K0, bool)
    died = np.full(K0, np.nan)
    for k in range(1, T):
        nxt = frames[k]
        a_idx = np.nonzero(alive)[0]
        if len(nxt) == 0:
            died[a_idx] = times[k]
            alive[:] = False
            continue
        d = np.sqrt(((cur[a_idx][:, None, :] - nxt[None, :, :]) ** 2).sum(-1))
        order = np.argsort(d, axis=None)
        used_a, used_n = set(), set()
        new = np.full((K0, 2), np.nan)
        for flat in order:
            i, j = divmod(int(flat), d.shape[1])
            if d[i, j] > TRACK_M:
                break
            if i in used_a or j in used_n:
                continue
            used_a.add(i)
            used_n.add(j)
            new[a_idx[i]] = nxt[j]
        for i in range(len(a_idx)):
            if i not in used_a:
                alive[a_idx[i]] = False
                died[a_idx[i]] = times[k]
        cur = np.where(np.isnan(new), cur, new)
        out[k] = new
    return times, out, died


def strikes(times, pos):
    """Sample indices where >= STRIKE_MEN men jump; consecutive ones within STRIKE_S are one strike (its first)."""
    v = np.sqrt(((pos[1:] - pos[:-1]) ** 2).sum(-1)) / np.diff(times)[:, None]
    fast = (np.nan_to_num(v) > FAST_MPS).sum(1)
    out = []
    for k in np.nonzero(fast >= STRIKE_MEN)[0]:
        if not out or times[k] - times[out[-1]] > STRIKE_S:
            out.append(int(k))
    return out, v


def per_man(times, pos, died, k0, v):
    """The men's throws in the strike starting at sample k0."""
    t0 = times[k0]
    before = (times >= t0 - 1.0) & (times <= t0)
    base = np.nanmean(pos[before], axis=0)                              # [K, 2]
    win = (times > t0) & (times <= t0 + STRIKE_S)
    kw = np.nonzero(win)[0]
    vv = v[kw[0] - 1:kw[-1]]                                             # speeds into the window's samples
    fast = np.nan_to_num(vv) > FAST_MPS                                  # [w, K]
    present = ~np.isnan(base[:, 0]) & (np.isnan(died) | (died > t0))
    thrown = fast.any(0) & present
    jumpers = np.nonzero(thrown)[0]
    centre = np.nanmean(base[jumpers], axis=0) if len(jumpers) else np.nanmean(base, axis=0)
    rows = []
    for m in np.nonzero(present)[0]:
        r = {"d0": float(np.hypot(*(base[m] - centre))), "thrown": bool(thrown[m]), "died": None}
        if not np.isnan(died[m]) and died[m] <= t0 + STRIKE_S:
            r["died"] = float(died[m] - t0)
        if thrown[m]:
            j = int(np.nonzero(fast[:, m])[0][0])                        # the first fast step
            s = [x for x in range(j + 1, len(kw)) if np.nan_to_num(vv[x, m], nan=0) < STILL_MPS]
            if s:
                k_stop = kw[0] + s[0]
                disp = pos[k_stop, m] - base[m]
                r["throw_m"] = float(np.hypot(*disp)) if np.isfinite(disp).all() else None
                rad = base[m] - centre
                n = np.hypot(*rad)
                if r["throw_m"] and n > 0.3:
                    r["away"] = float(np.dot(disp, rad) / (n * max(r["throw_m"], 1e-6)))
                r["t_jump"] = float(times[kw[0] + j] - t0)
                mv = [x for x in range(s[0] + 1, len(kw)) if np.nan_to_num(vv[x, m], nan=0) > MOVE_MPS]
                if mv:
                    r["lying_s"] = float(times[kw[0] + mv[0]] - times[k_stop])
                    r["lie_from"] = float(times[k_stop] - t0)
        rows.append(r)
    return rows, centre


def lane_report(lane):
    men = lane.get("men") or []
    if len(men) < 20:
        return None
    times, pos, died = track(men)
    ks, v = strikes(times, pos)
    out = {"kind": lane["spec"].get("kind"), "strikes": []}
    for k0 in ks:
        rows, centre = per_man(times, pos, died, k0, v)
        out["strikes"].append({"t": float(times[k0]), "men": rows})
    return out


def summary(rep):
    men = [r for s in rep["strikes"] for r in s["men"]]
    if not men:
        return {}
    thrown = [r for r in men if r["thrown"]]
    q = lambda xs: [round(float(np.percentile(xs, p)), 2) for p in (10, 50, 90)] if xs else None
    reach = max([r["d0"] for r in thrown], default=0.0)
    near = [r for r in men if r["d0"] <= reach]
    # deaths a man-second while lying / while up, after his throw (the men within the reach)
    lie_s = sum(r.get("lying_s", 0.0) for r in thrown)
    die_lying = sum(1 for r in thrown if r["died"] is not None and r.get("lie_from") is not None
                    and r["lie_from"] <= r["died"] <= r["lie_from"] + r.get("lying_s", STRIKE_S))
    up_s = sum(STRIKE_S for r in near) - lie_s
    die_up = sum(1 for r in near if r["died"] is not None) - die_lying
    return {"kind": rep["kind"], "strikes": len(rep["strikes"]), "men": len(men), "thrown": len(thrown),
            "thrown_share": len(thrown) / len(men), "reach_m": reach,
            "thrown_share_by_d": {f"{lo}-{hi}": (sum(1 for r in men if lo <= r["d0"] < hi and r["thrown"])
                                               / max(1, sum(1 for r in men if lo <= r["d0"] < hi)))
                                  for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 9), (9, 15))},
            "throw_m": q([r["throw_m"] for r in thrown if r.get("throw_m") is not None]),
            "away": q([r["away"] for r in thrown if r.get("away") is not None]),
            "lying_s": q([r["lying_s"] for r in thrown if r.get("lying_s") is not None]),
            "deaths_per_s_lying": die_lying / max(lie_s, 1e-9), "deaths_per_s_up": die_up / max(up_s, 1e-9),
            "lying_man_s": lie_s}


def melee_check(lanes):
    """The clanrats' HP a second (the extras' samples) after each strike of the melee lane, the control's at the same
    times."""
    by = {ln["spec"].get("kind"): ln for ln in lanes}
    cast, ctrl = by.get("b6_wl_swords_melee"), by.get("b6_control_melee")
    if not cast or not ctrl or not cast.get("extras") or not ctrl.get("extras"):
        return None
    rep = lane_report(cast)

    def hp_rate(ln, t0, t1):
        t = np.array([x[0] for x in ln["extras"]])
        hp = np.array([float((x[1][0] or {}).get("hp") or np.nan) for x in ln["extras"]])
        a, b = cp.at(t, hp, t0), cp.at(t, hp, t1)
        return (a - b) / (t1 - t0) if np.isfinite(a) and np.isfinite(b) else None
    out = []
    for s in rep["strikes"]:
        t = s["t"]
        thrown = sum(r["thrown"] for r in s["men"])
        out.append({"t": t, "thrown": thrown, "men": len(s["men"]),
                    "cast": [hp_rate(cast, t + a, t + b) for a, b in ((0, 3), (3, 6), (6, 12))],
                    "control": [hp_rate(ctrl, t + a, t + b) for a, b in ((0, 3), (3, 6), (6, 12))]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    lanes = [ln for d in a.runs for ln in cp.load_run(Path(d))]
    reps = [r for r in (lane_report(ln) for ln in lanes) if r]
    res = {"lanes": [summary(r) for r in reps], "melee": melee_check(lanes)}
    for s in res["lanes"]:
        print(json.dumps(s, default=float))
    print("melee:", json.dumps(res["melee"], default=float))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": res, "reports": reps}, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
