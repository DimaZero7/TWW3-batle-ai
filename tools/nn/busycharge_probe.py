"""The busycharge plan's table (tools/nn/charge_probe.py, entries/charge_probe.lua): a charge into clanrats already
fighting our spearmen (the anvil) from their front / right flank / back, against the same charge into free clanrats.

Per lane (game or the simulator on the same lanes, the same row): the charger's speed on the way in (the last 30 / 10 m,
the peak), the clanrats' HP lost in 0-2 / 0-5 s after the charger's first contact minus what the anvil alone took from
them at the rate of the 3 s before it ('extra': the charge's own share), the charger's kills in 0-5 s, the clanrats'
morale (MoralePercent, 0-1) at the contact and its drop in 0-5 / 0-10 s, their rout (s after the contact), and the
morale effects that came on in the first 5 s (game: the engine's keys; the simulator: its effect names).

    python -m tools.nn.busycharge_probe [runs...] [--sim] [--layout] [--set charge.free_target_only=0]
        runs: build/charge-probe/runs/* of the plan by default; --sim adds the simulator on the recorded lanes,
        --layout the simulator alone on the plan's own places (no game run needed); both need torch (the container).
Writes build/charge-probe/busycharge.json (not in Git).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp

PLAN = "busycharge"
PRE_S = 3.0                   # the anvil's own rate: the clanrats' HP lost a second over the 3 s before the contact
IMPACT_HP = 50.0            # a charge's impact: the target loses more than this in half a second ...
IMPACT_X = 2.5             # ... and over this many times its rate before (the anvil's blows)
REACH_M = 3.0              # ... with the charger within reach (the centres: half depths + this)
COLS = ("speed_last30", "speed_last10", "speed_peak30", "tg_hp_0_2", "tg_hp_0_5", "extra_0_2", "extra_0_5",
        "a_k_0_5", "flag_lag_s", "mp_c", "mp_drop_5", "mp_drop_10", "rout_s")


def _at(t, v, when):
    return cp.at(t, v, when)


def impact(lane):
    """The charge's impact (s on the lane's clock): the first half second after the charger's order in which the target
    loses more than IMPACT_HP and over IMPACT_X times its rate of the PRE_S before (the anvil's own blows), up to the
    charger's melee flag + 1 s; else that flag. (The game's melee flag of a unit that charges a back or a flank comes
    2.5-3 s after its first blows - busycharge runs: the clanrats' health falls from 3 s before it; in front it comes
    with them.)"""
    c = lane["contacts"].get(1)
    if c is None:
        return None
    s = lane["samples"]
    t = np.array([x[0] for x in s])
    hp = cp.series(s, "tg", "hp")
    go = next((p["t"] / 1000 for p in lane.get("phases", []) if p["phase"] == "a_go"), float(t[0]))
    # within reach: the centres closer than the two half depths (a flank: the target's half width and the charger's
    # half depth) + REACH_M
    spec = lane["spec"]
    half = spec["t_width"] / 2 if spec.get("a_side") == 90 else spec["t_depth"] / 2
    reach = half + spec["a_depth"] / 2 + REACH_M
    d = np.hypot(cp.series(s, "a", "x") - cp.series(s, "tg", "x"), cp.series(s, "a", "z") - cp.series(s, "tg", "z"))
    for i in range(1, len(t)):
        if t[i] < go + 1.0 or t[i - 1] > c + 1.0 + 1e-6 or not d[i] <= reach:
            continue
        drop = hp[i - 1] - hp[i]
        dt = t[i] - t[i - 1]
        before = (_at(t, hp, t[i - 1] - PRE_S) - hp[i - 1]) / PRE_S if t[i - 1] - PRE_S >= t[0] else 0.0
        if np.isfinite(drop) and drop > IMPACT_HP * dt / 0.5 and drop > IMPACT_X * max(before, 0.0) * dt:
            return float(t[i - 1])
    return c


def measure(lane):
    """One lane's row (game or simulator): the windows from the charge's impact (impact())."""
    spec = lane["spec"]
    flag = lane["contacts"].get(1)
    hit = impact(lane)
    out = cp.measure(dict(lane, contacts={1: hit}) if hit is not None else lane)
    out.update(kind=spec.get("kind"), charger=cp.SHORT[spec["a_key"]], src="sim" if lane["run"] == "sim" else "game",
               flag_lag_s=None if hit is None else flag - hit)
    c = out.get("contact_s")
    if c is None:
        return out
    s = lane["samples"]
    t = np.array([x[0] for x in s])
    hp = cp.series(s, "tg", "hp")
    pre = (_at(t, hp, c - PRE_S) - _at(t, hp, c)) / PRE_S
    out["pre_rate"] = pre
    for hi in (2, 5):
        # (from the impact itself, game and simulator alike: the impact is the sample before the drop)
        v = out[f"tg_hp_0_{hi}"] = _at(t, hp, c) - _at(t, hp, c + hi)
        if np.isfinite(v) and np.isfinite(pre):
            out[f"extra_0_{hi}"] = v - pre * hi
    k = cp.series(s, "a", "k")
    if np.isfinite(k).sum() > 1:
        out["a_k_0_5"] = _at(t, k, c + 5) - _at(t, k, c - 0.5)
    dd = cp.series(s, "a", "dd")
    if np.isfinite(dd).sum() > 1:
        out["a_dd_0_5"] = _at(t, dd, c + 5) - _at(t, dd, c - 0.5)
    mp = cp.series(s, "tg", "mp")
    if np.isfinite(mp).sum() > 1:
        out["mp_c"] = _at(t, mp, c)
        for hi in (5, 10):
            out[f"mp_drop_{hi}"] = out["mp_c"] - _at(t, mp, c + hi)
    r = cp.series(s, "tg", "r")
    after = np.nonzero((t >= c) & (r > 0.5))[0]
    out["rout_s"] = float(t[after[0]] - c) if len(after) else None
    fx0 = set()
    fx5 = set()
    for tm, _, tg, _ in s:
        keys = set(filter(None, ((tg or {}).get("fx") or "").split(",")))
        if c - 1.0 <= tm <= c:
            fx0 |= keys
        if c < tm <= c + 5:
            fx5 |= keys
    out["fx_new"] = sorted(fx5 - fx0)
    return out


def summary(rows):
    """{(src, charger, kind): {n, col means}}."""
    groups = {}
    for r in rows:
        if r.get("contact_s") is None:
            continue
        groups.setdefault((r["src"], r["charger"], r["kind"]), []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        row = {"n": len(rs)}
        for col in COLS:
            v = [x[col] for x in rs if x.get(col) is not None and np.isfinite(x[col])]
            row[col] = float(np.mean(v)) if v else None
        fx = {}
        for x in rs:
            for f in x.get("fx_new") or []:
                fx[f] = fx.get(f, 0) + 1
        row["fx_new"] = fx
        out[key] = row
    return out


def _params(sets):
    if not sets:
        return None
    from tools.nn.sim.stepreplay import _params as sp
    return sp(sets)


def report(run_dirs, sim=False, layout=False, sets=(), device="cpu", copies=4, out=None):
    lanes = [ln for d in run_dirs for ln in cp.load_run(d) if ln.get("plan") == PLAN]
    rows = [measure(ln) for ln in lanes]
    params = _params(sets)
    if sim and lanes:
        rows += [measure(ln) for ln in cp.sim_lanes([ln for ln in lanes if ln["contacts"].get(1) is not None],
                                                   params=params, device=device, copies=copies)]
    if layout:
        rows += [measure(ln) for ln in cp.sim_lanes(cp.layout_lanes(PLAN), params=params, device=device, copies=copies)]
    table = summary(rows)
    f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
    print("A charge into clanrats busy with our spearmen (busy) or free: v30 / v10 / peak = the charger's m/s on the way "
          "in; hp2 / hp5 = the clanrats' HP lost 0-2 / 0-5 s after its contact, x2 / x5 = less the anvil's own rate "
          "(the charge's share); k5 = its kills 0-5 s, lag = its melee flag after the impact (s); mp = morale at the contact, "
          "d5 / d10 its drop; rout = s after the contact" + (f"; sim with {list(sets)}" if sets else ""))
    print(f"  {'src':4} {'who':8} {'kind':8} {'n':>2} {'v30':>5} {'v10':>5} {'peak':>5} {'hp2':>6} {'hp5':>6} {'x2':>6} "
          f"{'x5':>6} {'k5':>5} {'lag':>6} {'mp':>5} {'d5':>5} {'d10':>5} {'rout':>5}  new effects")
    for (src, who, kind), r in table.items():
        print(f"  {src:4} {who:8} {kind:8} {r['n']:>2} {f(r['speed_last30'], 2):>5} {f(r['speed_last10'], 2):>5} "
              f"{f(r['speed_peak30'], 2):>5} {f(r['tg_hp_0_2'], 0):>6} {f(r['tg_hp_0_5'], 0):>6} {f(r['extra_0_2'], 0):>6} "
              f"{f(r['extra_0_5'], 0):>6} {f(r['a_k_0_5']):>5} {f(r['flag_lag_s'], 1):>6} {f(r['mp_c'], 2):>5} "
              f"{f(r['mp_drop_5'], 2):>5} {f(r['mp_drop_10'], 2):>5} {f(r['rout_s']):>5}  "
              + ",".join(f"{k}:{v}" for k, v in sorted(r["fx_new"].items())))
    result = {"lanes": rows, "summary": [dict(r, source=k[0], charger=k[1], kind=k[2]) for k, r in table.items()]}
    out = out or cp.ROOT / "busycharge.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=lambda v: None if v is None else float(v)), encoding="utf-8")
    print(out)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--layout", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="section.key=json over config/nn/sim.json (the twin)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    run_dirs = a.runs or ([] if a.layout and not a.sim else cp.runs())
    report(run_dirs, sim=a.sim, layout=a.layout, sets=a.set, device=a.device, copies=a.copies, out=a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
