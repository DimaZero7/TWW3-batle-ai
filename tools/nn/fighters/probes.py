"""The melee probe's lanes (tools/nn/charge_probe.py) three ways: the game's recording, the simulator's twin (the
rectangle) and the same twin with the lab's men (docs/en/training/fighters_lab.md).

  python -m tools.nn.fighters.probes --plans reform,reform2,dmgmelee,damaged,wave,armour --copies 4 --out build/fighters/probes.json

Per lane: charge_probe.measure's HP and men lost after the first contact (both units) and the men within 1.5 / 2.5 /
3.5 m of an enemy (the game's men and the lab's); per second after contact the shape (shape.py: men within reach,
the front's gap, depth, width, wrap) - the game's and the lab's; the rectangle's depth and width from its box. Needs
torch (the container)."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools import config as project
from tools.nn import charge_probe as cp
from tools.nn.fighters import shape

DMG_KEYS = ("tg_hp_0_1", "tg_hp_0_2", "tg_hp_0_5", "tg_hp_0_10", "tg_hp_5_15", "tg_hp_15_30", "tg_hp_steady",
            "a_hp_0_1", "a_hp_0_5", "a_hp_5_15", "a_hp_15_30", "a_hp_steady",
            "tg_men_0_10", "tg_men_5_15", "tg_men_15_30", "tg_men_steady", "a_men_0_10", "a_men_5_15", "a_men_15_30",
            "a_men_steady")
BINS = ((0, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, 40), (40, 90), (90, 240))
SHAPE_KEYS = ("near_3.15", "near_2.5", "front_gap", "gap", "depth_m", "width_m", "wrap", "contact")


def twin(lanes, fighters, copies=4, device="cpu", seed=0, params=None):
    """charge_probe.sim_lanes on lanes; fighters: with the lab's men (their places once a second in each lane's 'men',
    as the game's probe_men: decimetre lists of the attacker's and the target's living men)."""
    import torch
    from tools.nn.sim import scenario as sim_scenario
    from tools.nn.sim.params import load
    params = (params or load()).with_cal("noise", blows=False)
    if not fighters:
        return cp.sim_lanes(lanes, params=params, device=device, copies=copies, seed=seed)
    from tools.nn.fighters import hook, state as S
    seen = []
    orig_slots = sim_scenario.slots

    def slots(army, per_side):
        out = orig_slots(army, per_side)
        seen.append(out)
        return out
    rec = {"t": [], "pos": []}

    def listen(st, F):
        t = float(st.t.max())
        if rec["t"] and t + 1e-6 < rec["t"][-1] + 1.0:
            return
        px, pz = S.positions(F, st.u)
        live = S.alive(F)
        rec["t"].append(t)
        rec["pos"].append((px.cpu().numpy(), pz.cpu().numpy(), live.cpu().numpy(), F.uid.cpu().numpy()))
    sim_scenario.slots = slots
    try:
        with hook.installed(seed=seed) as h:
            h.listeners.append(listen)
            out = cp.sim_lanes(lanes, params=params, device=device, copies=copies, seed=seed)
    finally:
        sim_scenario.slots = orig_slots
    B = len(out)
    for b, ln in enumerate(out):
        sl = seen[b]
        A, T = sl.get("a", -1), sl.get("tg", -1)
        men = []
        for t, (px, pz, live, uid) in zip(rec["t"], rec["pos"]):
            def flat(n):
                if n < 0:
                    return []
                m = live[b] & (uid[b] == n)
                return np.round(np.stack([px[b][m], pz[b][m]], 1).reshape(-1) * 10).astype(int).tolist()
            men.append((t, flat(A), flat(T)))
        ln["men"] = men
    return out


def _bins(series):
    acc = defaultdict(list)
    for t, r in series:
        for lo, hi in BINS:
            if lo <= t < hi:
                acc[(lo, hi)].append(r)
    out = {}
    for k, rs in acc.items():
        out[f"{k[0]}-{k[1]}"] = {key: float(np.nanmean([r.get(key, np.nan) for r in rs])) for key in SHAPE_KEYS}
    return out


def _rect_shape(lane):
    """The rectangle's depth and width (geometry.dims from the passport: files = floor(width / h), ranks =
    ceil(men / files)) per second after contact, from the twin's samples."""
    sp = lane["spec"]
    out = []
    c = (lane.get("contacts") or {}).get(1)
    if c is None:
        return out
    for who, key, width in (("a", sp["a_key"], sp.get("a_width")), ("tg", sp["t_key"], sp.get("t_width"))):
        pas = cp.passport(key) or {}
        h, v = (pas.get("spacing") or {}).get("h", 1.5), (pas.get("spacing") or {}).get("v", 1.5)
        files = max(1, int(math.floor((width or 30) / h + 1e-4)))
        for t, *rows in lane["samples"]:
            row = rows[0] if who == "a" else rows[1]
            if row is None or t < c:
                continue
            men = max(1.0, float(row["men"]))
            out.append((who, t - c, {"depth_m": math.ceil(men / min(files, men)) * v, "width_m": min(files, men) * h}))
    return out


ROUT_BINS = ((0, 3), (3, 7), (7, 15), (15, 25), (25, 45))


def router_series(lane, rect=False):
    """[(s since the rout, {along, across})] of the lane's target once it routs (its phase 'rout' / 'rout_forced'):
    the mob's stretch along its flight and across it (shape.router_second, the places 3 s on giving the flight); rect:
    the box's (p90 - p10 of men spread evenly over it: 0.8 of its depth and front)."""
    rout = [p["t"] / 1000 for p in lane.get("phases", []) if p["phase"] in ("rout", "rout_forced")]
    if not rout:
        return []
    r0 = rout[0]
    out = []
    if rect:
        sp = lane["spec"]
        pas = cp.passport(sp["t_key"]) or {}
        h, v = (pas.get("spacing") or {}).get("h", 1.5), (pas.get("spacing") or {}).get("v", 1.5)
        files0 = max(1, int(math.floor((sp.get("t_width") or 30) / h + 1e-4)))
        for t, _, row, *rest in lane["samples"]:
            if row is None or t < r0 or not row.get("r"):
                continue
            men = max(1.0, float(row["men"]))
            f = min(files0, men)
            out.append((t - r0, {"along": 0.8 * math.ceil(men / f) * v, "across": 0.8 * f * h}))
        return out
    men = lane.get("men") or []
    ts = np.array([m[0] for m in men])
    for i, (t, fa, ft) in enumerate(men):
        if t < r0:
            continue
        j = int(np.searchsorted(ts, t + 3.0))
        if j >= len(ts):
            break
        r = shape.router_second(shape.xy(ft), shape.xy(men[j][2]))
        if r:
            out.append((t - r0, r))
    return out


def router_table(lanes_by_src):
    """{bin: {src: (along, across, n)}} medians over lanes' seconds."""
    out = {}
    for src, series in lanes_by_src.items():
        for lo, hi in ROUT_BINS:
            v = [r for ser in series for t, r in ser if lo <= t < hi]
            if v:
                out.setdefault(f"{lo}-{hi}", {})[src] = (float(np.median([r["along"] for r in v])),
                                                          float(np.median([r["across"] for r in v])), len(v))
    return out


def routers(run_dirs, copies=4, device="cpu", seed=0):
    """The routing mob's stretch in the game, the rectangle and the lab: the routmob / routmob2 / passby lanes."""
    lanes = [ln for d in run_dirs for ln in cp.load_run(d)]
    lanes = [ln for ln in lanes if (ln.get("contacts") or {}).get(1) is not None
             and not (ln["spec"].get("at_rout") == "away" or ln["spec"].get("rally_friends"))]
    rect = twin(lanes, False, copies, device, seed)
    figh = twin(lanes, True, copies, device, seed)
    return router_table({"game": [router_series(ln) for ln in lanes], "fighters": [router_series(x) for x in figh],
                         "rect": [router_series(x, rect=True) for x in rect]})


def compare(run_dirs, copies=4, device="cpu", seed=0):
    lanes = [ln for d in run_dirs for ln in cp.load_run(d)]
    lanes = [ln for ln in lanes if (ln.get("contacts") or {}).get(1) is not None]
    game = [cp.measure(ln) for ln in lanes]
    rect = twin(lanes, False, copies, device, seed)
    figh = twin(lanes, True, copies, device, seed)
    rows = []
    for i, ln in enumerate(lanes):
        r = {"run": ln["run"], "lane": ln["spec"]["name"], "plan": ln["plan"], "cell": game[i]["cell"], "game": {},
             "rect": {}, "fighters": {}}
        rs = [cp.measure(x) for x in rect[i * copies:(i + 1) * copies]]
        fs = [cp.measure(x) for x in figh[i * copies:(i + 1) * copies]]
        for k in DMG_KEYS:
            g = game[i].get(k)
            r["game"][k] = g if g is not None and np.isfinite(g) else None
            for name, src in (("rect", rs), ("fighters", fs)):
                v = [x[k] for x in src if x.get(k) is not None and np.isfinite(x[k])]
                r[name][k] = float(np.mean(v)) if v else None
        for name, src in (("game", [game[i]]), ("fighters", fs)):
            for k in cp.NEAR_KEYS:
                v = [x[k] for x in src if x.get(k)]
                if v:
                    r[name][k] = [float(a) for a in np.mean(v, axis=0)]
        # the shape per second, both units
        r["shape"] = {}
        for who, oth in (("a", "tg"), ("tg", "a")):
            r["shape"][f"game_{who}"] = _bins(shape.lane_series(ln, who, oth))
            r["shape"][f"fighters_{who}"] = _bins([p for x in figh[i * copies:(i + 1) * copies]
                                                    for p in shape.lane_series(x, who, oth)])
            r["shape"][f"rect_{who}"] = _bins([(t, s) for x in rect[i * copies:(i + 1) * copies]
                                                for w, t, s in _rect_shape(x) if w == who])
        rows.append(r)
    return rows


def score(rows, keys=DMG_KEYS):
    """Mean |log(sim / game)| per key over lanes with both, for the rectangle and the lab."""
    out = {}
    for k in keys:
        for name in ("rect", "fighters"):
            v = [abs(math.log(r[name][k] / r["game"][k])) for r in rows
                 if (r["game"].get(k) or 0) > 1 and (r[name].get(k) or 0) > 0]
            if v:
                out.setdefault(k, {})[name] = (float(np.mean(v)), len(v))
        ratio = {}
        for name in ("rect", "fighters"):
            g = sum(r["game"][k] for r in rows if (r["game"].get(k) or 0) > 1 and r[name].get(k) is not None)
            s = sum(r[name][k] for r in rows if (r["game"].get(k) or 0) > 1 and r[name].get(k) is not None)
            if g > 0:
                ratio[name] = s / g
        if ratio:
            out.setdefault(k, {})["sum_ratio"] = ratio
    return out


def shape_table(rows):
    """Lane-mean shape per time bin: game / fighters / rect."""
    out = {}
    for lo, hi in BINS:
        b = f"{lo}-{hi}"
        for key in SHAPE_KEYS:
            for src in ("game", "fighters", "rect"):
                v = [r["shape"][f"{src}_{w}"][b][key] for r in rows for w in ("a", "tg")
                     if b in r["shape"].get(f"{src}_{w}", {}) and np.isfinite(r["shape"][f"{src}_{w}"][b].get(key, np.nan))]
                if v:
                    out.setdefault(b, {}).setdefault(key, {})[src] = float(np.mean(v))
    return out


MISSILE_TIMES = (10, 20, 40, 80, 120)


def _missile_row(lane):
    """A missile probe lane's target: HP lost, men lost, men per HP (men's worth), the men left (and its HP) at
    MISSILE_TIMES s after the first volley."""
    from tools.nn import missile_probe as mp
    s = lane["samples"]
    t = np.array([x[0] for x in s])
    hp, men, ammo = mp.series(s, "tg", "hp"), mp.series(s, "tg", "men"), mp.series(s, "s", "ammo")
    vs = mp.volleys(t, ammo, mp.series(s, "s", "men"))
    ok = np.isfinite(hp) & np.isfinite(men)
    if not vs or ok.sum() < 3:
        return {}
    men0 = float(men[ok][0])
    hp_man = float((cp.passport(lane["spec"]["t_key"]) or {}).get("hp_per_man") or 1.0)
    out = {"hp_lost": float(hp[ok][0] - hp[ok][-1]), "men_lost": float(men[ok][0] - men[ok][-1]), "men0": men0}
    if men0 > 1 and out["hp_lost"] > 0:
        out["eta"] = out["men_lost"] / (out["hp_lost"] / hp_man)        # men lost per HP lost in men's worth
    t1 = vs[0][0]
    for dt in MISSILE_TIMES:
        if t1 + dt <= t[ok][-1]:
            out[f"men_{dt}"] = float(np.interp(t1 + dt, t[ok], men[ok]))
            out[f"hp_{dt}"] = float(np.interp(t1 + dt, t[ok], hp[ok]))
    return out


def missile_compare(run_dirs, copies=4, device="cpu", seed=0):
    """The missile probe's lanes (tools/nn/missile_probe.py) three ways: the target's men and health (module rows)."""
    from tools.nn import missile_probe as mp
    from tools.nn.sim.params import load
    lanes = [ln for d in run_dirs for ln in mp.load_run(d)]
    params = load().with_cal("noise", blows=False)
    rect = mp.sim_lanes(lanes, params=params, device=device, copies=copies, seed=seed)
    from tools.nn.fighters import hook
    with hook.installed(seed=seed) as h:
        figh = mp.sim_lanes(lanes, params=params, device=device, copies=copies, seed=seed)
        traced = (h.stats["traced"], h.stats["wanted"])
    rows = []
    for i, ln in enumerate(lanes):
        r = {"run": ln["run"], "plan": ln["plan"], "cell": mp.cell(ln["spec"]), "game": _missile_row(ln)}
        for name, src in (("rect", rect), ("fighters", figh)):
            xs = [_missile_row(x) for x in src[i * copies:(i + 1) * copies]]
            keys = set(k for x in xs for k in x)
            r[name] = {k: float(np.mean([x[k] for x in xs if k in x])) for k in keys}
        rows.append(r)
    return rows, traced


def missile_score(rows):
    out = {}
    keys = ["eta", "hp_lost", "men_lost"] + [f"men_{dt}" for dt in MISSILE_TIMES] + [f"hp_{dt}" for dt in MISSILE_TIMES]
    for k in keys:
        for name in ("rect", "fighters"):
            pairs = [(r["game"][k], r[name][k]) for r in rows if r["game"].get(k) and r[name].get(k) is not None
                     and r["game"][k] > 0 and r[name][k] > 0]
            if pairs:
                g = sum(p[0] for p in pairs)
                s = sum(p[1] for p in pairs)
                out.setdefault(k, {})[name] = {"log": float(np.mean([abs(math.log(b / a)) for a, b in pairs])),
                                               "ratio": s / g, "n": len(pairs)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--missile", nargs="*", default=None, help="missile probe run folders (instead of the melee probe)")
    ap.add_argument("--plans", default="reform,reform2,dmgmelee,damaged,wave,armour")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(project.BUILD / "fighters" / "probes.json"))
    ap.add_argument("--routers", action="store_true", help="the routing mob's stretch (plans routmob, routmob2, passby)")
    a = ap.parse_args(argv)
    if a.routers:
        plans = set(a.plans.split(","))
        dirs = [d for d in cp.runs() if json.loads((d / "manifest.json").read_text(encoding="utf-8"))["config"]
                .get("plan") in plans]
        tab = routers(dirs, a.copies, a.device)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(tab, indent=1, default=float), encoding="utf-8")
        print("the routing mob, s after the rout: along / across the flight (m, p90 - p10), seconds")
        for b, d in tab.items():
            print(f"  {b:6} " + "  ".join(f"{s} {v[0]:.1f} / {v[1]:.1f} ({v[2]})" for s, v in d.items()))
        return
    if a.missile is not None:
        dirs = [Path(x) for x in a.missile]
        rows, traced = missile_compare(dirs, a.copies, a.device)
        sc = missile_score(rows)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"lanes": rows, "score": sc, "traced": traced}, indent=1, default=float),
                       encoding="utf-8")
        print(f"{len(rows)} lanes; projectiles traced {traced[0]:.0f} of {traced[1]:.0f}")
        print("the target: mean |log(sim/game)| over lanes and sum sim/game")
        for k, v in sc.items():
            print(f"  {k:10} " + "  ".join(f"{nm} {v[nm]['log']:.3f} / {v[nm]['ratio']:.2f} (n {v[nm]['n']})"
                                           for nm in ("rect", "fighters") if nm in v))
        for r in rows:
            print(f"  {r['cell'][:40]:40} eta g {r['game'].get('eta', float('nan')):.2f} r "
                  f"{r['rect'].get('eta', float('nan')):.2f} f {r['fighters'].get('eta', float('nan')):.2f} | men_40 g "
                  f"{r['game'].get('men_40', float('nan')):.0f} r {r['rect'].get('men_40', float('nan')):.0f} f "
                  f"{r['fighters'].get('men_40', float('nan')):.0f} | hp_40 g {r['game'].get('hp_40', float('nan')):.2f} r "
                  f"{r['rect'].get('hp_40', float('nan')):.2f} f {r['fighters'].get('hp_40', float('nan')):.2f}")
        print(out)
        return
    plans = set(a.plans.split(","))
    dirs = [Path(d) for d in a.runs] if a.runs else [d for d in cp.runs() if
                                                       json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                                                       ["config"].get("plan") in plans]
    rows = compare(dirs, a.copies, a.device)
    sc = score(rows)
    sh = shape_table(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lanes": rows, "score": sc, "shape": sh}, indent=1, default=float), encoding="utf-8")
    print(f"{len(rows)} lanes of {len(dirs)} runs")
    print("HP / men lost after contact: mean |log(sim/game)| (lanes) and sum sim/game")
    nan = (float("nan"), 0)
    for k, v in sc.items():
        print(f"  {k:14} rect {v.get('rect', nan)[0]:.3f}  fighters {v.get('fighters', nan)[0]:.3f}  "
              f"sum ratio rect {v.get('sum_ratio', {}).get('rect', float('nan')):.2f} "
              f"fighters {v.get('sum_ratio', {}).get('fighters', float('nan')):.2f}  n {v.get('rect', nan)[1]}")
    print("shape per bin (game / fighters / rect):")
    for b, d in sh.items():
        print("  " + b + ": " + "; ".join(f"{k} " + "/".join(f"{d[k].get(s, float('nan')):.2f}" for s in
                                                             ("game", "fighters", "rect")) for k in d))
    print(out)


if __name__ == "__main__":
    main()
