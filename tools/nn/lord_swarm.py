"""The lord swarm probe (src/entries/lord_swarm.lua): a lord standing, attacked by 1-4 infantry
units from different sides, and with an armour-piercing unit among them (user, 01.10.2026:
damage is dealt by the soldiers that reach and fight, so surrounding a lord may gain little).

Two lanes 600 m apart on the arena's flat map: the Empire General (ours) attacked by Skaven
clanrat spearmen and Stormvermin halberds (AP), the Skaven Warlord attacked by Empire spearmen
and halberdiers (AP). Every trial runs in both lanes at once; spear units rotate so each fights
few trials. Both sides are held by script, fearless.

    python -m tools.build lord-swarm                 # writes scenarios/lord_swarm.xml, builds
    python -m tools.nn.lord_swarm                    # analyse build/lord-swarm/runs/* (table)

Analysis: per trial and lane, the lord's HP lost per second in the steady window (from
STEADY_FROM_S after his first contact to the trial's end) and in the first STEADY_FROM_S s,
the attackers' soldiers within each radius of him (mean over the steady window), each
attacking unit's damage dealt and men lost. Writes build/lord-swarm/analysis.json (not in Git).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools import config as project
from tools.nn import scenario as nn_scenario

SCENARIO = project.SCENARIOS / "lord_swarm.xml"
RUNS = project.BUILD / "lord-swarm" / "runs"
OUT = project.BUILD / "lord-swarm" / "analysis.json"
GENERAL, WARLORD = "wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0"
SPEAR, HALBERD = "wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_halberdiers"
CLANRAT, STORM = "wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_stormvermin_0"
N_SPEARS, N_AP = 8, 2
WIDTH_M = 30
SPACING_M = 1.5
STEADY_FROM_S = 15.0       # the charge fades over 13 s (charge_decay_duration)
RADII = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
# Trials: the sides of the lord the attackers come from; "ap" is the lane's armour-piercing unit.
LAYOUTS = [
    ("s1", [("front", "spear")]),
    ("s2", [("front", "spear"), ("back", "spear")]),
    ("s2_adjacent", [("front", "spear"), ("left", "spear")]),
    ("s3", [("front", "spear"), ("left", "spear"), ("right", "spear")]),
    ("s4", [("front", "spear"), ("back", "spear"), ("left", "spear"), ("right", "spear")]),
    ("ap1", [("front", "ap")]),
    ("s1_ap1", [("front", "spear"), ("back", "ap")]),
    ("s3_ap1", [("front", "spear"), ("left", "spear"), ("right", "spear"), ("back", "ap")]),
]
# The other side's lord attacks (alone or with units): one lane at a time, the rival leaves his own.
LORD_LAYOUTS = [
    ("lord", [("front", "lord")]),
    ("lord_s1", [("front", "lord"), ("back", "spear")]),
    ("lord_s3", [("front", "lord"), ("back", "spear"), ("left", "spear"), ("right", "spear")]),
    ("lord_ap1", [("front", "lord"), ("back", "ap")]),
]
PLANS = ("infantry", "lords", "all")


def arena():
    """The probe's armies in the shape of a named arena (tools/nn/scenario.py)."""
    base = nn_scenario.load_arena()

    def army(faction, lord, spear, spear_men, ap, ap_men):
        units = [{"slot": "lord", "key": lord, "men": 1, "general": True, "forward": -60, "lateral": 0, "width": 5}]
        units += [{"slot": f"spear_{i}", "key": spear, "men": spear_men, "forward": 0, "lateral": 36 * (i - 4.5),
                   "width": WIDTH_M} for i in range(1, N_SPEARS + 1)]
        units += [{"slot": f"ap_{i}", "key": ap, "men": ap_men, "forward": -40, "lateral": 36 * (i - 1.5),
                   "width": WIDTH_M} for i in range(1, N_AP + 1)]
        return {"faction": faction, "units": units}
    out = {k: v for k, v in base.items() if k not in ("faction", "units", "description")}
    out.update(name="lord_swarm", gap_m=400, defend_radius_m=150,
               sides={"own": army("wh_main_emp_empire", GENERAL, SPEAR, 120, HALBERD, 120),
                      "enemy": army("wh2_main_skv_skaven", WARLORD, CLANRAT, 160, STORM, 160)})
    return out


def depth(men, width=WIDTH_M, spacing=SPACING_M):
    files = max(1, round(width / spacing))
    return -(-men // files) * spacing


def write_scenario(path=SCENARIO):
    a = arena()
    path.write_text(nn_scenario.scenario_xml(a, "enemy", 3600), encoding="utf-8", newline="\n")
    return a


def run_config(repeats=2, fight_s=40, max_s=90, plan="infantry"):
    """The entry's config and the model seconds it takes at most. plan: the infantry layouts
    (both lanes at once), the rival-lord layouts (one lane at a time) or both."""
    assert plan in PLANS
    a = arena()
    widths, depths = {}, {}
    for side in nn_scenario.SIDES:
        for u in a["sides"][side]["units"]:
            name = f"{side}_{u['slot']}"
            widths[name] = u["width"]
            depths[name] = depth(u["men"]) if u["men"] > 1 else 0
    names = lambda side, kind, n: [f"{side}_{kind}_{i}" for i in range(1, n + 1)]
    lanes = [{"name": "general", "lord": "own_lord", "rival": "enemy_lord", "x": -300, "z": 0, "bearing": 0,
              "spears": names("enemy", "spear", N_SPEARS), "ap": names("enemy", "ap", N_AP),
              "park": {"x": -700, "z": 600, "bearing": 0}},
             {"name": "warlord", "lord": "enemy_lord", "rival": "own_lord", "x": 300, "z": 0, "bearing": 0,
              "spears": names("own", "spear", N_SPEARS), "ap": names("own", "ap", N_AP),
              "park": {"x": 250, "z": -600, "bearing": 180}}]
    trials = []
    for r in range(repeats):
        if plan in ("infantry", "all"):
            order = LAYOUTS if r % 2 == 0 else LAYOUTS[::-1]
            trials += [{"name": name, "attackers": [{"side": s, "kind": k} for s, k in places]}
                       for name, places in order]
        if plan in ("lords", "all"):
            for lane in ("warlord", "general") if r % 2 == 0 else ("general", "warlord"):
                trials += [{"name": name, "lanes": [lane], "attackers": [{"side": s, "kind": k} for s, k in places]}
                           for name, places in LORD_LAYOUTS]
    settle_ms = 3000
    config = {"start_m": 20, "settle_ms": settle_ms, "fight_s": fight_s, "max_s": max_s, "min_hp": 0.3,
              "soldier_ms": 1000, "radii": RADII, "near_m": 6.0, "lanes": lanes, "trials": trials,
              "widths": widths, "depths": depths}
    model_s = len(trials) * (max_s + 2 * settle_ms / 1000) + 30
    return config, model_s


# ---------------------------------------------------------------- analysis

def events(run_dir):
    rows = []
    for line in (Path(run_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if '"swarm_' in line or '"event":"result"' in line:
            rows.append(json.loads(line))
    return rows


def trials_of(run_dir):
    """[{trial, name, lane, attackers, samples [(t, lord, att)], men [(t, att counts)], end}]."""
    out = {}
    for r in events(run_dir):
        ev = r["event"]
        if ev == "swarm_trial":
            for lane in r["lanes"]:
                out[(r["trial"], lane["lane"])] = {"run": Path(run_dir).name, "trial": r["trial"], "name": r["name"],
                                                   "lane": lane["lane"], "attackers": lane["attackers"],
                                                   "samples": [], "men": [], "end": None}
        elif ev == "swarm_sample":
            for lane in r["lanes"]:
                k = (r["trial"], lane["lane"])
                if k in out:
                    out[k]["samples"].append(lane)
        elif ev == "swarm_men":
            for lane in r["lanes"]:
                k = (r["trial"], lane["lane"])
                if k in out:
                    out[k]["men"].append(lane)
        elif ev == "swarm_lane_end":
            k = (r["trial"], r["lane"])
            if k in out:
                out[k]["end"] = r
    return list(out.values())


def measure(tr):
    """One lane of one trial: the lord's loss and who reached him."""
    s = tr["samples"]
    if len(s) < 5:
        return None
    t = np.array([x["t"] / 1000 for x in s])
    hp = np.array([x["lord"].get("hp") if x["lord"].get("hp") is not None else np.nan for x in s], dtype=float)
    melee = np.array([bool(x["lord"].get("m")) for x in s])
    if not melee.any():
        return None
    c = int(np.argmax(melee))
    t0 = t[c]
    steady = t >= t0 + STEADY_FROM_S
    first = (t >= t0) & (t < t0 + STEADY_FROM_S)
    out = {"run": tr["run"], "trial": tr["trial"], "name": tr["name"], "lane": tr["lane"],
           "n_units": len(tr["attackers"]), "kinds": [a["kind"] for a in tr["attackers"]],
           "contact_s": round(float(t0), 1), "end": tr["end"] and tr["end"].get("why")}

    def rate(mask):
        idx = np.nonzero(mask & np.isfinite(hp))[0]
        if len(idx) < 2 or t[idx[-1]] - t[idx[0]] < 3:
            return None
        return float((hp[idx[0]] - hp[idx[-1]]) / (t[idx[-1]] - t[idx[0]]))
    out["lord_hp_per_s"] = rate(steady)
    out["lord_hp_per_s_first"] = rate(first)
    out["steady_s"] = float(t[steady][-1] - t[steady][0]) if steady.sum() > 1 else 0.0
    # Per attacking unit: damage dealt (CCO DamageDealt, cumulative), men lost, in melee share.
    units = []
    for j, a in enumerate(tr["attackers"]):
        def series(key, j=j):
            return np.array([(x["att"][j].get(key) if j < len(x["att"]) and x["att"][j].get(key) is not None
                              else np.nan) for x in s], dtype=float)
        dd, men = series("dd"), series("men")
        m = np.array([bool(x["att"][j].get("m")) if j < len(x["att"]) else False for x in s])
        idx = np.nonzero(steady & np.isfinite(dd))[0]
        u = {"name": a["name"], "kind": a["kind"], "side": a["side"], "men0": float(men[0]),
             "in_melee": float(m[steady].mean()) if steady.any() else None}
        if len(idx) >= 2 and t[idx[-1]] > t[idx[0]]:
            u["damage_per_s"] = float((dd[idx[-1]] - dd[idx[0]]) / (t[idx[-1]] - t[idx[0]]))
        idx = np.nonzero(steady & np.isfinite(men))[0]
        if len(idx) >= 2 and t[idx[-1]] > t[idx[0]]:
            u["men_lost_per_s"] = float((men[idx[0]] - men[idx[-1]]) / (t[idx[-1]] - t[idx[0]]))
        units.append(u)
    # Soldiers near the lord (mean over the steady window), per unit and in all.
    near = [x for x in tr["men"] if x["t"] / 1000 >= t0 + STEADY_FROM_S]
    if near:
        per = np.zeros((len(near), len(tr["attackers"]), len(RADII)))
        for i, x in enumerate(near):
            by = {a["n"]: a["c"] for a in x["att"]}
            for j, a in enumerate(tr["attackers"]):
                if a["name"] in by:
                    per[i, j] = by[a["name"]][:len(RADII)]
        mean = per.mean(axis=0)
        for j, u in enumerate(units):
            u["near"] = dict(zip(map(str, RADII), [round(float(v), 2) for v in mean[j]]))
        out["near_total"] = dict(zip(map(str, RADII), [round(float(v), 2) for v in mean.sum(axis=0)]))
    out["units"] = units
    return out


def summary(rows):
    """Mean over the repeats per lane and layout."""
    groups = {}
    for r in rows:
        groups.setdefault((r["lane"], r["name"]), []).append(r)
    out = []
    for (lane, name), rs in sorted(groups.items()):
        def m(key, rs=rs):
            v = [r[key] for r in rs if r.get(key) is not None]
            return (float(np.mean(v)), float(np.min(v)), float(np.max(v))) if v else (None, None, None)
        near = {}
        for rad in map(str, RADII):
            v = [r["near_total"][rad] for r in rs if r.get("near_total")]
            near[rad] = round(float(np.mean(v)), 2) if v else None
        share = []
        for r in rs:
            dd = [u.get("damage_per_s") or 0.0 for u in r["units"]]
            tot = sum(dd)
            if tot > 0:
                share.append(sorted((x / tot for x in dd), reverse=True))
        out.append({"lane": lane, "layout": name, "repeats": len(rs), "n_units": rs[0]["n_units"],
                    "kinds": rs[0]["kinds"], "lord_hp_per_s": m("lord_hp_per_s"),
                    "lord_hp_per_s_first": m("lord_hp_per_s_first"), "near_total": near,
                    "damage_share_sorted": [round(float(x), 2) for x in np.mean(share, axis=0)] if share else None,
                    "men_near_per_unit": [round(float(np.mean([u["near"]["2.5"] for r in rs for u in r["units"]
                                                               if u.get("near") and u["side"] == side])), 2)
                                          for side in ("front", "back", "left", "right")
                                          if any(u["side"] == side and u.get("near") for r in rs for u in r["units"])]})
    return out


# ---------------------------------------------------------------- the simulator on the same trials

def sim_army(tr, keys):
    """A trial's lane as a simulator army: the lord (side 1) and his attackers (side 2) at their
    places of the trial's first sample; nobody uses abilities (both sides were held by script)."""
    s0 = tr["samples"][0]
    lord = s0["lord"]
    lord_key, lord_faction, att_faction = ((GENERAL, "wh_main_emp_empire", "wh2_main_skv_skaven") if tr["lane"] == "general"
                                           else (WARLORD, "wh2_main_skv_skaven", "wh_main_emp_empire"))
    att = []
    for a, row in zip(tr["attackers"], s0["att"]):
        lord_att = a["kind"] == "lord"
        att.append({"key": keys[a["name"]], "x": row["x"], "z": row["z"], "b": row["b"],
                    "width": None if lord_att else WIDTH_M, "general": lord_att, "men": row["men"], "name": a["name"]})
    return {"attacker": 2, "sides": {
        1: {"faction": lord_faction, "ai": False, "units": [{"key": lord_key, "x": lord["x"], "z": lord["z"],
                                                               "b": lord["b"], "general": True, "name": "lord"}]},
        2: {"faction": att_faction, "ai": False, "units": att}}}


def simulate(trials, params=None, device="cpu", until_s=None):
    """The simulator on the recorded trials: every attacker attacks the lord (run), the lord holds;
    fearless (leadership out of reach). Returns per trial {lord_hp_per_s, lord_hp_per_s_first,
    contact_s} measured as in the game (measure)."""
    import torch
    from tools.nn.sim import battle, orders as O, scenario as sim_scenario
    from tools.nn.sim.params import load
    params = params or load()
    keys = {}
    for side in nn_scenario.SIDES:
        for u in arena()["sides"][side]["units"]:
            keys[f"{side}_{u['slot']}"] = u["key"]
    armies = [sim_army(tr, keys) for tr in trials]
    H = max(len(a["sides"][2]["units"]) for a in armies)
    st = sim_scenario.build(armies, params, device=device, per_side=H)
    u = st.u
    u["leadership"] = u["leadership"] + 1e4
    u["morale"] = u["morale"] + 1e4

    def policy(st):
        o = O.hold(st.B, st.N, st.device)
        att = (st.u["side"] == 2) & (st.u["men"] > 0)
        o.kind = torch.where(att, O.ATTACK, O.HOLD)
        o.target = torch.where(att, torch.zeros_like(o.target), o.target)
        o.run = att.clone()
        return o
    rec_t, rec_hp, rec_m = [], [], []

    def record(st):
        rec_t.append(float(st.t.max()))
        rec_hp.append(st.u["hp_abs"][:, 0].detach().cpu().numpy().copy())
        rec_m.append(st.u["m"][:, 0].detach().cpu().numpy().copy())
    battle.run(st, policy, params, until_s=until_s or 100.0, record=record)
    t = np.array(rec_t)
    hp, m = np.stack(rec_hp), np.stack(rec_m)
    out = []
    for b, tr in enumerate(trials):
        fight = (tr["end"] or {}).get("t", 0) / 1000 - ((tr["end"] or {}).get("contact_ms") or 0) / 1000
        samples = [{"t": int(1000 * tk), "lord": {"hp": float(hp[k, b]), "m": bool(m[k, b])}, "att": []}
                   for k, tk in enumerate(t)]
        if m[:, b].any():
            c = int(np.argmax(m[:, b]))
            samples = [x for x in samples if x["t"] / 1000 <= t[c] + max(fight, STEADY_FROM_S + 5)]
        r = measure({"run": "sim", "trial": tr["trial"], "name": tr["name"], "lane": tr["lane"],
                     "attackers": [], "samples": samples, "men": [], "end": None})
        out.append(r)
    return out


def compare(run_dirs=None, params=None, device="cpu"):
    """Game against simulator per lane and layout: lord HP/s steady and in the first 15 s."""
    trials = [tr for d in (run_dirs or runs()) for tr in trials_of(d)]
    trials = [tr for tr in trials if measure(tr)]
    game = [measure(tr) for tr in trials]
    sim = simulate(trials, params, device)
    groups = {}
    for g, s in zip(game, sim):
        groups.setdefault((g["lane"], g["name"]), []).append((g, s))
    rows = []
    for (lane, name), items in sorted(groups.items()):
        row = {"lane": lane, "layout": name, "n": len(items)}
        for key in ("lord_hp_per_s", "lord_hp_per_s_first"):
            gv = [g[key] for g, _ in items if g.get(key) is not None]
            sv = [s[key] for _, s in items if s and s.get(key) is not None]
            row[f"game_{key}"] = float(np.mean(gv)) if gv else None
            row[f"sim_{key}"] = float(np.mean(sv)) if sv else None
        rows.append(row)
    return rows


ORDER = ("s1", "s2", "s2_adjacent", "s3", "s4", "ap1", "s1_ap1", "s3_ap1", "lord", "lord_s1", "lord_s3", "lord_ap1")
LANES = {"general": "Empire General attacked by clanrats / Stormvermin halberds",
         "warlord": "Skaven Warlord attacked by Empire spearmen / halberdiers"}


def plot(table, compare_rows, path):
    """The lord's HP/s by layout (bars: the game's mean, whiskers: min-max; dots: the simulator)
    and the attackers' soldiers within 2.5 m of him (numbers over the bars)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sim = {(r["lane"], r["layout"]): r["sim_lord_hp_per_s"] for r in compare_rows or []}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, lane in zip(axes, LANES):
        rows = {r["layout"]: r for r in table if r["lane"] == lane}
        names = [n for n in ORDER if n in rows]
        x = range(len(names))
        mean = [rows[n]["lord_hp_per_s"][0] for n in names]
        lo = [rows[n]["lord_hp_per_s"][0] - rows[n]["lord_hp_per_s"][1] for n in names]
        hi = [rows[n]["lord_hp_per_s"][2] - rows[n]["lord_hp_per_s"][0] for n in names]
        colour = ["#4C78A8" if not n.startswith("lord") and "ap" not in n else
                  ("#E45756" if not n.startswith("lord") else "#B279A2") for n in names]
        ax.bar(x, mean, yerr=[lo, hi], color=colour, capsize=3, label="game (mean, min-max)")
        if sim:
            ax.plot(list(x), [sim.get((lane, n)) for n in names], "ko", ms=5, label="simulator")
        for k, n in zip(x, names):
            near = rows[n]["near_total"].get("2.5")
            if near is not None:
                ax.text(k, 1, f"{near:.1f}", ha="center", va="bottom", color="white", fontsize=8)
        ax.set_title(LANES[lane], fontsize=10)
        ax.set_ylabel("lord HP lost / s (steady)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[-1].set_xticks(range(len(names)))
    axes[-1].set_xticklabels(names, rotation=30, ha="right")
    fig.text(0.5, 0.005, "s1-s4: 1-4 spear units (front, back, left, right); ap: a halberd unit; lord: the other lord; "
             "white numbers: attacking soldiers within 2.5 m", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def runs(root=RUNS):
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if (d / "events.jsonl").exists())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run folders (default: every run in build/lord-swarm/runs)")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sim", action="store_true", help="also the simulator on the same trials (needs torch)")
    parser.add_argument("--plot", type=Path, help="draw the table (and the simulator from compare.json) to this png")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.sim:
        rows = compare(args.runs or None)
        print(f"{'lane':8} {'layout':12} {'n':>2} {'game HP/s':>10} {'sim':>7} {'err':>6} {'game first 15 s':>16} {'sim':>7}")
        for r in rows:
            f = lambda v: "-" if v is None else f"{v:.1f}"
            g, s = r["game_lord_hp_per_s"], r["sim_lord_hp_per_s"]
            err = "-" if not g or s is None else f"{100 * (s - g) / g:+.0f}%"
            print(f"{r['lane']:8} {r['layout']:12} {r['n']:>2} {f(g):>10} {f(s):>7} {err:>6} "
                  f"{f(r['game_lord_hp_per_s_first']):>16} {f(r['sim_lord_hp_per_s_first']):>7}")
        out = args.out.with_name("compare.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        return 0
    rows = []
    for d in args.runs or runs():
        rows += [m for m in (measure(tr) for tr in trials_of(d)) if m]
    table = summary(rows)
    if args.plot:
        cmp_path = args.out.with_name("compare.json")
        plot(table, json.loads(cmp_path.read_text(encoding="utf-8")) if cmp_path.exists() else None, args.plot)
    print(f"{'lane':8} {'layout':12} {'n':>2} {'rep':>3} {'HP/s steady':>12} {'(min-max)':>13} {'HP/s first 15 s':>16} "
          f"{'men <=2 / 2.5 / 3 m':>20}  damage share")
    for r in table:
        hp, hp1 = r["lord_hp_per_s"], r["lord_hp_per_s_first"]
        f = lambda v: "-" if v is None else f"{v:.1f}"
        nt = r["near_total"]
        print(f"{r['lane']:8} {r['layout']:12} {r['n_units']:>2} {r['repeats']:>3} {f(hp[0]):>12} "
              f"{f(hp[1]) + '-' + f(hp[2]):>13} {f(hp1[0]):>16} "
              f"{f(nt['2.0']) + ' / ' + f(nt['2.5']) + ' / ' + f(nt['3.0']):>20}  {r['damage_share_sorted']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"trials": rows, "summary": table}, indent=1), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
