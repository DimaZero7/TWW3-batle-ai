"""The melee probe (src/entries/charge_probe.lua): one melee indicator, one scenario, in the game and in
the simulator alike. Lanes far apart, in each an attacker and a target gap_m apart (front to front),
everyone held by script and fearless; the attacker gets one kind of order (attack at a run or at a
walk, a move order into the target, a recharge), the target stands (answering with an attack order at
contact), holds (braced, never ordered), attacks too, or stands facing away.

Plans (each a few battles of 2-5 lanes; lanes swap places between battles):
  charge  A: swordsmen -> clanrats, the four orders (4 battles); B: clanrats -> braced spearmen
          (charge reflection: run / walk / spears facing away / swordsmen without reflection; 2);
          C: the General and the Warlord charging, and the charge-speed rush: spearmen attacking
          skavenslave spearmen from 150 m and from 20 m (2 battles) - 8 battles;
  hit     formation pairs over a span of attack - defence (swordsmen, greatswords, spearmen with
          shields on unarmoured skavenslaves, flagellants on clanrats) and clanrat spearmen on Empire
          spearmen with the General behind them using Stand Your Ground at contact (battle 1) or not
          (battle 2, the control) - 2 battles;
  move    turning and leaving melee (build/movelords): battle 1 turns - spearmen turn in place 90 deg and
          back, then 180 deg (a facing order); spearmen run to a point 150 m behind them and 120 m to their
          side; the General and the Warlord turn in place 180 and 90 deg, then run to a point behind them
          (soldier places all the way); battle 2 melee exit - swordsmen / spearmen withdraw 10 s after contact
          and never change the order, chased by clanrats ordered to attack them or left standing; the General
          attacks clanrats and uses Foe-Seeker 50 s after contact (vigour) - 2 battles;
  vv      T-E3 of the effects (build/effects/spec.md 5): the Warlord attacks swordsmen and uses Verminous Valour
          20 s after contact (its 25 m blast: are the men around him thrown back, do they stop striking?); the
          soldiers' places all the fight; the control is the charge plan's Warlord lane - 1 battle.

    python -m tools.nn.charge_probe plan [--plan charge|hit]          # the battles
    python -m tools.build charge-probe --probe-plan hit --probe-battle 1   # one battle's build
    python -m tools.nn.charge_probe run --plan hit [--battles 1,2]      # build + launch each, in turn
    python -m tools.nn.charge_probe report [runs...]                    # the game's table
    python -m tools.nn.charge_probe report --sim                        # + the simulator on the same lanes

Measures per lane (the game's recording, and the simulator's run of the same lane from the same start):
the attacker's speed on the way in (the last 30 m, the peak), the first contact; HP lost by the target
and by the attacker in 0-1, 0-2, 0-5, 5-15, 15-30 s after contact and per second from 15 s to the end
(steady); men lost per second steady; for a recharge the same after the second contact; from the
soldiers' places (game only) how many men of each side stand within 1.5 / 2.5 / 3.5 m of an enemy
soldier, per second after contact. Writes build/charge-probe/analysis.json (not in Git).
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools import config as project
from tools.nn import scenario as nn_scenario

ROOT = project.BUILD / "charge-probe"
RUNS = ROOT / "runs"
OUT = ROOT / "analysis.json"
LAUNCHER = project.ROOT / "tools" / "launcher"
UNITS_JSON = project.ROOT / "config" / "nn" / "units.json"
EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"
# short name -> (unit key, men, faction)
UNITS = {
    "swords": ("wh_main_emp_inf_swordsmen", 120, EMP),
    "gs": ("wh_main_emp_inf_greatswords", 120, EMP),
    "spear": ("wh_main_emp_inf_spearmen_0", 120, EMP),
    "spearsh": ("wh_main_emp_inf_spearmen_1", 120, EMP),
    "flag": ("wh_dlc04_emp_inf_flagellants_0", 120, EMP),
    "general": ("wh_main_emp_cha_general_0", 1, EMP),
    "clanrat": ("wh2_main_skv_inf_clanrats_1", 160, SKV),
    "cspear": ("wh2_main_skv_inf_clanrat_spearmen_0", 160, SKV),
    "slave": ("wh2_main_skv_inf_skavenslaves_0", 180, SKV),
    "sspear": ("wh2_main_skv_inf_skavenslave_spearmen_0", 180, SKV),
    "warlord": ("wh2_main_skv_cha_warlord_0", 1, SKV),
}
LORDS = {EMP: "general", SKV: "warlord"}
SHORT = {key: short for short, (key, _, _) in UNITS.items()}
FACTION = {key: faction for key, _, faction in UNITS.values()}
SYG = "wh_main_character_abilities_stand_your_ground"
FOE_SEEKER = "wh_main_character_abilities_foe_seeker"
WIDTH_M = 30
LANE_DX = 240          # lanes this far apart across x (the General's auras reach 35-40 m)
GAP_M = 80
SETTLE_MS = 4000
TICK_MS = 500
MEN_MS = 1000
WINDOWS = ((0, 1), (0, 2), (0, 5), (5, 15), (15, 30))
STEADY_FROM_S = 15.0   # the charge bonus fades over 13 s (charge_decay_duration)
RADII = (1.5, 2.5, 3.5)
NEAR_WINDOWS = ((0, 5), (5, 15), (15, 30), (30, 90))
NEAR_KEYS = tuple(f"{who}_near_{lo}_{hi}" for who in ("a", "tg") for lo, hi in NEAR_WINDOWS)
PLANS = ("charge", "hit", "move", "vv")
VV = "wh2_main_character_abilities_verminous_valour"
TURN_TICK_MS = 250     # the turning battle: 0.25 s samples (a lord turns 180 deg in 1-2 s)


def lane(attacker, target, mode="attack_run", target_mode="stand", gap_m=GAP_M, fight_s=40, **extra):
    """One lane's spec (before it gets a place)."""
    out = {"attacker": attacker, "target": target, "mode": mode, "target_mode": target_mode, "gap_m": gap_m,
           "fight_s": fight_s, "answer": target_mode == "stand"}
    if mode == "recharge":
        out.update(recharge_after_s=10, back_m=40, recharge_max_s=25, fight_s=max(fight_s, 70))
    if mode == "withdraw":
        out.update(recharge_after_s=10, back_m=150, recharge_max_s=10 ** 6)
    out.update(extra)
    return out


def rotate(items, k):
    k %= len(items)
    return items[k:] + items[:k]


def battles(plan):
    """[[lane spec]] of a plan: each inner list one battle."""
    assert plan in PLANS, plan
    out = []
    if plan == "charge":
        a = [lane("swords", "clanrat", m) for m in ("attack_run", "attack_walk", "move_run", "recharge")]
        out += [rotate(a, k) for k in range(4)]
        b = [lane("clanrat", "spear", "attack_run", "hold"), lane("clanrat", "spear", "attack_walk", "hold"),
             lane("clanrat", "spear", "attack_run", "rear"), lane("clanrat", "swords", "attack_run", "hold")]
        out += [rotate(b, k) for k in (0, 2)]
        rush = [lane("spear", "sspear", "attack_run", gap_m=150), lane("spear", "sspear", "attack_run", gap_m=20)]
        out += [[lane("general", "clanrat", "attack_run"), lane("warlord", "swords", "attack_run")] + rush,
                [lane("general", "clanrat", "attack_walk"), lane("warlord", "swords", "recharge")] + rush[::-1]]
    elif plan == "move":
        # bearings: the attacker starts facing 180 (towards -z, its target 250 m away, never ordered)
        far = dict(target_mode="hold", gap_m=250, answer=False)
        spin = [{"at_s": 1, "kind": "face", "bearing": 270}, {"at_s": 16, "kind": "face", "bearing": 180},
                {"at_s": 31, "kind": "face", "bearing": 0}]
        lord_spin = [{"at_s": 1, "kind": "face", "bearing": 0}, {"at_s": 11, "kind": "face", "bearing": 270},
                     {"at_s": 21, "kind": "move", "dx": 100, "dz": 0, "run": True}]
        out.append([
            lane("spear", "clanrat", "script", steps=spin, men_all_s=50, max_s=50, **far),
            lane("spear", "slave", "script", steps=[{"at_s": 1, "kind": "move", "dx": 0, "dz": 150, "run": True}],
                 men_all_s=25, max_s=60, **far),
            lane("spear", "sspear", "script", steps=[{"at_s": 1, "kind": "move", "dx": 120, "dz": 0, "run": True}],
                 men_all_s=25, max_s=55, **far),
            lane("general", "clanrat", "script", steps=lord_spin, max_s=50, **far),
            lane("warlord", "swords", "script", steps=lord_spin, max_s=50, **far)])
        out.append([
            lane("swords", "clanrat", "withdraw", gap_m=40, fight_s=75),
            lane("swords", "clanrat", "withdraw", target_mode="hold", gap_m=40, fight_s=50),
            lane("spear", "clanrat", "withdraw", gap_m=40, fight_s=75),
            lane("general", "clanrat", "attack_run", fight_s=100, a_ability=FOE_SEEKER, a_ability_after_s=50)])
    elif plan == "vv":
        # (a second lane: one Warlord a battle; the swordsmen on clanrats only fill the plan's two-lane frame)
        out.append([lane("warlord", "swords", "attack_run", fight_s=45, a_ability=VV, a_ability_after_s=20),
                    lane("swords", "clanrat", "attack_run", fight_s=45)])
    else:
        pairs = [lane("swords", "slave", gap_m=40, fight_s=90), lane("gs", "slave", gap_m=40, fight_s=90),
                 lane("spearsh", "slave", gap_m=40, fight_s=90), lane("flag", "clanrat", gap_m=40, fight_s=90)]
        for k, ability in enumerate((SYG, "")):
            syg = lane("cspear", "spear", gap_m=40, fight_s=60, lord={"name": "general", "ability": ability, "dz": 8})
            out.append(rotate(pairs, 2 * k) + [syg])
    return out


def spacing(key):
    sp = json.loads(UNITS_JSON.read_text(encoding="utf-8"))["units"][key]["spacing"]
    return float(sp["h"]), float(sp["v"])


def depth(key, men, width=WIDTH_M):
    """The close block's depth, m (the database's spacing); 0 for a lone man."""
    if men <= 1:
        return 0.0
    h, v = spacing(key)
    files = max(1, int(width // h))
    return math.ceil(men / files) * v


def layout(specs):
    """A battle's lanes with places and script names, and the arena (tools/nn/scenario.py shape).
    Every lane's units get their own slots; each side's lord is always there (parked far unless a
    lane uses him)."""
    sides = {EMP: [], SKV: []}
    side_of = {EMP: "own", SKV: "enemy"}
    lanes, used_lords = [], set()

    def add(short, k):
        key, men, faction = UNITS[short]
        if short in LORDS.values():
            used_lords.add(short)
            return f"{side_of[faction]}_lord"
        slot = f"{short}_{k}"
        sides[faction].append({"slot": slot, "key": key, "men": men, "forward": -20 - 40 * (len(sides[faction]) // 4),
                               "lateral": 40 * (len(sides[faction]) % 4 - 1.5), "width": WIDTH_M})
        return f"{side_of[faction]}_{slot}"
    n = len(specs)
    for k, spec in enumerate(specs, 1):
        a_key, a_men, a_fac = UNITS[spec["attacker"]]
        t_key, t_men, t_fac = UNITS[spec["target"]]
        assert a_fac != t_fac, spec
        x = LANE_DX * (k - 1 - (n - 1) / 2)
        row = dict(spec, name=f"L{k}", x=round(x, 1), z=0.0, attacker=add(spec["attacker"], k),
                   target=add(spec["target"], k), a_key=a_key, t_key=t_key,
                   a_depth=round(depth(a_key, a_men), 2), t_depth=round(depth(t_key, t_men), 2),
                   a_width=5 if a_men == 1 else WIDTH_M, t_width=5 if t_men == 1 else WIDTH_M,
                   move_beyond_m=5.0, max_s=spec.get("max_s") or round(spec["gap_m"] / 1.4 + spec["fight_s"] + 40))
        if spec.get("lord"):
            row["lord"] = dict(spec["lord"], name=add(spec["lord"]["name"], k))
        lanes.append(row)
    base = nn_scenario.load_arena()
    armies = {}
    for faction, units in sides.items():
        lord = LORDS[faction]
        key, _, _ = UNITS[lord]
        armies[side_of[faction]] = {"faction": faction, "units": [
            {"slot": "lord", "key": key, "men": 1, "general": True, "forward": -80, "lateral": 0, "width": 5}] + units}
    park = [{"name": f"{side_of[f]}_lord", "x": (-700 if f == EMP else 700), "z": -400, "bearing": 0, "width": 5}
            for f, lord in LORDS.items() if lord not in used_lords]
    arena = {k: v for k, v in base.items() if k not in ("faction", "units", "description")}
    arena.update(name="charge_probe", gap_m=400, defend_radius_m=150, sides=armies)
    return lanes, park, arena


def run_config(plan, index):
    """(entry config, model seconds, arena) of battle `index` (1-based) of a plan."""
    specs = battles(plan)[index - 1]
    lanes, park, arena = layout(specs)
    turn = plan == "move" and any(l["mode"] == "script" for l in lanes)
    config = {"plan": plan, "battle": index, "settle_ms": SETTLE_MS, "tick_ms": TURN_TICK_MS if turn else TICK_MS,
              "men_ms": 500 if turn else MEN_MS,
              "men_near_m": 60,
              # the soldiers' places: the first 30 s (the charge plan) or the whole fight (hit: men in contact)
              "men_after_s": 90 if plan in ("hit", "move", "vv") else 30, "lanes": lanes, "park": park}
    model_s = max(l["max_s"] for l in lanes) + SETTLE_MS / 1000 + 20
    return config, model_s, arena


def write_scenario(plan, index, path=None):
    _, _, arena = run_config(plan, index)
    path = path or ROOT / f"charge_probe_{plan}_{index}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nn_scenario.scenario_xml(arena, "enemy", 3600), encoding="utf-8", newline="\n")
    return path


def run(plan, which=None, python=sys.executable, dry=False, deadline=900):
    """Builds and launches each battle in turn (tools/launcher/launch.ps1: Normal difficulty set and
    restored by the launcher)."""
    done = []
    for i in which or range(1, len(battles(plan)) + 1):
        build = [python, "-m", "tools.build", "charge-probe", "--probe-plan", plan, "--probe-battle", str(i),
                 "--deadline", str(deadline)]
        launch = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAUNCHER / "launch.ps1"),
                  "-Target", "charge-probe"]
        print(f"--- {plan} {i}", flush=True)
        if dry:
            print(" ".join(build), "&&", " ".join(launch))
            continue
        code = subprocess.run(build, cwd=project.ROOT, stdout=subprocess.DEVNULL).returncode
        if code == 0:
            code = subprocess.run(launch, cwd=project.ROOT).returncode
        done.append((plan, i, code))
    return done


# ---------------------------------------------------------------- the recordings

def load_run(run_dir):
    """{lane name: {spec, samples [(t s, a row, tg row, lord row)], men [(t, a xz, tg xz)], contacts, end,
    abilities, phases}} of one run."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    lanes = {l["name"]: {"run": run_dir.name, "plan": cfg.get("plan"), "battle": cfg.get("battle"), "spec": l,
                         "samples": [], "men": [], "contacts": {}, "end": None, "abilities": [], "phases": []}
             for l in cfg.get("lanes", [])}
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if '"probe_' not in line:
            continue
        r = json.loads(line)
        ev = r["event"]
        if ev == "probe_sample":
            for x in r["lanes"]:
                if x["lane"] in lanes:
                    lanes[x["lane"]]["samples"].append((x["t"] / 1000, x["a"], x["tg"], x.get("l")))
        elif ev == "probe_men":
            for x in r["lanes"]:
                if x["lane"] in lanes:
                    lanes[x["lane"]]["men"].append((x["t"] / 1000, x.get("a") or [], x.get("tg") or []))
        elif ev == "probe_contact":
            lanes[r["lane"]]["contacts"][r["n"]] = r["t"] / 1000
        elif ev == "probe_lane_end":
            lanes[r["lane"]]["end"] = r
        elif ev == "probe_ability":
            lanes[r["lane"]]["abilities"].append(r)
        elif ev == "probe_phase":
            lanes[r["lane"]]["phases"].append(r)
    return [x for x in lanes.values() if x["samples"]]


def series(samples, who, key):
    i = {"a": 1, "tg": 2, "l": 3}[who]
    return np.array([np.nan if (s[i] or {}).get(key) is None else float(s[i][key]) for s in samples])


def at(t, v, when):
    """v interpolated at time `when` (NaN outside the recording)."""
    ok = np.isfinite(v)
    if ok.sum() < 2 or when < t[ok][0] or when > t[ok][-1]:
        return np.nan
    return float(np.interp(when, t[ok], v[ok]))


def near_counts(a, b, radii=RADII):
    """Soldiers of a within each radius of their nearest soldier of b (flat decimetre lists)."""
    if len(a) < 2 or len(b) < 2:
        return [0] * len(radii)
    pa = np.asarray(a, float).reshape(-1, 2) / 10
    pb = np.asarray(b, float).reshape(-1, 2) / 10
    d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return [int((d <= r).sum()) for r in radii]


def measure(lane):
    """One lane's numbers (game or simulator: the same row format)."""
    spec, s = lane["spec"], lane["samples"]
    t = np.array([x[0] for x in s])
    out = {"run": lane["run"], "lane": spec["name"], "attacker": SHORT[spec["a_key"]],
           "target": SHORT[spec["t_key"]], "a_key": spec["a_key"], "t_key": spec["t_key"],
           "mode": spec["mode"], "target_mode": spec["target_mode"], "gap_m": spec["gap_m"],
           "ability": (spec.get("lord") or {}).get("ability"), "cell": cell(spec)}
    ax, az, tx, tz = (series(s, w, k) for w, k in (("a", "x"), ("a", "z"), ("tg", "x"), ("tg", "z")))
    c = lane["contacts"].get(1)
    out["contact_s"] = c
    if c is None:
        return out
    # the way in: the attacker's mean speed over distance bands before contact (the engine moves a unit in
    # steps, so speeds come from the times the distance crossed each band, not from 0.5 s differences)
    dist = np.hypot(ax - tx, az - tz)
    before = (t <= c) & np.isfinite(dist)
    tb, db = t[before], dist[before]
    if len(tb) > 3:
        reach = db[-1]
        left = db - reach                      # metres still to go
        def crossed(m):
            k = np.nonzero(left <= m)[0]
            if not len(k) or k[0] == 0:
                return None
            i = k[0]
            return float(np.interp(m, [left[i], left[i - 1]], [tb[i], tb[i - 1]]))
        t30, t60, t10 = crossed(30), crossed(60), crossed(10)
        out["speed_last30"] = 30 / (tb[-1] - t30) if t30 is not None and tb[-1] > t30 else None
        out["speed_30_60"] = 30 / (t30 - t60) if None not in (t30, t60) and t30 > t60 else None
        out["speed_last10"] = 10 / (tb[-1] - t10) if t10 is not None and tb[-1] > t10 else None
        # the peak: the fastest 2 s on the way in
        w = int(round(2.0 / max(1e-6, float(np.median(np.diff(tb))))))
        if len(tb) > w:
            out["speed_peak30"] = float(np.max((db[:-w] - db[w:]) / (tb[w:] - tb[:-w])))
    out["approach_s"] = float(c - t[0])
    end_t = t[-1]
    # a window from the contact starts at the sample before it (the game's first contact sample already holds the
    # blows struck since that sample: a lord's first blow; the simulator's contact time is that sample already)
    before_c = t[t < c - 1e-6]
    c0 = float(before_c[-1]) if lane["run"] != "sim" and len(before_c) else c
    for who in ("tg", "a"):
        hp, men = series(s, who, "hp"), series(s, who, "men")
        for lo, hi in WINDOWS:
            if c + hi <= end_t + 1e-6:
                start = c0 if lo == 0 else c + lo
                out[f"{who}_hp_{lo}_{hi}"] = at(t, hp, start) - at(t, hp, c + hi)
                out[f"{who}_men_{lo}_{hi}"] = at(t, men, start) - at(t, men, c + hi)
        stop = min(end_t, c + spec["fight_s"])
        if lane["contacts"].get(2) is not None:
            stop = min(stop, lane["contacts"][2] - 1)
            phases = [p["t"] / 1000 for p in lane.get("phases", []) if p["phase"] == "out"]
            if phases:
                stop = min(stop, phases[0])
        if stop - (c + STEADY_FROM_S) >= 5:
            out[f"{who}_hp_steady"] = (at(t, hp, c + STEADY_FROM_S) - at(t, hp, stop)) / (stop - c - STEADY_FROM_S)
            out[f"{who}_men_steady"] = (at(t, men, c + STEADY_FROM_S) - at(t, men, stop)) / (stop - c - STEADY_FROM_S)
        c2 = lane["contacts"].get(2)
        if c2 is not None:
            for lo, hi in ((0, 1), (0, 3), (0, 5), (5, 15)):
                if c2 + hi <= end_t + 1e-6:
                    out[f"{who}_hp2_{lo}_{hi}"] = at(t, hp, c2 + lo) - at(t, hp, c2 + hi)
        if who == "tg" and lane.get("abilities"):
            out["ability_status"] = lane["abilities"][0].get("status")
            if c + 18 <= end_t:
                out["tg_hp_0_18"] = at(t, hp, c) - at(t, hp, c + 18)
                out["a_hp_0_18"] = at(t, series(s, "a", "hp"), c) - at(t, series(s, "a", "hp"), c + 18)
    out["contact2_s"] = lane["contacts"].get(2)
    # men in contact (soldiers' places): per second after contact, the mean over 0-5, 5-15, 15-30 s
    if lane.get("men"):
        rows = [(tm - c, near_counts(a, b), near_counts(b, a)) for tm, a, b in lane["men"] if tm >= c - 0.01]
        for lo, hi in NEAR_WINDOWS:
            sel = [r for r in rows if lo <= r[0] < hi]
            if sel:
                out[f"a_near_{lo}_{hi}"] = [round(float(np.mean([r[1][k] for r in sel])), 1) for k in range(len(RADII))]
                out[f"tg_near_{lo}_{hi}"] = [round(float(np.mean([r[2][k] for r in sel])), 1) for k in range(len(RADII))]
    return out


def cell(spec):
    """The lane's experimental cell: who on whom, how."""
    a = SHORT[spec["a_key"]] if "a_key" in spec else spec["attacker"]
    t = SHORT[spec["t_key"]] if "t_key" in spec else spec["target"]
    name = f"{a}>{t} {spec['mode']}/{spec['target_mode']}"
    if spec["gap_m"] not in (GAP_M, 40):
        name += f" gap{spec['gap_m']}"
    ab = (spec.get("lord") or {}).get("ability")
    if ab is not None:
        name += " SYG" if ab else " noSYG"
    return name


def runs(root=RUNS):
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if (d / "events.jsonl").exists() and (d / "manifest.json").exists())


# ---------------------------------------------------------------- the simulator on the same lanes

def sim_lanes(lanes, params=None, device="cpu", copies=8, jitter_m=1.0, seed=0):
    """The simulator on recorded lanes: each lane its own battle (its units at their places at the
    probe's go), the same scripted orders as the entry's modes, everyone fearless. Returns lanes in
    the game's shape (samples every 0.5 s, contacts, phases), `copies` per recorded lane."""
    import torch
    from tools.nn.sim import abilities as sim_abilities, battle, orders as O, scenario as sim_scenario
    from tools.nn.sim.params import load
    params = params or load()
    armies, meta = [], []
    for ln in lanes:
        spec = ln["spec"]
        s0 = ln["samples"][0]
        sides = {1: {"faction": EMP, "ai": False, "units": []}, 2: {"faction": SKV, "ai": False, "units": []}}
        roles = {}
        for role, row, key, width in (("a", s0[1], spec["a_key"], spec["a_width"]),
                                      ("tg", s0[2], spec["t_key"], spec["t_width"])):
            fac = 1 if FACTION[key] == EMP else 2
            lord = key in (UNITS["general"][0], UNITS["warlord"][0])
            sides[fac]["units"].append({"key": key, "x": row["x"], "z": row["z"], "b": row["b"], "men": row["men"],
                                        "width": None if lord else width, "general": lord, "name": role})
            roles[role] = fac
        if spec.get("lord") and s0[3]:
            r = s0[3]
            sides[1]["units"].append({"key": UNITS["general"][0], "x": r["x"], "z": r["z"], "b": r["b"],
                                      "general": True, "name": "l"})
        army = {"attacker": roles["a"], "sides": sides}
        for _ in range(copies):
            armies.append(army)
            meta.append(ln)
    per_side = max(len(a["sides"][s]["units"]) for a in armies for s in (1, 2))
    st = sim_scenario.build(armies, params, device=device, per_side=per_side)
    gen = torch.Generator().manual_seed(seed)
    for k in ("x", "z"):
        st.u[k] = st.u[k] + ((torch.rand(st.u[k].shape, generator=gen) * 2 - 1) * jitter_m).to(device)
    st.u["leadership"] = st.u["leadership"] + 1e4
    st.u["morale"] = st.u["morale"] + 1e4
    B, N = st.B, st.N
    slot = {r: torch.tensor([sim_scenario.slots(a, per_side).get(r, -1) for a in armies], device=device)
            for r in ("a", "tg", "l")}
    specs = [ln["spec"] for ln in meta]
    mode = [sp["mode"] for sp in specs]
    tmode = [sp["target_mode"] for sp in specs]
    syg_slot = sim_abilities.slot_keys(UNITS["general"][0], params.units, params.abilities).index(SYG) \
        if SYG in sim_abilities.slot_keys(UNITS["general"][0], params.units, params.abilities) else -1
    st_ = {"contact": [None] * B, "contact2": [None] * B, "phase": ["in"] * B, "out_t": [0.0] * B,
           "out_from": [None] * B, "fired": [False] * B, "last": {}, "phases": [[] for _ in range(B)],
           "a_ability": [False] * B, "start": [None] * B, "faced": set()}
    rec = {"t": [], "rows": []}
    fields = ("x", "z", "b", "men", "hp_abs", "m", "fatigue", "fat", "k")
    a_slot = {}
    for sp in specs:
        if sp.get("a_ability") and sp["a_key"] not in a_slot:
            keys = sim_abilities.slot_keys(sp["a_key"], params.units, params.abilities)
            a_slot[sp["a_key"]] = keys.index(sp["a_ability"]) if sp["a_ability"] in keys else -1

    def policy(st):
        u = st.u
        t = float(st.t.max())
        b_idx = torch.arange(B, device=device)
        m_a = u["m"][b_idx, slot["a"]].bool().cpu().numpy()
        m_t = u["m"][b_idx, slot["tg"]].bool().cpu().numpy()
        ax, az = u["x"][b_idx, slot["a"]].cpu().numpy(), u["z"][b_idx, slot["a"]].cpu().numpy()
        o = O.hold(B, N, device)
        kind, target, run_, x, z, ab = (o.kind.clone(), o.target.clone(), o.run.clone(), o.x.clone(), o.z.clone(),
                                        o.ability.clone())
        for b in range(B):
            A, T, L = int(slot["a"][b]), int(slot["tg"][b]), int(slot["l"][b])
            sp = specs[b]
            # the contact began in the step that ended now (its blows are in this sample already): its start
            if st_["contact"][b] is None and (m_a[b] or m_t[b]):
                st_["contact"][b] = t - params.dt
                if L >= 0 and sp["lord"].get("ability") and syg_slot >= 0:
                    ab[b, L] = syg_slot
            c = st_["contact"][b]
            if st_["start"][b] is None:
                st_["start"][b] = (float(ax[b]), float(az[b]))
            # the attacker's own ability (Foe-Seeker) a_ability_after_s after the contact, once
            if sp.get("a_ability") and c is not None and not st_["a_ability"][b]                     and t - c >= sp.get("a_ability_after_s", 0) and a_slot.get(sp["a_key"], -1) >= 0:
                ab[b, A] = a_slot[sp["a_key"]]
                st_["a_ability"][b] = True
            # the attacker
            if mode[b] == "script":
                # a 'move' step: MOVE to its start + (dx, dz); a 'face' step has no order in the simulator (it has no
                # facing order): the unit holds and the twin turns it to the step's bearing at once (so that a later
                # move starts from the game's facing)
                due = [x for x in sp.get("steps", []) if t >= x["at_s"]]
                for j, x_ in enumerate(due):
                    if x_["kind"] == "face" and (b, j) not in st_["faced"]:
                        st_["faced"].add((b, j))
                        st.u["b"][b, A] = float(x_["bearing"])
                if due and due[-1]["kind"] == "move":
                    sx, sz = st_["start"][b]
                    kind[b, A], x[b, A], z[b, A] = O.MOVE, sx + due[-1].get("dx", 0), sz + due[-1].get("dz", 0)
                    run_[b, A] = bool(due[-1].get("run"))
            elif mode[b] in ("recharge", "withdraw"):
                ph = st_["phase"][b]
                if ph == "in" and c is not None and t - c >= sp["recharge_after_s"]:
                    st_["phase"][b], st_["out_t"][b], st_["out_from"][b] = "out", t, (float(ax[b]), float(az[b]))
                    st_["phases"][b].append({"phase": "out", "t": t * 1000})
                elif ph == "out":
                    fx, fz = st_["out_from"][b]
                    moved = math.hypot(ax[b] - fx, az[b] - fz)
                    if mode[b] == "recharge" and ((not m_a[b] and moved >= sp["back_m"] - 5)
                                                  or t - st_["out_t"][b] >= sp["recharge_max_s"]):
                        st_["phase"][b] = "back"
                        st_["phases"][b].append({"phase": "back", "t": t * 1000})
                if st_["phase"][b] == "back" and st_["contact2"][b] is None and m_a[b]:
                    st_["contact2"][b] = t - params.dt
                if st_["phase"][b] == "out":
                    fx, fz = st_["out_from"][b]
                    kind[b, A], x[b, A], z[b, A], run_[b, A] = O.MOVE, fx, fz + sp["back_m"], True
                else:
                    kind[b, A], target[b, A], run_[b, A] = O.ATTACK, T, True
            elif mode[b] in ("attack_run", "attack_walk"):
                kind[b, A], target[b, A], run_[b, A] = O.ATTACK, T, mode[b] == "attack_run"
            elif mode[b] == "move_run":
                kind[b, A], x[b, A], z[b, A], run_[b, A] = O.MOVE, sp["x"], sp["z"] - sp["move_beyond_m"], True
            # the target
            if tmode[b] == "both" or (sp.get("answer") and c is not None):
                kind[b, T], target[b, T], run_[b, T] = O.ATTACK, A, tmode[b] == "both"
        # KEEP the orders in force (re-issuing every step would be a new order each time)
        key = torch.stack([kind.float(), target.float(), run_.float(), x, z], -1)
        last = st_["last"].get("key")
        if last is not None:
            same = (key == last).all(-1)
            kind = torch.where(same, torch.full_like(kind, O.KEEP), kind)
        st_["last"]["key"] = key
        return O.Orders(kind=kind, x=x, z=z, target=target, run=run_, ability=ab)

    next_t = [0.0]

    def record(st):
        t = float(st.t.max())
        if t + 1e-6 < next_t[0]:
            return
        next_t[0] = t + TICK_MS / 1000
        rec["t"].append(t)
        rec["rows"].append({k: st.u[k].detach().cpu().numpy().copy() for k in fields if k in st.u})
    until = max((ln["end"] or {}).get("t", 0) / 1000 for ln in lanes) + 5 if any(ln["end"] for ln in lanes) else 200
    battle.run(st, policy, params, until_s=until, record=record)
    out = []
    for b, ln in enumerate(meta):
        A, T, L = int(slot["a"][b]), int(slot["tg"][b]), int(slot["l"][b])
        samples = []
        for t, r in zip(rec["t"], rec["rows"]):
            def row(i):
                if i < 0:
                    return None
                out = {"x": float(r["x"][b, i]), "z": float(r["z"][b, i]), "b": float(r["b"][b, i]),
                       "men": float(r["men"][b, i]), "hp": float(r["hp_abs"][b, i]), "m": bool(r["m"][b, i])}
                for k in ("fatigue", "fat", "k"):
                    if k in r:
                        out[k] = float(r[k][b, i])
                return out
            samples.append((t, row(A), row(T), row(L)))
        contacts = {}
        if st_["contact"][b] is not None:
            contacts[1] = st_["contact"][b]
        if st_["contact2"][b] is not None:
            contacts[2] = st_["contact2"][b]
        out.append({"run": "sim", "spec": ln["spec"], "samples": samples, "men": [], "contacts": contacts,
                    "end": None, "abilities": ln.get("abilities") and [{"status": "sim"}], "phases": st_["phases"][b]})
    return out


# ---------------------------------------------------------------- turning (plan move, battle 1)

def _ang(a):
    """Degrees wrapped to -180..180."""
    return (np.asarray(a, float) + 180) % 360 - 180


def turn_measure(lane, run_speed=None):
    """A script lane's turns and starts, per step: the bearing's change over time (s to within 10 deg of the
    facing ordered, or of the way to the point), the centre's speed 1..6 s after the order (share of the run) and
    when it first reaches 0.9 of the run; from the soldiers' places (game): how far the men moved in the first
    10 s and whether the formation turned as a rigid block (each man keeps his rank: +1) or about-faced in place
    (the front rank becomes the back: -1)."""
    spec, smp = lane["spec"], lane["samples"]
    t = np.array([x[0] for x in smp])
    b, x, z = series(smp, "a", "b"), series(smp, "a", "x"), series(smp, "a", "z")
    starts = {}
    for ph in lane.get("phases", []):
        if str(ph.get("phase", "")).startswith("step"):
            starts[int(ph["phase"][4:].split(":")[0])] = ph["t"] / 1000
    steps = spec.get("steps") or []
    out = []
    for i, step in enumerate(steps, 1):
        t0 = starts.get(i, step["at_s"])
        t1 = starts.get(i + 1, steps[i]["at_s"] if i < len(steps) else t[-1])
        sel = (t >= t0 - 1e-6) & (t <= t1 + 1e-6) & np.isfinite(b)
        if sel.sum() < 2:
            continue
        tt, bb, xx, zz = t[sel] - t0, b[sel], x[sel], z[sel]
        if step["kind"] == "face":
            goal = float(step["bearing"])
        else:
            goal = float(np.degrees(np.arctan2(step.get("dx", 0), step.get("dz", 0))) % 360)
        off = np.abs(_ang(bb - goal))
        done = np.nonzero(off <= 10)[0]
        row = {"step": i, "kind": step["kind"], "goal_deg": round(goal), "turn_deg": round(float(abs(_ang(bb[0] - goal)))),
               "s_to_10deg": round(float(tt[done[0]]), 2) if len(done) else None,
               "bearing": [(round(float(a), 2), round(float(v))) for a, v in zip(tt, bb)][:40]}
        if step["kind"] == "move":
            d = np.hypot(xx - xx[0], zz - zz[0])
            moved = np.nonzero(d > 1.0)[0]
            row["s_to_move_1m"] = round(float(tt[moved[0]]), 2) if len(moved) else None
            sp = []
            for k in range(1, 9):
                lo, hi = at(tt, d, k - 1.0), at(tt, d, k + 0.0)
                sp.append(None if not np.isfinite(lo) or not np.isfinite(hi) else round(hi - lo, 2))
            row["speed_by_s"] = sp
            if run_speed:
                row["run_share_by_s"] = [None if v is None else round(v / run_speed, 2) for v in sp]
                fast = [k + 1 for k, v in enumerate(sp) if v is not None and v >= 0.9 * run_speed]
                row["s_to_0.9_run"] = fast[0] if fast else None
        men = [(tm - t0, a) for tm, a, _ in lane.get("men", []) if t0 - 0.6 <= tm <= t0 + 12 and len(a) >= 4]
        if len(men) >= 3:
            p0 = np.asarray(men[0][1], float).reshape(-1, 2) / 10
            p_end = np.asarray(men[-1][1], float).reshape(-1, 2) / 10
            n = min(len(p0), len(p_end))
            p0, p_end = p0[:n], p_end[:n]
            row["men_moved_m"] = round(float(np.median(np.hypot(*(p_end - p0).T))), 1)
            # each man's place along the facing, before (old facing) and after (new facing, from the centre)
            fb, fe = np.radians(bb[0]), np.radians(bb[-1])
            c0, ce = p0.mean(0), p_end.mean(0)
            along0 = (p0 - c0) @ np.array([np.sin(fb), np.cos(fb)])
            along1 = (p_end - ce) @ np.array([np.sin(fe), np.cos(fe)])
            if np.std(along0) > 0.1 and np.std(along1) > 0.1:
                row["rank_kept_corr"] = round(float(np.corrcoef(along0, along1)[0, 1]), 2)
            row["men_window_s"] = round(float(men[-1][0]), 1)
        out.append(row)
    return out


def turn_report(run_dirs, sim=False, device="cpu", copies=4, out=None, params=None):
    """The turning lanes (mode script) of the runs: the game's table and, with sim, the simulator's on the same
    lanes. Writes build/charge-probe/turns.json (not in Git)."""
    units = json.loads(UNITS_JSON.read_text(encoding="utf-8"))["units"]
    lanes = [ln for d in run_dirs for ln in load_run(d) if ln["spec"]["mode"] == "script"]
    result = {"game": [], "sim": []}
    for ln in lanes:
        run = units[ln["spec"]["a_key"]]["speed"]["run"]
        result["game"].append({"cell": cell(ln["spec"]), "run": ln["run"], "steps": turn_measure(ln, run)})
    if sim and lanes:
        for ln in sim_lanes(lanes, params=params, device=device, copies=copies):
            run = units[ln["spec"]["a_key"]]["speed"]["run"]
            result["sim"].append({"cell": cell(ln["spec"]), "run": "sim", "steps": turn_measure(ln, run)})
    keys = ("turn_deg", "s_to_10deg", "s_to_move_1m", "s_to_0.9_run", "run_share_by_s", "men_moved_m", "rank_kept_corr")
    for who in ("game", "sim"):
        for r in result[who]:
            for st in r["steps"]:
                print(who, r["cell"], st["step"], st["kind"], {k: st.get(k) for k in keys if st.get(k) is not None})
    out = out or ROOT / "turns.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(out)
    return result


# ---------------------------------------------------------------- melee exit and vigour (plan move, battle 2)

FAT_LEVELS = ("threshold_fresh", "threshold_active", "threshold_winded", "threshold_tired", "threshold_very_tired",
              "threshold_exhausted")


def _level(v):
    """A fatigue level index from the game's name or the simulator's index."""
    if v is None:
        return np.nan
    if isinstance(v, str):
        return float(FAT_LEVELS.index(v)) if v in FAT_LEVELS else np.nan
    return float(v)


def exit_measure(lane):
    """A withdraw lane after its order ('out'): seconds the leaver stays in melee (the last melee sample), its kills
    and the HP its enemy lost in 0-24 s and 26-50 s after the order, its own HP lost in 0-24 s; an attack lane with
    a_ability: the attacker's fatigue level times (first sample at each level after contact) and the ability's time."""
    spec, smp = lane["spec"], lane["samples"]
    t = np.array([x[0] for x in smp])
    out = {"cell": cell(spec), "run": lane["run"]}
    outs = [p["t"] / 1000 for p in lane.get("phases", []) if p["phase"] == "out"]
    if spec["mode"] == "withdraw" and outs:
        o = outs[0]
        m = np.array([bool((x[1] or {}).get("m")) for x in smp])
        after = (t >= o) & m
        out["in_melee_s"] = round(float(t[after][-1] - o), 1) if after.any() else 0.0
        k, ehp, hp = series(smp, "a", "k"), series(smp, "tg", "hp"), series(smp, "a", "hp")
        def r(v, p=0):
            return None if not np.isfinite(v) else round(float(v), p)
        for lo, hi in ((0, 24), (26, 50)):
            out[f"kills_{lo}_{hi}"] = r(at(t, k, o + hi) - at(t, k, o + lo), 1)
            out[f"enemy_hp_{lo}_{hi}"] = r(at(t, ehp, o + lo) - at(t, ehp, o + hi))
        out["own_hp_0_24"] = r(at(t, hp, o) - at(t, hp, o + 24))
    if spec.get("a_ability"):
        c = lane["contacts"].get(1) or 0.0
        lv = np.array([_level((x[1] or {}).get("fat")) for x in smp])
        out["ability_at_s"] = spec.get("a_ability_after_s")
        out["level_at_s"] = {FAT_LEVELS[j][10:]: round(float(t[np.nonzero(lv >= j)[0][0]] - c), 1)
                             for j in range(1, 6) if (lv >= j).any()}
        pts = series(smp, "a", "fatigue")
        if np.isfinite(pts).any():
            out["points_by_10s"] = [None if not np.isfinite(at(t, pts, c + s)) else round(at(t, pts, c + s))
                                    for s in range(0, 101, 10)]
    return out


def exit_report(run_dirs, sim=False, device="cpu", copies=4, params=None):
    """The melee-exit and vigour lanes (plan move battle 2) of the runs: the game's numbers and, with sim, the
    simulator's on the same lanes."""
    lanes = [ln for d in run_dirs for ln in load_run(d)
             if ln["spec"]["mode"] == "withdraw" or ln["spec"].get("a_ability")]
    rows = [exit_measure(ln) for ln in lanes]
    if sim and lanes:
        rows += [exit_measure(ln) for ln in sim_lanes(lanes, params=params, device=device, copies=copies)]
    for r in rows:
        print(r)
    return rows


# ---------------------------------------------------------------- tables

KEYS = ("contact_s", "speed_30_60", "speed_last30", "speed_last10", "speed_peak30",
        "tg_hp_0_1", "tg_hp_0_2", "tg_hp_0_5", "tg_hp_5_15", "tg_hp_15_30", "tg_hp_steady", "tg_men_steady",
        "a_hp_0_1", "a_hp_0_2", "a_hp_0_5", "a_hp_5_15", "a_hp_15_30", "a_hp_steady", "a_men_steady",
        "tg_hp2_0_5", "a_hp2_0_3", "tg_hp_0_18", "a_hp_0_18")


def summary(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["cell"], []).append(r)
    out = []
    for name, rs in sorted(groups.items()):
        row = {"cell": name, "n": len(rs)}
        for k in KEYS:
            v = [r[k] for r in rs if r.get(k) is not None and np.isfinite(r[k])]
            if v:
                row[k] = (float(np.mean(v)), float(np.min(v)), float(np.max(v)))
        for k in NEAR_KEYS:
            v = [r[k] for r in rs if r.get(k)]
            if v:
                row[k] = [round(float(x), 1) for x in np.mean(v, axis=0)]
        out.append(row)
    return out


def report(run_dirs, sim=False, out=OUT, device="cpu", copies=8):
    lanes = [ln for d in run_dirs for ln in load_run(d)]
    game = [measure(ln) for ln in lanes]
    table = summary(game)
    result = {"lanes": game, "summary": table}
    sim_table = None
    if sim:
        sims = sim_lanes([ln for ln in lanes if ln["contacts"].get(1) is not None], device=device, copies=copies)
        sim_rows = [measure(ln) for ln in sims]
        sim_table = {r["cell"]: r for r in summary(sim_rows)}
        result["sim_summary"] = list(sim_table.values())
    f = lambda v, p=0: "-" if v is None else f"{v:.{p}f}"
    print("HP lost after the first contact (target tg / attacker a), mean of the cell's lanes; sim = the simulator "
          "on the same lanes")
    for r in table:
        print(f"\n{r['cell']}  (n={r['n']})")
        s = (sim_table or {}).get(r["cell"], {})
        for k in KEYS:
            if k in r:
                g = r[k]
                sv = s.get(k)
                extra = "" if not sim_table else f"   sim {f(sv[0], 2 if 'speed' in k or 'men' in k else 0)}"
                p = 2 if "speed" in k or "men" in k else 0
                print(f"  {k:14} {f(g[0], p):>8} ({f(g[1], p)}-{f(g[2], p)}){extra}")
        for k in NEAR_KEYS:
            if k in r:
                print(f"  {k:14} men within {RADII} m of an enemy: {r[k]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("report", "plan", "run", "turns", "exits"), default="report")
    parser.add_argument("runs", nargs="*", type=Path, help="report: run folders (default: build/charge-probe/runs/*)")
    parser.add_argument("--plan", choices=PLANS, default="charge")
    parser.add_argument("--battles", help="run: battle numbers, comma separated (default: all of the plan)")
    parser.add_argument("--sim", action="store_true", help="report: also the simulator on the same lanes (torch)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "plan":
        for i, b in enumerate(battles(args.plan), 1):
            print(i, [cell(l) for l in b])
        return 0
    if args.command == "run":
        which = [int(x) for x in args.battles.split(",")] if args.battles else None
        done = run(args.plan, which, dry=args.dry)
        print("done:", done)
        return 0 if all(d[-1] == 0 for d in done) else 1
    if args.command == "exits":
        exit_report(args.runs or runs(), sim=args.sim, device=args.device, copies=min(args.copies, 4))
        return 0
    if args.command == "turns":
        turn_report(args.runs or runs(), sim=args.sim, device=args.device, copies=min(args.copies, 4),
                    out=None if args.out == OUT else args.out)
        return 0
    report(args.runs or runs(), sim=args.sim, out=args.out, device=args.device, copies=args.copies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
