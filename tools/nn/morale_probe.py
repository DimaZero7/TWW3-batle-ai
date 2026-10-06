"""The morale probe (src/entries/morale_probe.lua): one morale term, one scenario, in the game and in the
simulator alike. Lanes far apart (350-450 m: no enemy of another lane within 150 m), in each a few units
placed and ordered by script; the side under test keeps its morale, the other side is fearless (its routs
never lift ours). Each sample (0.5 s) reads every unit's MoralePercent and the game's strongest morale effect
(CCO MoraleGreatestEffect, its text), so a term's start, size (points = MoralePercent x the card's
leadership) and end are seen directly (build/morale_spec/spec.md section 5).

Plans (battles):
  flank   T-A (2 battles, lanes swapped): spearmen hold, clanrats attack them in front; 20 s after the
          contact skavenslaves attack their flank (lane F) or rear (lane R) and withdraw 20 s after their own
          contact; lane C: the same front fight with two friends 34 m at the spearmen's sides (flanks secure
          in melee?);
  charge  T-B (1): swordsmen attack standing clanrats at a run from 150 m and 40 m, at a walk from 80 m, and
          walk to 10 m then attack at a walk (the morale charge bonus: start, length, blocks);
  secure  T-C (1): spearmen stand; skavenslave spearmen come closer by steps 300 -> 200 -> 160 -> 140 -> 120
          -> 100 -> 70 m (centre to centre), 15-25 s each; lanes: alone, a friend at the side 32 m, a friend
          in front 30 m, only the lord at the side (flanks secure);
  shoot   T-D (1): archers shoot fearless clanrats at 120 m and skavenslaves at 100 m for 30 s; fearless slave
          slingers shoot spearmen at 90 m for 30 s (winning the fight by shooting; its window);
  rally   T-E (1): skavenslaves (tested) against fearless Empire spearmen until they rout; then the enemy halts
          (lane 0) or keeps 50 / 90 / 150 m from the routers (rally: distance, wait, morale at rally).

    python -m tools.nn.morale_probe plan [--plan flank]            # the battles
    python -m tools.build morale-probe --morale-plan flank --morale-battle 1
    python -m tools.nn.morale_probe run --plan flank [--battles 1,2]
    python -m tools.nn.morale_probe report [runs...] [--sim]        # the game's table (+ the simulator twin)
    python -m tools.nn.morale_probe twin --plan rally [--out f.json] # the simulator alone on a plan's lanes

Writes build/morale-probe/analysis.json (not in Git).
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools import config as project
from tools.nn import scenario as nn_scenario

ROOT = project.BUILD / "morale-probe"
RUNS = ROOT / "runs"
OUT = ROOT / "analysis.json"
LAUNCHER = project.ROOT / "tools" / "launcher"
UNITS_JSON = project.ROOT / "config" / "nn" / "units.json"
EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"
UNITS = {
    "spear": ("wh_main_emp_inf_spearmen_0", 120, EMP),
    "spearsh": ("wh_main_emp_inf_spearmen_1", 120, EMP),
    "swords": ("wh_main_emp_inf_swordsmen", 120, EMP),
    "archers": ("wh2_dlc13_emp_inf_archers_0", 90, EMP),
    "gs": ("wh_main_emp_inf_greatswords", 120, EMP),
    "general": ("wh_main_emp_cha_general_0", 1, EMP),
    "clanrat": ("wh2_main_skv_inf_clanrats_1", 160, SKV),
    "slave": ("wh2_main_skv_inf_skavenslaves_0", 180, SKV),
    "sspear": ("wh2_main_skv_inf_skavenslave_spearmen_0", 180, SKV),
    "slingers": ("wh2_main_skv_inf_skavenslave_slingers_0", 140, SKV),
    "storm": ("wh2_main_skv_inf_stormvermin_0", 160, SKV),
    "warlord": ("wh2_main_skv_cha_warlord_0", 1, SKV),
    "nr": ("wh2_main_skv_inf_night_runners_1", 120, SKV),
}
LORDS = {EMP: "general", SKV: "warlord"}
RESERVES = {EMP: ("gs", "gs"), SKV: ("storm", "storm")}     # parked far: the armies' strength (no army losses)
SHORT = {key: short for short, (key, _, _) in UNITS.items()}
SETTLE_MS = 4000
TICK_MS = 500
WIDTH_M = 30
PARK_Z = 650
PLANS = ("flank", "charge", "secure", "shoot", "rally", "strong")
# the game's strongest-effect texts (Russian client) this probe reads
FLANK, REAR, CHARGE, WIN, SECURE = "Атакованы с фланга", "Атакованы с тыла", "Натиск", "Одерживают верх", "Фланги прикрыты"


def spacing(key):
    sp = json.loads(UNITS_JSON.read_text(encoding="utf-8"))["units"][key]["spacing"]
    return float(sp["h"]), float(sp["v"])


def depth(short, width=WIDTH_M):
    key, men, _ = UNITS[short]
    if men <= 1:
        return 0.0
    h, v = spacing(key)
    return math.ceil(men / max(1, int(width // h))) * v


def leadership(key):
    return float(json.loads(UNITS_JSON.read_text(encoding="utf-8"))["units"][key]["leadership"])


def u(role, short, x, z, b, fearless=False):
    """A lane unit: its centre (lane frame, m) and bearing (0: facing +z)."""
    return {"role": role, "short": short, "x": round(float(x), 1), "z": round(float(z), 1), "b": b,
            "fearless": fearless}


def s(on, do, unit=None, of=None, delay_s=0.0, **extra):
    """A scripted step (morale_probe.lua): when `on` (of role / step) happened and delay_s passed, do."""
    out = {"on": on, "do": do, "delay_s": float(delay_s)}
    if unit is not None:
        out["unit"] = unit
    if of is not None:
        out["of"] = of
    out.update(extra)
    return out


def flank_lane(kind):
    """T-A: U (spearmen with shields) holds; A (clanrats) attacks its front from 40 m; 20 s after U's
    contact B (skavenslaves) attacks its flank / rear from 70 m and walks away 20 s after B's contact
    (kind 'F', 'R'); 'C': no B, two friends at U's sides (34 m centre to centre)."""
    dU, dA, dB = depth("spearsh"), depth("clanrat"), depth("slave")
    units = [u("U", "spearsh", 0, -dU / 2, 0), u("A", "clanrat", 0, 40 + dA / 2, 180, True)]
    steps = [s("go", "attack", "A", target="U")]
    if kind == "F":
        units.append(u("B", "slave", 70 + 15, -dU / 2, 270, True))
        steps += [s("contact", "attack", "B", of="U", delay_s=20, target="U"),
                  s("contact", "move", "B", of="B", delay_s=20, x=170, z=-dU / 2),
                  s("step", "end", of=3, delay_s=40)]
    elif kind == "R":
        units.append(u("B", "slave", 0, -dU - 70 - dB / 2, 0, True))
        steps += [s("contact", "attack", "B", of="U", delay_s=20, target="U"),
                  s("contact", "move", "B", of="B", delay_s=20, x=0, z=-dU - 170),
                  s("step", "end", of=3, delay_s=40)]
    else:
        units += [u("F1", "spear", 34, -dU / 2, 0), u("F2", "spear", -34, -dU / 2, 0)]
        steps += [s("contact", "end", of="U", delay_s=70)]
    return {"kind": kind, "units": units, "steps": steps, "max_s": 200}


def charge_lane(kind):
    """T-B: S (swordsmen, tested) attacks T (clanrats, standing; answers at the contact): 'run150', 'run40'
    (an attack order at a run from that gap, front to front), 'walk80' (an attack order at a walk), 'step10'
    (a move at a walk to 10 m, then an attack order at a walk)."""
    dS, dT = depth("swords"), depth("clanrat")
    gap = {"run150": 150, "run40": 40, "walk80": 80, "step10": 40}[kind]
    units = [u("S", "swords", 0, gap + dS / 2, 180), u("T", "clanrat", 0, -dT / 2, 0, True)]
    if kind == "step10":
        steps = [s("go", "move_walk", "S", x=0, z=10 + dS / 2), s("go", "attack_walk", "S", delay_s=35, target="T")]
    else:
        steps = [s("go", "attack_walk" if kind == "walk80" else "attack", "S", target="T")]
    steps += [s("contact", "attack_walk", "T", of="S", target="S"), s("contact", "end", of="S", delay_s=25)]
    return {"kind": kind, "units": units, "steps": steps, "max_s": 150}


SECURE_STEPS = ((15, 200), (55, 160), (80, 140), (105, 120), (130, 100), (155, 70))
SECURE_END_S = 185


def secure_lane(kind):
    """T-C: U (spearmen) stands; E (skavenslave spearmen) at 300 m centre to centre moves (at a run) to
    each SECURE_STEPS distance at its time; kinds: 'alone', 'side' (a friend 32 m to the side), 'front'
    (a friend 30 m in front), 'lord' (only the lord 30 m to the side)."""
    dU, dE = depth("spear"), depth("sspear")
    z0 = -dU / 2
    units = [u("U", "spear", 0, z0, 0), u("E", "sspear", 0, z0 + 300, 180, True)]
    if kind == "side":
        units.append(u("F", "spear", 32, z0, 0))
    elif kind == "front":
        units.append(u("F", "spear", 0, z0 + 30, 0))
    elif kind == "lord":
        units.append(u("F", "general", 30, z0, 0))
    steps = [s("go", "move", "E", delay_s=t, x=0, z=round(z0 + d, 1)) for t, d in SECURE_STEPS]
    steps.append(s("go", "end", delay_s=SECURE_END_S))
    return {"kind": kind, "units": units, "steps": steps, "max_s": SECURE_END_S + 10}


def shoot_lane(kind):
    """T-D: 'arch_cr' archers (tested) shoot clanrats 120 m away (centre to centre) for 30 s, 'arch_sl'
    skavenslaves 100 m away; 'sling' fearless slave slingers shoot spearmen (tested) 90 m away. Then the
    fire stops; the lane runs 45 s more (the window after the last volley)."""
    shooter, target, d = {"arch_cr": ("archers", "clanrat", 120), "arch_sl": ("archers", "slave", 100),
                          "sling": ("slingers", "spear", 90)}[kind]
    fear_s = shooter == "slingers"
    units = [u("S", shooter, 0, 0, 0, fear_s), u("T", target, 0, d, 180, not fear_s)]
    steps = [s("go", "shoot", "S", target="T"), s("go", "stop_fire", "S", delay_s=30), s("go", "end", delay_s=75)]
    return {"kind": kind, "units": units, "steps": steps, "max_s": 85}


def rally_lane(kind):
    """T-E: U (skavenslaves, tested) holds; E (Empire spearmen, fearless) attacks it from 40 m; at U's rout E
    halts and then keeps `kind` metres from U (0: stays where it halted); at U's rally E halts. The lane ends
    60 s after the rally or 150 s after the rout."""
    dU, dE = depth("slave"), depth("spear")
    units = [u("U", "slave", 0, -dU / 2, 0), u("E", "spear", 0, 40 + dE / 2, 180, True)]
    steps = [s("go", "attack", "E", target="U"), s("rout", "halt", "E", of="U")]
    if kind:
        steps.append(s("step", "shadow", "E", of=2, delay_s=1.0, target="U", d=float(kind)))
    steps += [s("rally", "halt", "E", of="U"), s("rally", "end", of="U", delay_s=60),
              s("rout", "end", of="U", delay_s=150)]
    return {"kind": kind, "units": units, "steps": steps, "max_s": 330}


STRONG_STEPS = ((0, 60), (60, 40), (85, 25))
STRONG_END_S = 115


def strong_lane(kind):
    """'Enemies superior in strength and speed' (T-F): U (tested) stands; E (fearless) 200 m away (centre to centre)
    runs to 60 m, then 40 m, then 25 m and stands (STRONG_STEPS); kind 'U:E' (short names)."""
    a, b = kind.split(":")
    dU = depth(a)
    z0 = -dU / 2
    units = [u("U", a, 0, z0, 0), u("E", b, 0, z0 + 200, 180, True)]
    steps = [s("go", "move", "E", delay_s=t, x=0, z=round(z0 + d, 1)) for t, d in STRONG_STEPS]
    steps.append(s("go", "end", delay_s=STRONG_END_S))
    return {"kind": kind, "units": units, "steps": steps, "max_s": STRONG_END_S + 10}


def rotate(items, k):
    k %= len(items)
    return items[k:] + items[:k]


def battles(plan):
    """[[lane spec]] of a plan: each inner list one battle."""
    assert plan in PLANS, plan
    if plan == "flank":
        a = [flank_lane(k) for k in ("F", "R", "C")]
        return [a, rotate(a, 1)]
    if plan == "charge":
        return [[charge_lane(k) for k in ("run150", "run40", "walk80", "step10")]]
    if plan == "secure":
        return [[secure_lane(k) for k in ("alone", "side", "front", "lord")]]
    if plan == "shoot":
        return [[shoot_lane(k) for k in ("arch_cr", "sling", "arch_sl")]]
    if plan == "strong":
        return [[strong_lane(k) for k in ("archers:warlord", "archers:nr", "spear:clanrat", "slave:general",
                                          "slave:swords")]]
    return [[rally_lane(k) for k in (0, 50, 90, 150)]]


def lane_dx(plan):
    return 450 if plan == "secure" else 330 if plan == "strong" else 350


def layout(specs, plan):
    """A battle's lanes with places, script names and fearless units, and the arena (tools/nn/scenario.py)."""
    sides = {EMP: [], SKV: []}
    side_of = {EMP: "own", SKV: "enemy"}
    used, fearless, lanes = set(), [], []

    def add(short, k):
        key, men, faction = UNITS[short]
        if short in LORDS.values():
            used.add(short)
            return f"{side_of[faction]}_lord"
        slot = f"{short}_{k}_{len(sides[faction])}"
        sides[faction].append({"slot": slot, "key": key, "men": men, "forward": -20 - 40 * (len(sides[faction]) // 4),
                               "lateral": 40 * (len(sides[faction]) % 4 - 1.5), "width": WIDTH_M})
        return f"{side_of[faction]}_{slot}"
    n, dx = len(specs), lane_dx(plan)
    tested = set()
    for k, spec in enumerate(specs, 1):
        x = dx * (k - 1 - (n - 1) / 2)
        units = []
        for un in spec["units"]:
            name = add(un["short"], k)
            row = dict(un, name=name, key=UNITS[un["short"]][0], width=5 if UNITS[un["short"]][1] == 1 else WIDTH_M)
            units.append(row)
            if un["fearless"]:
                fearless.append(name)
            else:
                tested.add(UNITS[un["short"]][2])
        lanes.append({"name": f"L{k}", "kind": spec["kind"], "x": round(x, 1), "z": 0.0, "max_s": spec["max_s"],
                      "units": units, "steps": spec["steps"]})
    park = []
    for faction in (EMP, SKV):
        sign = -1 if faction == EMP else 1
        for i, short in enumerate(RESERVES[faction]):
            name = add(short, 9)
            fearless.append(name)
            park.append({"name": name, "x": sign * (850 - 60 * i), "z": PARK_Z, "bearing": 0, "width": WIDTH_M})
        if LORDS[faction] not in used:
            park.append({"name": f"{side_of[faction]}_lord", "x": sign * 960, "z": PARK_Z, "bearing": 0, "width": 5})
            if faction not in tested:
                fearless.append(f"{side_of[faction]}_lord")
    base = nn_scenario.load_arena()
    armies = {}
    for faction, units in sides.items():
        key = UNITS[LORDS[faction]][0]
        armies[side_of[faction]] = {"faction": faction, "units": [
            {"slot": "lord", "key": key, "men": 1, "general": True, "forward": -80, "lateral": 0, "width": 5}] + units}
    arena = {k: v for k, v in base.items() if k not in ("faction", "units", "description")}
    arena.update(name="morale_probe", gap_m=400, defend_radius_m=150, sides=armies)
    return lanes, park, sorted(set(fearless)), arena


def run_config(plan, index):
    """(entry config, model seconds, arena) of battle `index` (1-based) of a plan."""
    lanes, park, fearless, arena = layout(battles(plan)[index - 1], plan)
    config = {"plan": plan, "battle": index, "settle_ms": SETTLE_MS, "tick_ms": TICK_MS, "lanes": lanes,
              "park": park, "fearless": fearless}
    model_s = max(l["max_s"] for l in lanes) + SETTLE_MS / 1000 + 20
    return config, model_s, arena


def write_scenario(plan, index, path=None):
    _, _, arena = run_config(plan, index)
    path = path or ROOT / f"morale_probe_{plan}_{index}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nn_scenario.scenario_xml(arena, "enemy", 3600), encoding="utf-8", newline="\n")
    return path


def run(plan, which=None, python=sys.executable, dry=False, deadline=1200):
    """Builds and launches each battle in turn (tools/launcher/launch.ps1: Normal difficulty set and restored by
    the launcher)."""
    done = []
    for i in which or range(1, len(battles(plan)) + 1):
        build = [python, "-m", "tools.build", "morale-probe", "--morale-plan", plan, "--morale-battle", str(i),
                 "--deadline", str(deadline)]
        launch = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAUNCHER / "launch.ps1"),
                  "-Target", "morale-probe"]
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
    """[{run, plan, battle, spec, t [s], rows {role: [row]}, events {kind: {role: s}}, steps {i: s}, end}]."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    lanes = {l["name"]: {"run": run_dir.name, "plan": cfg.get("plan"), "battle": cfg.get("battle"), "spec": l,
                         "t": [], "rows": {x["role"]: [] for x in l["units"]},
                         "events": {"contact": {}, "rout": {}, "rally": {}}, "steps": {}, "end": None}
             for l in cfg.get("lanes", [])}
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if '"probe_' not in line:
            continue
        r = json.loads(line)
        ev = r["event"]
        if ev == "probe_sample":
            for x in r["lanes"]:
                ln = lanes.get(x["lane"])
                if ln is None:
                    continue
                ln["t"].append(x["t"] / 1000)
                for role in ln["rows"]:
                    ln["rows"][role].append((x.get("u") or {}).get(role) or {})
        elif ev == "probe_event":
            lanes[r["lane"]]["events"][r["kind"]].setdefault(r["role"], r["t"] / 1000)
        elif ev == "probe_step":
            lanes[r["lane"]]["steps"][r["i"]] = r["t"] / 1000
        elif ev == "probe_lane_end":
            lanes[r["lane"]]["end"] = r
    return [x for x in lanes.values() if x["t"]]


def runs(root=RUNS):
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if (d / "events.jsonl").exists() and (d / "manifest.json").exists())


def col(lane, role, key):
    return np.array([np.nan if r.get(key) is None else float(r[key]) for r in lane["rows"][role]], float)


def flag(lane, role, key):
    return np.array([bool(r.get(key)) for r in lane["rows"][role]])


def clean(text):
    """The game's effect text without its colour markup ([[col:red]]...[[/col]])."""
    return re.sub(r"\[\[/?col[^\]]*\]\]", "", text).strip() if text else text


def labels(lane, role):
    return [clean(r.get("mge")) for r in lane["rows"][role]]


def points(lane, role):
    """MoralePercent x the card's leadership (the game) or the simulator's points."""
    if lane.get("sim"):
        return col(lane, role, "pts")
    key = next(x["key"] for x in lane["spec"]["units"] if x["role"] == role)
    return col(lane, role, "mp") * leadership(key)


def at(t, v, when, how="interp"):
    ok = np.isfinite(v)
    if when is None or ok.sum() < 2 or when < t[ok][0] or when > t[ok][-1]:
        return None
    return round(float(np.interp(when, t[ok], v[ok])), 2)


def runs_of(lab, name):
    """[(start index, end index exclusive)] of consecutive samples showing the label."""
    out, start = [], None
    for i, x in enumerate(list(lab) + [None]):
        if x == name and start is None:
            start = i
        elif x != name and start is not None:
            out.append((start, i))
            start = None
    return out


def dist(lane, a, b):
    return np.hypot(col(lane, a, "x") - col(lane, b, "x"), col(lane, a, "z") - col(lane, b, "z"))


def measure(lane):
    """One lane's numbers (game or simulator: the same row format)."""
    spec, t = lane["spec"], np.array(lane["t"])
    ev, steps = lane["events"], lane["steps"]
    plan = lane["plan"]
    out = {"run": lane["run"], "lane": spec["name"], "plan": plan, "kind": spec["kind"]}
    r1 = lambda v: None if v is None else round(float(v), 1)
    if plan == "flank":
        pts, lab = points(lane, "U"), labels(lane, "U")
        c = ev["contact"].get("U")
        out["contact_s"] = r1(c)
        if c is not None:
            # the front fight before B: contact + 5 ... + 18 s
            sel = (t >= c + 5) & (t <= c + 18)
            out["pts_front_5_18"] = r1(np.nanmean(pts[sel])) if sel.any() else None
            out["label_front_5_18"] = most(lab, sel)
        cb = ev["contact"].get("B")
        out["b_contact_s"] = r1(cb)
        leave = steps.get(3)
        out["b_leave_s"] = r1(leave)
        if cb is not None:
            for dt_ in (-1, 2, 4, 8, 15):
                out[f"pts_b{dt_:+d}"] = at(t, pts, cb + dt_)
            name = FLANK if spec["kind"] == "F" else REAR
            spans = [(t[i], t[j - 1] + 0.5) for i, j in runs_of(lab, name) if t[i] >= cb - 2]
            out["label_spans"] = [(r1(a - cb), r1(b - cb)) for a, b in spans]
            out["label_s_total"] = r1(sum(b - a for a, b in spans))
            out["label_last_after_leave_s"] = r1(spans[-1][1] - leave) if spans and leave is not None else None
            sim_term = lane.get("terms", {}).get("flank")
            if sim_term is not None:
                on = np.abs(sim_term) > 0
                sp = [(t[i], t[j - 1] + 0.5) for i, j in runs_of(list(on), True) if t[i] >= cb - 2]
                out["term_spans"] = [(r1(a - cb), r1(b - cb)) for a, b in sp]
                out["term_s_total"] = r1(sum(b - a for a, b in sp))
        if leave is not None:
            for dt_ in (-1, 5, 10, 20):
                out[f"pts_leave{dt_:+d}"] = at(t, pts, leave + dt_)
    elif plan == "charge":
        pts, lab = points(lane, "S"), labels(lane, "S")
        c = ev["contact"].get("S")
        out["contact_s"] = r1(c)
        spans = [(t[i], t[j - 1] + 0.5) for i, j in runs_of(lab, CHARGE)]
        out["label_spans_vs_contact"] = [(r1(a - c), r1(b - a)) for a, b in spans] if c is not None else None
        fast = flag(lane, "S", "fast")
        out["first_fast_s"] = r1(t[np.argmax(fast)]) if fast.any() else None
        for dt_ in (-6, -4, -2, 0, 2, 6, 12):
            out[f"pts_c{dt_:+d}"] = at(t, pts, None if c is None else c + dt_)
        out["pts_go"] = at(t, pts, 0.0)
        term = lane.get("terms", {}).get("charge")
        if term is not None and c is not None:
            sp = [(t[i], t[j - 1] + 0.5) for i, j in runs_of(list(np.abs(term) > 0), True)]
            out["term_spans_vs_contact"] = [(r1(a - c), r1(b - a)) for a, b in sp]
    elif plan == "secure":
        pts, lab = points(lane, "U"), labels(lane, "U")
        d = dist(lane, "U", "E")
        holds = [(0.0, SECURE_STEPS[0][0])] + [(SECURE_STEPS[i][0], SECURE_STEPS[i + 1][0])
                                               for i in range(len(SECURE_STEPS) - 1)] + [(SECURE_STEPS[-1][0], SECURE_END_S)]
        out["steps"] = []
        for a, b in holds:
            sel = (t >= b - 5) & (t < b)
            if sel.any():
                out["steps"].append({"enemy_m": r1(np.nanmedian(d[sel])), "pts": r1(np.nanmedian(pts[sel])),
                                     "label": most(lab, sel)})
    elif plan == "strong":
        pts, lab = points(lane, "U"), labels(lane, "U")
        d = dist(lane, "U", "E")
        out["steps"] = []
        edges = [0.0] + [t for t, _ in STRONG_STEPS[1:]] + [STRONG_END_S]
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (t >= b - 5) & (t < b)
            if sel.any():
                out["steps"].append({"enemy_m": r1(np.nanmedian(d[sel])), "pts": r1(np.nanmedian(pts[sel])),
                                     "label": most(lab, sel)})
        sel = (t >= 0) & (t < 3)
        out["pts_go"] = r1(np.nanmedian(pts[sel])) if sel.any() else None
    elif plan == "shoot":
        tested = "S" if not next(x for x in spec["units"] if x["role"] == "S")["fearless"] else "T"
        pts, lab = points(lane, tested), labels(lane, tested)
        stop = steps.get(2)
        out["tested"] = tested
        out["pts_go"] = at(t, pts, 0.0)
        for a, b in ((5, 15), (15, 30), (30, 40), (40, 50), (50, 75)):
            sel = (t >= a) & (t < b)
            out[f"pts_{a}_{b}"] = r1(np.nanmean(pts[sel])) if sel.any() else None
            out[f"label_{a}_{b}"] = most(lab, sel)
        fire = flag(lane, "S", "fire")
        out["last_fire_s"] = r1(t[np.nonzero(fire)[0][-1]]) if fire.any() else None
        out["win_last_s"] = r1(t[[i for i, x in enumerate(lab) if x == WIN][-1]]) if WIN in lab else None
        out["hp_lost_T"] = r1(1 - np.nanmin(col(lane, "T", "hp"))) if "hp" in lane["rows"]["T"][0] else None
    elif plan == "rally":
        pts = points(lane, "U")
        rout, rally = ev["rout"].get("U"), ev["rally"].get("U")
        out["rout_s"], out["rally_s"] = r1(rout), r1(rally)
        d = dist(lane, "U", "E")
        if rout is not None:
            after = t >= rout
            far = after & (d > 80)
            out["free80_s"] = r1(t[np.argmax(far)]) if far.any() else None
            out["pts_rout"] = at(t, pts, rout)
            for dt_ in (5, 10, 20, 30):
                out[f"pts_rout+{dt_}"] = at(t, pts, rout + dt_)
        if rally is not None:
            out["pts_rally"] = at(t, pts, rally)
            out["mp_rally"] = at(t, col(lane, "U", "mp"), rally) if not lane.get("sim") else \
                at(t, pts / lane["L"]["U"], rally)
            out["enemy_m_rally"] = at(t, d, rally)
            out["wait_after_free_s"] = r1(rally - out["free80_s"]) if out.get("free80_s") is not None else None
            r = flag(lane, "U", "r")
            back = (t > rally + 0.5) & r
            out["rerout_after_s"] = r1(t[np.argmax(back)] - rally) if back.any() else None
        out["lost_hp"] = r1(1 - np.nanmin(col(lane, "U", "hp")))
    return out


def most(lab, sel):
    from collections import Counter
    xs = [x for x, k in zip(lab, sel) if k and x]
    return Counter(xs).most_common(1)[0][0] if xs else None


# ---------------------------------------------------------------- the simulator on the same lanes

def sim_lanes(lanes, params=None, device="cpu", copies=4, jitter_m=1.0, seed=0, collapse=False):
    """The simulator on the probe's lanes (recorded, or a plan's: then the planned places): each lane its own
    battle, its units at their places at the probe's go, the same scripted steps by the same triggers, the
    fearless units given +10000 leadership, both sides a far reserve unit (a lane alone must not end the battle
    when its tested unit routs); army losses off unless `collapse` (in the game the whole army counts: the
    probe's runs show no 'army losses'). Returns lanes in the recordings' shape plus the simulator's points
    ('pts'), its leadership (L) and, if the simulator exposes it (tools.nn.sim.morale.TRACE), each morale
    term per sample (lane['terms'])."""
    import torch
    from tools.nn.sim import battle, morale as sim_morale, orders as O, scenario as sim_scenario
    from tools.nn.sim.params import load
    params = params or load()
    if not collapse and (params.sim["morale"].get("collapse") or {}).get("on"):
        params.sim["morale"]["collapse"] = dict(params.sim["morale"]["collapse"], on=False)
    armies, meta = [], []
    for ln in lanes:
        spec = ln["spec"]
        sides = {1: {"faction": EMP, "ai": False, "units": []}, 2: {"faction": SKV, "ai": False, "units": []}}
        first = {role: rows[0] for role, rows in ln["rows"].items()} if ln.get("rows") else {}
        for un in spec["units"]:
            fac = 1 if UNITS[un["short"]][2] == EMP else 2
            r0 = first.get(un["role"]) or {}
            x = r0.get("x", spec["x"] + un["x"])
            z = r0.get("z", spec["z"] + un["z"])
            b = r0.get("b", un["b"])
            lord = un["short"] in LORDS.values()
            sides[fac]["units"].append({"key": un["key"], "x": x, "z": z, "b": b, "men": UNITS[un["short"]][1],
                                        "width": None if lord else WIDTH_M, "general": lord, "name": un["role"]})
        for fac, short in ((1, "gs"), (2, "storm")):
            sides[fac]["units"].append({"key": UNITS[short][0], "x": (-1 if fac == 1 else 1) * 900, "z": 900, "b": 0,
                                        "men": UNITS[short][1], "width": WIDTH_M, "name": f"res{fac}"})
        army = {"attacker": 1, "sides": sides}
        for _ in range(copies):
            armies.append(army)
            meta.append(ln)
    per_side = max(len(a["sides"][sd]["units"]) for a in armies for sd in (1, 2))
    st = sim_scenario.build(armies, params, device=device, per_side=per_side)
    B, N = st.B, st.N
    gen = torch.Generator().manual_seed(seed)
    for k in ("x", "z"):
        st.u[k] = st.u[k] + ((torch.rand(st.u[k].shape, generator=gen) * 2 - 1) * jitter_m).to(device)
    roles = sorted({un["role"] for ln in meta for un in ln["spec"]["units"]})
    slot = {r: [sim_scenario.slots(a, per_side).get(r, -1) for a in armies] for r in roles + ["res1", "res2"]}
    fear = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b, ln in enumerate(meta):
        for un in ln["spec"]["units"]:
            if un["fearless"]:
                fear[b, slot[un["role"]][b]] = True
        fear[b, slot["res1"][b]] = fear[b, slot["res2"][b]] = True
    # the probe turns fire at will off: only a unit with a 'shoot' step shoots (the simulator has no such switch)
    for b, ln in enumerate(meta):
        shooters = {stp["unit"] for stp in ln["spec"]["steps"] if stp["do"] == "shoot"}
        for un in ln["spec"]["units"]:
            if un["role"] not in shooters:
                st.u["a"][b, slot[un["role"]][b]] = 0
    st.u["leadership"] = torch.where(fear, st.u["leadership"] + 1e4, st.u["leadership"])
    st.u["morale"] = torch.where(fear, st.u["morale"] + 1e4, st.u["morale"])
    L = st.u["leadership"].detach().cpu().numpy()
    book = [{"go": 0.0, "contact": {}, "rout": {}, "rally": {}, "step": {}} for _ in range(B)]
    shadows = [dict() for _ in range(B)]
    last = {}
    rec = {"t": [], "rows": [], "terms": []}
    fields = ("x", "z", "b", "men", "hp", "m", "r", "morale", "f", "fire")

    trace = []
    has_trace = hasattr(sim_morale, "TRACE")

    def policy(st):
        uu = st.u
        t = float(st.t.max())
        m = uu["m"].bool().cpu().numpy()
        r = uu["r"].bool().cpu().numpy()
        x, z = uu["x"].cpu().numpy(), uu["z"].cpu().numpy()
        o = O.hold(B, N, device)
        kind, target, run_, ox, oz = o.kind.clone(), o.target.clone(), o.run.clone(), o.x.clone(), o.z.clone()
        kind[:] = O.KEEP
        for b, ln in enumerate(meta):
            spec, ev = ln["spec"], book[b]
            if ev.get("ended"):
                continue
            for un in spec["units"]:
                i = slot[un["role"]][b]
                if m[b, i] and un["role"] not in ev["contact"]:
                    ev["contact"][un["role"]] = t
                if r[b, i] and un["role"] not in ev["rout"]:
                    ev["rout"][un["role"]] = t
                if un["role"] in ev["rout"] and not r[b, i] and un["role"] not in ev["rally"]:
                    ev["rally"][un["role"]] = t
            for k, stp in enumerate(spec["steps"], 1):
                if k in ev["step"]:
                    continue
                at_ = ev["go"] if stp["on"] == "go" else ev[stp["on"]].get(stp.get("of"))
                if at_ is None or t - at_ < stp.get("delay_s", 0.0) - 1e-6:
                    continue
                ev["step"][k] = t
                d = stp["do"]
                if d == "end":
                    ev["ended"] = t
                    continue
                i = slot[stp["unit"]][b]
                j = slot[stp["target"]][b] if stp.get("target") else -1
                if d in ("attack", "attack_walk", "shoot"):
                    kind[b, i], target[b, i], run_[b, i] = O.ATTACK, j, d == "attack"
                elif d in ("move", "move_walk"):
                    kind[b, i], ox[b, i], oz[b, i], run_[b, i] = O.MOVE, spec["x"] + stp["x"], spec["z"] + stp["z"], d == "move"
                elif d in ("halt", "stop_fire"):
                    shadows[b].pop(stp["unit"], None)
                    kind[b, i] = O.HOLD
                    if d == "stop_fire":
                        # the game's probe turns fire at will off; the simulator has no such switch: no ammunition
                        uu["a"][b, i] = 0
                elif d == "shadow":
                    shadows[b][stp["unit"]] = (stp["target"], float(stp["d"]))
            for role, (tr, dd) in shadows[b].items():
                i, j = slot[role][b], slot[tr][b]
                vx, vz = x[b, i] - x[b, j], z[b, i] - z[b, j]
                n = math.hypot(vx, vz) or 1.0
                px, pz = float(x[b, j] + vx / n * dd), float(z[b, j] + vz / n * dd)
                if math.hypot(x[b, i] - px, z[b, i] - pz) > 8:
                    kind[b, i], ox[b, i], oz[b, i], run_[b, i] = O.MOVE, px, pz, True
                    last[(b, role)] = "move"
                elif last.get((b, role)) == "move":
                    kind[b, i] = O.HOLD
                    last[(b, role)] = "hold"
        return O.Orders(kind=kind, x=ox, z=oz, target=target, run=run_, ability=o.ability)

    def record(st):
        rec["t"].append(float(st.t.max()))
        rec["rows"].append({k: st.u[k].detach().float().cpu().numpy().copy() for k in fields})
        if has_trace and trace:
            rec["terms"].append({k: v.detach().float().cpu().numpy().copy() for k, v in trace[-1].items()})
            trace.clear()
    until = max(ln["spec"]["max_s"] for ln in meta) + 1
    if has_trace:
        sim_morale.TRACE = trace
    try:
        record(st)
        battle.run(st, policy, params, until_s=until, record=record, compile=False)
    finally:
        if has_trace:
            sim_morale.TRACE = None
    out = []
    t_all = np.array(rec["t"])
    for b, ln in enumerate(meta):
        spec, ev = ln["spec"], book[b]
        end = ev.get("ended", t_all[-1])
        keep = t_all <= end + 1e-6
        rows = {}
        for un in spec["units"]:
            i = slot[un["role"]][b]
            rows[un["role"]] = [{"x": float(r["x"][b, i]), "z": float(r["z"][b, i]), "b": float(r["b"][b, i]),
                                 "men": float(r["men"][b, i]), "hp": float(r["hp"][b, i]), "m": bool(r["m"][b, i]),
                                 "r": bool(r["r"][b, i]), "fast": bool(r["f"][b, i]),
                                 "pts": float(r["morale"][b, i]), "fire": bool(r["fire"][b, i])}
                                for r, k in zip(rec["rows"], keep) if k]
        terms = {}
        if rec["terms"]:
            tested = next((un for un in spec["units"] if not un["fearless"]), None)
            if tested is not None:
                i = slot[tested["role"]][b]
                names = rec["terms"][0].keys()
                # term samples are one per step after the first record
                tt = [None] + rec["terms"]
                terms = {k: np.array([0.0 if x is None else float(x[k][b, i]) for x, kk in zip(tt, keep) if kk])
                         for k in names}
                lab = []
                for q in range(len(next(iter(terms.values())))):
                    best = max(terms, key=lambda k: abs(terms[k][q]))
                    lab.append(best if abs(terms[best][q]) > 0 else None)
                rows[tested["role"]] = [dict(r, mge=SIM_LABEL.get(lb, lb)) for r, lb in zip(rows[tested["role"]], lab)]
        out.append({"run": "sim", "plan": ln["plan"], "battle": ln.get("battle"), "spec": spec, "sim": True,
                    "t": list(t_all[keep]), "rows": rows, "terms": terms,
                    "events": {k: dict(ev[k]) for k in ("contact", "rout", "rally")},
                    "steps": dict(ev["step"]), "L": {un["role"]: float(L[b, slot[un["role"]][b]]) for un in spec["units"]}})
    return out


# the simulator's term names (tools/nn/sim/morale.py terms) as the game's labels
SIM_LABEL = {"flank": None, "charge": CHARGE, "secure": SECURE, "combat": None}


def plan_lanes(plan):
    """A plan's lanes in the recordings' shape without samples (the simulator twin from the planned places)."""
    out = []
    for i, specs in enumerate(battles(plan), 1):
        lanes, _, _, _ = layout(specs, plan)
        out += [{"run": "plan", "plan": plan, "battle": i, "spec": ln, "rows": None} for ln in lanes]
    return out


# ---------------------------------------------------------------- tables

def report(run_dirs, sim=False, out=OUT, device="cpu", copies=4):
    lanes = [ln for d in run_dirs for ln in load_run(d)]
    game = [measure(ln) for ln in lanes]
    result = {"lanes": game}
    if sim:
        sims = sim_lanes(lanes, device=device, copies=copies)
        result["sim"] = [measure(ln) for ln in sims]
    for r in game:
        print(json.dumps(r, ensure_ascii=False, default=float))
    for r in result.get("sim", []):
        print("sim", json.dumps(r, ensure_ascii=False, default=float))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(out)
    return result


def twin(plans, out=None, device="cpu", copies=4):
    lanes = [ln for p in plans for ln in plan_lanes(p)]
    sims = sim_lanes(lanes, device=device, copies=copies)
    rows = [measure(ln) for ln in sims]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False, default=float))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("report", "plan", "run", "twin"), default="report")
    parser.add_argument("runs", nargs="*", type=Path, help="report: run folders (default: build/morale-probe/runs/*)")
    parser.add_argument("--plan", default="flank", help="a plan, or several comma separated (twin)")
    parser.add_argument("--battles", help="run: battle numbers, comma separated (default: all of the plan)")
    parser.add_argument("--sim", action="store_true", help="report: also the simulator on the same lanes (torch)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "plan":
        for p in args.plan.split(","):
            for i, b in enumerate(battles(p), 1):
                print(p, i, [ln["kind"] for ln in b])
        return 0
    if args.command == "run":
        which = [int(x) for x in args.battles.split(",")] if args.battles else None
        done = []
        for p in args.plan.split(","):
            done += run(p, which, dry=args.dry)
        print("done:", done)
        return 0 if all(d[-1] == 0 for d in done) else 1
    if args.command == "twin":
        twin(args.plan.split(","), out=args.out, device=args.device, copies=args.copies)
        return 0
    report(args.runs or runs(), sim=args.sim, out=args.out or OUT, device=args.device, copies=args.copies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
