"""The missile probe (src/entries/missile_probe.lua): one shooting indicator, one scenario, in the game and in the
simulator alike. Lanes far apart, in each a shooter and a target whose centre stands d m away at t_angle degrees
off the shooter's facing, turned t_rot degrees from facing it; everyone held by script and fearless; the
shooter fires at will (or is given an attack order) until out of ammo or the target is dead.

Plans (each battle 4-5 lanes):
  dist     each shooter on an unarmoured target straight ahead (Empire archers and militia on skavenslaves,
           slave slingers and Night Runners on flagellants), one distance a battle: archers 40 / 80 / 120 / 128,
           militia 30 / 50 / 70 / 88, slingers 40 / 80 / 110 / 118, Night Runners 40 / 80 / 120 / 138
           (4 battles): the reload (volley intervals), hits per projectile by distance and by men left;
  arc      the per-man fire arc: archers (46 m front) on a narrow target (40 skavenslaves, 6 m wide) 60 m away
           at 35 deg, on a wide one (180, 60 m wide) at 35 deg, on the narrow one at 50 deg; militia (32 m) on the
           narrow one at 30 deg (1 battle): first-volley size, turn;
  range    the target steps 2 m closer every 2 s from 150 m until the first arrow: clanrat spearmen 60 / 30 m wide,
           skavenslave spearmen 15 m wide, slave slingers (1 battle): centre or edge rule;
  targets  at 100 m: archers on the Warlord (his back to them), slingers on the General (his back), archers on
           skavenslaves and on clanrat spearmen (1 battle): lone men vs formations;
  shield   archers at 90 m on clanrats (shield 35) facing them, on clanrat spearmen (no shield, the same body)
           facing them, both turned 90 deg, and clanrats turned 180 (1 battle);
  moving   skavenslaves running at the archers (145 -> 25 m), crossing at a walk at ~100 m (-40 -> +40 m across),
           running away (40 -> 170 m); flagellants running at Night Runners (150 -> 25 m) (1 battle);
  rank     archers rank 0 / 9 on skavenslaves, slingers rank 0 / 9 on flagellants, all at 100 m (1 battle);
  thin     P1 holes or re-formed: archers at ~100 m on skavenslaves 180 (control, thinned by the fire), fresh 90,
           fresh 45 (30 m wide: fewer ranks), and 180 re-formed by a 5 m move order once down to 60 (2 battles);
  pistol   P2 militia's range: on skavenslaves at ~86 / 90 / 94 / 98 m centre, and a 2-rank target (80 m wide) at
           ~94 m (1 battle);
  moving2  P3: skavenslaves running at the archers (160 -> 40 m, twice), running away (50 -> 175 m, twice), a
           standing twin at ~120 m: hits per volley at the same distance (1 battle);
  lof      P4 the line of fire past friends: militia, Empire spearmen 40 m in front of them offset 0 / 15 / 22.5 m
           sideways, skavenslaves ~70 m away; and no friend (1 battle): who fires, the friends' HP lost.

    python -m tools.nn.missile_probe plan [--plan dist]
    python -m tools.build missile-probe --mprobe-plan dist --mprobe-battle 1   # one battle's build
    python -m tools.nn.missile_probe run --plan dist [--battles 1,2]           # build + launch each, in turn
    python -m tools.nn.missile_probe report [runs...] [--sim] [--out FILE]     # the game (and its sim twin)

Measures per lane (the game's recording, and the simulator's run of the same lane from the same start): volleys
from the shooter's ammo (a volley: the drops within 1.5 s of each other), their size and the median interval of
whole volleys; the first shot's time and centre distance; the shooter's turn; projectiles fired, the target's HP
lost per projectile (all, and while it still has 3/4 of its men), and hits per projectile = that / the rule's
HP a hit (melee.per_hit: armour, overkill). Writes build/missile/probe.json (not in Git).
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

ROOT = project.BUILD / "missile-probe"
RUNS = ROOT / "runs"
OUT = project.BUILD / "missile" / "probe.json"
LAUNCHER = project.ROOT / "tools" / "launcher"
EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"
UNITS = {
    "archers": ("wh2_dlc13_emp_inf_archers_0", 90, EMP),
    "militia": ("wh_dlc04_emp_inf_free_company_militia_0", 120, EMP),
    "slingers": ("wh2_main_skv_inf_skavenslave_slingers_0", 140, SKV),
    "nr": ("wh2_main_skv_inf_night_runners_1", 120, SKV),
    "slave": ("wh2_main_skv_inf_skavenslaves_0", 180, SKV),
    "sspear": ("wh2_main_skv_inf_skavenslave_spearmen_0", 180, SKV),
    "cspear": ("wh2_main_skv_inf_clanrat_spearmen_0", 160, SKV),
    "clanrat": ("wh2_main_skv_inf_clanrats_1", 160, SKV),
    "flag": ("wh_dlc04_emp_inf_flagellants_0", 120, EMP),
    "spear": ("wh_main_emp_inf_spearmen_0", 120, EMP),
    "general": ("wh_main_emp_cha_general_0", 1, EMP),
    "warlord": ("wh2_main_skv_cha_warlord_0", 1, SKV),
}
LORDS = {EMP: "general", SKV: "warlord"}
SHORT = {key: short for short, (key, _, _) in UNITS.items()}
FACTION = {key: faction for key, _, faction in UNITS.values()}
LANE_DX = 240
Z0 = -80               # the shooters' line
SETTLE_MS = 4000
TICK_MS = 500
PLANS = ("dist", "arc", "range", "targets", "shield", "moving", "rank", "thin", "pistol", "moving2", "lof")
DIST = {"archers": (40, 80, 120, 128), "militia": (30, 50, 70, 88), "slingers": (40, 80, 110, 118),
        "nr": (40, 80, 120, 138)}
TARGET_OF = {"archers": "slave", "militia": "slave", "slingers": "flag", "nr": "flag"}


def lane(shooter, target, d, mode="fire", target_mode="stand", s_width=30, t_width=30, **extra):
    out = {"shooter": shooter, "target": target, "d": float(d), "mode": mode, "target_mode": target_mode,
           "s_width": s_width, "t_width": t_width, "t_angle": 0.0, "t_rot": 0.0, "max_s": 300, "idle_s": 30,
           "settle_s": 6}
    out.update(extra)
    return out


def rotate(items, k):
    k %= len(items)
    return items[k:] + items[:k]


def battles(plan):
    """[[lane spec]] of a plan: each inner list one battle."""
    assert plan in PLANS, plan
    if plan == "dist":
        return [rotate([lane(s, TARGET_OF[s], DIST[s][k]) for s in DIST], k) for k in range(4)]
    if plan == "arc":
        return [[lane("archers", "slave", 60, s_width=46, t_width=6, t_men=40, t_angle=35),
                 lane("archers", "slave", 60, s_width=46, t_width=60, t_angle=35),
                 lane("archers", "slave", 60, s_width=46, t_width=6, t_men=40, t_angle=50),
                 lane("militia", "slave", 60, s_width=32, t_width=6, t_men=40, t_angle=30)]]
    if plan == "range":
        step = dict(target_mode="step", start_d=150, step_m=2, step_s=2)
        return [[lane("archers", "cspear", 150, t_width=60, **step), lane("archers", "cspear", 150, **step),
                 lane("archers", "sspear", 150, t_width=15, **step),
                 lane("archers", "slingers", 150, t_width=62, **step)]]
    if plan == "targets":
        return [[lane("archers", "warlord", 100, t_rot=180), lane("slingers", "general", 100, t_rot=180),
                 lane("archers", "slave", 100), lane("archers", "cspear", 100)]]
    if plan == "shield":
        return [[lane("archers", "clanrat", 90), lane("archers", "cspear", 90), lane("archers", "clanrat", 90, t_rot=90),
                 lane("archers", "cspear", 90, t_rot=90), lane("archers", "clanrat", 90, t_rot=180)]]
    if plan == "moving":
        return [[lane("archers", "slave", 145, target_mode="move", move_fwd=25, move_lat=0, move_run=True),
                 lane("archers", "slave", round(math.hypot(100, 40), 1), target_mode="move",
                      t_angle=round(-math.degrees(math.atan2(40, 100)), 1), t_rot=round(math.degrees(math.atan2(40, 100)), 1) - 90,
                      move_fwd=100, move_lat=40, move_run=False),
                 lane("archers", "slave", 40, target_mode="move", t_rot=180, move_fwd=170, move_lat=0, move_run=True),
                 lane("nr", "flag", 150, target_mode="move", move_fwd=25, move_lat=0, move_run=True)]]
    if plan == "thin":
        base = [lane("archers", "slave", 90), lane("archers", "slave", 90, t_men=90),
                lane("archers", "slave", 90, t_men=45), lane("archers", "slave", 90, reform_men=60)]
        return [base, rotate(base, 2)]
    if plan == "pistol":
        return [[lane("militia", "slave", d) for d in (76, 80, 84, 88)] + [lane("militia", "slave", 84, t_width=80)]]
    if plan == "moving2":
        at = dict(target_mode="move", move_fwd=40, move_lat=0, move_run=True)
        away = dict(target_mode="move", t_rot=180, move_fwd=175, move_lat=0, move_run=True)
        return [[lane("archers", "slave", 150, **at), lane("archers", "slave", 40, **away), lane("archers", "slave", 110),
                 lane("archers", "slave", 150, **at), lane("archers", "slave", 40, **away)]]
    if plan == "lof":
        fr = dict(friend="spear", friend_fwd=40, friend_width=30)
        return [[lane("militia", "slave", 60, friend_lat=0, **fr), lane("militia", "slave", 60, friend_lat=15, **fr),
                 lane("militia", "slave", 60, friend_lat=22.5, **fr), lane("militia", "slave", 60)]]
    return [[lane("archers", "slave", 100), lane("archers", "slave", 100, s_rank=9),
             lane("slingers", "flag", 100), lane("slingers", "flag", 100, s_rank=9)]]


def layout(specs):
    """A battle's lanes with places and script names, and the arena (tools/nn/scenario.py shape). Each side's
    lord is always there (parked far unless a lane uses him)."""
    sides = {EMP: [], SKV: []}
    side_of = {EMP: "own", SKV: "enemy"}
    lanes, used_lords, ranks = [], set(), {}

    def add(short, k, men=None):
        key, men0, faction = UNITS[short]
        if short in LORDS.values():
            used_lords.add(short)
            return f"{side_of[faction]}_lord"
        slot = f"{short}_{k}"
        sides[faction].append({"slot": slot, "key": key, "men": men or men0,
                               "forward": -20 - 40 * (len(sides[faction]) // 4),
                               "lateral": 40 * (len(sides[faction]) % 4 - 1.5), "width": 30})
        return f"{side_of[faction]}_{slot}"
    n = len(specs)
    for k, spec in enumerate(specs, 1):
        s_key, _, s_fac = UNITS[spec["shooter"]]
        t_key, _, t_fac = UNITS[spec["target"]]
        assert s_fac != t_fac, spec
        x = LANE_DX * (k - 1 - (n - 1) / 2)
        row = dict(spec, name=f"L{k}", x=round(x, 1), z=Z0, s_b=0.0, shooter=add(spec["shooter"], k),
                   target=add(spec["target"], k, spec.get("t_men")), s_key=s_key, t_key=t_key)
        if spec.get("friend"):
            row["friend"] = add(spec["friend"], k)
            row["f_key"] = UNITS[spec["friend"]][0]
        if t_key in (UNITS["general"][0], UNITS["warlord"][0]):
            row["t_width"] = 5
        if spec.get("s_rank"):
            ranks[row["shooter"]] = int(spec["s_rank"])
        lanes.append(row)
    base = nn_scenario.load_arena()
    armies = {}
    for faction, units in sides.items():
        key, _, _ = UNITS[LORDS[faction]]
        armies[side_of[faction]] = {"faction": faction, "units": [
            {"slot": "lord", "key": key, "men": 1, "general": True, "forward": -80, "lateral": 0, "width": 5}] + units}
    park = [{"name": f"{side_of[f]}_lord", "x": (-700 if f == EMP else 700), "z": -400, "bearing": 0, "width": 5}
            for f, lord in LORDS.items() if lord not in used_lords]
    arena = {k: v for k, v in base.items() if k not in ("faction", "units", "description")}
    arena.update(name="missile_probe", gap_m=400, defend_radius_m=150, sides=armies)
    return lanes, park, arena, ranks


def run_config(plan, index):
    """(entry config, model seconds, arena, ranks) of battle `index` (1-based) of a plan."""
    specs = battles(plan)[index - 1]
    lanes, park, arena, ranks = layout(specs)
    config = {"plan": plan, "battle": index, "settle_ms": SETTLE_MS, "tick_ms": TICK_MS, "lanes": lanes, "park": park}
    model_s = max(l["max_s"] + l["settle_s"] for l in lanes) + SETTLE_MS / 1000 + 20
    return config, model_s, arena, ranks


def write_scenario(plan, index, path=None):
    _, _, arena, ranks = run_config(plan, index)
    path = path or ROOT / f"missile_probe_{plan}_{index}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = nn_scenario.scenario_xml(arena, "enemy", 3600)
    for name, rank in ranks.items():           # experience: the unit's rank (0-9)
        text, n = re.subn(rf'(script_name="{re.escape(name)}">.*?<unit_experience level=")0("/>)',
                          rf"\g<1>{rank}\g<2>", text, count=1, flags=re.S)
        assert n == 1, name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def run(plan, which=None, python=sys.executable, dry=False, deadline=900):
    """Builds and launches each battle in turn (tools/launcher/launch.ps1: Normal difficulty set and restored by
    the launcher)."""
    done = []
    for i in which or range(1, len(battles(plan)) + 1):
        build = [python, "-m", "tools.build", "missile-probe", "--mprobe-plan", plan, "--mprobe-battle", str(i),
                 "--deadline", str(deadline)]
        launch = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAUNCHER / "launch.ps1"),
                  "-Target", "missile-probe"]
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
    """[{run, plan, battle, spec, samples [(t s, s row, tg row)], end, first}] of one run."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    lanes = {l["name"]: {"run": run_dir.name, "plan": cfg.get("plan"), "battle": cfg.get("battle"), "spec": l,
                         "samples": [], "end": None} for l in cfg.get("lanes", [])}
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if '"probe_' not in line:
            continue
        r = json.loads(line)
        if r["event"] == "probe_sample":
            for x in r["lanes"]:
                if x["lane"] in lanes:
                    lanes[x["lane"]]["samples"].append((x["t"] / 1000, x["s"], x["tg"], x.get("d"), x.get("f")))
        elif r["event"] == "probe_lane_end":
            lanes[r["lane"]]["end"] = r
    return [x for x in lanes.values() if x["samples"]]


def series(samples, who, key):
    i = {"s": 1, "tg": 2, "f": 4}[who]
    return np.array([np.nan if (s[i] or {}).get(key) is None else float(s[i][key]) for s in samples])


def volleys(t, ammo, men, gap_s=1.5):
    """[(start t, projectiles)] from an ammo series: drops within gap_s of each other are one volley."""
    out = []
    last = None
    for k in range(1, len(t)):
        if not (np.isfinite(ammo[k]) and np.isfinite(ammo[k - 1])):
            continue
        drop = ammo[k - 1] - ammo[k]
        if drop <= 0.5:
            continue
        if out and last is not None and t[k] - last <= gap_s:
            out[-1][1] += drop
        else:
            out.append([t[k], drop])
        last = t[k]
    return [(a, b) for a, b in out]


def measure(lane, per_hit=None):
    """One lane's numbers (game or simulator: the same row format). per_hit: the rule's HP a hit (or None)."""
    spec, s = lane["spec"], lane["samples"]
    t = np.array([x[0] for x in s])
    out = {"run": lane["run"], "lane": spec["name"], "cell": cell(spec), "shooter": spec["shooter_short"]
           if "shooter_short" in spec else SHORT[spec["s_key"]], "target": SHORT[spec["t_key"]], "d": spec["d"]}
    ammo, smen = series(s, "s", "ammo"), series(s, "s", "men")
    hp, tmen = series(s, "tg", "hp"), series(s, "tg", "men")
    sx, sz, sb = series(s, "s", "x"), series(s, "s", "z"), series(s, "s", "b")
    tx, tz = series(s, "tg", "x"), series(s, "tg", "z")
    vs = volleys(t, ammo, smen)
    out["volleys"] = len(vs)
    if not vs:
        return out
    t1 = vs[0][0]
    k1 = int(np.searchsorted(t, t1))
    out["first_s"] = float(t1)
    out["first_d"] = float(np.hypot(sx[k1] - tx[k1], sz[k1] - tz[k1]))
    m1 = smen[max(0, k1 - 1)]
    out["first_share"] = float(vs[0][1] / m1) if m1 else None
    b0 = sb[np.isfinite(sb)][0]
    out["turn_deg"] = float(abs((sb[k1] - b0 + 180) % 360 - 180))
    sizes = np.array([v[1] for v in vs])
    whole = [v for v in vs if v[1] >= 0.5 * np.median(sizes)]
    if len(whole) >= 2:
        out["interval_s"] = float(np.median(np.diff([v[0] for v in whole])))
        out["intervals"] = [round(float(x), 1) for x in np.diff([v[0] for v in whole])]
    out["volley_share"] = float(np.median([v[1] / max(1.0, smen[min(len(t) - 1, int(np.searchsorted(t, v[0])))])
                                           for v in vs]))
    fired = float(np.nanmax(ammo) - np.nanmin(ammo))
    out["shots"] = fired
    # the steady rate: projectiles after the first volley over the man-seconds from the first shot to the last
    # (whole volleys n x men in (n - 1) reloads, or a steady stream: the same per man); its inverse the reload
    drops = np.nonzero(np.concatenate([[False], np.diff(ammo) < -0.5]))[0]
    if len(drops) >= 2 and len(vs) >= 2:
        t_last = t[drops[-1]]
        span = (t >= t1) & (t <= t_last)
        man_s = float(np.nanmean(smen[span]) * (t_last - t1))
        if man_s > 0:
            out["reload_s"] = man_s / max(1.0, fired - vs[0][1])
        # ... and while the target still has 3/4 of its men (the volleys started before it fell below)
        men0_t = tmen[np.isfinite(tmen)][0]
        below = np.nonzero(np.isfinite(tmen) & (tmen < 0.75 * men0_t))[0]
        t_end = t[below[0]] if (men0_t > 1 and len(below)) else t_last
        starts = [v for v in vs if v[0] < t_end]
        if len(starts) >= 3:
            t_e = starts[-1][0]
            span = (t >= t1) & (t <= t_e)
            man_s = float(np.nanmean(smen[span]) * (t_e - t1))
            shots_e = float(ammo[np.nonzero(np.isfinite(ammo))[0][0]] - ammo[min(len(t) - 1, int(np.searchsorted(t, t_e)) - 1)])
            if man_s > 0 and shots_e > 0:
                out["reload_s_full"] = man_s / shots_e        # the projectiles of the volleys before the last
    ok = np.isfinite(hp)
    lost = float(hp[ok][0] - hp[ok][-1]) if ok.sum() > 1 else None
    out["hp_lost"] = lost
    if lost is not None and fired > 0:
        out["hp_per_shot"] = lost / fired
        # the full phase: until the target is down to 3/4 of its men (projectiles until then; HP 3 s later)
        men0 = tmen[np.isfinite(tmen)][0]
        down = np.nonzero(np.isfinite(tmen) & (tmen < 0.75 * men0))[0]
        if men0 > 1 and len(down):
            kd = down[0]
            shots_full = float(np.nanmax(ammo[:kd + 1]) - np.nanmin(ammo[:kd + 1]))
            k3 = min(len(t) - 1, int(np.searchsorted(t, t[kd] + 3.0)))
            if shots_full > 0:
                out["hp_per_shot_full"] = float(hp[ok][0] - hp[k3]) / shots_full if np.isfinite(hp[k3]) else None
        elif men0 > 1 or men0 <= 1:
            out["hp_per_shot_full"] = out["hp_per_shot"]
        if per_hit:
            out["per_hit"] = per_hit
            out["hits_per_shot"] = out["hp_per_shot"] / per_hit
            if out.get("hp_per_shot_full") is not None:
                out["hits_per_shot_full"] = out["hp_per_shot_full"] / per_hit
    # hits per projectile by the target's men left (bands of its starting men: the thinned unit): projectiles while
    # it is in the band, HP it lost from 3 s after it entered the band to 3 s after it left it (the flight)
    if per_hit and np.isfinite(tmen).sum() > 2:
        men0 = tmen[np.isfinite(tmen)][0]
        share = tmen / max(1.0, men0)
        bands = []
        for lo, hi in ((0.75, 1.01), (0.5, 0.75), (0.25, 0.5), (0.0, 0.25)):
            idx = np.nonzero(np.isfinite(share) & (share >= lo) & (share < hi) & (t >= t1))[0]
            if len(idx) < 2 or men0 <= 1:
                bands.append(None)
                continue
            k0, k1 = idx[0], idx[-1]
            shots_b = float(ammo[k0] - ammo[k1]) if np.isfinite(ammo[k0]) and np.isfinite(ammo[k1]) else 0.0
            j0 = min(len(t) - 1, int(np.searchsorted(t, t[k0] + 3.0)))
            j1 = min(len(t) - 1, int(np.searchsorted(t, t[k1] + 3.0)))
            lost_b = float(hp[j0] - hp[j1]) if np.isfinite(hp[j0]) and np.isfinite(hp[j1]) else None
            bands.append(None if (shots_b < 50 or lost_b is None) else round(lost_b / shots_b / per_hit, 3))
        out["hits_by_men"] = bands
    # the target's motion while shot (a moving lane): mean speed and the centre distance at the volleys
    if spec["target_mode"] == "move":
        dist = np.hypot(sx - tx, sz - tz)
        out["d_at_volleys"] = [round(float(dist[min(len(t) - 1, int(np.searchsorted(t, v[0])))]), 0) for v in vs]
    out["end"] = (lane.get("end") or {}).get("why")
    return out


def cell(spec):
    s = SHORT.get(spec.get("s_key"), spec.get("shooter"))
    t = SHORT.get(spec.get("t_key"), spec.get("target"))
    name = f"{s}>{t} d{spec['d']:.0f}"
    if spec.get("t_angle"):
        name += f" at{spec['t_angle']:.0f}"
    if spec.get("t_rot"):
        name += f" rot{spec['t_rot']:.0f}"
    if spec.get("t_men"):
        name += f" men{spec['t_men']}"
    if spec.get("t_width", 30) not in (30, 5):
        name += f" w{spec['t_width']}"
    if spec["target_mode"] != "stand":
        name += f" {spec['target_mode']}"
        if spec["target_mode"] == "move":
            name += f" to({spec['move_fwd']},{spec['move_lat']}){' run' if spec.get('move_run') else ''}"
    if spec.get("s_rank"):
        name += f" rank{spec['s_rank']}"
    if spec.get("reform_men"):
        name += f" reform{spec['reform_men']}"
    if spec.get("friend"):
        name += f" friend({spec.get('friend_fwd')},{spec.get('friend_lat')})"
    if spec["mode"] != "fire":
        name += f" {spec['mode']}"
    return name


def runs(root=RUNS):
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if (d / "events.jsonl").exists() and (d / "manifest.json").exists())


# ---------------------------------------------------------------- the simulator on the same lanes

def rule_per_hit(s_key, t_key, params):
    """The rule's HP a hit of s_key's projectile takes from t_key (tools/nn/measure.py per_hit: armour, overkill; x
    the target's missile + physical resistance, at most 90 %)."""
    from tools.nn.measure import per_hit
    m, u = params.units[s_key]["missile"], params.units[t_key]
    res = u.get("damage_resist") or {}
    resist = min(0.9, (res.get("missile", 0) + res.get("physical", 0) + res.get("all", 0)) / 100)
    return per_hit(m["damage"] * (1 - resist), m["ap_damage"] * (1 - resist), u["armour"], u["hp_per_man"],
                   single=u["men"] <= 1)


def sim_lanes(lanes, params=None, device="cpu", copies=4, jitter_m=0.5, seed=0):
    """The simulator on recorded lanes: each lane its own battle (its units at their places at the probe's go),
    the shooter holding (fire at will) or attacking, everyone fearless; a stepping target steps by the lane's
    schedule until the simulator's first shot, a moving target follows the game's recorded path. Returns lanes in
    the game's shape (samples every 0.5 s), `copies` per recorded lane."""
    import torch
    from tools.nn.sim import battle, orders as O, scenario as sim_scenario
    from tools.nn.sim.params import load
    params = params or load()
    armies, meta = [], []
    for ln in lanes:
        spec = ln["spec"]
        s0 = ln["samples"][0]
        sides = {1: {"faction": EMP, "ai": False, "units": []}, 2: {"faction": SKV, "ai": False, "units": []}}
        for role, row, key, width in (("s", s0[1], spec["s_key"], spec["s_width"]),
                                      ("tg", s0[2], spec["t_key"], spec["t_width"])):
            fac = 1 if FACTION[key] == EMP else 2
            lord = key in (UNITS["general"][0], UNITS["warlord"][0])
            sides[fac]["units"].append({"key": key, "x": row["x"], "z": row["z"], "b": row["b"], "men": row["men"],
                                        "width": None if lord else width, "general": lord, "name": role})
        if spec.get("friend") and len(s0) > 4 and s0[4]:
            r = s0[4]
            sides[1]["units"].append({"key": spec["f_key"], "x": r["x"], "z": r["z"], "b": r["b"], "men": r["men"],
                                      "width": spec.get("friend_width", 30), "general": False, "name": "f"})
        army = {"attacker": 1, "sides": sides}
        for _ in range(copies):
            armies.append(army)
            meta.append(ln)
    per_side = max(len(a["sides"][k]["units"]) for a in armies for k in (1, 2))
    st = sim_scenario.build(armies, params, device=device, per_side=per_side)
    B, N = st.B, st.N
    slot = {r: torch.tensor([sim_scenario.slots(a, per_side).get(r, -1) for a in armies], device=device)
            for r in ("s", "tg", "f")}
    gen = torch.Generator().manual_seed(seed)
    jit = ((torch.rand((B, 2), generator=gen) * 2 - 1) * jitter_m).to(device)
    b_idx = torch.arange(B, device=device)
    st.u["x"][b_idx, slot["s"]] += jit[:, 0]
    st.u["z"][b_idx, slot["s"]] += jit[:, 1]
    st.u["leadership"] = st.u["leadership"] + 1e4
    st.u["morale"] = st.u["morale"] + 1e4
    specs = [ln["spec"] for ln in meta]
    # a moving target's recorded path (t, x, z, b), a stepping target's schedule
    paths = []
    for ln in meta:
        s = ln["samples"]
        paths.append((np.array([x[0] for x in s]), np.array([x[2]["x"] for x in s], float),
                      np.array([x[2]["z"] for x in s], float), np.array([x[2]["b"] for x in s], float)))
    fired = [False] * B
    rec = {"t": [], "rows": []}
    fields = ("x", "z", "b", "men", "hp_abs", "a")
    a0 = st.u["a"][b_idx, slot["s"]].clone()

    def policy(st):
        t = float(st.t.max())
        u = st.u
        o = O.hold(B, N, device)
        a_now = u["a"][b_idx, slot["s"]]
        for b in range(B):
            S, T = int(slot["s"][b]), int(slot["tg"][b])
            sp = specs[b]
            if not fired[b] and float(a_now[b]) < float(a0[b]):
                fired[b] = True
            if sp["mode"] == "attack":
                o.kind[b, S], o.target[b, S] = O.ATTACK, T
            if sp["target_mode"] == "step" and not fired[b]:
                k = int(t // sp["step_s"])
                d = sp["start_d"] - sp["step_m"] * k
                sx, sz = float(u["x"][b, S]), float(u["z"][b, S])
                ang = math.radians(sp.get("s_b", 0.0) + sp.get("t_angle", 0.0))
                u["x"][b, T], u["z"][b, T] = sp["x"] + d * math.sin(ang), sp["z"] + d * math.cos(ang)
                del sx, sz
            elif sp["target_mode"] == "move":
                pt, px, pz, pb = paths[b]
                u["x"][b, T] = float(np.interp(t, pt, px))
                u["z"][b, T] = float(np.interp(t, pt, pz))
                u["b"][b, T] = float(pb[min(len(pb) - 1, int(np.searchsorted(pt, t)))])
        return o

    next_t = [0.0]

    def record(st):
        t = float(st.t.max())
        if t + 1e-6 < next_t[0]:
            return
        next_t[0] = t + TICK_MS / 1000
        rec["t"].append(t)
        rec["rows"].append({k: st.u[k].detach().cpu().numpy().copy() for k in fields})
    until = max(ln["samples"][-1][0] for ln in lanes) + 2
    battle.run(st, policy, params, until_s=until, record=record, compile=False)
    out = []
    for b, ln in enumerate(meta):
        S, T, F = int(slot["s"][b]), int(slot["tg"][b]), int(slot["f"][b])
        end = ln["samples"][-1][0]
        samples = []
        for t, r in zip(rec["t"], rec["rows"]):
            if t > end + 1e-6:
                break
            samples.append((t, {"x": float(r["x"][b, S]), "z": float(r["z"][b, S]), "b": float(r["b"][b, S]),
                                "men": float(r["men"][b, S]), "ammo": float(r["a"][b, S])},
                            {"x": float(r["x"][b, T]), "z": float(r["z"][b, T]), "b": float(r["b"][b, T]),
                             "men": float(r["men"][b, T]), "hp": float(r["hp_abs"][b, T])}, None,
                            None if F < 0 else {"x": float(r["x"][b, F]), "z": float(r["z"][b, F]),
                                                "men": float(r["men"][b, F]), "hp": float(r["hp_abs"][b, F])}))
        out.append({"run": "sim", "spec": ln["spec"], "samples": samples, "end": None})
    return out


# ---------------------------------------------------------------- tables

KEYS = ("first_s", "first_d", "first_share", "turn_deg", "interval_s", "reload_s", "reload_s_full", "volley_share", "shots", "hp_per_shot",
        "hp_per_shot_full", "hits_per_shot", "hits_per_shot_full")


def summary(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["cell"], []).append(r)
    out = {}
    for name, rs in sorted(groups.items()):
        row = {"cell": name, "n": len(rs)}
        for k in KEYS:
            v = [r[k] for r in rs if r.get(k) is not None and np.isfinite(r[k])]
            if v:
                row[k] = float(np.mean(v))
        bands = [r["hits_by_men"] for r in rs if r.get("hits_by_men")]
        if bands:
            row["hits_by_men"] = [None if not [b[i] for b in bands if b[i] is not None] else
                                  round(float(np.mean([b[i] for b in bands if b[i] is not None])), 3) for i in range(4)]
        for k in ("d_at_volleys",):
            if rs and rs[0].get(k):
                row[k] = rs[0][k]
        out[name] = row
    return out


def report(run_dirs, sim=False, out=OUT, device="cpu", copies=4, label="sim", missile=None):
    """missile: {key: value} over config/nn/sim.json missile for the simulator (e.g. accuracy, reload_s)."""
    from tools.nn.sim.params import load
    params = load()
    if missile:
        params = params.with_cal("missile", **missile)
    lanes = [ln for d in run_dirs for ln in load_run(d)]
    ph = {}
    for ln in lanes:
        sp = ln["spec"]
        ph[(sp["s_key"], sp["t_key"])] = rule_per_hit(sp["s_key"], sp["t_key"], params)
    game = [measure(ln, ph[(ln["spec"]["s_key"], ln["spec"]["t_key"])]) for ln in lanes]
    table = summary(game)
    result = {"lanes": game, "summary": list(table.values())}
    sim_table = {}
    if sim:
        sims = sim_lanes(lanes, params, device=device, copies=copies)
        sim_rows = [measure(ln, ph[(ln["spec"]["s_key"], ln["spec"]["t_key"])]) for ln in sims]
        sim_table = summary(sim_rows)
        result[label] = list(sim_table.values())
    f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"
    print(f"{'cell':58} {'':4} {'first_s':>7} {'first_d':>7} {'share1':>6} {'turn':>5} {'interv':>6} {'reload':>6} {'vshare':>6} "
          f"{'shots':>6} {'HP/sh':>6} {'HPfull':>6} {'hit/sh':>6} {'hitful':>6}")
    for name, r in table.items():
        for who, row in (("game", r), (label[:4], sim_table.get(name))):
            if row is None:
                continue
            print(f"{name if who == 'game' else '':58} {who:4} {f(row.get('first_s'), 1):>7} {f(row.get('first_d'), 1):>7} "
                  f"{f(row.get('first_share')):>6} {f(row.get('turn_deg'), 0):>5} {f(row.get('interval_s'), 1):>6} "
                  f"{f(row.get('reload_s'), 1):>6} "
                  f"{f(row.get('volley_share')):>6} {f(row.get('shots'), 0):>6} {f(row.get('hp_per_shot')):>6} "
                  f"{f(row.get('hp_per_shot_full')):>6} {f(row.get('hits_per_shot')):>6} {f(row.get('hits_per_shot_full')):>6}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("report", "plan", "run"), default="report")
    parser.add_argument("runs", nargs="*", type=Path, help="report: run folders (default: build/missile-probe/runs/*)")
    parser.add_argument("--plan", choices=PLANS, default="dist")
    parser.add_argument("--battles", help="run: battle numbers, comma separated (default: all of the plan)")
    parser.add_argument("--sim", action="store_true", help="report: also the simulator on the same lanes (torch)")
    parser.add_argument("--label", default="sim", help="report: the simulator's name in the output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--missile", help="report --sim: JSON over sim.json missile, e.g. "
                                          "'{\"accuracy\": {\"k\": 1.1, \"units\": \"rates\"}}'")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "plan":
        for p in ([args.plan] if args.plan else PLANS):
            for i, b in enumerate(battles(p), 1):
                print(p, i, [cell(dict(l, s_key=UNITS[l["shooter"]][0], t_key=UNITS[l["target"]][0])) for l in b])
        return 0
    if args.command == "run":
        which = [int(x) for x in args.battles.split(",")] if args.battles else None
        done = run(args.plan, which, dry=args.dry)
        print("done:", done)
        return 0 if all(d[-1] == 0 for d in done) else 1
    report(args.runs or runs(), sim=args.sim, out=args.out, device=args.device, copies=args.copies, label=args.label,
           missile=json.loads(args.missile) if args.missile else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
