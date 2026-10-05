"""Lord killed or routed: how much morale does his army lose? (src/entries/lord_fall.lua)

The user (05.10.2026): a killed lord shocks the army (Skaven most; Vampire Counts crumble). The
simulator has no such shock (config/nn/sim.json morale lord_fall 0 / 0) because all 75 recorded lord
falls were routs; the database has ume_concerned_general_died_recently -16 / general_dead -10 and
general_fled_recently -16 (config/nn/game_rules.json).

One battle = one treatment of one army (side 1, ours in the file): its lord is KILLED, ROUTED or left
alone (the control) at a fixed moment, 20 s after the first contact. The other side is fearless (no
routs that would lift the treated army). The treated army: the lord 20 m behind his line, two infantry
units in melee within his aura ('fight'), two standing idle out of the aura and 115+ m from any enemy
('idle': the shock alone, no melee drift). The game's database gives the units (land_units /
main_units, db.pack): Empire spearmen 120 men (leadership 60), Skaven clanrat spearmen 160 (45),
Vampire Counts skeleton spearmen 160 (35; not in the simulator's data).

    python -m tools.build lord-fall --faction skv --treatment kill     # one battle's build
    python -m tools.nn.lord_fall plan                                    # the night's battles
    python -m tools.nn.lord_fall run                                     # build + launch them all
    python -m tools.nn.lord_fall                                         # the table from the runs

Analysis: per treated unit the change of MoralePercent from the last sample before the moment to
+1 ... +60 s, in points (MoralePercent x the unit's leadership, as docs/en/training/measurements.md),
by faction, treatment and role; the control's change of the same role is subtracted; routed share
within 10 / 60 s, the idle units' health lost (crumbling), the strongest morale effect (CCO
MoraleGreatestEffect) seen after the moment. Writes build/lord-fall/analysis.json (not in Git).
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from tools import config as project
from tools.nn import scenario as nn_scenario

SCENARIO = project.SCENARIOS / "lord_fall.xml"
ROOT = project.BUILD / "lord-fall"
RUNS = ROOT / "runs"
OUT = ROOT / "analysis.json"
LAUNCH = project.ROOT / "tools" / "launcher" / "launch.ps1"

# faction -> faction key, lord, infantry, men, the infantry's leadership (land_units.morale).
ARMIES = {
    "emp": {"faction": "wh_main_emp_empire", "lord": "wh_main_emp_cha_general_0",
            "inf": "wh_main_emp_inf_spearmen_0", "men": 120, "leadership": 60},
    "skv": {"faction": "wh2_main_skv_skaven", "lord": "wh2_main_skv_cha_warlord_0",
            "inf": "wh2_main_skv_inf_clanrat_spearmen_0", "men": 160, "leadership": 45},
    "vmp": {"faction": "wh_main_vmp_vampire_counts", "lord": "wh_main_vmp_cha_vampire_lord_0",
            "inf": "wh_main_vmp_inf_skeleton_warriors_1", "men": 160, "leadership": 35},
}
# The treated army's (fearless) opponent.
OPPONENT = {"emp": "skv", "skv": "emp", "vmp": "emp"}
TREATMENTS = ("kill", "rout", "none")
GAP_M = 80                 # the two lines' fronts (zones reach 40 m ahead: no overlap)
LORD_BACK_M = 20           # the lord behind his line: ~63 m from the melee (aura full to 70 m)
FIGHT_LATERAL = (-18, 18)  # the two fight units of each side, facing each other
IDLE_LATERAL = (130, 166)  # the idle units: own to the south, the enemy's to the north
DEPLOYMENT_M = 400
AURA_M = (70, 105)         # general_aura_radius, faded out at x1.5
TREAT_AFTER_MS, TREAT_LATEST_MS, OBSERVE_MS = 20000, 60000, 60000
TICK_MS = 500
OFFSETS_S = (1, 2, 5, 10, 20, 40, 60)


def arena(faction):
    """The battle in the shape of a named arena (tools/nn/scenario.py): side 'own' is treated."""
    assert faction in ARMIES, faction
    base = nn_scenario.load_arena()

    def army(key, mirror):
        a = ARMIES[key]
        units = [{"slot": "lord", "key": a["lord"], "men": 1, "general": True, "forward": -LORD_BACK_M,
                  "lateral": 0, "width": 5}]
        # the enemy's lateral axis is mirrored (scenario.frame): -lateral puts its fight unit k opposite ours
        units += [{"slot": f"fight_{k}", "key": a["inf"], "men": a["men"], "forward": 0,
                   "lateral": -lat if mirror else lat, "width": 30} for k, lat in enumerate(FIGHT_LATERAL, 1)]
        units += [{"slot": f"idle_{k}", "key": a["inf"], "men": a["men"], "forward": 0, "lateral": lat,
                   "width": 30} for k, lat in enumerate(IDLE_LATERAL, 1)]
        return {"faction": a["faction"], "units": units}
    out = {k: v for k, v in base.items() if k not in ("faction", "units", "description")}
    out.update(name=f"lord_fall_{faction}", gap_m=GAP_M, deployment_m=DEPLOYMENT_M, defend_radius_m=150,
               sides={"own": army(faction, False), "enemy": army(OPPONENT[faction], True)})
    return out


def scenario_path(faction):
    """emp: the target's own file in scenarios/; the others next to the build."""
    return SCENARIO if faction == "emp" else ROOT / f"lord_fall_{faction}.xml"


def write_scenario(faction, path=None):
    path = path or scenario_path(faction)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nn_scenario.scenario_xml(arena(faction), "enemy", 3600), encoding="utf-8", newline="\n")
    return path


def run_config(faction, treatment):
    """The entry's config and the model seconds the battle takes at most."""
    assert treatment in TREATMENTS, treatment
    names = lambda side, kind: [f"{side}_{kind}_{k}" for k in (1, 2)]
    config = {"faction": faction, "opponent": OPPONENT[faction], "treatment": treatment, "treated_side": 1,
              "factions": {"own": ARMIES[faction]["faction"], "enemy": ARMIES[OPPONENT[faction]]["faction"]},
              "lords": {"own": "own_lord", "enemy": "enemy_lord"},
              "fight": [list(p) for p in zip(names("own", "fight"), names("enemy", "fight"))],
              "idle": names("own", "idle") + names("enemy", "idle"),
              "treat_after_ms": TREAT_AFTER_MS, "treat_latest_ms": TREAT_LATEST_MS, "observe_ms": OBSERVE_MS,
              "leadership": ARMIES[faction]["leadership"]}
    model_s = (TREAT_LATEST_MS + OBSERVE_MS) / 1000 + 30
    return config, model_s


def plan(kill=2, rout=2, none=1, factions=("emp", "skv", "vmp")):
    """[(faction, treatment)] in rounds, so a night cut short still has every cell once."""
    want = {"kill": kill, "rout": rout, "none": none}
    out = []
    for r in range(max(want.values())):
        for f in factions:
            out += [(f, t) for t in TREATMENTS if r < want[t]]
    return out


def run(battles, python=sys.executable, dry=False):
    """Builds and launches each battle in turn (one game launch each); returns [(faction, treatment, exit)]."""
    done = []
    for i, (faction, treatment) in enumerate(battles, 1):
        build = [python, "-m", "tools.build", "lord-fall", "--faction", faction, "--treatment", treatment]
        launch = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAUNCH), "-Target", "lord-fall"]
        print(f"--- {i}/{len(battles)}: {faction} {treatment}", flush=True)
        if dry:
            print(" ".join(build), "&&", " ".join(launch))
            continue
        code = subprocess.run(build, cwd=project.ROOT, stdout=subprocess.DEVNULL).returncode
        if code == 0:
            code = subprocess.run(launch, cwd=project.ROOT).returncode
        done.append((faction, treatment, code))
    return done


# ---------------------------------------------------------------- analysis

def load_run(run_dir):
    """{config, samples [(t, {name: row})], fall, result} of one run, or None if it has no moment."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    out = {"run": run_dir.name, "config": manifest["config"], "samples": [], "fall": None, "result": None,
           "fallback": None}
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if '"lord_fall' not in line and '"fall_sample"' not in line and '"event":"result"' not in line:
            continue
        r = json.loads(line)
        if r["event"] == "fall_sample":
            out["samples"].append((r["t"], r["treated"], {u["n"]: u for u in r["units"]}))
        elif r["event"] == "lord_fall":
            out["fall"] = r
        elif r["event"] == "lord_fall_fallback":
            out["fallback"] = r
        elif r["event"] == "result":
            out["result"] = r
    return out if out["fall"] and out["samples"] else None


def measure(run):
    """Per treated unit (side 1, role fight / idle): MoralePercent change at OFFSETS_S, points, routs, crumbling."""
    cfg, t0 = run["config"], run["fall"]["t"]
    before = [s for s in run["samples"] if not s[1] and s[0] <= t0]
    if not before:
        return []
    base = before[-1][2]
    after = [s for s in run["samples"] if s[0] > t0]
    lead = cfg.get("leadership") or ARMIES[cfg["faction"]]["leadership"]
    out = []
    for name, row0 in base.items():
        if row0.get("side") != cfg.get("treated_side", 1) or row0.get("role") not in ("fight", "idle"):
            continue
        if row0.get("mp") is None:
            continue
        u = {"run": run["run"], "faction": cfg["faction"], "treatment": cfg["treatment"], "unit": name,
             "role": row0["role"], "mp0": row0["mp"], "routing0": bool(row0.get("r") or row0.get("s"))}
        for k in OFFSETS_S:
            row = next((s[2].get(name) for s in after if s[0] >= t0 + 1000 * k), None)
            mp = row and row.get("mp")
            u[f"d{k}"] = None if mp is None else round((mp - row0["mp"]) * lead, 2)
        for k in (10, 60):
            u[f"routed{k}"] = any((s[2].get(name) or {}).get("r") or (s[2].get(name) or {}).get("s")
                                  for s in after if s[0] <= t0 + 1000 * k)
        last = next((s[2].get(name) for s in reversed(after)), None)
        u["hp_lost"] = None if not last or last.get("hp") is None or row0.get("hp") is None else round(row0["hp"] - last["hp"], 4)
        u["mge_after"] = Counter((s[2].get(name) or {}).get("mge") for s in after
                                 if s[0] <= t0 + 20000 and (s[2].get(name) or {}).get("mge")).most_common(2)
        out.append(u)
    return out


def lord_state(run):
    """The treated lord 2 s after the moment: dead (men 0 / IsAlive false), routing, or standing."""
    t0, lord = run["fall"]["t"], run["fall"]["lord"]
    row = next((s[2].get(lord) for s in run["samples"] if s[0] >= t0 + 2000), None)
    if not row:
        return "?"
    if row.get("men") == 0 or row.get("alive") is False:
        return "dead"
    return "routing" if row.get("r") or row.get("s") else "standing"


def summary(units):
    """Mean by faction, treatment and role, with the control's mean subtracted."""
    groups = {}
    for u in units:
        groups.setdefault((u["faction"], u["treatment"], u["role"]), []).append(u)

    def mean(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None]
        return round(sum(v) / len(v), 2) if v else None
    rows = []
    for (faction, treatment, role), rs in sorted(groups.items()):
        ctrl = groups.get((faction, "none", role), [])
        row = {"faction": faction, "treatment": treatment, "role": role, "units": len(rs),
               "runs": len({r["run"] for r in rs})}
        for k in OFFSETS_S:
            m, c = mean(rs, f"d{k}"), mean(ctrl, f"d{k}")
            row[f"d{k}"] = m
            row[f"net{k}"] = None if m is None or c is None or treatment == "none" else round(m - c, 2)
        row["routed10"] = round(sum(r["routed10"] for r in rs) / len(rs), 2)
        row["routed60"] = round(sum(r["routed60"] for r in rs) / len(rs), 2)
        row["hp_lost"] = mean(rs, "hp_lost")
        row["mge"] = Counter(m for r in rs for m, _ in r["mge_after"]).most_common(2)
        rows.append(row)
    return rows


def runs(root=RUNS):
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if (d / "events.jsonl").exists() and (d / "manifest.json").exists())


def report(run_dirs, out=OUT):
    loaded = [r for r in (load_run(d) for d in run_dirs) if r]
    units = [u for r in loaded for u in measure(r)]
    table = summary(units)
    f = lambda v: "-" if v is None else f"{v:+.1f}"
    print(f"{len(loaded)} runs with a moment; points = MoralePercent x leadership; 'net' = minus the control")
    for r in loaded:
        print(f"  {r['run']}  {r['config']['faction']} {r['config']['treatment']:5} lord after 2 s: {lord_state(r):9}"
              f" method {r['fall'].get('method')}{' + uc:kill' if r['fallback'] else ''}"
              f"  status {(r['result'] or {}).get('status')}")
    head = "".join(f"{f'+{k}s':>7}" for k in OFFSETS_S)
    print(f"{'faction':7} {'treat':5} {'role':5} {'n':>3} {'runs':>4} {head}   {'net +2s':>7} {'+10s':>6} {'+40s':>6}"
          f"  rout10 rout60  hp lost  strongest effect after")
    for r in table:
        cols = "".join(f"{f(r[f'd{k}']):>7}" for k in OFFSETS_S)
        print(f"{r['faction']:7} {r['treatment']:5} {r['role']:5} {r['units']:>3} {r['runs']:>4} {cols}   "
              f"{f(r['net2']):>7} {f(r['net10']):>6} {f(r['net40']):>6}  {r['routed10']:>6} {r['routed60']:>6}"
              f"  {'-' if r['hp_lost'] is None else r['hp_lost']:>7}  {r['mge']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"units": units, "summary": table}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(out)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("report", "plan", "run"), default="report")
    parser.add_argument("runs", nargs="*", type=Path, help="report: run folders (default: build/lord-fall/runs/*)")
    parser.add_argument("--kill", type=int, default=2, help="plan/run: battles a faction with the lord killed")
    parser.add_argument("--rout", type=int, default=2, help="plan/run: ... with the lord routed")
    parser.add_argument("--none", type=int, default=1, help="plan/run: ... control (nothing done)")
    parser.add_argument("--factions", default="emp,skv,vmp")
    parser.add_argument("--dry", action="store_true", help="run: print the commands only")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    battles = plan(args.kill, args.rout, args.none, tuple(args.factions.split(",")))
    if args.command == "plan":
        for b in battles:
            print(*b)
        print(f"{len(battles)} battles, ~2 min each with the game's loading")
        return 0
    if args.command == "run":
        done = run(battles, dry=args.dry)
        if args.dry:
            return 0
        print("done:", done)
        report(runs())
        return 0 if all(c == 0 for *_, c in done) else 1
    report(args.runs or runs())
    return 0


if __name__ == "__main__":
    sys.exit(main())
