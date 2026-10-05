"""Lord duel: our network's lord against the same lord under one plain attack order (no other units).

How much do the network's extra moves and orders cost in a duel? Side 1 (ours) is the network in the
companion (entries.nn_arena own_ai 'net', the gate's path: tools/launcher/watch.ps1); side 2 is the
same lord type, mirrored, taken by script and given ONE attack order on our lord at the start
(enemy_ai 'scripted': given again only when the engine drops it - not in melee, no target for 3 s).
The control ('scripted' on both sides) shows what two plain lords do to each other: the trade a
network that did nothing clever would get. Lords that the network has seen in training: the Empire
General and the Skaven Warlord.

    python -m tools.build lord-duel --duel emp                         # one battle's build (network side)
    python -m tools.nn.lord_duel plan                                   # the night's battles
    python -m tools.nn.lord_duel run --checkpoint build/nn-train/<label>/m15.pt
    python -m tools.nn.lord_duel                                        # the table from the runs

Analysis per battle: both lords' health (unary_hitpoints) at the end and lost per second of melee,
the HP trade (enemy lost - ours lost: positive = our lord trades better), the winner, the time to
contact, the network's orders to its lord (given / kept / keeps, by kind), its abilities used, the
metres our lord and the enemy walked, the script's re-issued attacks. Writes build/lord-duel/analysis.json.
"""
import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

from tools import config as project
from tools.nn import scenario as nn_scenario

SCENARIO = project.SCENARIOS / "lord_duel.xml"
ROOT = project.BUILD / "lord-duel"
RUNS = ROOT / "runs"
OUT = ROOT / "analysis.json"
LAUNCHER = project.ROOT / "tools" / "launcher"
LORDS = {"emp": ("wh_main_emp_empire", "wh_main_emp_cha_general_0"),
         "skv": ("wh2_main_skv_skaven", "wh2_main_skv_cha_warlord_0")}
GAP_M = 100
TIMEOUT_S = 900


def arena(kind):
    """Both sides the same lord alone, gap_m apart, facing each other."""
    faction, key = LORDS[kind]
    base = nn_scenario.load_arena()
    army = {"faction": faction, "units": [{"slot": "lord", "key": key, "men": 1, "general": True,
                                           "forward": 0, "lateral": 0, "width": 5}]}
    out = {k: v for k, v in base.items() if k not in ("faction", "units", "description")}
    out.update(name=f"lord_duel_{kind}", gap_m=GAP_M, defend_radius_m=150,
               sides={"own": army, "enemy": json.loads(json.dumps(army))})
    return out


def write_scenario(kind="emp", path=SCENARIO, defender="enemy", duration_s=3600):
    path.parent.mkdir(parents=True, exist_ok=True)
    nn_scenario.write_scenario(defender, arena(kind), path, duration_s)
    return path


def build_config(kind, own_ai, own_role, decide_ms, poll_ms, timeout_s):
    """The battle file (next to the build) and the entry's config (entries.nn_arena)."""
    assert own_ai in ("net", "scripted"), own_ai
    a = arena(kind)
    enemy_role = "defend" if own_role == "attack" else "attack"
    path = ROOT / f"{a['name']}_{own_role}.xml"
    write_scenario(kind, path, "enemy" if enemy_role == "defend" else "own", max(3600, timeout_s + 60))
    config = dict(nn_scenario.run_config(a), duel=kind, own_ai=own_ai, enemy_ai="scripted",
                  own_role=own_role, enemy_role=enemy_role)
    if own_ai == "net":
        config.update(decide_ms=decide_ms, poll_ms=poll_ms)
    return path, config


def plan(net=4, control=2, kinds=("emp", "skv")):
    """[(kind, own_ai, own_role)]: the network attacking and defending in turn, then the controls."""
    out = []
    for r in range(max(net, control)):
        for k in kinds:
            if r < net:
                out.append((k, "net", "attack" if r % 2 == 0 else "defend"))
            if r < control:
                out.append((k, "scripted", "attack"))
    return out


def run(battles, checkpoint, python=sys.executable, dry=False, greedy=False):
    """Builds and launches each battle (the network's through watch.ps1 with the companion)."""
    done = []
    for i, (kind, own_ai, role) in enumerate(battles, 1):
        build = [python, "-m", "tools.build", "lord-duel", "--duel", kind, "--own-ai", own_ai, "--own-role", role]
        ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
        if own_ai == "net":
            launch = ps + [str(LAUNCHER / "watch.ps1"), "-Target", "lord-duel", "-NoBuild", "-LingerSeconds", "0",
                           "-Checkpoint", checkpoint] + (["-Greedy"] if greedy else [])
        else:
            launch = ps + [str(LAUNCHER / "launch.ps1"), "-Target", "lord-duel"]
        print(f"--- {i}/{len(battles)}: {kind} {own_ai} {role}", flush=True)
        if dry:
            print(" ".join(build), "&&", " ".join(launch))
            continue
        code = subprocess.run(build, cwd=project.ROOT, stdout=subprocess.DEVNULL).returncode
        if code == 0:
            code = subprocess.run(launch, cwd=project.ROOT).returncode
        done.append((kind, own_ai, role, code))
    return done


# ---------------------------------------------------------------- analysis

def load_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    out = {"run": run_dir.name, "config": manifest["config"], "samples": [], "orders": [], "abilities": [],
           "reissues": [], "result": None}
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not any(k in line for k in ('"nn_sample"', '"nn_final"', '"nn_orders"', '"nn_ability"',
                                       '"scripted_order"', '"event":"result"')):
            continue
        r = json.loads(line)
        ev = r["event"]
        if ev in ("nn_sample", "nn_final"):
            out["samples"].append((r["t"], {u["n"]: u for u in r["units"]}))
        elif ev == "nn_orders":
            out["orders"].append(r)
        elif ev == "nn_ability":
            out["abilities"].append(r)
        elif ev == "scripted_order":
            out["reissues"].append(r)
        elif ev == "result":
            out["result"] = r
    return out if out["samples"] else None


def walked(series):
    """Metres along the recorded places [(x, z)]."""
    d = 0.0
    for (x0, z0), (x1, z1) in zip(series, series[1:]):
        if None not in (x0, z0, x1, z1):
            d += math.hypot(x1 - x0, z1 - z0)
    return round(d, 1)


def measure(run, own="own_lord", enemy="enemy_lord"):
    cfg, s = run["config"], run["samples"]
    first, last = s[0][1], s[-1][1]
    res = run["result"] or {}
    hp = lambda row, k: (row.get(k) or {}).get("hp")
    melee = [(t, u) for t, u in s if (u.get(own) or {}).get("m") or (u.get(enemy) or {}).get("m")]
    contact = melee[0][0] / 1000 if melee else None
    out = {"run": run["run"], "duel": cfg.get("duel"), "own_ai": cfg.get("own_ai"), "role": cfg.get("own_role"),
           "mode": "control" if cfg.get("own_ai") == "scripted" else f"net-{cfg.get('own_role')}",
           "status": res.get("status"), "winner": res.get("winner"),
           "t_end": round(s[-1][0] / 1000, 1), "contact_s": contact, "melee_s": len(melee) * 1.0}
    for side, name in (("own", own), ("enemy", enemy)):
        h0, h1 = hp(first, name), hp(last, name)
        out[f"{side}_hp_end"] = h1
        out[f"{side}_lost"] = None if h0 is None or h1 is None else round(h0 - h1, 4)
        row = last.get(name) or {}
        out[f"{side}_down"] = "dead" if (row.get("men") == 0) else ("routing" if row.get("r") or row.get("s") else "")
        out[f"{side}_walked_m"] = walked([((u.get(name) or {}).get("x"), (u.get(name) or {}).get("z")) for _, u in s])
    out["trade"] = (None if out["own_lost"] is None or out["enemy_lost"] is None
                    else round(out["enemy_lost"] - out["own_lost"], 4))
    given, kinds = 0, Counter()
    for o in run["orders"]:
        for x in o.get("orders", []):
            if x.get("u") == own and x.get("status") == "given":
                given += 1
                kinds[x.get("k")] += 1
    out["orders_given"], out["order_kinds"] = given, dict(kinds)
    out["orders_per_min"] = round(60 * given / max(1.0, s[-1][0] / 1000), 1)
    out["abilities"] = dict(Counter(a.get("key") for a in run["abilities"] if a.get("status") == "used"))
    out["reissues"] = sum(1 for r in run["reissues"] if r.get("why") == "lost")
    return out


def summary(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["duel"], r["mode"]), []).append(r)
    out = []
    for (duel, mode), rs in sorted(groups.items()):
        def mean(key, rs=rs):
            v = [r[key] for r in rs if r.get(key) is not None]
            return round(sum(v) / len(v), 3) if v else None
        out.append({"duel": duel, "mode": mode, "battles": len(rs), "own_wins": sum(r["winner"] == 1 for r in rs),
                    "enemy_wins": sum(r["winner"] == 2 for r in rs), "trade": mean("trade"),
                    "own_lost": mean("own_lost"), "enemy_lost": mean("enemy_lost"), "contact_s": mean("contact_s"),
                    "orders_per_min": mean("orders_per_min"), "own_walked_m": mean("own_walked_m"),
                    "enemy_walked_m": mean("enemy_walked_m"),
                    "abilities": dict(sum((Counter(r["abilities"]) for r in rs), Counter()))})
    return out


def runs(root=RUNS):
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if (d / "events.jsonl").exists() and (d / "manifest.json").exists())


def report(run_dirs, out=OUT):
    rows = [measure(r) for r in (load_run(d) for d in run_dirs) if r]
    f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"
    print("health is unary (1 = full); trade = enemy lost - ours lost (positive: our lord trades better)")
    print(f"{'run':16} {'duel':4} {'mode':11} {'status':9} {'win':>3} {'end s':>6} {'contact':>7} {'own hp':>6} "
          f"{'en hp':>6} {'trade':>6} {'own m':>6} {'en m':>6} {'ord/min':>7}  orders, abilities")
    for r in rows:
        print(f"{r['run']:16} {r['duel'] or '-':4} {r['mode']:11} {r['status'] or '-':9} {r['winner'] if r['winner'] is not None else '-':>3} "
              f"{r['t_end']:>6} {f(r['contact_s'], 0):>7} {f(r['own_hp_end']):>6} {f(r['enemy_hp_end']):>6} "
              f"{f(r['trade']):>6} {r['own_walked_m']:>6.0f} {r['enemy_walked_m']:>6.0f} {r['orders_per_min']:>7}  "
              f"{r['order_kinds']} {r['abilities']} {r['own_down']}{'/' if r['enemy_down'] else ''}{r['enemy_down']}")
    table = summary(rows)
    print(f"{'duel':4} {'mode':11} {'n':>2} {'wins own:enemy':>14} {'trade':>6} {'own lost':>8} {'en lost':>7} "
          f"{'contact':>7} {'ord/min':>7} {'own m':>6} {'en m':>6}  abilities")
    for r in table:
        print(f"{r['duel']:4} {r['mode']:11} {r['battles']:>2} {str(r['own_wins']) + ':' + str(r['enemy_wins']):>14} "
              f"{f(r['trade']):>6} {f(r['own_lost']):>8} {f(r['enemy_lost']):>7} {f(r['contact_s'], 0):>7} "
              f"{f(r['orders_per_min'], 1):>7} {f(r['own_walked_m'], 0):>6} {f(r['enemy_walked_m'], 0):>6}  {r['abilities']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"battles": rows, "summary": table}, indent=1), encoding="utf-8")
    print(out)
    return rows, table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("report", "plan", "run"), default="report")
    parser.add_argument("runs", nargs="*", type=Path, help="report: run folders (default: build/lord-duel/runs/*)")
    parser.add_argument("--net", type=int, default=4, help="plan/run: network battles a lord type (attack/defend in turn)")
    parser.add_argument("--control", type=int, default=2, help="plan/run: script-against-script battles a lord type")
    parser.add_argument("--kinds", default="emp,skv")
    parser.add_argument("--checkpoint", help="run: the network's weights, a path inside the repository")
    parser.add_argument("--greedy", action="store_true", help="run: the most likely order (the gate samples)")
    parser.add_argument("--dry", action="store_true", help="run: print the commands only")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    battles = plan(args.net, args.control, tuple(args.kinds.split(",")))
    if args.command == "plan":
        for b in battles:
            print(*b)
        print(f"{len(battles)} battles, ~2-3 min each with the game's loading")
        return 0
    if args.command == "run":
        if not args.checkpoint and any(b[1] == "net" for b in battles):
            parser.error("run needs --checkpoint for the network's battles")
        done = run(battles, args.checkpoint, dry=args.dry, greedy=args.greedy)
        if args.dry:
            return 0
        print("done:", done)
        report(runs())
        return 0 if all(d[-1] == 0 for d in done) else 1
    report(args.runs or runs())
    return 0


if __name__ == "__main__":
    sys.exit(main())
