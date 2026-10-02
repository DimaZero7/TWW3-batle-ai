"""The in-game gate: the network against the game's AI at fair Normal difficulty, on generated
battles it has never seen (docs/en/launch/gate.md). tools/launcher/gate.ps1 runs it:

    python -m tools.nn.gate plan --battles 4            # JSON: the battles (seed, role, limits)
    python -m tools.nn.gate summary build/nn-gate/<time>   # summary.json + a table; exit 1 unless passed

Battles come from the gate block of the generator's EVAL seeds (GATE_BLOCK), never used in
training. The plan walks the block in order and takes seeds by the size of our army, in turn
1-4, 5-9, 10-14 and 15-19 units besides the lord, so even a short gate has small and large
armies. Roles alternate: our network attacks in battles 1, 3, ..., defends in 2, 4, ...

The outcome of a battle: the engine's `result` event (src/entries/nn_arena.lua). The game gives
the defender the win when time is out, so `timeout` (our limit, the simulator's battle length)
and `stalled` (nobody took damage for 10 minutes of battle) are the defender's wins. `deadline`
(real time ran out), `incomplete`, a crash or no result: no result, not a win. A battle not at
Normal difficulty (launch.json battle_difficulty 1) does not count either.

Plain python (numpy through the generator): runs in the project's .venv.
"""
import argparse
import json
import math
import sys
from pathlib import Path

from tools import config as project

GATE_BLOCK = range(1_000_900_000, 1_001_000_000)   # the last 100 000 of generate.EVAL_SEEDS
SIZE_BINS = ((0, 4), (5, 9), (10, 14), (15, 19))  # our units besides the lord
ROLES = ("attack", "defend")
PASS_SHARE = 0.75                                  # 4 battles: 3 wins
REAL_SPEED = 5                                     # the slowest engine speed the deadline allows (x20 gives ~x9)
SIM = project.CONFIG_DIR / "nn" / "sim.json"
OUT = project.BUILD / "nn-gate"


def battle_limit_s():
    """The battle's time limit: the simulator's (config/nn/sim.json battle_limit_s), as in training."""
    return int(json.loads(SIM.read_text(encoding="utf-8"))["battle_limit_s"]["value"])


def deadline_s(timeout_s):
    """Real seconds a battle may take before the script ends it."""
    return int(timeout_s / REAL_SPEED) + 120


def _size(arena):
    return sum(not u.get("general") for u in arena["sides"]["own"]["units"])


def plan(battles, offset=0, block=GATE_BLOCK):
    """[{battle, seed, role, own_units, enemy_units, factions}] for battles offset+1 .. offset+battles."""
    from tools.nn.armies import generate
    assert block.start >= generate.EVAL_SEEDS.start and block.stop <= generate.EVAL_SEEDS.stop
    total = offset + battles
    want = [SIZE_BINS[i % len(SIZE_BINS)] for i in range(total)]
    picked = [None] * total
    queues = {b: [] for b in SIZE_BINS}
    seeds = iter(block)
    for i, bin_ in enumerate(want):
        while not queues[bin_]:
            seed = next(seeds)
            arena = generate.battle(seed)
            n = _size(arena)
            for b in SIZE_BINS:
                if b[0] <= n <= b[1]:
                    queues[b].append((seed, arena))
        seed, arena = queues[bin_].pop(0)
        picked[i] = (seed, arena)
    out = []
    for i in range(offset, total):
        seed, arena = picked[i]
        out.append({"battle": i + 1, "seed": seed, "role": ROLES[i % 2],
                    "own_units": len(arena["sides"]["own"]["units"]),
                    "enemy_units": len(arena["sides"]["enemy"]["units"]),
                    "factions": {s: arena["sides"][s]["faction"] for s in ("own", "enemy")},
                    "budget": arena["budget"],
                    "side_budget": {s: arena["sides"][s]["budget"] for s in ("own", "enemy")}})
    return out


def min_wins(battles):
    return math.ceil(PASS_SHARE * battles - 1e-9)


def _json(path):
    """A JSON file (the launcher's are UTF-8 with a BOM), or {} when missing."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def read_events(path):
    """(the result event or None, error messages) of a run's events.jsonl."""
    result, errors = None, []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if '"event":"result"' in line:
                    result = json.loads(line)
                elif '"event":"error"' in line:
                    errors.append(json.loads(line).get("message"))
    except OSError:
        pass
    return result, errors


def outcome(result, own_role):
    """(winner side 1 | 2 | None, how) from the result event; timeouts go to the defender."""
    if not result:
        return None, "no_result"
    status, winner = result.get("status"), result.get("winner") or 0
    if status == "completed" and winner in (1, 2):
        return winner, "completed"
    if status in ("timeout", "stalled"):
        return (1 if own_role == "defend" else 2), status
    return None, status or "no_result"


def battle_row(entry):
    """One battle of the gate from its run folder (build/nn-arena/runs/<time>/)."""
    run = Path(entry["run"]) if entry.get("run") else None
    row = {"battle": entry["battle"], "seed": entry["seed"], "role": entry["role"],
           "attempts": entry.get("attempts", 1), "run": run.name if run else None}
    if not run or not run.exists():
        row.update(outcome="no_result", how="no_run", winner=None)
        return row
    cfg = _json(run / "manifest.json").get("config", {})
    launch, status = _json(run / "launch.json"), _json(run / "status.json")
    result, errors = read_events(run / "events.jsonl")
    army = cfg.get("army", {})
    role = cfg.get("own_role", entry["role"])
    winner, how = outcome(result, role)
    fair = launch.get("battle_difficulty") == 1
    if not fair:
        winner, how = None, "not_normal_difficulty"
    if army.get("seed") not in (None, entry["seed"]) or role != entry["role"]:
        winner, how = None, "run_does_not_match_plan"
    result = result or {}
    row.update({
        "factions": cfg.get("factions"), "template": army.get("template"), "budget": army.get("budget"),
        "side_budget": army.get("side_budget"), "cost": army.get("cost"),
        "units": {s: len(cfg.get("units", {}).get(s, [])) for s in ("own", "enemy")},
        "men_start": army.get("men"),
        "men_left": {"own": result.get("side_1_men"), "enemy": result.get("side_2_men")},
        "standing_left": {"own": result.get("side_1_standing_units"), "enemy": result.get("side_2_standing_units")},
        "winner": {1: "net", 2: "game_ai"}.get(winner), "how": how,
        "outcome": "win" if winner == 1 else "loss" if winner == 2 else "no_result",
        "duration_s": round(result["duration_model_ms"] / 1000, 1) if result.get("duration_model_ms") else None,
        "wall_s": result.get("duration_wall_s"),
        "nn": {k[3:]: result.get(k) for k in ("nn_moves", "nn_answered", "nn_missed", "nn_orders_given",
                                               "nn_keeps", "nn_bad_files")},
        "lua_errors": errors, "battle_difficulty": launch.get("battle_difficulty"),
        "launch_status": status.get("status"), "preferences_restored": bool(status.get("preferences_restored")),
        "build": cfg.get("build")})
    return row


def summarize(gate_dir):
    """Reads gate_dir/battles.json (written by gate.ps1), writes gate_dir/summary.json; returns it."""
    gate_dir = Path(gate_dir)
    doc = _json(gate_dir / "battles.json")
    rows = [battle_row(e) for e in doc.get("battles", [])]
    planned = doc.get("planned", len(rows))
    count = {k: sum(r["outcome"] == k for r in rows) for k in ("win", "loss", "no_result")}
    need = doc.get("min_wins") or min_wins(planned)
    summary = {"checkpoint": doc.get("checkpoint"), "greedy": doc.get("greedy"), "speed": doc.get("speed"),
               "timeout_s": doc.get("timeout_s"), "planned": planned, "played": len(rows),
               "wins": count["win"], "losses": count["loss"], "no_result": count["no_result"],
               "min_wins": need, "passed": len(rows) == planned and count["win"] >= need,
               "fair": all(r.get("battle_difficulty") == 1 for r in rows),
               "preferences_restored": all(r.get("preferences_restored") for r in rows),
               "battles": rows}
    (gate_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    return summary


def _short(faction):
    return {"wh_main_emp_empire": "EMP", "wh2_main_skv_skaven": "SKV"}.get(faction, (faction or "?")[-8:])


def table(summary):
    """The summary as text lines."""
    head = (f"{'#':>2} {'seed':>10} {'factions':>9} {'units':>7} {'role':>6} {'winner':>7} {'how':>9} "
            f"{'battle s':>8} {'men left':>11} {'nn moves/miss/orders':>20}")
    lines = [head, "-" * len(head)]
    for r in summary["battles"]:
        f = r.get("factions") or {}
        u, m, nn = r.get("units") or {}, r.get("men_left") or {}, r.get("nn") or {}
        lines.append(f"{r['battle']:>2} {r['seed']:>10} {_short(f.get('own')) + '-' + _short(f.get('enemy')):>9} "
                     f"{str(u.get('own', '?')) + 'v' + str(u.get('enemy', '?')):>7} {r['role']:>6} "
                     f"{r.get('winner') or '-':>7} {r.get('how') or '-':>9} "
                     f"{r.get('duration_s') if r.get('duration_s') is not None else '-':>8} "
                     f"{str(m.get('own', '-')) + '/' + str(m.get('enemy', '-')):>11} "
                     f"{str(nn.get('moves', '-')) + '/' + str(nn.get('missed', '-')) + '/' + str(nn.get('orders_given', '-')):>20}")
    verdict = "PASSED" if summary["passed"] else "NOT PASSED"
    lines.append(f"wins {summary['wins']}, losses {summary['losses']}, no result {summary['no_result']} "
                 f"of {summary['planned']} (need {summary['min_wins']}): {verdict}; "
                 f"fair Normal difficulty: {summary['fair']}; preferences restored: {summary['preferences_restored']}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="the battles of a gate as JSON")
    p.add_argument("--battles", type=int, default=4)
    p.add_argument("--offset", type=int, default=0, help="skip this many battles of the plan (another set)")
    p.add_argument("--timeout", type=int, default=0, help="battle limit, model s (0: the simulator's)")
    s = sub.add_parser("summary", help="summary.json and a table of a gate folder")
    s.add_argument("gate_dir", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.cmd == "plan":
        if args.battles < 1 or args.offset < 0:
            parser.error("--battles must be 1 or more, --offset 0 or more")
        timeout = args.timeout or battle_limit_s()
        print(json.dumps({"timeout_s": timeout, "deadline_s": deadline_s(timeout), "min_wins": min_wins(args.battles),
                          "battles": plan(args.battles, args.offset)}))
        return 0
    summary = summarize(args.gate_dir)
    for line in table(summary):
        print(line)
    print("summary:", args.gate_dir / "summary.json")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
