"""The in-game gate: the network against the game's AI at fair Normal difficulty, on generated
battles it has never seen (docs/en/launch/gate.md). tools/launcher/gate.ps1 runs it:

    python -m tools.nn.gate plan --battles 4            # JSON: the battles (seed, role, limits)
    python -m tools.nn.gate summary build/nn-gate/<time>   # summary.json + a table; exit 1 unless passed

Battles come from the gate block of the generator's EVAL seeds (GATE_BLOCK), never used in
training, in swapped pairs (docs/en/training/training.md "Network evaluation: fair metrics"):
battles 1 and 2 are one seed, our network on the generator's own army, then on its enemy army
(build --army-swap), the same army attacking, so our network attacks in battles 1, 3, ... and
defends in 2, 4, ... A pair is won both / split / lost both; the faction matchup weighs on both of
its battles alike. The plan walks the block in order and takes a pair's seed by the size of the
generator's own army, in turn 1-4, 10-14, 5-9 and 15-19 units besides the lord (PAIR_BINS), so even
a short gate has small and large armies. No script baseline in the game (it would take the game's
AI against itself).

The outcome of a battle: the engine's `result` event (src/entries/nn_arena.lua). The game gives
the defender the win when time is out, so `timeout` (our limit, the simulator's battle length)
and `stalled` (nobody took damage for 10 minutes of battle) are the defender's wins. `deadline`
(real time ran out), `incomplete`, a crash or no result: no result, not a win. A battle not at
Normal difficulty (launch.json battle_difficulty 1) does not count either.

Liveliness (measured only, docs/en/launch/gate.md): from each recording our network's given orders
(nn_orders: changes, attack-target switches, A->B->A flips, move jitter, twitching out of melee, as
tools/nn/train/behaviour.py counts them in the simulator) and both sides' own current targets every
second (nn_sample): the game's AI's target switches are the reference band.

Plain python (numpy through the generator): runs in the project's .venv.
"""
import argparse
import json
import math
import sys
from pathlib import Path

from tools import config as project
from tools.nn.train import matchups, skill

GATE_BLOCK = range(1_000_900_000, 1_001_000_000)   # the last 100 000 of generate.EVAL_SEEDS
SIZE_BINS = ((0, 4), (5, 9), (10, 14), (15, 19))  # our units besides the lord
PAIR_BINS = (0, 2, 1, 3)                           # the size bins of pairs 1, 2, 3, 4 (then again)
ROLES = ("attack", "defend")
PASS_SHARE = 0.75                                  # 4 battles: 3 wins
REAL_SPEED = 5                                     # the slowest engine speed the deadline allows (x20 gives ~x9)
SIM = project.CONFIG_DIR / "nn" / "sim.json"
UNITS = project.CONFIG_DIR / "nn" / "units.json"
ROUT_SHARE = 0.5                                   # = tools/nn/train/reward.py Weights.rout_share (no torch here)
OUT = project.BUILD / "nn-gate"


def battle_limit_s():
    """The battle's time limit: the simulator's (config/nn/sim.json battle_limit_s), as in training."""
    return int(json.loads(SIM.read_text(encoding="utf-8"))["battle_limit_s"]["value"])


def deadline_s(timeout_s):
    """Real seconds a battle may take before the script ends it."""
    return int(timeout_s / REAL_SPEED) + 120


def _size(arena):
    return sum(not u.get("general") for u in arena["sides"]["own"]["units"])


# symmetric: a seed plays 4 battles, (swap, role) in turn - so each army attacks and defends under
# either pair of hands (in the plain plan the generator's own army always attacks: our network
# defends only with the other army, a faction matchup's role never turns round).
SYMMETRIC = ((False, "attack"), (True, "defend"), (False, "defend"), (True, "attack"))


def plan(battles, offset=0, block=GATE_BLOCK, symmetric=False):
    """[{battle, pair, swap, seed, role, own_units, enemy_units, factions, budget, side_budget}] for
    battles offset+1 .. offset+battles; own / enemy: our network's army and the game AI's (swapped in
    the second battle of a pair). symmetric: 4 battles a seed (SYMMETRIC; pairs 2k+1 and 2k+2: in the
    second pair the generator's enemy army attacks), else 2 (its own army attacks)."""
    from tools.nn.armies import generate
    assert block.start >= generate.EVAL_SEEDS.start and block.stop <= generate.EVAL_SEEDS.stop
    total = offset + battles
    per_seed = 4 if symmetric else 2
    want = [SIZE_BINS[PAIR_BINS[k % len(PAIR_BINS)]] for k in range((total + per_seed - 1) // per_seed)]
    picked = [None] * len(want)
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
        seed, arena = picked[i // per_seed]
        swap, role = SYMMETRIC[i % 4] if symmetric else (i % 2 == 1, ROLES[i % 2])
        ours, theirs = ("enemy", "own") if swap else ("own", "enemy")
        out.append({"battle": i + 1, "pair": i // 2 + 1, "swap": swap, "seed": seed, "role": role,
                    "own_units": len(arena["sides"][ours]["units"]),
                    "enemy_units": len(arena["sides"][theirs]["units"]),
                    "factions": {"own": arena["sides"][ours]["faction"], "enemy": arena["sides"][theirs]["faction"]},
                    "budget": arena["budget"],
                    "side_budget": {"own": arena["sides"][ours]["budget"], "enemy": arena["sides"][theirs]["budget"]}})
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


def final_units(path):
    """The units of a run's last nn_final (else last nn_sample) event: [{n, side, men, hp, r, s, ...}] or None."""
    final = sample = None
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if '"event":"nn_final"' in line:
                    final = line
                elif '"event":"nn_sample"' in line:
                    sample = line
    except OSError:
        return None
    line = final or sample
    return json.loads(line).get("units") if line else None


def _costs():
    """{unit key: multiplayer cost} from the passports (config/nn/units.json)."""
    units = _json(UNITS).get("units", {})
    return {k: u.get("multiplayer_cost") for k, u in units.items()}


def gold(cfg, units, costs, rout_share=ROUT_SHARE):
    """{lost: [side 1, side 2], start: [side 1, side 2]}: the gold each side lost at the end, as the
    simulator's reward counts it (tools/nn/train/reward.py gold_lost): a unit's cost x the health share
    lost; dead (no men) or shattered: whole; routing: rout_share of what it has left besides; a unit
    missing from the last sample: lost whole. None without the units' keys, costs or the end state."""
    listed = cfg.get("units") or {}
    if not units:
        return None
    end = {u.get("n"): u for u in units}
    lost, start = [0.0, 0.0], [0.0, 0.0]
    for i, s in enumerate(("own", "enemy")):
        for spec in listed.get(s) or []:
            cost = costs.get(spec.get("key"))
            if not cost or not spec.get("script_name"):
                return None
            start[i] += cost
            u = end.get(spec["script_name"])
            if u is None or (u.get("men") or 0) <= 0 or u.get("s"):
                share = 1.0
            else:
                hp = min(max(float(u.get("hp") or 0.0), 0.0), 1.0)
                share = (1 - hp) + hp * rout_share * bool(u.get("r"))
            lost[i] += cost * share
    if not start[0] or not start[1]:
        return None
    return {"lost": [round(x, 1) for x in lost], "start": start}


# Liveliness, as tools/nn/train/behaviour.py measures it in the simulator (its constants; no torch here).
MOVE_M, REPEAT_M, FLIP_S, TWITCH_PER_MIN, FREE_MIN_S = 10.0, 5.0, 10.0, 6.0, 30.0
LIVELY_COUNTS = ("unit_s", "changes", "switches", "flips", "move_s", "repoints", "repoint_m", "free_s", "free_changes",
                 "twitch_units", "free_units", "eng_s", "eng_switches", "eng_flips")
POINT_KINDS = ("move", "withdraw")


def _point(o):
    return o is not None and o.get("k") in POINT_KINDS and o.get("x") is not None and o.get("z") is not None


def _dist(a, b):
    return math.hypot(a["x"] - b["x"], a["z"] - b["z"])


def _same_order(a, b):
    """The order b is a (A -> B -> A): the same kind, target, a point within MOVE_M."""
    if a is None or b is None or a.get("k") != b.get("k"):
        return False
    if a.get("k") == "attack":
        return a.get("tg") == b.get("tg")
    return _dist(a, b) <= MOVE_M if _point(a) and _point(b) else True


def liveliness(path):
    """Liveliness counts of a run's events.jsonl: {"net": counts, "game_ai": counts} (LIVELY_COUNTS) or
    None without samples. Our network's orders: the bridge's nn_orders (only orders that changed are
    given; a change: a new kind, attack target or a point more than MOVE_M away; a switch: a new attack
    target while the old one stands; a flip: back to the order before the previous one within FLIP_S;
    move jitter: successive points of a moving unit more than REPEAT_M apart; twitching: FREE_MIN_S or
    more out of melee, TWITCH_PER_MIN or more changes a minute there). Both sides, from the nn_sample
    rows (every second): the units' own current target (t) switching (eng_switches) and flipping back
    within FLIP_S (eng_flips), a standing unit's seconds (eng_s): the game's AI as a reference."""
    c = {s: dict.fromkeys(LIVELY_COUNTS, 0.0) for s in ("net", "game_ai")}
    last = {}                  # name -> the last sample row
    order, before, changed = {}, {}, {}      # our units: the order in force, the one before it, when it changed
    eng = {}                   # name -> (target, previous target, when it changed)
    free_s, free_ch = {}, {}
    t_prev = None
    seen = False
    try:
        f = open(path, encoding="utf-8")
    except OSError:
        return None
    with f:
        for line in f:
            if '"event":"nn_sample"' in line:
                ev = json.loads(line)
                t = ev.get("t", 0) / 1000
                dt = 0.0 if t_prev is None else min(max(t - t_prev, 0.0), 2.0)
                t_prev = t
                seen = True
                for u in ev.get("units") or []:
                    who = "net" if u.get("side") == 1 else "game_ai"
                    name = u.get("n")
                    last[name] = u
                    up = (u.get("men") or 0) > 0 and not u.get("r") and not u.get("s")
                    tg = (u.get("t") or "") if up else ""
                    if up:
                        c[who]["eng_s"] += dt
                        if who == "net":
                            c[who]["unit_s"] += dt
                            if _point(order.get(name)):
                                c[who]["move_s"] += dt
                            if not u.get("m"):
                                free_s[name] = free_s.get(name, 0.0) + dt
                    old, older, when = eng.get(name, ("", "", -1e9))
                    if tg and old and tg != old:
                        c[who]["eng_switches"] += 1
                        if tg == older and t - when <= FLIP_S:
                            c[who]["eng_flips"] += 1
                        eng[name] = (tg, old, t)
                    elif tg != old:
                        eng[name] = (tg, older, when)
            elif '"event":"nn_orders"' in line:
                ev = json.loads(line)
                t = ev.get("t", 0) / 1000
                for o in ev.get("orders") or []:
                    if o.get("status") != "given":
                        continue
                    name = o.get("u")
                    cur = order.get(name)
                    new_target = o.get("k") == "attack" and cur is not None and o.get("tg") != cur.get("tg")
                    moved = _point(o) and _point(cur) and _dist(o, cur) > MOVE_M
                    if cur is None or o.get("k") != cur.get("k") or new_target or moved:
                        c["net"]["changes"] += 1
                        if cur is not None and cur.get("k") == "attack" and o.get("k") == "attack":
                            old = last.get(cur.get("tg")) or {}
                            if (old.get("men") or 0) > 0 and not old.get("r") and not old.get("s"):
                                c["net"]["switches"] += 1
                        if _same_order(o, before.get(name)) and t - changed.get(name, -1e9) <= FLIP_S:
                            c["net"]["flips"] += 1
                        if not (last.get(name) or {}).get("m"):
                            free_ch[name] = free_ch.get(name, 0) + 1
                        before[name], changed[name] = cur, t
                    if _point(o) and _point(cur) and _dist(o, cur) > REPEAT_M:
                        c["net"]["repoints"] += 1
                        c["net"]["repoint_m"] += _dist(o, cur)
                    order[name] = o
    if not seen:
        return None
    c["net"]["free_s"] = sum(free_s.values())
    c["net"]["free_changes"] = float(sum(free_ch.values()))
    for name, s in free_s.items():
        if s >= FREE_MIN_S:
            c["net"]["free_units"] += 1
            c["net"]["twitch_units"] += free_ch.get(name, 0) / (s / 60) >= TWITCH_PER_MIN
    return {k: {n: round(v, 3) for n, v in x.items()} for k, x in c.items()}


def lively_rates(c):
    """Rates of liveliness counts (one battle's or a sum): the names of tools/nn/train/behaviour.py."""
    if not c:
        return {}

    def per(k, d, scale=60.0):
        return c[k] / c[d] * scale if c.get(d) else None
    return {"order_changes_per_min": per("changes", "unit_s"), "target_switches_per_min": per("switches", "unit_s"),
            "flips_per_min": per("flips", "unit_s"), "move_jitter_m": per("repoint_m", "repoints", 1.0),
            "move_repoints_per_min": per("repoints", "move_s"), "free_changes_per_min": per("free_changes", "free_s"),
            "twitch_share": per("twitch_units", "free_units", 1.0), "engine_switches_per_min": per("eng_switches", "eng_s"),
            "engine_flips_per_min": per("eng_flips", "eng_s")}


def lively_summary(rows):
    """{"net": {role: rates}, "game_ai": rates, "band": {"net"|"game_ai": [min, median, max] of the
    battles' engine_switches_per_min}} over the battles with a recording."""
    have = [r for r in rows if r.get("lively")]
    if not have:
        return None

    def total(rs, who):
        return {k: sum(r["lively"][who][k] for r in rs) for k in LIVELY_COUNTS}
    out = {"net": {role: lively_rates(total([r for r in have if r["role"] == role], "net"))
                   for role in ROLES if any(r["role"] == role for r in have)},
           "game_ai": {k: v for k, v in lively_rates(total(have, "game_ai")).items() if k.startswith("engine_")},
           "band": {}}
    for who in ("net", "game_ai"):
        v = sorted(x for x in (lively_rates(r["lively"][who]).get("engine_switches_per_min") for r in have)
                   if x is not None)
        if v:
            out["band"][who] = [round(v[0], 3), round(v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2, 3),
                                round(v[-1], 3)]
    return out


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


def battle_row(entry, costs=None):
    """One battle of the gate from its run folder (build/nn-arena/runs/<time>/). costs: {unit key: gold}
    (default: the passports)."""
    run = Path(entry["run"]) if entry.get("run") else None
    row = {"battle": entry["battle"], "pair": entry.get("pair"), "swap": bool(entry.get("swap")), "seed": entry["seed"],
           "role": entry["role"], "attempts": entry.get("attempts", 1), "run": run.name if run else None,
           "factions": entry.get("factions")}
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
    if army.get("seed") not in (None, entry["seed"]) or role != entry["role"] \
            or bool(army.get("swap")) != bool(entry.get("swap")):
        winner, how = None, "run_does_not_match_plan"
    result = result or {}
    g = gold(cfg, final_units(run / "events.jsonl"), _costs() if costs is None else costs)
    if g:
        budget = army.get("budget") or sum(g["start"]) / 2
        w = winner if winner in (1, 2) else 0                    # our network is side 1 (own)
        g.update(destroyed=g["lost"][1], own_lost=g["lost"][0], budget=budget,
                 trade=round((g["lost"][1] - g["lost"][0]) / budget, 4),
                 margin=round(float(skill.margin([w], [1], [g["lost"]], [g["start"]])[0]), 4))
    row.update({
        "factions": cfg.get("factions") or entry.get("factions"), "template": army.get("template"), "budget": army.get("budget"),
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
        "gold": g, "lively": liveliness(run / "events.jsonl"), "lua_errors": errors, "battle_difficulty": launch.get("battle_difficulty"),
        "launch_status": status.get("status"), "preferences_restored": bool(status.get("preferences_restored")),
        "build": cfg.get("build")})
    return row


def summarize(gate_dir):
    """Reads gate_dir/battles.json (written by gate.ps1), writes gate_dir/summary.json; returns it."""
    gate_dir = Path(gate_dir)
    doc = _json(gate_dir / "battles.json")
    costs = _costs()
    rows = [battle_row(e, costs) for e in doc.get("battles", [])]
    planned = doc.get("planned", len(rows))
    count = {k: sum(r["outcome"] == k for r in rows) for k in ("win", "loss", "no_result")}
    need = doc.get("min_wins") or min_wins(planned)
    summary = {"checkpoint": doc.get("checkpoint"), "greedy": doc.get("greedy"), "speed": doc.get("speed"),
               "timeout_s": doc.get("timeout_s"), "planned": planned, "played": len(rows),
               "wins": count["win"], "losses": count["loss"], "no_result": count["no_result"],
               "min_wins": need, "passed": len(rows) == planned and count["win"] >= need,
               "fair": all(r.get("battle_difficulty") == 1 for r in rows),
               "preferences_restored": all(r.get("preferences_restored") for r in rows),
               "by_faction": by_faction(rows), "pairs": pairs(rows), "pair_gold": pair_gold(rows),
               "liveliness": lively_summary(rows), "battles": rows}
    (gate_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    return summary


def by_faction(rows):
    """Wins by our faction and role and by matchup (ours first: EMP-SKV), battle counts and no-result
    battles (counted as games, not wins); tools/nn/train/matchups.py."""
    fac = [r.get("factions") or {} for r in rows]
    return matchups.group([r["outcome"] == "win" for r in rows], [f.get("own") for f in fac],
                          [f.get("enemy") for f in fac], [r["role"] == "attack" for r in rows],
                          void=[r["outcome"] == "no_result" for r in rows])


def pairs(rows):
    """The swapped pairs: skill.pairs (won both / split / lost both, pair_score; a pair with a battle
    without a result or a missing half is incomplete) and "each": [{pair, seed, outcomes, result}]."""
    out = skill.pairs([r["outcome"] == "win" for r in rows], [r.get("pair") or -1 for r in rows],
                      [r["outcome"] == "no_result" for r in rows])
    each = {}
    for r in rows:
        if r.get("pair"):
            p = each.setdefault(r["pair"], {"pair": r["pair"], "seed": r["seed"], "outcomes": []})
            p["outcomes"].append(r["outcome"])
    for p in each.values():
        o = p["outcomes"]
        p["result"] = ("incomplete" if len(o) != 2 or "no_result" in o
                       else {2: "won both", 1: "split", 0: "lost both"}[o.count("win")])
    out["each"] = list(each.values())
    return out


def pair_gold(rows):
    """skill.pair_gold over the swapped pairs: our network played the generator's own army (side 1) in
    the first battle of a pair and its enemy army (side 2) in the swapped one; a battle without a result
    or without the gold (no end state, no unit keys) makes its pair incomplete."""
    g = [r.get("gold") or {} for r in rows]
    nan = float("nan")
    return skill.pair_gold([r.get("pair") or -1 for r in rows], [2 if r.get("swap") else 1 for r in rows],
                           [x.get("destroyed", nan) for x in g], [x.get("own_lost", nan) for x in g],
                           [x.get("budget", nan) for x in g], [x.get("margin", nan) for x in g],
                           [r["outcome"] == "no_result" for r in rows])


_short = matchups.short


def table(summary):
    """The summary as text lines."""
    head = (f"{'#':>2} {'pair':>4} {'seed':>10} {'factions':>9} {'units':>7} {'role':>6} {'winner':>7} {'how':>9} "
            f"{'battle s':>8} {'men left':>11} {'nn moves/miss/orders':>20}")
    lines = [head, "-" * len(head)]
    for r in summary["battles"]:
        f = r.get("factions") or {}
        u, m, nn = r.get("units") or {}, r.get("men_left") or {}, r.get("nn") or {}
        pair = (str(r["pair"]) + ("s" if r.get("swap") else "")) if r.get("pair") else "-"
        lines.append(f"{r['battle']:>2} {pair:>4} {r['seed']:>10} {_short(f.get('own')) + '-' + _short(f.get('enemy')):>9} "
                     f"{str(u.get('own', '?')) + 'v' + str(u.get('enemy', '?')):>7} {r['role']:>6} "
                     f"{r.get('winner') or '-':>7} {r.get('how') or '-':>9} "
                     f"{r.get('duration_s') if r.get('duration_s') is not None else '-':>8} "
                     f"{str(m.get('own', '-')) + '/' + str(m.get('enemy', '-')):>11} "
                     f"{str(nn.get('moves', '-')) + '/' + str(nn.get('missed', '-')) + '/' + str(nn.get('orders_given', '-')):>20}")
    verdict = "PASSED" if summary["passed"] else "NOT PASSED"
    lines.append(f"wins {summary['wins']}, losses {summary['losses']}, no result {summary['no_result']} "
                 f"of {summary['planned']} (need {summary['min_wins']}): {verdict}; "
                 f"fair Normal difficulty: {summary['fair']}; preferences restored: {summary['preferences_restored']}")
    if summary.get("by_faction"):
        lines.append(f"by faction (wins/battles): {matchups.text(summary['by_faction'])}")
    p = summary.get("pairs")
    if p and p.get("each"):
        score = "-" if p.get("pair_score") is None else f"{p['pair_score']:+.2f}"
        lines.append("pairs (one seed, our network on either army): "
                     + "; ".join(f"{x['pair']} {x['result']}" for x in p["each"])
                     + f"; pair score {score} over {p['pairs']} complete")
    g = summary.get("pair_gold")
    if g and g.get("pairs"):
        pg, ex, weak = skill.gold_cells(g)
        lines.append(f"pair gold (ours - game AI's, same armies, / budget): {pg}; exchange {ex}; "
                     f"weak army destroyed/lost net vs game AI: {weak}; over {g['pairs']} pairs")
    lines += lively_lines(summary.get("liveliness"))
    return lines


def lively_lines(lv):
    """The liveliness lines of a summary (lively_summary) or []."""
    if not lv:
        return []
    f = (lambda v, fmt="{:.2f}": "-" if v is None else fmt.format(v))
    net = lv["net"]
    roles = [r for r in ROLES if r in net]
    cell = (lambda k, fmt="{:.2f}": " / ".join(f(net[r].get(k), fmt) for r in roles))
    band = (lambda w: "-" if w not in lv["band"] else "{:.2f} [{:.2f}, {:.2f}]".format(*[lv["band"][w][i] for i in (1, 0, 2)]))
    return [f"liveliness of our network in the game ({' / '.join(roles)}; per unit-minute): order changes "
            f"{cell('order_changes_per_min')}, attack-target switches {cell('target_switches_per_min')}, flips A-B-A "
            f"{cell('flips_per_min', '{:.3f}')}, move jitter {cell('move_jitter_m', '{:.1f}')} m, out of melee "
            f"{cell('free_changes_per_min')}, twitching units {cell('twitch_share', '{:.2f}')}",
            f"units' own target switches / unit-min (every second; median [min, max] of the battles): ours "
            f"{band('net')}, the game's AI {band('game_ai')} (flips {f(lv['game_ai'].get('engine_flips_per_min'), '{:.3f}')})"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="the battles of a gate as JSON")
    p.add_argument("--battles", type=int, default=4)
    p.add_argument("--offset", type=int, default=0, help="skip this many battles of the plan (another set)")
    p.add_argument("--timeout", type=int, default=0, help="battle limit, model s (0: the simulator's)")
    p.add_argument("--symmetric", action="store_true",
                   help="4 battles a seed: each army attacks and defends under either side (SYMMETRIC)")
    s = sub.add_parser("summary", help="summary.json and a table of a gate folder")
    s.add_argument("gate_dir", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.cmd == "plan":
        if args.battles < 1 or args.offset < 0:
            parser.error("--battles must be 1 or more, --offset 0 or more")
        timeout = args.timeout or battle_limit_s()
        print(json.dumps({"timeout_s": timeout, "deadline_s": deadline_s(timeout), "min_wins": min_wins(args.battles),
                          "battles": plan(args.battles, args.offset, symmetric=args.symmetric)}))
        return 0
    summary = summarize(args.gate_dir)
    for line in table(summary):
        print(line)
    print("summary:", args.gate_dir / "summary.json")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
