"""The lord against the game's AI: does the game's battle AI make its lord hit harder than a plain lord?

In the gate battles the game AI's lord took our lord's health ~1.4x faster than the simulator says, while
our lord hit his as the simulator says (build/lordmorale). Is the AI's lord stronger (better card stats:
level, skills, the difficulty), does it use its abilities, or does it fight better (charges again, takes
the flank)? Here the lords fight alone (tools/nn/lord_duel.py's arena, mirrored: Empire General v General,
Skaven Warlord v Warlord). Side 1 is under script: one attack order on the other lord, no abilities. Side 2
is the game's own battle AI ('ai'), or, the control, the same script ('control'; also the scripted-v-
scripted solo duels of build/lord-duel/runs). Both lords' unit cards (CCO UnitDetailsContext.StatList,
rank, experience: nn_card on change), active effects (nn_effects) and abilities ready (nn_ability_ready:
a use shows as ready true -> false) are recorded for both sides (entries.nn_arena observe, cards).

    python -m tools.nn.lord_ai plan                     # the battles
    python -m tools.nn.lord_ai run --ai 4 --control 2   # builds and launches them (launch.ps1 -Target lord-ai)
    python -m tools.nn.lord_ai                          # the table from build/lord-ai/runs + the old controls

Duel seconds: both lords in melee, neither routing nor shattered. The rate on a lord: his health lost
(unary) in duel seconds over their length, %HP/s; 'fresh': the first FRESH_S duel seconds. The AI's
advantage: the enemy lord's pooled rate on ours in 'ai' over the same in 'control', per lord type, a 95 %
bootstrap interval over the battles. Writes build/lord-ai/analysis.json.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from tools import config as project
from tools.nn import lord_duel

ROOT = project.BUILD / "lord-ai"
RUNS = ROOT / "runs"
OUT = ROOT / "analysis.json"
OWN, ENEMY = "own_lord", "enemy_lord"
FRESH_S = 30
TIMEOUT_S = 900
BOOT = 4000
EVENTS = ('"nn_sample"', '"nn_final"', '"nn_card"', '"nn_effects"', '"nn_ability_ready"', '"event":"result"')


def build_config(kind, own_role, enemy, timeout_s=TIMEOUT_S):
    """tools.build lord-ai: the battle file and the entry's config (our lord scripted, theirs the game's AI
    or scripted; cards, effects and abilities recorded)."""
    return lord_duel.build_config(kind, "scripted", own_role, 0, 0, timeout_s, "solo", enemy_ai=enemy, root=ROOT,
                                  record=True)


def plan(ai=4, control=2, kinds=("emp", "skv")):
    """[(kind, enemy, own_role)]: the game's AI attacking and defending in turn, the controls attack."""
    out = []
    for r in range(max(ai, control)):
        for k in kinds:
            if r < ai:
                out.append((k, "game", "attack" if r % 2 == 0 else "defend"))
            if r < control:
                out.append((k, "scripted", "attack"))
    return out


def run(battles, python=sys.executable, dry=False):
    done = []
    for i, (kind, enemy, role) in enumerate(battles, 1):
        build = [python, "-m", "tools.build", "lord-ai", "--duel", kind, "--own-role", role, "--duel-enemy", enemy]
        launch = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                  str(project.ROOT / "tools" / "launcher" / "launch.ps1"), "-Target", "lord-ai"]
        print(f"--- {i}/{len(battles)}: {kind} {enemy} {role}", flush=True)
        if dry:
            print(" ".join(build), "&&", " ".join(launch))
            continue
        code = subprocess.run(build, cwd=project.ROOT, stdout=subprocess.DEVNULL).returncode
        if code == 0:
            code = subprocess.run(launch, cwd=project.ROOT).returncode
        done.append((kind, enemy, role, code))
    return done


# ---------------------------------------------------------------- analysis

def load_run(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    out = {"run": run_dir.name, "config": config, "samples": [], "cards": [], "effects": [], "ready": [],
           "result": None}
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not any(k in line for k in EVENTS):
            continue
        r = json.loads(line)
        ev = r["event"]
        if ev in ("nn_sample", "nn_final"):
            out["samples"].append((r["t"] / 1000, {u["n"]: u for u in r["units"]}))
        elif ev == "nn_card":
            out["cards"].append(r)
        elif ev == "nn_effects":
            out["effects"].append(r)
        elif ev == "nn_ability_ready":
            out["ready"].append(r)
        elif ev == "result":
            out["result"] = r
    return out if out["samples"] else None


def mode_of(config):
    """'ai' (side 2 the game's AI), 'control' (both scripted, solo) or None (not this experiment)."""
    if config.get("own_ai") != "scripted" or config.get("variant", "solo") != "solo":
        return None
    return "ai" if config.get("duel_enemy") == "game" else "control"


def down(row):
    return not row or row.get("r") or row.get("s") or row.get("men") == 0


def duel_steps(samples):
    """[(t0, dt, own_lost, enemy_lost, own_row, enemy_row)] for each step that starts with both lords in melee
    and neither down."""
    out = []
    for (t0, a), (t1, b) in zip(samples, samples[1:]):
        o, e = a.get(OWN), a.get(ENEMY)
        if down(o) or down(e) or not (o.get("m") and e.get("m")):
            continue
        lost = lambda n: (a[n].get("hp") or 0.0) - ((b.get(n) or {}).get("hp") or 0.0)
        out.append((t0, t1 - t0, lost(OWN), lost(ENEMY), o, e))
    return out


def card_stats(cards, unit):
    """The unit's first card {stat key: (Value, ValueBase)} and its fields, and the stats' later changes
    [(t s, key, Value)]."""
    rows = [c for c in cards if c.get("u") == unit]
    if not rows:
        return None, []
    def stats(c):
        return {s["k"]: (s.get("v"), s.get("b")) for s in c.get("stats") or [] if isinstance(s, dict)}
    first = stats(rows[0])
    card = {"stats": first, **{k: rows[0].get(k) for k in ("rank", "has_rank", "xp", "hpmax")}}
    changes, last = [], first
    for c in rows[1:]:
        now = stats(c)
        changes += [(round(c["t"] / 1000, 1), k, v[0]) for k, v in now.items() if last.get(k) != v]
        last = now
    return card, changes


def ability_uses(ready, windows, unit, merge_s=2.0):
    """[(t s, key)]: the unit's ability uses - an ability's phase appearing in his active effects (the game's
    AI side: can_perform_special_ability stays true for it, 06.10.2026), or its ready turning true -> false
    (ours); the two readings of one use (within merge_s) count once."""
    out, was = [], set()
    for t, fx in windows:
        now = {p for p in fx if is_ability(p)}
        out += [(round(t, 1), k) for k in sorted(now - was)]
        was = now
    flags = {}
    for r in ready:
        if r.get("u") != unit:
            continue
        t = round(r["t"] / 1000, 1)
        if flags.get(r["key"]) is True and r.get("ready") is False and not any(
                k == r["key"] and abs(t - t0) <= merge_s for t0, k in out):
            out.append((t, r["key"]))
        flags[r["key"]] = r.get("ready")
    return sorted(out)


def effect_windows(effects, unit):
    """[(t s, [phase keys])] of the unit's active effects, in time order."""
    return [(r["t"] / 1000, r["fx"] if isinstance(r.get("fx"), list) else []) for r in effects if r.get("u") == unit]


def effects_at(windows, t):
    now = []
    for t0, fx in windows:
        if t0 > t:
            break
        now = fx
    return now


def is_ability(phase):
    """An active ability's phase (the passives' phases, '_passive_', are always on)."""
    return ("_abilities_" in phase or "_ability_" in phase) and "_passive_" not in phase


def rate(lost, seconds):
    return round(100 * lost / seconds, 4) if seconds > 0 else None


def measure(run):
    cfg, s = run["config"], run["samples"]
    steps = duel_steps(s)
    res = run["result"] or {}
    out = {"run": run["run"], "duel": cfg.get("duel"), "mode": mode_of(cfg), "role": cfg.get("own_role"),
           "status": res.get("status"), "winner": res.get("winner"), "t_end": round(s[-1][0], 1)}
    duel_s = sum(st[1] for st in steps)
    out["contact_s"] = steps[0][0] if steps else None
    out["duel_s"] = round(duel_s, 1)
    out["own_lost"] = round(sum(st[2] for st in steps), 4)
    out["enemy_lost"] = round(sum(st[3] for st in steps), 4)
    out["on_own"], out["on_enemy"] = rate(out["own_lost"], duel_s), rate(out["enemy_lost"], duel_s)
    fresh = [st for st in steps if st[0] - steps[0][0] < FRESH_S] if steps else []
    fs = sum(st[1] for st in fresh)
    out["fresh_s"] = round(fs, 1)
    out["fresh_own_lost"] = round(sum(st[2] for st in fresh), 4)
    out["fresh_enemy_lost"] = round(sum(st[3] for st in fresh), 4)
    out["fresh_on_own"] = rate(out["fresh_own_lost"], fs)
    out["fresh_on_enemy"] = rate(out["fresh_enemy_lost"], fs)
    # behaviour: the enemy lord leaving melee and coming back (a new charge), our lord's flanks threatened
    back, was, started = 0, False, False
    for _, u in s:
        m = bool((u.get(ENEMY) or {}).get("m"))
        back += started and m and not was
        started, was = started or m, m
    out["enemy_reengaged"] = back
    flank = [st for st in steps if st[4].get("lf") or st[4].get("rf") or st[4].get("bf")]
    rear = [st for st in steps if st[4].get("bf")]
    out["own_flank_share"] = round(sum(st[1] for st in flank) / duel_s, 3) if duel_s else None
    out["own_rear_share"] = round(sum(st[1] for st in rear) / duel_s, 3) if duel_s else None
    # abilities and effects of both lords; the enemy's rate on ours with an ability of his active, and without
    for side, unit in (("own", OWN), ("enemy", ENEMY)):
        windows = effect_windows(run["effects"], unit)
        uses = ability_uses(run["ready"], windows, unit)
        c0 = out["contact_s"]
        out[f"{side}_abilities"] = [(k, t, None if c0 is None else round(t - c0, 1)) for t, k in uses]
        seen = sorted({p for _, fx in windows for p in fx})
        out[f"{side}_effects"] = seen
        card, changes = card_stats(run["cards"], unit)
        out[f"{side}_card"], out[f"{side}_card_changes"] = card, changes
    windows = effect_windows(run["effects"], ENEMY)
    on = [st for st in steps if any(is_ability(p) for p in effects_at(windows, st[0]))]
    off = [st for st in steps if st not in on]
    out["ability_on_s"] = round(sum(st[1] for st in on), 1)
    out["on_own_ability"] = rate(sum(st[2] for st in on), sum(st[1] for st in on))
    out["on_own_no_ability"] = rate(sum(st[2] for st in off), sum(st[1] for st in off))
    out["recorded"] = bool(run["cards"] or run["effects"] or run["ready"])
    return out


def pooled(rows, lost, seconds):
    s = sum(r[seconds] for r in rows)
    return rate(sum(r[lost] for r in rows), s)


def ratio_ci(ai, control, lost="own_lost", seconds="duel_s", boot=BOOT, seed=0):
    """The pooled rate in 'ai' over the pooled rate in 'control' and its 95 % bootstrap interval (the
    battles of each group resampled)."""
    a, c = pooled(ai, lost, seconds), pooled(control, lost, seconds)
    if not ai or not control or not a or not c:
        return None, None, None
    rnd, draws = random.Random(seed), []
    for _ in range(boot):
        x = pooled([rnd.choice(ai) for _ in ai], lost, seconds)
        y = pooled([rnd.choice(control) for _ in control], lost, seconds)
        if x is not None and y:
            draws.append(x / y)
    draws.sort()
    lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]
    return round(a / c, 3), round(lo, 3), round(hi, 3)


def summary(rows):
    out = []
    for duel in sorted({r["duel"] for r in rows}):
        g = {m: [r for r in rows if r["duel"] == duel and r["mode"] == m and r["duel_s"] > 0] for m in ("ai", "control")}
        row = {"duel": duel, "n_ai": len(g["ai"]), "n_control": len(g["control"])}
        for m in ("ai", "control"):
            row[f"{m}_duel_s"] = round(sum(r["duel_s"] for r in g[m]), 1)
            row[f"{m}_on_own"] = pooled(g[m], "own_lost", "duel_s")
            row[f"{m}_on_enemy"] = pooled(g[m], "enemy_lost", "duel_s")
            row[f"{m}_fresh_on_own"] = pooled(g[m], "fresh_own_lost", "fresh_s")
            row[f"{m}_fresh_on_enemy"] = pooled(g[m], "fresh_enemy_lost", "fresh_s")
            row[f"{m}_enemy_wins"] = sum(r["winner"] == 2 for r in g[m])
            row[f"{m}_reengaged"] = (round(sum(r["enemy_reengaged"] for r in g[m]) / len(g[m]), 2) if g[m] else None)
            fl = [r["own_rear_share"] for r in g[m] if r["own_rear_share"] is not None]
            row[f"{m}_own_rear_share"] = round(sum(fl) / len(fl), 3) if fl else None
        row["ratio_on_own"], row["ratio_lo"], row["ratio_hi"] = ratio_ci(g["ai"], g["control"])
        row["ratio_fresh"], row["fresh_lo"], row["fresh_hi"] = ratio_ci(g["ai"], g["control"], "fresh_own_lost", "fresh_s")
        row["ratio_on_enemy"], row["enemy_lo"], row["enemy_hi"] = ratio_ci(g["ai"], g["control"], "enemy_lost")
        uses = {}
        for r in g["ai"]:
            for k, _, rel in r["enemy_abilities"]:
                uses.setdefault(k, []).append(rel)
        row["ai_abilities"] = {k: {"uses": len(v), "per_battle": round(len(v) / len(g["ai"]), 2),
                                   "after_contact_s": sorted(x for x in v if x is not None)} for k, v in uses.items()}
        on = [r for r in g["ai"] if r["ability_on_s"] > 0]
        row["ai_on_own_ability"] = rate(sum(r["on_own_ability"] * r["ability_on_s"] / 100 for r in on),
                                        sum(r["ability_on_s"] for r in on)) if on else None
        out.append(row)
    return out


def runs(*roots):
    out = []
    for root in roots or (RUNS, lord_duel.RUNS):
        if root.exists():
            out += sorted(d for d in root.iterdir() if (d / "events.jsonl").exists() and (d / "manifest.json").exists())
    return out


def card_line(card):
    if not card:
        return "-"
    st = " ".join(f"{k.replace('stat_', '').replace('scalar_', '')}={v[0]}/{v[1]}" for k, v in card["stats"].items())
    return f"{st} rank={card.get('rank')} xp={card.get('xp')} hpmax={card.get('hpmax')}"


def report(run_dirs, out=OUT):
    rows = []
    for d in run_dirs:
        r = load_run(d)
        if r and mode_of(r["config"]):
            rows.append(measure(r))
    f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
    print("rates: %HP a duel second (both lords in melee, neither routing); on own = the enemy lord's damage on ours")
    print(f"{'run':16} {'duel':4} {'mode':7} {'role':6} {'win':>3} {'contact':>7} {'duel s':>6} {'on own':>6} "
          f"{'on en':>6} {'fr own':>6} {'fr en':>6} {'back':>4} {'rear':>5}  enemy abilities (key, t, after contact)")
    for r in rows:
        print(f"{r['run']:16} {r['duel']:4} {r['mode']:7} {r['role'] or '-':6} {r['winner'] if r['winner'] is not None else '-':>3} "
              f"{f(r['contact_s'], 0):>7} {r['duel_s']:>6.0f} {f(r['on_own']):>6} {f(r['on_enemy']):>6} "
              f"{f(r['fresh_on_own']):>6} {f(r['fresh_on_enemy']):>6} {r['enemy_reengaged']:>4} {f(r['own_rear_share'], 2):>5}  "
              f"{[(k.split('_abilities_')[-1], t, rel) for k, t, rel in r['enemy_abilities']]}")
    print("cards (Value/ValueBase), first reading:")
    for r in rows:
        if r["recorded"]:
            print(f"  {r['run']} {r['duel']} {r['mode']}: own   {card_line(r['own_card'])}")
            print(f"  {' ' * len(r['run'])} {r['duel']} {r['mode']}: enemy {card_line(r['enemy_card'])}")
            if r["enemy_card_changes"]:
                print(f"  {' ' * len(r['run'])} enemy card changes (t, stat, value): {r['enemy_card_changes'][:12]}")
            if r["enemy_effects"]:
                print(f"  {' ' * len(r['run'])} enemy effects seen: {r['enemy_effects']}")
    table = summary(rows)
    print("pooled per lord type: ai / control; ratio = ai / control with a 95 % bootstrap interval over battles")
    for t in table:
        print(f"{t['duel']}: n {t['n_ai']}/{t['n_control']}, duel s {t['ai_duel_s']:.0f}/{t['control_duel_s']:.0f}; "
              f"on own {f(t['ai_on_own'])}/{f(t['control_on_own'])} ratio {f(t['ratio_on_own'], 2)} "
              f"[{f(t['ratio_lo'], 2)}, {f(t['ratio_hi'], 2)}]; fresh {f(t['ai_fresh_on_own'])}/{f(t['control_fresh_on_own'])} "
              f"ratio {f(t['ratio_fresh'], 2)} [{f(t['fresh_lo'], 2)}, {f(t['fresh_hi'], 2)}]; on enemy "
              f"{f(t['ai_on_enemy'])}/{f(t['control_on_enemy'])} ratio {f(t['ratio_on_enemy'], 2)} "
              f"[{f(t['enemy_lo'], 2)}, {f(t['enemy_hi'], 2)}]; enemy wins {t['ai_enemy_wins']}/{t['control_enemy_wins']}; "
              f"re-engaged {t['ai_reengaged']}/{t['control_reengaged']}; our rear threatened {t['ai_own_rear_share']}/"
              f"{t['control_own_rear_share']}")
        print(f"     AI abilities: {t['ai_abilities']}; AI rate on own with an ability active {f(t['ai_on_own_ability'])}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"battles": rows, "summary": table}, indent=1), encoding="utf-8")
    print(out)
    return rows, table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("report", "plan", "run"), default="report")
    parser.add_argument("runs", nargs="*", type=Path,
                        help="report: run folders (default: build/lord-ai/runs/* and build/lord-duel/runs/*)")
    parser.add_argument("--ai", type=int, default=4, help="plan/run: battles against the game's AI a lord type")
    parser.add_argument("--control", type=int, default=2, help="plan/run: scripted-v-scripted battles a lord type")
    parser.add_argument("--kinds", default="emp,skv")
    parser.add_argument("--dry", action="store_true", help="run: print the commands only")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    battles = plan(args.ai, args.control, tuple(args.kinds.split(",")))
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
        return 0 if all(d[-1] == 0 for d in done) else 1
    report(args.runs or runs())
    return 0


if __name__ == "__main__":
    sys.exit(main())
