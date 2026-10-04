"""The standard 5-minute test of a training change (docs/en/training/training.md "Test protocol").

    DOCK_NAME=t2-test5 bash tools/nn/dock.sh tools.nn.train.test5 --label baseline
    bash tools/nn/dock.sh tools.nn.train.test5 --label mychange -- --lord-rout 0.5   # run.py options after --

The GPU lock: the test waits while build/gpu-train.lock exists (another heavy GPU job: training, a
test, a big evaluation; polled every 30 s), then holds it (its label and start time) until it ends,
also on failure. A test started within 60 s of a release (build/gpu-train.released) waits out those
60 s first, so the jobs already waiting go before it. Training is a fixed number of updates (--updates 36, ~5 minutes on a free GPU), so
a test on a busy GPU takes longer but learns as much; --minutes only caps the time.

1. "before": the starting network (--init, default build/nn-train/runs/long_ai/best.pt) is evaluated
   on EVAL_SEEDS: --eval battles of random armies (up to 19 units a side) per opponent, half of them
   in each role, against ai_like, nearest and hold_shoot; the same battles every time: --eval / 2
   seeds in swapped pairs (the network on either side), with the script baselines (cached) and the
   rating (docs/en/training/training.md "Network evaluation: fair metrics"; the report's and
   trend.md's first block, "skill").
2. 36 updates of PPO (--updates; ~5 minutes) from it with the current code and the protocol's settings
   (PROTOCOL below; run.py's defaults otherwise), options after `--` go to run.py and override them
   (a task passes its own new settings there, e.g. `-- --lord-rout 0.5`). report.json keeps all of them (train_args). build/nn-train/latest.pt is not touched.
3. "after": the trained network, the same evaluation.
4. The behaviour report, before -> after, printed and written to build/nn-train/test5/<label>/
   (report.json; before.json, after.json: the whole evaluations): win rates by opponent and role;
   order kinds; own lord deaths; seconds missile units spend in melee; the share of own melee
   seconds struck in flank or rear; the share of decisions with a pile (more than 2 own units on
   one enemy while another enemy strikes an own unit in flank or rear); the share of own melee
   seconds striking an enemy's flank or rear; attack target switches a minute; ability uses
   (tools/nn/train/behaviour.py).
--before PATH reuses a "before" evaluation (a before.json of the same --init and code).

With the drills' teacher (`-- --drill-teach '{"kiting": 0.5}'`, run.py) a "teacher" block per taught
drill from the training log: the imitation weight, its cross-entropy and the agreement (the policy's
most likely action = the skilled script's label) over the updates since the last point. With the
adaptive teacher (`-- --drill-teach auto`, tools/nn/train/teach_auto.py) the test makes it: its first
shares from the "before" evaluation's drills, new ones after every evaluation (the hook), and per point a
table drill | net win | skilled win | deficit | teacher share (was -> now) | agreement.

The drills block also carries, per drill, the win rate on the clean / broad frame (run.py --drill-broad,
drills.BROAD) and the TRANSFER of its skill to the normal evaluation battles, network / ai_like
(tools/nn/train/drills/transfer.py; report.json before / after / trend -> "transfer").

Two more blocks (measured only): "liveliness" per opponent and role (order changes, attack-target
switches, A->B->A flips, move-point jitter, twitching units out of melee, the units' own target
switches; tools/nn/train/behaviour.py), and "capacity and forgetting" (tools/nn/train/capacity.py:
the final evaluation against the previous iteration's (--prev-report, else found) and against minute
0, the training log's signals, a verdict ok / watch / widen-candidate); report.json "capacity".
--report-only rebuilds trend.md and report.json's blocks of a finished --label from its evaluations
(no training, no GPU; the liveliness rows of an evaluation older than them stay "-").

A trend run: --updates 0 --minutes M --every K trains M minutes and runs the same evaluation every K
minutes of training too (its time not counted), keeping each network (m<minute>.pt) and evaluation
(eval_m<minute>.json); report.json and trend.md then hold the table minute 0 / K / ... / M:

    DOCK_NAME=t0-trend bash tools/nn/dock.sh tools.nn.train.test5 --label gold30 --updates 0 --minutes 30 --every 10 --eval 256
"""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch

from tools.nn.train import cadence as cad
from tools.nn.train import capacity, checkpoint, drills, evaluate, matchups, run, skill, teach_auto

OUT = checkpoint.DIR / "test5"
LOCK = checkpoint.DIR.parent / "gpu-train.lock"
INIT = "build/nn-train/runs/long_ai/best.pt"
OPPONENTS = ("ai_like", "nearest", "hold_shoot")
PROTOCOL = ["--small", "0.35:6", "--critic-warmup", "3",
            "--lr", "1.5e-4", "--entropy", "0.003", "--entropy-end", "0.001", "--anchor", "0.06", "--anchor-end", "0.03",
            "--pool-extra", "build/nn-train/runs/long19/latest.pt", "--snapshot-every", "10", "--no-eval"]


@contextlib.contextmanager
def gpu_lock(label, path=LOCK, poll_s=30.0, gap_s=60.0):
    """Wait until no other heavy GPU job holds `path`, hold it while the block runs, then free it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    released = path.with_suffix(".released")
    # Fair turns: a job that starts within `gap_s` of a release waits that long first, so the jobs
    # already waiting (polling every poll_s) take the lock before it.
    with contextlib.suppress(FileNotFoundError, ValueError):
        since = time.time() - float(released.read_text(encoding="utf-8").split()[0])
        if -gap_s < since < gap_s:              # (a stamp a little ahead: rounding, another clock)
            since = max(since, 0.0)
            print(f"the GPU lock was freed {since:.0f} s ago: waiting {gap_s - since:.0f} s for the queue", flush=True)
            time.sleep(gap_s - since)
    said = False
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if not said:
                held = path.read_text(encoding="utf-8", errors="replace").strip() if path.exists() else ""
                print(f"waiting for the GPU lock {path} ({held})", flush=True)
                said = True
            time.sleep(poll_s)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{label} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            released.write_text(f"{time.time():.3f} {label}\n", encoding="utf-8", newline="\n")
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def evaluation(actor, args, device, cadence):
    """The evaluation of every point; cadence: the training's (run.py --decide-s, --order-latency)."""
    t = time.time()
    res = evaluate.play(actor, opponents=OPPONENTS, device=device, generated=args.eval, max_units=19, seed=1,
                        together=True, paired=True, baseline=True, cadence=cadence)
    if args.drill_eval:
        # the drills (tools/nn/train/drills, the verified ones): win rate and gold trade per drill
        res["drills"] = evaluate.play_drills(actor, args.drill_eval, device, cadence=cadence)
    res["seconds"] = round(time.time() - t)
    res["cadence"] = cadence.meta()
    return res


def distance(run_dir, since, until):
    """The distance from the start over the training updates (since, until] of a run (run.py's
    log.jsonl): start_kl, the mean per-unit KL of the actor to the network the run started from
    (ppo.distance on the training batch), and its last value; anchor_kl, the mean KL to the leash's
    reference (moved by --anchor-roll); rolls, the reference's renewals so far. None without a log."""
    path = Path(run_dir) / "log.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if since < r.get("update", 0) <= until]
    if not rows:
        return None
    mean = (lambda k: sum(r.get(k, 0.0) for r in rows) / len(rows))
    return {"start_kl": round(mean("start_kl"), 4), "start_kl_last": round(rows[-1].get("start_kl", 0.0), 4),
            "anchor_kl": round(mean("anchor_kl"), 4), "rolls": rows[-1].get("anchor_rolls", 0),
            "updates": [rows[0]["update"], rows[-1]["update"]]}


def teacher(run_dir, since, until):
    """The drills' teacher over the training updates (since, until] of a run (run.py's log.jsonl "teach",
    drills/teach.py): {drill: {weight (mean), ce (mean), agree (mean, first, last), agree_kind (mean),
    agree_active (over the active labels of all the updates: any kind but hold; None without), updates}};
    None without a log or a teacher."""
    path = Path(run_dir) / "log.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if since < r.get("update", 0) <= until and r.get("teach")]
    if not rows:
        return None
    out = {}
    for name in dict.fromkeys(n for r in rows for n in r["teach"]):
        xs = [r["teach"][name] for r in rows if name in r["teach"] and r["teach"][name].get("units")]
        if not xs:
            continue
        mean = (lambda k: round(sum(x[k] for x in xs) / len(xs), 4))
        act = sum(x.get("active", 0) for x in xs)
        hit = sum((x.get("agree_active") or 0.0) * x.get("active", 0) for x in xs)
        out[name] = {"weight": mean("weight"), "ce": mean("ce"), "agree": mean("agree"),
                     "agree_first": xs[0]["agree"], "agree_last": xs[-1]["agree"], "agree_kind": mean("agree_kind"),
                     "agree_active": round(hit / act, 4) if act else None, "updates": len(xs)}
        if all("labelled" in x for x in xs):
            out[name]["labelled"] = mean("labelled")
    return out or None


def teach_block(points, heads, join=None):
    """The teacher block (teacher() per drill) over metrics() points: a column per point, or one column
    joined by `join`; [] when no point has it."""
    names = list(dict.fromkeys(n for p in points for n in (p.get("teach") or {})))
    if not names:
        return []
    f = (lambda fmt, v: "-" if v is None else fmt.format(v))
    rows = []
    for n in names:
        get = (lambda p, k: ((p.get("teach") or {}).get(n) or {}).get(k))
        for title, cell in (("weight (mean)", lambda p: f("{:.3f}", get(p, "weight"))),
                            ("share of the units labelled (mean)", lambda p: f("{:.3f}", get(p, "labelled"))),
                            ("imitation cross-entropy (mean)", lambda p: f("{:.3f}", get(p, "ce"))),
                            ("agreement, policy argmax = script (first / last update)",
                             lambda p: f"{f('{:.3f}', get(p, 'agree_first'))} / {f('{:.3f}', get(p, 'agree_last'))}"),
                            ("agreement on the kind (mean)", lambda p: f("{:.3f}", get(p, "agree_kind"))),
                            ("agreement on the active labels (not hold)", lambda p: f("{:.3f}", get(p, "agree_active")))):
            cells = [cell(p) for p in points]
            rows.append((f"teacher {n}: {title}", [join.join(cells)] if join is not None else cells))
    return (["", "the drills' teacher (run.py --drill-teach; training batch, run.py log):", "",
             "| teacher | " + " | ".join(heads) + " |", "|---" * (len(heads) + 1) + "|"]
            + [f"| {t} | " + " | ".join(c) + " |" for t, c in rows])


def auto_block(points, heads):
    """The adaptive teacher's tables (teach_auto.table), one per point that has one: the drill numbers of
    that evaluation and the shares chosen from them; [] when no point has one."""
    lines = []
    if any(p.get("teach_auto") for p in points):
        lines += ["", "the drills' adaptive teacher (run.py --drill-teach auto; tools/nn/train/teach_auto.py): per "
                  "evaluation the network's and the skilled script's drill win rates (an embedded frame: gold-trade "
                  "scores, naive 0, skilled 1), deficit = max(0, skilled - net) / skilled, the share of the drill's "
                  "battles labelled until the next point, the agreement since the last point:"]
        for p, h in zip(points, heads):
            if p.get("teach_auto"):
                lines += [""] + teach_auto.table(p["teach_auto"], f"{h}:")
    if any(p.get("teach_normal") for p in points):
        lines += ["", "the teacher in normal battles (run.py --teach-normal; teach_auto.Transfer): per evaluation the "
                  "APPLIED share (higher is better) of the network and of ai_like in the normal battles, gap = "
                  "max(0, ai_like - net) / ai_like, the share of the normal battles labelled at the drill's moments "
                  "until the next point, the agreement since the last point:"]
        for p, h in zip(points, heads):
            if p.get("teach_normal"):
                lines += [""] + teach_auto.table(p["teach_normal"], f"{h}:", ref="ai_like applied", net="net applied")
    return lines


def metrics(res):
    """{"opponent/role": {name: value}} of one evaluation; "opponent/all": per opponent; "opponent/factions":
    win rates by our faction and role and by matchup, with gold (tools/nn/train/matchups.py); "skill": the
    fair metrics (tools/nn/train/skill.py summary: rating, pairs, pair gold, advantage over the script,
    margin)."""
    out = {"skill": skill.summary(res)}
    if res.get("drills"):
        out["drills"] = res["drills"]
    if res.get("transfer"):
        out["transfer"] = res["transfer"]
    if res.get("distance") is not None:
        out["distance"] = res["distance"]
    if res.get("teach"):
        out["teach"] = res["teach"]
    if res.get("teach_auto"):
        out["teach_auto"] = res["teach_auto"]
    if res.get("teach_normal"):
        out["teach_normal"] = res["teach_normal"]
    for opp, o in res["by_opponent"].items():
        for role, x in o["roles"].items():
            if not x.get("games"):
                continue
            b = x.get("behaviour", {})
            out[f"{opp}/{role}"] = {"win": x["win_rate"], "lord_dead_own": x.get("lord_dead_own"),
                                    "timeouts": x["timeouts"], "gold_destroyed": x.get("gold_destroyed"),
                                    "gold_lost": x.get("gold_lost"), "gold_ratio": x.get("gold_ratio"),
                                    "gold_trade": x.get("gold_trade"), **b}
        out[f"{opp}/all"] = {"switches_per_min": o.get("switches_per_minute"), "orders_per_min": o["orders_per_minute"],
                             **{f"kind_{k}": v for k, v in o["kinds"].items()}}
        if "matchups" in o:
            out[f"{opp}/factions"] = {"factions": o["factions"], "matchups": o["matchups"]}
    return out


ROWS = (("win", "win rate", "{:.3f}"), ("gold_destroyed", "enemy gold destroyed / battle", "{:.0f}"),
        ("gold_lost", "own gold lost / battle", "{:.0f}"), ("gold_ratio", "gold exchange ratio", "{:.2f}"),
        ("lord_dead_own", "own lord dead", "{:.3f}"),
        ("missile_melee_s", "missile s in melee / battle", "{:.1f}"),
        ("missile_melee_share", "missile time in melee", "{:.3f}"),
        ("flanked_share", "own melee hit flank/rear", "{:.3f}"),
        ("crowding_share", "decisions with a pile", "{:.3f}"),
        ("flank_attack_share", "own melee into flank/rear", "{:.3f}"),
        ("abilities_per_battle", "ability uses / battle", "{:.2f}"), ("timeouts", "timeouts", "{:.3f}"))
LIVELY = (("order_changes_per_min", "order changes / unit-min", "{:.2f}"),
          ("target_switches_per_min", "attack-target switches / unit-min", "{:.2f}"),
          ("flips_per_min", "flips A→B→A ≤10 s / unit-min", "{:.3f}"),
          ("move_jitter_m", "move-point jitter (keeps moving), m", "{:.1f}"),
          ("move_repoints_per_min", "move re-points / moving min", "{:.2f}"),
          ("free_changes_per_min", "changes out of melee / unit-min", "{:.2f}"),
          ("twitch_share", "units twitching out of melee", "{:.3f}"),
          ("engine_switches_per_min", "own target switches / unit-min (1 s)", "{:.2f}"),
          ("opp_engine_switches_per_min", "opponent's target switches / unit-min (1 s)", "{:.2f}"))
ALL = (("switches_per_min", "target switches / min", "{:.2f}"), ("orders_per_min", "order changes / min", "{:.2f}"),
       ("kind_hold", "kind hold", "{:.2f}"), ("kind_move", "kind move", "{:.2f}"), ("kind_attack", "kind attack", "{:.2f}"),
       ("kind_withdraw", "kind withdraw", "{:.2f}"), ("kind_keep", "kind keep", "{:.2f}"))


def table(before, after):
    """Lines: metric, then before -> after per opponent and role."""
    keys = [k for k in before if k.endswith(("/attack", "/defend"))]
    lines = ["| metric | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, title, fmt in ROWS:
        cells = []
        for k in keys:
            a, b = before[k].get(name), after.get(k, {}).get(name)
            f = (lambda v: "-" if v is None else fmt.format(v))
            cells.append(f"{f(a)} → {f(b)}")
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    opps = [k for k in before if k.endswith("/all")]
    lines += ["", "| metric | " + " | ".join(opps) + " |", "|---" * (len(opps) + 1) + "|"]
    for name, title, fmt in ALL:
        f = (lambda v: "-" if v is None else fmt.format(v))
        lines.append(f"| {title} | " + " | ".join(f"{f(before[k].get(name))} → {f(after.get(k, {}).get(name))}"
                                                  for k in opps) + " |")
    by = [(f"{title}, {k.split('/')[0]}", cells) for k in before if k.endswith("/factions")
          for title, cells in matchups.rows([before[k], after.get(k)])]
    if by:
        lines += ["", "| by faction | before → after |", "|---|---|"]
        lines += [f"| {title} | {' → '.join(cells)} |" for title, cells in by]
    return (skill_block([before.get("skill"), after.get("skill")], ["before → after"], " → ") + lines
            + lively_block([before, after], ["before → after"], " → ")
            + drill_block([before, after], ["before → after"], " → ")
            + teach_block([before, after], ["before → after"], " → ")
            + auto_block([before, after], ["before", "after"]))


def transfer_cell(p, n, key="share"):
    """'network / ai_like' transfer numbers of drill n at a metrics() point (drills/transfer.py): key "share" the
    APPLIED share, "mistake" the MISTAKE share; '-' without."""
    x = (p.get("transfer") or {}).get(n) or {}
    g = (lambda d: "-" if not d or d.get(key) is None else f"{d[key]:.2f}")
    return "-" if not x else f"{g(x.get('network'))} / {g(x.get('ai_like'))}"


def drill_block(points, heads, join=None):
    """The drills block (win rate and gold trade per drill, the drill's naive / skilled scripts beside; clean /
    broad frames; the transfer to the normal battles, network / ai_like) over metrics() points: a column per
    point, or one column joined by `join`; [] when no point has drills."""
    names = list(dict.fromkeys(n for p in points for n in list(p.get("drills") or {}) + list(p.get("transfer") or {})))
    if not names:
        return []
    f = (lambda fmt, v: "-" if v is None else fmt.format(v))
    rows = []
    for n in names:
        for key, title, fmt in (("win_rate", "win rate", "{:.3f}"), ("gold_trade", "gold trade", "{:+.3f}")):
            cells = [f(fmt, ((p.get("drills") or {}).get(n) or {}).get(key)) for p in points]
            ref = next(((p.get("drills") or {}).get(n, {}).get("scripts") for p in points
                        if (p.get("drills") or {}).get(n, {}).get("scripts")), None)
            ref_s = (f" (naive {fmt.format(ref['naive'][key])}, skilled {fmt.format(ref['skilled'][key])})"
                     if ref else "")
            rows.append((f"drill {n}: {title}{ref_s}", [join.join(cells)] if join is not None else cells))
        # the frames apart (drills.FRAMES: clean, broad, embedded; drills.BROAD, drills.EMBED), the scripts' beside
        d = (lambda p: (p.get("drills") or {}).get(n) or {})
        kinds = [k for k in drills.FRAMES if any(d(p).get(k) for p in points)]
        if len(kinds) > 1:
            cells = [" / ".join(f"{f('{:.3f}', (d(p).get(k) or {}).get('win_rate'))} {f('{:+.3f}', (d(p).get(k) or {}).get('gold_trade'))}"
                                for k in kinds) for p in points]
            ref = next((d(p)["scripts"] for p in points
                        if all((d(p).get("scripts") or {}).get("skilled", {}).get(k) for k in kinds)), None)
            ref_s = ("" if not ref else " (" + ", ".join(
                f"{w} " + " / ".join(f"{ref[w][k]['win_rate']:.3f} {ref[w][k]['gold_trade']:+.3f}" for k in kinds)
                for w in ("naive", "skilled")) + ")")
            rows.append((f"drill {n}: win rate and gold trade by frame, {' / '.join(kinds)}{ref_s}",
                         [join.join(cells)] if join is not None else cells))
        # the transfer: the skill in the normal evaluation battles (drills/transfer.py)
        if any((p.get("transfer") or {}).get(n) for p in points):
            cells = [transfer_cell(p, n) for p in points]
            rows.append((f"drill {n}: TRANSFER to normal battles, APPLIED share (higher is better), network / ai_like "
                         f"(share of the situation's unit-s where the unit uses the skill)",
                         [join.join(cells)] if join is not None else cells))
            cells = [transfer_cell(p, n, "mistake") for p in points]
            rows.append((f"drill {n}: TRANSFER to normal battles, MISTAKE share (lower is better), network / ai_like "
                         f"(share of the situation's unit-s where the unit makes the drill's mistake)",
                         [join.join(cells)] if join is not None else cells))
        # what our units do (drills/metrics.py): unit-seconds by order, all / the last 100 s before the limit
        def play(p):
            return ((p.get("drills") or {}).get(n) or {}).get("play")
        if not any(play(p) for p in points):
            continue
        sk = next((((p.get("drills") or {}).get(n, {}).get("scripts") or {}).get("skilled", {}).get("play")
                   for p in points if (((p.get("drills") or {}).get(n, {}).get("scripts") or {}).get("skilled") or {}).get("play")), None)
        for key, title in (("correct", "on the correct target"), ("bad", "on its counter"), ("other", "on another enemy"),
                           ("hold", "holding"), ("move", "moving"), ("melee", "in melee")):
            g = (lambda x, part: "-" if not x or x.get(part, {}).get(key) is None else f"{x[part][key]:.2f}")
            cells = [f"{g(play(p), 'all')} / {g(play(p), 'tail')}" for p in points]
            ref = f" (skilled {g(sk, 'all')} / {g(sk, 'tail')})" if sk else ""
            rows.append((f"drill {n}: unit-s {title}, all / last 100 s{ref}", [join.join(cells)] if join is not None else cells))
        fc = (lambda x: "-" if not x or x.get("first_correct_s") is None else f"{x['first_correct_s']:.0f} s ({x.get('ever_correct', 0):.2f})")
        cells = [fc(play(p)) for p in points]
        rows.append((f"drill {n}: first order on the correct target, median (share of units ever){' (skilled ' + fc(sk) + ')' if sk else ''}",
                     [join.join(cells)] if join is not None else cells))
    return (["", "drills (tools/nn/train/drills; the network against each drill's enemy, DRILL_EVAL_SEEDS; "
             "in brackets the drill's check scripts on the same battles):", "",
             "| drill | " + " | ".join(heads) + " |", "|---" * (len(heads) + 1) + "|"]
            + [f"| {t} | " + " | ".join(c) + " |" for t, c in rows])


def lively_block(points, heads, join=None):
    """The liveliness block (LIVELY, attack / defend per opponent) over metrics() points: a column per
    point, or one column of the cells joined by `join`; [] when no point has it."""
    if not any("order_changes_per_min" in v for p in points for k, v in p.items() if k.endswith(("/attack", "/defend"))):
        return []
    opps = list(dict.fromkeys(k.split("/")[0] for p in points for k in p if k.endswith(("/attack", "/defend"))))
    f = (lambda fmt, v: "-" if v is None else fmt.format(v))
    rows = []
    for name, title, fmt in LIVELY:
        for o in opps:
            cells = [f"{f(fmt, p.get(f'{o}/attack', {}).get(name))} / {f(fmt, p.get(f'{o}/defend', {}).get(name))}"
                     for p in points]
            rows.append((f"{title}, {o}", [join.join(cells)] if join is not None else cells))
    return (["", "liveliness (attack / defend; measured only: tools/nn/train/behaviour.py):", "",
             "| liveliness | " + " | ".join(heads) + " |", "|---" * (len(heads) + 1) + "|"]
            + [f"| {t} | " + " | ".join(c) + " |" for t, c in rows])


def skill_block(points, heads, join=None):
    """The compact skill block (tools/nn/train/skill.py rows): a column per point, or one column of the
    cells joined by `join`; [] when no point has it."""
    if not any(points):
        return []
    rows = skill.rows(points)
    if join is not None:
        rows = [(t, [join.join(c)]) for t, c in rows]
    return (["skill (fair metrics: docs/en/training/training.md):", "", "| skill | " + " | ".join(heads) + " |",
             "|---" * (len(heads) + 1) + "|"] + [f"| {t} | " + " | ".join(c) + " |" for t, c in rows] + [""])


def distance_block(points):
    """Lines: the distance from the start next to the rating, a column per minute ([] when no point has it)."""
    if not any(m.get("distance") for _, m in points[1:]):
        return []
    d = [m.get("distance") or {} for _, m in points]
    f = (lambda v, fmt="{:.3f}": "-" if v is None else fmt.format(v))
    rows = [("rating, overall (logit)", [f(((((m.get("skill") or {}).get("rating") or {}).get("overall")) or {}).get("value"), "{:+.2f}")
                                         for _, m in points]),
            ("distance from start: KL to the init network (mean / last of the updates since the last point)",
             ["0" if i == 0 else f"{f(x.get('start_kl'))} / {f(x.get('start_kl_last'))}" for i, x in enumerate(d)]),
            ("anchor KL to the leash's reference (mean)", ["-" if i == 0 else f(x.get("anchor_kl")) for i, x in enumerate(d)]),
            ("reference renewals (--anchor-roll), so far", ["-" if i == 0 else f(x.get("rolls"), "{:d}")
                                                           for i, x in enumerate(d)])]
    mins = [m for m, _ in points]
    return (["distance from start (training batch, run.py log; growing with the rating = the search works):", "",
             "| distance | " + " | ".join(f"min {m}" for m in mins) + " |", "|---" * (len(mins) + 1) + "|"]
            + [f"| {t} | " + " | ".join(c) + " |" for t, c in rows] + [""])


def trend(points):
    """Lines: a metric per opponent (attack / defend), a column per minute. points: [(minute, metrics())]."""
    mins = [m for m, _ in points]
    first = points[0][1]
    opps = sorted({k.split("/")[0] for k in first if "/" in k},
                  key=lambda o: OPPONENTS.index(o) if o in OPPONENTS else 99)
    lines = skill_block([m.get("skill") for _, m in points], [f"min {m}" for m in mins])
    lines += distance_block(points)
    lines += ["| metric (attack / defend) | " + " | ".join(f"min {m}" for m in mins) + " |", "|---" * (len(mins) + 1) + "|"]
    f = (lambda fmt, v: "-" if v is None else fmt.format(v))
    for name, title, fmt in ROWS:
        for o in opps:
            cells = [f"{f(fmt, m.get(f'{o}/attack', {}).get(name))} / {f(fmt, m.get(f'{o}/defend', {}).get(name))}"
                     for _, m in points]
            lines.append(f"| {title}, {o} | " + " | ".join(cells) + " |")
    for name, title, fmt in ALL:
        for o in opps:
            lines.append(f"| {title}, {o} | " + " | ".join(f(fmt, m.get(f"{o}/all", {}).get(name)) for _, m in points) + " |")
    by = [(f"{title}, {o}", cells) for o in opps
          for title, cells in matchups.rows([m.get(f"{o}/factions") for _, m in points])]
    if by:
        lines += ["", "by faction (our faction first):", "",
                  "| by faction | " + " | ".join(f"min {m}" for m in mins) + " |", "|---" * (len(mins) + 1) + "|"]
        lines += [f"| {title} | " + " | ".join(cells) + " |" for title, cells in by]
    return (lines + lively_block([m for _, m in points], [f"min {m}" for m in mins])
            + drill_block([m for _, m in points], [f"min {m}" for m in mins])
            + teach_block([m for _, m in points], [f"min {m}" for m in mins])
            + auto_block([m for _, m in points], [f"min {m}" for m in mins]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--init", default=INIT)
    ap.add_argument("--updates", type=int, default=36, help="PPO updates (~5 minutes on a free GPU)")
    ap.add_argument("--minutes", type=float, default=30.0, help="the training's time cap")
    ap.add_argument("--eval", type=int, default=512, help="EVAL_SEEDS battles per opponent (half in each role)")
    ap.add_argument("--before", help="reuse this before.json")
    ap.add_argument("--every", type=float, default=0,
                    help="minutes of training between full evaluations (a trend run; 0: before and after only)")
    ap.add_argument("--no-lock", action="store_true", help="do not take the GPU lock (small smoke runs only)")
    ap.add_argument("--drill-eval", type=int, default=128,
                    help="battles of each verified drill per evaluation (tools/nn/train/drills; 0: none)")
    ap.add_argument("--prev-report", help="the previous iteration's report.json or folder (default: found, "
                                          "tools/nn/train/capacity.py)")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild trend.md and report.json's blocks of a finished --label from its files")
    ap.add_argument("--baseline-canary", type=int, default=0,
                    help="on a baseline cache miss play this many pairs first and adopt an older version's file "
                         "when they come out identical (evaluate.CANARY; 0: play the whole baseline)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, rest = ap.parse_known_args()
    rest = [a for a in rest if a != "--"]
    if args.report_only:
        return report_only(args)
    evaluate.CANARY = max(0, args.baseline_canary)
    with (contextlib.nullcontext() if args.no_lock else gpu_lock(f"test5 {args.label}")):
        test(args, rest)


def test(args, rest):
    out = OUT / args.label
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    targs = run.parser().parse_args(["--name", f"test5_{args.label}", "--init", args.init, "--minutes",
                                     str(args.minutes), "--updates", str(args.updates), "--device", args.device]
                                    + PROTOCOL + rest)
    cadence = cad.of_args(targs)                 # the evaluations decide as the training does
    drills.BROAD = targs.drill_broad              # the drill evaluations play the training's mix of clean / broad frames
    drills.EMBED = targs.drill_embed              # ... and of embedded ones
    if args.before:
        before = json.loads(Path(args.before).read_text(encoding="utf-8"))
        if before.get("cadence", cad.STEP.meta()) != cadence.meta():
            print(f"WARNING: --before was evaluated at cadence {before.get('cadence', cad.STEP.meta())}, "
                  f"this run's is {cadence.meta()}", flush=True)
    else:
        before = evaluation(checkpoint.load_policy(args.init, device), args, device, cadence)
    (out / "before.json").write_text(json.dumps(before, indent=1), encoding="utf-8", newline="\n")
    print(f"before: {before.get('seconds')} s", flush=True)

    points = [(0, metrics(before))]
    run_log = checkpoint.DIR / "runs" / targs.name
    last = [0]                                   # the update of the last evaluation point
    # the adaptive teacher: first shares from the "before" drills (else the init's files), new after every point
    auto = None
    if targs.drill_teach == "auto":
        prior = ((before["drills"], "test5 before") if before.get("drills") else teach_auto.prior(args.init))
        auto = teach_auto.Auto(targs.drill_teach_k, targs.drill_teach_cap, targs.drill_teach_weight, *prior)

    # the teacher in normal battles, adaptive: first shares from the "before" transfer block, new after every point
    nauto = None
    fixed_normal, normal_names = run.normal_drills(targs.teach_normal)
    if normal_names and fixed_normal is None:
        prior_t = ((before["transfer"], "test5 before") if before.get("transfer") else teach_auto.prior(args.init, "transfer"))
        nauto = teach_auto.Transfer(targs.teach_normal_k, targs.teach_normal_cap, targs.teach_normal_weight, *prior_t,
                                    match=targs.teach_normal_match)

    def observe(res):
        """The adaptive teachers see an evaluation: new shares; their tables go into the evaluation."""
        agree = {n: v.get("agree") for n, v in (res.get("teach") or {}).items()}
        if auto is not None:
            if auto.history and not points[0][1].get("teach_auto"):
                before["teach_auto"] = points[0][1]["teach_auto"] = auto.history[0]
            res["teach_auto"] = auto.observe(res.get("drills"), agree)
        if nauto is not None:
            if nauto.history and not points[0][1].get("teach_normal"):
                before["teach_normal"] = points[0][1]["teach_normal"] = nauto.history[0]
            res["teach_normal"] = nauto.observe(res.get("transfer"), agree)

    def hook(actor, critic, minute, update):
        actor.eval()
        res = evaluation(actor, args, device, cadence)
        res["update"] = update
        res["distance"] = distance(run_log, last[0], update)
        res["teach"] = teacher(run_log, last[0], update)
        observe(res)
        last[0] = update
        m = f"{minute:g}"
        checkpoint.save(out / f"m{m}.pt", actor, critic, targs.preset, {"minute": minute, "update": update,
                                                                      "run": targs.name, "cadence": cadence.meta()})
        (out / f"eval_m{m}.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
        points.append((m, metrics(res)))
        print(f"evaluation at minute {m} (update {update}): {res['seconds']} s", flush=True)
        print("\n".join(trend(points)), flush=True)

    actor, summary, run_dir = run.train(targs, (args.every, hook) if args.every else None, teacher=auto, normal=nauto)
    actor.eval()
    after = evaluation(actor, args, device, cadence)
    after["distance"] = distance(run_dir, last[0], summary["updates"])
    after["teach"] = teacher(run_dir, last[0], summary["updates"])
    observe(after)
    if auto is not None or nauto is not None:
        (out / "before.json").write_text(json.dumps(before, indent=1), encoding="utf-8", newline="\n")
    (out / "after.json").write_text(json.dumps(after, indent=1), encoding="utf-8", newline="\n")
    if args.every:
        m = f"{args.minutes:g}"
        # with the critic (the run's latest.pt has it): the next run of the chain starts from this file
        critic = checkpoint.load_critic(run_dir / "latest.pt", device)
        checkpoint.save(out / f"m{m}.pt", actor, critic, targs.preset, {"minute": args.minutes, "update": summary["updates"],
                                                                      "run": targs.name, "cadence": cadence.meta()})
        points.append((m, metrics(after)))

    mb, ma = metrics(before), metrics(after)
    lines = table(mb, ma)
    report = {"label": args.label, "init": args.init, "updates": summary["updates"], "train_s": summary["seconds"],
              "eval_battles": args.eval, "protocol": PROTOCOL, "options": rest, "train_args": vars(targs), "training": summary, "run": str(run_dir),
              "before": mb, "after": ma, "table": lines}
    cap = capacity.report(out, args.prev_report, run_dir, start=before, end=after, rep=report)
    report["capacity"] = cap
    lines += [""] + capacity.lines(cap)
    if args.every:
        report["trend"] = {str(m): p for m, p in points}
        report["trend_table"] = trend(points) + [""] + capacity.lines(cap)
        (out / "trend.md").write_text("\n".join(report["trend_table"]) + "\n", encoding="utf-8", newline="\n")
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    print(f"\ntest5 {args.label}: {args.init}, {summary['updates']} updates in {summary['seconds']} s "
          f"({summary['battles']} battles); before -> after, {args.eval} EVAL_SEEDS battles per opponent", flush=True)
    print("\n".join(lines), flush=True)
    if args.every:
        print("\ntrend:\n" + "\n".join(report["trend_table"]), flush=True)
    print(f"written: {out / 'report.json'}", flush=True)


def report_only(args):
    """--report-only: the blocks of a finished folder from its evaluations (before.json, eval_m<minute>.json,
    after.json) and report.json; rewrites trend.md and report.json's before, after, table, trend,
    trend_table and capacity."""
    out = OUT / args.label
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    load = (lambda p: json.loads(p.read_text(encoding="utf-8")))
    before, after = load(out / "before.json"), load(out / "after.json")
    evs = sorted(out.glob("eval_m*.json"), key=lambda p: float(p.stem[6:]))
    mb, ma = metrics(before), metrics(after)
    cap = capacity.report(out, args.prev_report, None, start=before, end=after, rep=rep)
    rep.update(before=mb, after=ma, table=table(mb, ma) + [""] + capacity.lines(cap), capacity=cap)
    if evs:
        last = (rep.get("train_args") or {}).get("minutes")
        points = [(0, mb)] + [(p.stem[6:], metrics(load(p))) for p in evs]
        points.append((f"{last:g}" if last is not None else "end", ma))
        rep["trend"] = {str(m): p for m, p in points}
        rep["trend_table"] = trend(points) + [""] + capacity.lines(cap)
        (out / "trend.md").write_text("\n".join(rep["trend_table"]) + "\n", encoding="utf-8", newline="\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    print("\n".join(rep.get("trend_table") or rep["table"]), flush=True)
    print(f"written: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
