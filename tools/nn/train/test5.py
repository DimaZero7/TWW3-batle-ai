"""The standard 5-minute test of a training change (docs/en/training/training.md "Test protocol").

    DOCK_NAME=t2-test5 bash tools/nn/dock.sh tools.nn.train.test5 --label baseline
    bash tools/nn/dock.sh tools.nn.train.test5 --label mychange -- --unit-credit 0.5   # run.py options after --

The GPU lock: the test waits while build/gpu-train.lock exists (another heavy GPU job: training, a
test, a big evaluation; polled every 30 s), then holds it (its label and start time) until it ends,
also on failure. A test started within 60 s of a release (build/gpu-train.released) waits out those
60 s first, so the jobs already waiting go before it. Training is a fixed number of updates (--updates 36, ~5 minutes on a free GPU), so
a test on a busy GPU takes longer but learns as much; --minutes only caps the time.

1. "before": the starting network (--init, default build/nn-train/runs/long_ai/best.pt) is evaluated
   on EVAL_SEEDS: --eval battles of random armies (up to 19 units a side) per opponent, half of them
   in each role, against ai_like, nearest and hold_shoot; the same battles every time.
2. 36 updates of PPO (--updates; ~5 minutes) from it with the current code and the protocol's settings
   (PROTOCOL below: the long_ai2 continuation with the baseline's --unit-credit 0 pinned; run.py's
   defaults otherwise), options after `--` go to run.py and override them (a task passes its own
   new settings there, e.g. `-- --unit-credit 0.3`). report.json keeps all of them (train_args). build/nn-train/latest.pt is not touched.
3. "after": the trained network, the same evaluation.
4. The behaviour report, before -> after, printed and written to build/nn-train/test5/<label>/
   (report.json; before.json, after.json: the whole evaluations): win rates by opponent and role;
   order kinds; own lord deaths; seconds missile units spend in melee; the share of own melee
   seconds struck in flank or rear; the share of decisions with a pile (more than 2 own units on
   one enemy while another enemy strikes an own unit in flank or rear); the share of own melee
   seconds striking an enemy's flank or rear; attack target switches a minute; ability uses
   (tools/nn/train/behaviour.py).
--before PATH reuses a "before" evaluation (a before.json of the same --init and code).

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

from tools.nn.train import checkpoint, evaluate, run

OUT = checkpoint.DIR / "test5"
LOCK = checkpoint.DIR.parent / "gpu-train.lock"
INIT = "build/nn-train/runs/long_ai/best.pt"
OPPONENTS = ("ai_like", "nearest", "hold_shoot")
PROTOCOL = ["--armies", "generated", "--curriculum", "19:1", "--small", "0.35:6", "--critic-warmup", "3",
            "--lr", "1.5e-4", "--entropy", "0.003", "--entropy-end", "0.001", "--anchor", "0.06", "--anchor-end", "0.03",
            "--reference", "build/nn-train/runs/bcmix/bc.pt",
            "--pool-extra", "build/nn-train/runs/long19/latest.pt", "--snapshot-every", "10", "--no-eval",
            # pinned, so tests stay comparable when run.py's defaults change: the baseline's training
            # (no per-unit credit); a task passes its own settings after `--`
            "--unit-credit", "0"]


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


def evaluation(actor, args, device):
    t = time.time()
    res = evaluate.play(actor, opponents=OPPONENTS, device=device, generated=args.eval, max_units=19, seed=1,
                        together=True)
    res["seconds"] = round(time.time() - t)
    return res


def metrics(res):
    """{(opponent, role): {name: value}} of one evaluation."""
    out = {}
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
ALL = (("switches_per_min", "target switches / min", "{:.2f}"), ("orders_per_min", "order changes / min", "{:.2f}"),
       ("kind_hold", "kind hold", "{:.2f}"), ("kind_move", "kind move", "{:.2f}"), ("kind_attack", "kind attack", "{:.2f}"),
       ("kind_withdraw", "kind withdraw", "{:.2f}"), ("kind_keep", "kind keep", "{:.2f}"))


def table(before, after):
    """Lines: metric, then before -> after per opponent and role."""
    keys = [k for k in before if not k.endswith("/all")]
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
    return lines


def trend(points):
    """Lines: a metric per opponent (attack / defend), a column per minute. points: [(minute, metrics())]."""
    mins = [m for m, _ in points]
    first = points[0][1]
    opps = sorted({k.split("/")[0] for k in first}, key=lambda o: OPPONENTS.index(o) if o in OPPONENTS else 99)
    lines = ["| metric (attack / defend) | " + " | ".join(f"min {m}" for m in mins) + " |", "|---" * (len(mins) + 1) + "|"]
    f = (lambda fmt, v: "-" if v is None else fmt.format(v))
    for name, title, fmt in ROWS:
        for o in opps:
            cells = [f"{f(fmt, m.get(f'{o}/attack', {}).get(name))} / {f(fmt, m.get(f'{o}/defend', {}).get(name))}"
                     for _, m in points]
            lines.append(f"| {title}, {o} | " + " | ".join(cells) + " |")
    for name, title, fmt in ALL:
        for o in opps:
            lines.append(f"| {title}, {o} | " + " | ".join(f(fmt, m.get(f"{o}/all", {}).get(name)) for _, m in points) + " |")
    return lines


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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, rest = ap.parse_known_args()
    rest = [a for a in rest if a != "--"]
    with (contextlib.nullcontext() if args.no_lock else gpu_lock(f"test5 {args.label}")):
        test(args, rest)


def test(args, rest):
    out = OUT / args.label
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.before:
        before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    else:
        before = evaluation(checkpoint.load_policy(args.init, device), args, device)
    (out / "before.json").write_text(json.dumps(before, indent=1), encoding="utf-8", newline="\n")
    print(f"before: {before.get('seconds')} s", flush=True)

    targs = run.parser().parse_args(["--name", f"test5_{args.label}", "--init", args.init, "--minutes",
                                     str(args.minutes), "--updates", str(args.updates), "--device", args.device]
                                    + PROTOCOL + rest)
    points = [(0, metrics(before))]

    def hook(actor, critic, minute, update):
        actor.eval()
        res = evaluation(actor, args, device)
        res["update"] = update
        m = f"{minute:g}"
        checkpoint.save(out / f"m{m}.pt", actor, critic, targs.preset, {"minute": minute, "update": update,
                                                                      "run": targs.name})
        (out / f"eval_m{m}.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
        points.append((m, metrics(res)))
        print(f"evaluation at minute {m} (update {update}): {res['seconds']} s", flush=True)
        print("\n".join(trend(points)), flush=True)

    actor, summary, run_dir = run.train(targs, (args.every, hook) if args.every else None)
    actor.eval()
    after = evaluation(actor, args, device)
    (out / "after.json").write_text(json.dumps(after, indent=1), encoding="utf-8", newline="\n")
    if args.every:
        m = f"{args.minutes:g}"
        # with the critic (the run's latest.pt has it): the next run of the chain starts from this file
        critic = checkpoint.load_critic(run_dir / "latest.pt", device)
        checkpoint.save(out / f"m{m}.pt", actor, critic, targs.preset, {"minute": args.minutes, "update": summary["updates"],
                                                                      "run": targs.name})
        points.append((m, metrics(after)))

    mb, ma = metrics(before), metrics(after)
    lines = table(mb, ma)
    report = {"label": args.label, "init": args.init, "updates": summary["updates"], "train_s": summary["seconds"],
              "eval_battles": args.eval, "protocol": PROTOCOL, "options": rest, "train_args": vars(targs), "training": summary, "run": str(run_dir),
              "before": mb, "after": ma, "table": lines}
    if args.every:
        report["trend"] = {str(m): p for m, p in points}
        report["trend_table"] = trend(points)
        (out / "trend.md").write_text("\n".join(report["trend_table"]) + "\n", encoding="utf-8", newline="\n")
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    print(f"\ntest5 {args.label}: {args.init}, {summary['updates']} updates in {summary['seconds']} s "
          f"({summary['battles']} battles); before -> after, {args.eval} EVAL_SEEDS battles per opponent", flush=True)
    print("\n".join(lines), flush=True)
    if args.every:
        print("\ntrend:\n" + "\n".join(report["trend_table"]), flush=True)
    print(f"written: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
