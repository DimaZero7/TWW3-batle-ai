"""The script references of the evaluation, played on the CPU in parallel processes
(docs/en/training/workflow.md "Baseline cache and canary").

    DOCK_NAME=orch-baselines bash tools/nn/dock.sh tools.nn.train.refs            # what is missing, all cores
    python -m tools.nn.train.refs --jobs 4 --out build/tmp/refs                    # another folder (experiments)

Two kinds, cached in build/nn-train/baselines by version (tools/nn/train/version.py):
- the script baselines (evaluate.baselines): each of ai_like, nearest, hold_shoot against itself on the first
  256 pairs of the evaluation's battles, with the canary (evaluate.CANARY) against older files;
- the drills' check scripts (evaluate.drill_scripts): naive and skilled on 128 battles of each verified drill.

Each script's baseline and each drill script (naive, skilled: one file per drill, written when both are in)
is a job in its own process (multiprocessing spawn) with an equal share of the container's cores: the
simulator's uncompiled step on a few hundred battles is mostly per-operation overhead, so eight processes
of one thread do about eight times the work of one process of eight. The battles step compacted to the
running ones (evaluate.script_battles, drills/verify.run), the longest jobs first.

test5 starts the missing ones before its first evaluation (start(); the network's own battles go on on
the GPU meanwhile, evaluate.REFS_READY waits where the files are read): a cache miss costs ~8 min with
8 cores instead of ~30 on the GPU. The orchestrator can run them ahead (tools/ops/baselines.py: right
after a simulator change is ready, while the previous training step still runs) and the step finds them.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

SCRIPTS = ("ai_like", "nearest", "hold_shoot")          # the evaluation's script opponents (test5.OPPONENTS)
PAIRS = 256                                              # test5 --eval 1024: evaluate.pair_count
DRILL_BATTLES = 128                                      # test5 --drill-eval
MAX_UNITS, LIMIT_S = 19, 3600.0
# Longest first (a pool gives a free process the next job): one thread each, 8 cores, 05.10 (s): ai_like 472,
# hold_shoot 413, kiting 762 and hold_fire 615 for both scripts, nearest 210, counter 73 for both.
ORDER = ("ai_like", "hold_shoot", "drill:kiting:naive", "drill:kiting:skilled", "drill:hold_fire:naive",
         "drill:hold_fire:skilled", "nearest", "drill:counter:naive", "drill:counter:skilled")
DRILL_SCRIPTS = ("naive", "skilled")                     # evaluate.DRILL_SCRIPTS


def cpu_quota(cgroup=Path("/sys/fs/cgroup")):
    """The cores this process may use: the container's quota (docker --cpus: cgroup v2 cpu.max, v1 cfs),
    else the machine's (os.cpu_count sees the host's cores inside a container)."""
    try:
        quota, period = (cgroup / "cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        quota = int((cgroup / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((cgroup / "cpu" / "cpu.cfs_period_us").read_text())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def plan(tasks, cores, jobs=None):
    """(processes, threads per process) for the tasks on `cores`: a process per task up to the cores (or
    `jobs`), the cores shared out evenly."""
    n = max(1, min(len(tasks) or 1, jobs or cores, cores))
    return n, max(1, cores // n)


def jobs_of(tasks):
    """The jobs of the missing files (missing()): a script as it is, a drill "drill:<name>" as its two
    scripts "drill:<name>:naive", "drill:<name>:skilled"; the longest first (ORDER; unknown ones last)."""
    out = [f"{t}:{w}" for t in tasks if t.startswith("drill:") for w in DRILL_SCRIPTS]
    out += [t for t in tasks if not t.startswith("drill:")]
    rank = {t: i for i, t in enumerate(ORDER)}
    return sorted(out, key=lambda t: rank.get(t, len(ORDER)))


def _drills_setting(broad, embed):
    from tools.nn.train import drills as D
    if broad is not None:
        D.BROAD = broad
    if embed is not None:
        D.EMBED = embed
    return D


def missing(out=None, scripts=SCRIPTS, drills=None, pairs=PAIRS, drill_battles=DRILL_BATTLES, broad=None, embed=None):
    """The tasks whose files are not in `out` (default evaluate.BASELINES) for the current version: a script
    name, or "drill:<name>"."""
    from tools.nn.train import evaluate
    D = _drills_setting(broad, embed)
    folder = Path(out) if out else evaluate.BASELINES
    version, seeds = evaluate.sim_version(), evaluate.eval_seeds(pairs)
    tasks = []
    for n in scripts:
        p = folder / evaluate.baseline_path(n, MAX_UNITS, LIMIT_S, version).name
        try:
            ok = json.loads(p.read_text(encoding="utf-8"))["seed"][:pairs] == seeds
        except (OSError, ValueError, KeyError):
            ok = False
        if not ok:
            tasks.append(n)
    for n in (D.READY if drills is None else drills):
        if not (folder / evaluate.drill_ref_path(n, drill_battles).name).is_file():
            tasks.append(f"drill:{n}")
    return tasks


def _task(task, out, threads, canary, pairs, drill_battles, broad, embed):
    """One job in a worker process -> (job, seconds, what): a script's baseline is written by
    evaluate.baselines (what: its file's name), a drill script's result comes back (what: the dict)."""
    import torch
    torch.set_num_threads(threads)
    from tools.nn.train import evaluate
    _drills_setting(broad, embed)
    if out:
        evaluate.BASELINES = Path(out)
    evaluate.CANARY = canary
    t = time.time()
    if task.startswith("drill:"):
        _, name, which = task.split(":")
        what = evaluate.drill_script(name, drill_battles, which, "cpu")
    else:
        what = evaluate.baselines([task], pairs, MAX_UNITS, LIMIT_S, "cpu")[task]["cache"]
    return task, round(time.time() - t), what


class Pending:
    """The jobs started by start(), running in their processes; wait() blocks until every file is written
    (a drill's when both its scripts are in) -> {job: seconds} (once; later calls return the same)."""

    def __init__(self, pool, jobs, futures, t0, log, drill_path):
        self.pool, self.jobs, self.futures, self.t0, self.log, self.done = pool, jobs, futures, t0, log, None
        self.drill_path = drill_path             # drill name -> its file (evaluate.drill_ref_path in the folder)

    def wait(self):
        if self.done is None:
            done, drills = {}, {}
            try:
                for job, f in zip(self.jobs, self.futures):
                    try:
                        task, sec, what = f.result()
                    except Exception as e:           # noqa: BLE001 - the evaluation plays a missing file itself
                        self.log(f"reference {job}: FAILED ({type(e).__name__}: {e}); the evaluation plays it itself")
                        continue
                    done[task] = sec
                    if task.startswith("drill:"):
                        _, name, which = task.split(":")
                        drills.setdefault(name, {})[which] = what
                        what = f"win {what['win_rate']}, trade {what.get('gold_trade')}"
                        if set(drills[name]) == set(DRILL_SCRIPTS):
                            from tools.nn.train import evaluate
                            evaluate.write_drill_ref(self.drill_path(name), {w: drills[name][w] for w in DRILL_SCRIPTS})
                    self.log(f"reference {task}: {sec} s ({what})")
            finally:
                self.pool.shutdown()
            self.done = done
            self.log(f"script references ready in {time.time() - self.t0:.0f} s")
        return self.done


def start(out=None, jobs=None, canary=32, pairs=PAIRS, drill_battles=DRILL_BATTLES, scripts=SCRIPTS, drills=None,
          broad=None, embed=None, log=None):
    """Starts every missing reference (missing()) on the CPU, in parallel processes -> Pending, or None when
    nothing is missing. The caller goes on (test5: the network's own battles on the GPU) and waits before
    it reads the files (evaluate.REFS_READY)."""
    log = log or (lambda s: print(s, flush=True))
    D = _drills_setting(broad, embed)
    broad, embed = D.BROAD, D.EMBED                  # the shares in force here go to the workers (spawned: defaults)
    tasks = jobs_of(missing(out, scripts, drills, pairs, drill_battles, broad, embed))
    if not tasks:
        return None
    n, threads = plan(tasks, cpu_quota(), jobs)
    from tools.nn.train import evaluate
    log(f"script references missing for simulator version {evaluate.sim_version()} (drills {evaluate.drill_version()}): "
        f"{', '.join(tasks)}; playing them on the CPU, {n} processes x {threads} threads")
    pool = ProcessPoolExecutor(max_workers=n, mp_context=get_context("spawn"))
    futures = [pool.submit(_task, t, str(out) if out else None, threads, canary, pairs, drill_battles, broad, embed)
               for t in tasks]
    folder = Path(out) if out else evaluate.BASELINES
    return Pending(pool, tasks, futures, time.time(), log, lambda name: folder / evaluate.drill_ref_path(name, drill_battles).name)


def ensure(*args, **kwargs):
    """start() and wait for it -> {job: seconds}; {} when nothing is missing."""
    pending = start(*args, **kwargs)
    return pending.wait() if pending else {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="the cache folder (default build/nn-train/baselines)")
    ap.add_argument("--jobs", type=int, help="processes (default: one per task, up to the cores)")
    ap.add_argument("--canary", type=int, default=32, help="canary pairs against older files (0: off)")
    ap.add_argument("--scripts", default=",".join(SCRIPTS), help="comma-separated script opponents ('' none)")
    ap.add_argument("--drills", default=None, help="comma-separated drills (default drills.READY; '' none)")
    ap.add_argument("--drill-broad", type=float, default=None, help="drills.BROAD (default: as run.py)")
    ap.add_argument("--drill-embed", type=float, default=None, help="drills.EMBED (default: as run.py)")
    args = ap.parse_args(argv)
    split = (lambda s: tuple(x for x in s.split(",") if x))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
    t = time.time()
    done = ensure(args.out, args.jobs, args.canary, scripts=split(args.scripts),
                  drills=None if args.drills is None else split(args.drills), broad=args.drill_broad, embed=args.drill_embed)
    if not done:
        print("script references: all present", flush=True)
    print(f"written: {len(done)} job(s), {time.time() - t:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
