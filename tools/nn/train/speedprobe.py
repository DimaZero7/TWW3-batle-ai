"""Where a training update's time goes: run.py's training for a few updates with timers around its parts.

    DOCK_BUILD=<main>/build bash tools/nn/dock.sh tools.nn.train.speedprobe <run.py options> --updates 5 --no-eval
PROBE=time / --probe time (default): CUDA-synchronised inclusive timers per part (nested parts as a/b), from the 2nd update on (the
            1st compiles); the synchronisation itself costs a little (no CPU / GPU overlap)
PROBE=light: only collect, the branches, the update and start_kl's distance (no slowdown)
PROBE=torch / --probe torch: torch.profiler over update PROBE_AT's (default 3) PPO step (--probe-part update) or
            its decisions 10..13 (--probe-part step): kernel time against wall time, the top kernels
Output: build/speedprobe/<name>.json, <name>_torch.txt."""
import collections
import functools
import json
import os
import sys
import time

import torch

from tools import config as project
from tools.nn.train import ability_branch, ppo, rollout, run

MODE = os.environ.get("PROBE", "time")
AT = int(os.environ.get("PROBE_AT", "3"))
PART = "update"                                  # --probe-part update | step (decisions STEPS of the collection)
STEPS = (10, 13)
for _i, _a in enumerate(list(sys.argv)):          # (dock.sh passes no environment: --probe MODE, --probe-at N too)
    if _a in ("--probe", "--probe-at", "--probe-part") and _i + 1 < len(sys.argv):
        if _a == "--probe":
            MODE = sys.argv[_i + 1]
        elif _a == "--probe-part":
            PART = sys.argv[_i + 1]
        else:
            AT = int(sys.argv[_i + 1])
TRACE0 = "--probe-trace" in sys.argv
_OWN = ("--probe", "--probe-at", "--probe-part")
sys.argv = [a for i, a in enumerate(sys.argv) if a not in _OWN and (i == 0 or sys.argv[i - 1] not in _OWN)
            and a != "--probe-trace"]
OUT = project.BUILD / "speedprobe"
T = collections.defaultdict(float)
N = collections.Counter()
ON = [False]
STACK = []
UPD = [0]
NAME = ["probe"]
PROF = [None, 0.0]
TRACE = TRACE0
NL = chr(10)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(name, fn):
    @functools.wraps(fn)
    def w(*a, **k):
        if not ON[0] or torch.compiler.is_compiling():
            return fn(*a, **k)
        key = "/".join(STACK + [name])
        sync()
        t = time.perf_counter()
        STACK.append(name)
        try:
            return fn(*a, **k)
        finally:
            STACK.pop()
            sync()
            T[key] += time.perf_counter() - t
            N[key] += 1
    return w


def patch(obj, attr, name=None):
    setattr(obj, attr, timed(name or attr, getattr(obj, attr)))


def install():
    if MODE == "time":
        B = rollout.Battles
        for m in ("step", "observe", "_act", "_act_past", "_sim_step", "_reset", "scripted", "_teach_labels", "value",
                  "_play", "replay"):
            patch(B, m)
        # the compiled bookkeeping and network calls (fast(), observer()): timed outside the compiled code
        orig_fast, orig_obs = rollout.fast, rollout.observer
        rollout.fast = lambda fn, use: timed(fn.__name__, orig_fast(fn, use))
        rollout.observer = lambda device, compile=None: timed("look", orig_obs(device, compile))
        from tools.nn.fighters import hook
        patch(hook.Hook, "step", "sim+men")
        Br = ability_branch.Brancher
        for m in ("capture", "finish", "_play", "_criticise"):
            patch(Br, m)
        patch(ability_branch, "play_pair")
        from tools.nn.model import critic as mcritic
        from tools.nn.model import policy
        patch(policy.ActorV2, "sequence", "actor.sequence")
        patch(mcritic.Critic, "forward", "critic")
        patch(torch.Tensor, "backward")
        patch(torch.optim.Adam, "step", "adam")
        patch(torch.nn.utils, "clip_grad_norm_")
        for f in ("ability_epochs", "distance", "gae", "critic_stats", "full_obs"):
            patch(ppo, f)
        patch(torch.cuda, "empty_cache")
    if MODE == "light":                          # a few big parts only (three synchronisations an update)
        patch(ppo, "distance")
        patch(ability_branch.Brancher, "finish")
    orig_collect, orig_update = rollout.collect, ppo.update

    def start():
        sync()
        PROF[0] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA], with_stack=TRACE)
        PROF[0].__enter__()
        PROF[1] = time.perf_counter()

    def stop(what):
        sync()
        wall = time.perf_counter() - PROF[1]
        PROF[0].__exit__(None, None, None)
        report(PROF[0], wall, what)
        PROF[0] = None

    if MODE == "torch" and PART in ("step", "both"):       # decisions STEPS[0]..STEPS[1] of update AT's collection
        step = rollout.Battles.step
        count = [0]

        def stepped(self, *a, **k):
            if UPD[0] == AT and self.brancher is not None:
                count[0] += 1
                if count[0] == STEPS[0]:
                    start()
                out = step(self, *a, **k)
                if count[0] == STEPS[1]:
                    stop(f"decisions {STEPS[0]}..{STEPS[1]}")
                return out
            return step(self, *a, **k)
        rollout.Battles.step = stepped

    if MODE == "sync":       # every read of the GPU (synchronisation) in decisions STEPS of update AT, by the code line
        import traceback
        import warnings
        step = rollout.Battles.step
        count = [0]
        seen = collections.Counter()

        def show(message, category, filename, lineno, file=None, line=None):
            st = [f for f in traceback.extract_stack()[:-1] if "/repo/" in f.filename and "speedprobe" not in f.filename]
            seen[" <- ".join(f"{f.filename.split('/repo/')[-1]}:{f.lineno}" for f in st[-1:-4:-1])] += 1

        def stepped(self, *a, **k):
            if UPD[0] == AT and self.brancher is not None:
                count[0] += 1
                if count[0] == STEPS[0]:
                    sync()
                    warnings.showwarning = show
                    warnings.simplefilter("always")
                    torch.cuda.set_sync_debug_mode(1)
                out = step(self, *a, **k)
                if count[0] == STEPS[1]:
                    torch.cuda.set_sync_debug_mode(0)
                    n = STEPS[1] - STEPS[0] + 1
                    text = f"reads of the GPU per decision (decisions {STEPS[0]}..{STEPS[1]}):" + NL + NL.join(
                        f"{c / n:6.1f}  {k}" for k, c in seen.most_common())
                    OUT.mkdir(parents=True, exist_ok=True)
                    (OUT / f"{NAME[0]}_sync.txt").write_text(text, encoding="utf-8")
                    print(text, flush=True)
                return out
            return step(self, *a, **k)
        rollout.Battles.step = stepped

    def collect(*a, **k):
        UPD[0] += 1
        if MODE in ("time", "light") and UPD[0] == 2:
            ON[0] = True
        torch.cuda.reset_peak_memory_stats()
        out = timed("collect", orig_collect)(*a, **k)
        print(f"probe: collect peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, reserved "
              f"{torch.cuda.memory_reserved() / 2**30:.2f} GB", flush=True)
        return out

    if MODE == "torch" and PART in ("update", "both"):     # minibatches 5..7 of update AT (Adam's steps 4..7)
        adam = torch.optim.Adam.step
        n_step = [0]
        INU = [False]

        def stepping(self, *a, **k):
            out = adam(self, *a, **k)
            if INU[0]:
                n_step[0] += 1
                if n_step[0] == 4:
                    start()
                if n_step[0] == 7:
                    stop("minibatches 5..7 of ppo.update")
            return out
        torch.optim.Adam.step = stepping

    def mismatch(actor, batch, cfg):
        """The rollout's log-probability of the actions taken against the update's recomputed one, before any step:
        TF32 (as the update) and bf16 autocast (--amp bf16), on the batch's first rows (a part's worth)."""
        import dataclasses
        from tools.nn.model import heads as hd
        T, R = batch["reward"].shape
        out = {}
        for amp in ("", "bf16", "bf16h"):
            diffs, kls = [], []
            for lo in range(0, min(R, 96), 12):
                idx = torch.arange(lo, min(R, lo + 12), device=batch["reward"].device)
                obs = rollout.full_obs({k: v[:, idx] for k, v in batch["obs"].items()}, batch.get("abil_static"))
                with torch.no_grad(), ppo._amp(dataclasses.replace(cfg, amp=amp), actor):
                    lg, _ = actor.sequence(obs, batch["h0"][idx], batch["reset"][:, idx], ppo.chunk_action(batch["action"], idx))
                lg = {k: v.float().reshape(-1, *v.shape[2:]) for k, v in lg.items()}
                ctrl = obs["ctrl"].reshape(-1, obs["ctrl"].shape[-1])
                act = hd.Action(**{f: ppo._rows(getattr(batch["action"], f), idx) for f in batch["action"].parts()})
                lp, _ = hd.log_prob(lg, act, ctrl)
                old = ppo._rows(batch["lp"], idx)
                d = (lp - old)[ctrl]
                diffs.append(d.abs())
                kls.append(ppo.kl_k3(torch.exp(d)))
            d, kl = torch.cat(diffs), torch.cat(kls)
            out[amp or "tf32"] = (float(d.mean()), float(d.abs().max()), float(kl.mean()), float(d.mean()))
        txt = "; ".join(f"{k}: mean |dlp| {v[0]:.2e} max {v[1]:.2e} KL(k3) {v[2]:.2e}" for k, v in out.items())
        print("probe: rollout vs update log-probability " + txt, flush=True)

    def update(*a, **k):
        if MODE == "mismatch" and UPD[0] >= 2:
            mismatch(a[0], a[3], a[4])
        INU_ = MODE == "torch" and PART in ("update", "both") and UPD[0] == AT
        if INU_:
            INU[0] = True
        try:
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            out = timed("ppo.update", orig_update)(*a, **k)
            print(f"probe: update peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB (from "
                  f"{before / 2**30:.2f}), reserved {torch.cuda.memory_reserved() / 2**30:.2f} GB", flush=True)
            return out
        finally:
            if INU_:
                INU[0] = False
    rollout.collect = collect
    ppo.update = update


def gaps(path, out):
    """The GPU's idle gaps in a chrome trace (path): every gap of 0.3 ms or more between kernels, by the innermost repo
    Python function open on the CPU when the GPU ran dry -> text."""
    import bisect
    doc = json.loads(open(path, encoding="utf-8").read())
    ev = doc["traceEvents"]
    kern = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") == "kernel" and "dur" in e)
    py = sorted((e for e in ev if e.get("cat") == "python_function" and "dur" in e and "/repo/" in e.get("name", "")
                 and "speedprobe" not in e.get("name", "")), key=lambda e: e["ts"])
    starts = [e["ts"] for e in py]
    total = kern[-1][1] - kern[0][0] if kern else 0.0
    by = collections.Counter()
    idle, end = 0.0, None
    for s0, e0 in kern:
        if end is not None and s0 - end >= 300:
            gap = s0 - end
            idle += gap
            i = bisect.bisect_right(starts, end)
            open_ = [e for e in py[max(0, i - 3000):i] if e["ts"] + e["dur"] >= end]
            name = min(open_, key=lambda e: e["dur"])["name"].split("/repo/")[-1] if open_ else "?"
            by[name[:110]] += gap
        end = e0 if end is None else max(end, e0)
    lines = [f"GPU span {total / 1e3:.0f} ms, idle gaps >= 0.3 ms {idle / 1e3:.0f} ms ({idle / max(total, 1):.0%})"]
    lines += [f"{v / 1e3:8.1f} ms  {k}" for k, v in by.most_common(40)]
    text = NL.join(lines)
    open(out, "w", encoding="utf-8").write(text)
    return text


def report(p, wall, what):
    if TRACE:
        OUT.mkdir(parents=True, exist_ok=True)
        path = str(OUT / f"{NAME[0]}_{what.split()[0].strip('.')}_trace.json")
        p.export_chrome_trace(path)
        print(gaps(path, path.replace("_trace.json", "_gaps.txt")), flush=True)
    ka = p.key_averages()
    busy = sum(e.self_device_time_total for e in ka) / 1e6
    launches = sum(e.count for e in ka if e.self_device_time_total > 0)
    text = (f"{what}: wall {wall:.1f} s (profiled), GPU kernels busy {busy:.1f} s ({busy / wall:.0%}), "
            f"ops with kernels {launches}\n"
            + ka.table(sort_by="self_device_time_total", row_limit=50, max_name_column_width=70))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{NAME[0]}_{what.split()[0].strip('.')}_torch.txt").write_text(text, encoding="utf-8")
    print(text[:8000], flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = run.parser().parse_args()
    NAME[0] = args.name
    install()
    t0 = time.time()
    _, summary, _ = run.train(args)
    if MODE in ("time", "light"):
        n = max(1, UPD[0] - 1)
        rows = sorted(((k, T[k] / n, N[k] / n) for k in T), key=lambda r: r[0])
        tot = (T["collect"] + T["ppo.update"]) / n
        print(f"\nper update (updates 2..{UPD[0]}, means, inclusive; share of collect + update {tot:.1f} s):", flush=True)
        for k, s, c in rows:
            print(f"  {k:72s} {s:8.2f} s {s / tot:6.1%}  calls {c:8.1f}", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{args.name}.json").write_text(json.dumps({"updates": n, "per_update": {k: [s, c] for k, s, c in rows},
                                                           "summary": summary}, indent=1), encoding="utf-8")
    print(f"probe done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
