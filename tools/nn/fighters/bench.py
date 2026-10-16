"""The lab's speed against the simulator's: B battles of 20 x 20 units (the arena whole_emp_v_skv, every unit attacking
its nearest enemy, as the analyst's bench: build/... bench_sim2) on the GPU; ms a step (0.5 s) over blocks of 20 steps
from the start into the melee, CUDA events; peak memory.

  python -m tools.nn.fighters.bench --B 256 512 1024 [--steps 160]

Modes: sim (the simulator compiled, as training runs it), sim_eager, lab (the simulator eager with the men, hook.py:
the lab's step is eager - its Python control flow), and the lab's own share (lab - sim_eager). Needs torch + CUDA."""
import argparse
import statistics
import time

import torch

from tools.nn.sim import battle, replay, scenario
from tools.nn.sim.params import load


def fresh(B, params, dev, seed=0, per_side=20):
    army = scenario.from_arena("whole_emp_v_skv", "attack")
    st = scenario.build([army] * B, params, device=dev, per_side=per_side)
    g = torch.Generator(device="cpu").manual_seed(seed)
    st.u["x"] = st.u["x"] + (torch.rand(st.u["x"].shape, generator=g) * 20 - 10).to(dev)
    st.u["z"] = st.u["z"] + (torch.rand(st.u["z"].shape, generator=g) * 20 - 10).to(dev)
    return st


def timed(st, adv, params, steps, warm=3):
    orders = replay.nearest_attack(st)
    for _ in range(warm):
        adv(st, orders, params, params.dt)
    torch.cuda.synchronize()
    blocks, ev = [], []
    for k in range(steps):
        if k % 2 == 0:
            orders = replay.nearest_attack(st)
        a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        adv(st, orders, params, params.dt)
        e.record()
        ev.append((a, e))
        if len(ev) == 20:
            torch.cuda.synchronize()
            blocks.append(sum(x.elapsed_time(y) for x, y in ev) / len(ev))
            ev = []
    m = statistics.mean(blocks)
    ci = 2.0 * statistics.stdev(blocks) / len(blocks) ** 0.5 if len(blocks) > 1 else 0.0
    return m, ci, blocks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--B", nargs="+", type=int, default=[256, 512, 1024])
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--modes", default="sim,sim_eager,lab")
    a = ap.parse_args(argv)
    dev = "cuda"
    params = load().with_cal("noise", blows=False)
    for B in a.B:
        res = {}
        for mode in a.modes.split(","):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                if mode == "lab":
                    from tools.nn.fighters import hook
                    with hook.installed() as h:
                        st = fresh(B, params, dev)
                        m, ci, blocks = timed(st, battle.step, params, a.steps)
                        M = h.F.M
                else:
                    st = fresh(B, params, dev)
                    adv = battle.stepper(dev, compile=(mode == "sim"))
                    m, ci, blocks = timed(st, adv, params, a.steps)
                    M = None
            except torch.cuda.OutOfMemoryError:
                print(f"B={B} {mode}: out of memory", flush=True)
                continue
            peak = torch.cuda.max_memory_allocated() / 1e9
            res[mode] = m
            print(f"B={B} {mode}: {m:.1f} +- {ci:.1f} ms a step (blocks of 20: {', '.join(f'{b:.0f}' for b in blocks)}); "
                  f"peak {peak:.2f} GB; sim time {float(st.t.max()):.0f} s; men a battle {M}; "
                  f"wall {time.perf_counter() - t0:.0f} s", flush=True)
            del st
        if "lab" in res and "sim_eager" in res:
            print(f"B={B}: the lab's own part {res['lab'] - res['sim_eager']:.1f} ms a step; lab / sim compiled "
                  f"{res['lab'] / res.get('sim', float('nan')):.1f}x", flush=True)


if __name__ == "__main__":
    main()
