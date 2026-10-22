"""torch.compile with the eager numbers (docs/en/training/simulator.md "Compiled and eager step").

Every torch.compile of the project (the simulator's step: battle.stepper; the observation and the bookkeeping of a
decision: tools/nn/train/rollout.py; the fighters lab: tools/nn/fighters/hook.py) goes through compiled(), so that the
compiled code on CUDA computes as the eager code - the code the companion runs in the game (on the CPU, no compile).

Why: Triton lowers an fp32 division to the approximate div.full (up to 2 ulp); eager CUDA and the CPU divide exactly
(round to nearest). In the simulator's step `hp = hp_abs / hp0` of a whole unit then came out 1 - 6e-8 for ~13 % of the
units (4068 / 4068 = 0.99999994: 2672 of the integers 1..20000), the first compiled step "took" health from them and the
observation's damage clock (TIMERS: dealt and taken damage, since when) started at the first decision of every battle -
in the game it starts at the first real hit (build/diag_obs_compile). Inductor's eager_numerics.division_rounding makes
the division exact (Triton's div_rn; no cost seen in the step's speed). What stays apart (fused multiply-adds, the order
of a sum, transcendental functions) is the last digit: it changes the state only at exact ties (a run-up through exactly
the charge's speed share, a shot at the very edge of range).
"""
import torch

_DONE = []


def eager_numerics():
    """Inductor divides fp32 exactly, as eager (set once, before a compile; a global switch of torch._inductor)."""
    if _DONE:
        return
    import torch._inductor.config as inductor
    inductor.eager_numerics.division_rounding = True
    _DONE.append(True)


def compiled(fn, dynamic=False):
    """torch.compile(fn, dynamic=dynamic) with eager_numerics()."""
    eager_numerics()
    return torch.compile(fn, dynamic=dynamic)
