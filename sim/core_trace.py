"""The melee core's read-outs (melee.core, S0; sim/config/sim.json melee.core_why): a hook the step and melee.strikes
hand their pair state to when it is set - nothing is computed, kept or changed when it is None (the default).

HOOK: None, or an object with
  pre(st, u, pw, touch, strike, params, dt)   battle.step, before melee.strikes: the units, the pair geometry, the
                                              formations' contact (touch, before the strike masks) and the strike mask
  note(name, tensor)                          melee.strikes: F_geo (the men striking from the line, men cap, fill and
                                              contact mask - before every side, stack and answer cap), cover / ahead
                                              (sim/stack.py stacked, per pair), side_of, F_ans_in / F_pre (F before melee.answer_struck and before the wave), p_front (the cadence's chance)
  post(rate, hit, sector, F_men)              battle.step, after melee.strikes
The tracer itself (pair clock, senior, class codes, tables) is outside sim/ while the core is designed
(tmp/melee_core/harness/ctrace.py). With a hook set the step must run eager (torch.compile would trace the hook)."""
HOOK = None
