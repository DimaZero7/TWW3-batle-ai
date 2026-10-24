"""The learned group collection's stickiness and the tourney's mixed penalty (GeneralConfig.groups_sticky,
assign.keep_last, groups.mixed, tourney_train.plan_advantage; docs/ru/training/general.md "Сбор групп"): a re-plan by
the timer keeps our groups, one on an event chooses them again; the mixed groups cost the plan's advantage. torch: the
training container, on the CPU."""
import dataclasses

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import assign as gassign                 # noqa: E402
from tools.nn.general import executor                          # noqa: E402
from tools.nn.general import groups as ggroups                 # noqa: E402
from tools.nn.general import tourney_train as tt               # noqa: E402
from tools.nn.model import critic as mcritic, policy           # noqa: E402

from tests.tools.test_general_runtime_torch import GCFG, TINY, battles   # noqa: E402

G = 4


def test_keep_last_keeps_the_numbers_drops_the_units_out_and_places_a_unit_back():
    prev = torch.tensor([[2, 2, 0, 3, -1, 5], [1, 1, 0, 0, 2, -1]])
    who = torch.tensor([[True, True, True, False, True, False], [True] * 5 + [False]])
    sampled = torch.tensor([[0, 1, 1, -1, 0, -1], [3, 3, 2, 2, 1, -1]])
    logits = torch.zeros(2, 6, G)
    logits[0, 4, 1] = 9.0                       # (not a kept group: the unit back joins the kept group it likes most)
    logits[0, 4, 0] = 2.0
    keep = torch.tensor([True, False])
    a, kept = gassign.keep_last(sampled, logits, who, prev, torch.tensor([True, True]), keep, G)
    assert kept.tolist() == [True, False]
    assert a[0].tolist() == [2, 2, 0, -1, 0, -1]          # unit 3 out (dropped, group 3 gone), unit 4 back -> 0
    assert torch.equal(a[1], sampled[1])                  # not a timer re-plan: the sampled membership
    # no last plan: nothing kept
    _, kept = gassign.keep_last(sampled, logits, who, prev, torch.tensor([False, False]), torch.tensor([True, True]), G)
    assert not bool(kept.any())


def test_mixed_groups_are_shooters_and_melee_together():
    s, m, lord = (ggroups.ROLES.index(k) for k in ("shoot", "melee", "lord"))
    main = torch.tensor([[0, 0, 1, 1, 2, 5]])
    role = torch.tensor([[s, m, m, lord, s, s]])
    who = torch.tensor([[True] * 5 + [False]])
    assert ggroups.mixed(main, who, role, G)[0].tolist() == [True, False, False, False]
    assert abs(float(ggroups.mixed_share(main, who, role, G)[0]) - 1 / 3) < 1e-6


def test_the_mixed_penalty_lowers_the_mixed_plans_advantage():
    a = torch.zeros(4)
    share = torch.tensor([0.0, 0.5, 1.0, 1.0])
    chosen = torch.tensor([True, True, True, False])
    off = tt.plan_advantage(a, share, 0.0, chosen, 0.1)
    assert torch.equal(off, torch.zeros(4))
    on = tt.plan_advantage(a, share, 0.05, chosen, 0.1)
    assert float(on[0]) > float(on[1]) > float(on[2])
    assert abs(float(on[:3].mean())) < 1e-6             # less the chosen plans' mean
    assert abs(float(on[0]) - 0.25) < 1e-6               # 0.05 x 0.5 / 0.1


def _learned(sticky):
    torch.manual_seed(0)
    cfg = dataclasses.replace(GCFG, groups="learned", groups_sticky=sticky)
    return executor.wrap(policy.Actor(TINY).eval(), cfg).eval(), mcritic.Critic(TINY).eval()


def _own(env, row):
    b, s = row % env.B, row // env.B + 1
    u = env.st.u
    return (u["side"][b] == s) & (u["men"][b] > 0) & ~u["gone"][b] & (u["ms"][b].float() < 5.5)


def test_a_timer_replan_keeps_our_groups_and_an_event_chooses_them_again():
    env = battles(B=4, mix={"self": 1.0})
    actor, crit = _learned(True)
    env.step(actor, crit)
    rt = env.general_rt
    r = rt.r
    rt.b["K"][:] = 2                            # (two groups wanted: a membership to choose; K 1 has none)
    row = 0
    b, s = row % env.B, row // env.B + 1

    def event(lord):                            # a lord fallen: a re-plan on an event
        r["t_plan"][row] -= 5.0
        env.st.lord_dead_s[b, lord] = 0.5
        env.step(actor, crit)
    event(2 - s)                                # their lord
    assert int(r["n_replan"][row]) == 1 and not bool(r["kept"][row]) and float(r["lp_assign"][row]) < 0
    old = r["main"][row].clone()
    ch0 = rt.stats["churn_changed"]
    r["t_plan"][row] -= 31.0                    # the timer alone
    env.step(actor, crit)
    assert int(r["n_replan"][row]) == 2 and bool(r["kept"][row])
    assert float(r["lp_assign"][row]) == 0.0                                # not chosen: nothing to learn
    own = _own(env, row)
    assert torch.equal(r["main"][row][own], old[own])                       # the same groups, the same numbers
    assert rt.stats["assign_kept"] >= 1 and rt.stats["churn_changed"] == ch0     # no composition change counted
    event(s - 1)                                # our lord: chosen again
    assert int(r["n_replan"][row]) == 3 and not bool(r["kept"][row]) and float(r["lp_assign"][row]) < 0
    st = rt.take_stats()
    for k in ("group_changes_per_battle", "group_change_share", "mixed_share", "assign_kept"):
        assert k in st, k


def test_without_stickiness_a_timer_replan_chooses_again():
    env = battles(B=2, mix={"self": 1.0})
    actor, crit = _learned(False)
    env.step(actor, crit)
    env.general_rt.b["K"][:] = 2
    r = env.general_rt.r
    r["t_plan"][0] -= 31.0
    env.step(actor, crit)
    assert int(r["n_replan"][0]) == 1 and not bool(r["kept"][0]) and float(r["lp_assign"][0]) < 0
