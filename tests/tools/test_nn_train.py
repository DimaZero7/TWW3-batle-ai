"""tools.nn.train: PPO training in the simulator (docs/en/training/training.md).

torch is not in the project's .venv: these tests are skipped there and run in the training
container (snake-ai-trainer; its image has no pytest, so put a pure-Python pytest on PYTHONPATH).

Level 1: functions (GAE, clipping, reward, order changes, layout). Level 2: properties (masks are
respected, rewards are symmetric between the sides, KEEP keeps the order). Level 3: a tiny training
step end to end on the CPU, checkpoints, recordings.
"""
import dataclasses
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn import gamedata  # noqa: E402
from tools.nn.model import config, critic, heads, policy  # noqa: E402
from tools.nn.sim import battle, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.train import checkpoint, evaluate, imitate, league, opponents, ppo, randomise, reward, rollout, scenes  # noqa: E402,E501
from tools.nn.train import cadence as cad  # noqa: E402

CFG = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)
MIRROR = [("arena", "attack")]


def nets(seed=0):
    torch.manual_seed(seed)
    return policy.Actor(CFG).eval(), critic.Critic(CFG).eval()


def two_units(kind_first=O.HOLD):
    army = {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": "wh_main_emp_inf_spearmen_0", "x": -300, "z": 0, "b": 90}]},
        2: {"faction": "wh_main_emp_empire", "units": [{"key": "wh_main_emp_inf_spearmen_0", "x": 300, "z": 0, "b": 270}]}}}
    return scenario.build([army])


def line_army(attacker=1, gap=600.0, lord_ahead=0.0):
    """Each side: a general, two spearmen and archers behind; side 1 west, side 2 east."""
    def side(sign, faces):
        x0 = sign * gap / 2
        return {"faction": "wh_main_emp_empire", "units": [
            {"key": "wh_main_emp_cha_general_0", "general": True, "x": x0 + sign * (20 - lord_ahead), "z": 0, "b": faces},
            {"key": "wh_main_emp_inf_spearmen_0", "x": x0, "z": -40, "b": faces},
            {"key": "wh_main_emp_inf_spearmen_0", "x": x0, "z": 40, "b": faces},
            {"key": "wh2_dlc13_emp_inf_archers_0", "x": x0 + sign * 40, "z": 0, "b": faces}]}
    return scenario.build([{"attacker": attacker, "sides": {1: side(-1, 90), 2: side(1, 270)}}])


# --- level 1: functions ---

class TestFunctions:
    def test_gae_is_the_discounted_sum_with_lambda_one_and_stops_at_the_end_of_a_battle(self):
        r = torch.tensor([[1.0], [0.0], [2.0]])
        v = torch.zeros(3, 1)
        done = torch.tensor([[False], [True], [False]])
        adv, ret = ppo.gae(r, v, done, torch.tensor([10.0]), gamma=0.5, lam=1.0)
        assert ret[:, 0].tolist() == pytest.approx([1.0, 0.0, 2.0 + 0.5 * 10.0])
        adv, _ = ppo.gae(r, torch.ones(3, 1), done, torch.tensor([0.0]), gamma=0.5, lam=0.0)
        assert adv[:, 0].tolist() == pytest.approx([1.0 + 0.5 - 1, -1.0, 2.0 - 1])

    def test_the_entropy_floor_raises_the_weight_below_the_target_within_its_bounds(self):
        assert ppo.entropy_weight(0.01, 0.02, 0.1, 0.003, 0.1, rate=2.0) == pytest.approx(0.02)
        assert ppo.entropy_weight(0.08, 0.02, 0.1, 0.003, 0.1, rate=2.0) == pytest.approx(0.1)     # the ceiling
        assert ppo.entropy_weight(0.01, 0.2, 0.1, 0.003, 0.1, rate=2.0) == pytest.approx(0.005)
        assert ppo.entropy_weight(0.004, 0.2, 0.1, 0.003, 0.1, rate=2.0) == pytest.approx(0.003)   # the schedule

    def test_the_clipped_loss_stops_at_the_clip(self):
        lp_old = torch.zeros(1, 2)
        lp_new = torch.log(torch.tensor([[1.5, 0.5]])).requires_grad_()
        mask = torch.tensor([[True, True]])
        loss, clipped = ppo.policy_loss(lp_new, lp_old, torch.tensor([1.0]), mask, 0.2)
        assert float(loss.detach()) == pytest.approx(-(1.2 + 0.5) / 2)
        assert float(clipped) == 1.0
        loss.backward()
        assert lp_new.grad[0, 0] == 0 and lp_new.grad[0, 1] != 0       # the clipped unit gets no gradient

    def test_the_policy_loss_leaves_out_masked_units(self):
        loss, _ = ppo.policy_loss(torch.tensor([[0.0, 5.0]]), torch.zeros(1, 2), torch.tensor([1.0]),
                                  torch.tensor([[True, False]]), 0.2)
        assert float(loss) == pytest.approx(-1.0)

    def test_the_reward_is_win_plus_health_and_standing_differences(self):
        w = reward.Weights(win=1.0, hp=0.5, standing=0.5)
        before = torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.4, 0.3]]])
        after = torch.tensor([[[0.9, 1.0], [0.7, 0.8]], [[0.5, 0.0], [0.4, 0.3]]])
        r = reward.step(before, after, torch.tensor([False, True]), torch.tensor([0, 2]), torch.tensor([1, 1]), w)
        shaped = 0.5 * (0.3 - 0.1) + 0.5 * 0.2
        assert r[0].tolist() == pytest.approx([shaped, -shaped])
        assert r[1].tolist() == pytest.approx([-1.0 - 0.25, 1.0 + 0.25])

    def test_at_the_time_limit_the_attacker_loses_more_than_a_fight(self):
        w = reward.Weights(win=1.0, timeout=1.5, hp=0.0, standing=0.0)
        m = torch.ones(2, 2, 2)                                            # both sides still stand
        r = reward.step(m, m, torch.tensor([True, True]), torch.tensor([2, 1]), torch.tensor([1, 2]), w)
        assert r.tolist() == [[-1.5, 1.0], [1.0, -1.5]]

    def test_gold_lost_is_cost_times_the_share_lost(self):
        st = line_army()
        u = st.u
        cost = u["cost"][0].clone()
        assert reward.gold_lost(u)[0].abs().sum() == 0.0                             # nothing lost yet
        u["hp_abs"][0, 1] = u["hp0"][0, 1] * 0.5                                      # half its health
        u["hp_abs"][0, 2] = u["hp0"][0, 2] * 0.5
        u["r"][0, 2] = True                                                           # ... and routing
        u["r"][0, 3] = u["s"][0, 3] = True                                            # shattered, health full
        H = st.N // 2
        u["gone"][0, H + 1] = True                                                    # off the map
        u["hp_abs"][0, H + 2] = 0.0
        u["men"][0, H + 2] = 0.0                                                      # dead
        g = reward.gold_lost(u, rout_share=0.5)[0]
        assert float(g[1]) == pytest.approx(0.5 * float(cost[1]))
        assert float(g[2]) == pytest.approx(0.75 * float(cost[2]))                    # 0.5 + 0.5 x the rest
        for i in (3, H + 1, H + 2):
            assert float(g[i]) == pytest.approx(float(cost[i]))                       # lost whole
        assert float(g[0]) == 0.0 and float(g[H]) == 0.0
        assert float(g[u["side"][0] == 0].abs().sum()) == 0.0                         # empty slots
        sides = reward.gold_sides(u, 0.5)[0]
        assert sides.tolist() == pytest.approx([float(g[1] + g[2] + g[3]), float(g[H + 1] + g[H + 2])])
        c1, c2 = float(cost[u["side"][0] == 1].sum()), float(cost[u["side"][0] == 2].sum())
        assert float(reward.budget(u)[0]) == pytest.approx((c1 + c2) / 2)

    def test_the_gold_reward_is_zero_sum_and_a_rally_gives_it_back(self):
        st = line_army()
        u = st.u
        H = st.N // 2
        w = reward.Weights(win=1.0, gold=1.0, lord=0.0)
        no = torch.tensor([False])
        before = reward.measure(st)
        u["hp_abs"][0, H + 1] = u["hp0"][0, H + 1] * 0.6                              # an enemy spearman
        u["r"][0, 1] = True                                                           # an own spearman routs
        after = reward.measure(st)
        r = reward.step(before, after, no, torch.tensor([0]), torch.tensor([1]), w)
        bud = float(reward.budget(u)[0])
        want = (0.4 * float(u["cost"][0, H + 1]) - 0.5 * float(u["cost"][0, 1])) / bud
        assert r[0].tolist() == pytest.approx([want, -want], rel=1e-4)
        assert float(r.sum()) == pytest.approx(0.0, abs=1e-7)
        u["r"][0, 1] = False                                                          # it rallies
        back = reward.step(after, reward.measure(st), no, torch.tensor([0]), torch.tensor([1]), w)
        assert float(back[0, 0]) == pytest.approx(0.5 * float(u["cost"][0, 1]) / bud, rel=1e-4)
        # the hp and standing terms are off by default: gold counts those losses once
        assert reward.Weights().hp == 0.0 and reward.Weights().standing == 0.0

    def test_before_its_first_damage_the_attackers_idle_cost_is_exponential_in_time_and_capped(self):
        st = two_units()
        w = reward.Weights(idle=0.01, idle_tau_s=100.0, idle_cap=5.0)
        costs = []
        for t in (0.0, 10.0, 50.0, 100.0, 150.0, 1000.0):
            st.t[:] = t
            costs.append(reward.idle_cost(st, w)[0].tolist())
        att = [c[0] for c in costs]
        assert att == pytest.approx([0.0, 0.01 * (np.exp(0.1) - 1), 0.01 * (np.exp(0.5) - 1), 0.01 * (np.e - 1),
                                     0.01 * (np.exp(1.5) - 1), 0.05], rel=1e-5)          # capped at x5
        assert att[1] < 0.002 and all(a < b for a, b in zip(att, att[1:]))                # ~0 early, then grows
        assert all(c[1] == 0.0 for c in costs)
        # the defaults: almost nothing for the first minutes, the cap by 10 minutes
        d = reward.Weights()
        assert float(reward.idle_scale(torch.tensor([60.0]), torch.tensor([-1.0]), d)) < 0.5
        assert float(reward.idle_scale(torch.tensor([180.0]), torch.tensor([-1.0]), d)) < 2.5
        assert float(reward.idle_scale(torch.tensor([600.0]), torch.tensor([-1.0]), d)) == pytest.approx(d.idle_cap)

    def test_after_damage_the_idle_cost_is_zero_then_steps_up_every_pause_and_new_damage_resets_it(self):
        st = two_units()
        w = reward.Weights(idle=0.01, idle_tau_s=100.0, idle_pause_s=30.0, idle_step=0.5, idle_cap=20.0)
        hit = torch.tensor([100.0])
        cost = []
        for t in (100.0, 110.0, 129.9, 130.0, 159.9, 160.0, 190.0, 1000.0):
            st.t[:] = t
            cost.append(float(reward.idle_cost(st, w, hit)[0, 0]))
        assert cost == pytest.approx([0.0, 0.0, 0.0, 0.01 * (np.exp(0.5) - 1), 0.01 * (np.exp(0.5) - 1),
                                      0.01 * (np.e - 1), 0.01 * (np.exp(1.5) - 1), 0.2], rel=1e-5)
        st.t[:] = 1000.0
        assert reward.idle_cost(st, w, torch.tensor([1000.0]))[0].tolist() == [0.0, 0.0]   # new damage: 0 again
        assert reward.idle_cost(st, w, torch.tensor([980.0]))[0].tolist() == [0.0, 0.0]    # within the pause
        # the defaults: faster than the first curve at the same seconds, without its grace
        d = reward.Weights()
        for s in (30.0, 60.0, 90.0, 120.0, 180.0):
            t = torch.tensor([200.0 + s])
            after = float(reward.idle_scale(t, torch.tensor([200.0]), d))
            first = float(reward.idle_scale(torch.tensor([s]), torch.tensor([-1.0]), d))
            assert after > first
        assert float(reward.idle_scale(torch.tensor([200.0 + 30 * 20]), torch.tensor([200.0]), d)) == d.idle_cap

    def test_a_marching_attacker_pays_the_idle_cost(self):
        st = two_units()                                                              # side 1 at x -300 attacks
        w = reward.Weights(idle=0.01, idle_tau_s=100.0)
        st.t[:] = 100.0
        standing = reward.idle_cost(st, w)[0].tolist()
        st.u["vx"][0, 0] = 1.5                                                        # walking at the enemy
        assert reward.idle_cost(st, w)[0].tolist() == standing == pytest.approx([0.01 * (np.e - 1), 0.0])
        st = line_army(attacker=1)
        P = load()
        for _ in range(20):                                                           # 10 s at the enemy 600 m away
            battle.step(st, opponents.nearest(st), P)
        own = st.u["side"][0] == 1
        assert float(st.u["vx"][0][own].mean()) > 0.5 and not bool((st.u["m"] | st.u["fire"]).any())   # marching
        w = reward.Weights(idle=0.01, idle_tau_s=10.0)
        assert reward.idle_cost(st, w)[0].tolist() == pytest.approx([0.01 * (np.exp(float(st.t[0]) / 10) - 1), 0.0])

    def test_the_defender_never_pays_the_idle_cost(self):
        w = reward.Weights(idle=0.01, idle_tau_s=100.0)
        for attacker in (1, 2):
            st = two_units()
            H = st.N // 2
            st.attacker[:] = attacker
            d = 2 - attacker                                                          # the defender's column
            slot = 0 if d == 0 else H
            for t, vx, hit in ((10.0, 0.0, -1.0), (1000.0, 0.0, -1.0), (1000.0, 1.5, -1.0), (1000.0, -1.5, 900.0)):
                st.t[:] = t
                st.u["vx"][0, slot] = vx if d == 0 else -vx
                cost = reward.idle_cost(st, w, torch.tensor([hit]))[0]
                assert float(cost[d]) == 0.0
                assert float(cost[1 - d]) > 0.0                                       # the attacker stands: it pays

    def test_the_idle_attacker_pays_until_it_fights_or_shoots(self):
        st = two_units()
        st.t[:] = 100.0
        w = reward.Weights(idle=0.01, idle_tau_s=100.0)
        assert reward.idle_cost(st, w)[0].tolist() == pytest.approx([0.01 * (np.e - 1), 0.0])
        st.u["fire"][0, 0] = True
        assert reward.idle_cost(st, w)[0].tolist() == [0.0, 0.0]

    def test_late_in_the_battle_there_is_no_cost_beside_the_idle_cost(self):
        st = two_units()
        w = reward.Weights(idle=0.01, idle_tau_s=100.0, idle_cap=3.0)
        assert not hasattr(w, "tempo")
        st.t[:] = 1000.0
        st.u["fire"][0, 0] = True
        assert reward.idle_cost(st, w)[0].tolist() == [0.0, 0.0]                          # busy: nothing
        st.u["fire"][0, 0] = False
        assert reward.idle_cost(st, w)[0].tolist() == pytest.approx([0.03, 0.0])          # capped at x3
        assert reward.idle_cost(st, w, torch.tensor([1000.0]))[0].tolist() == [0.0, 0.0]  # damage: nothing

    def test_the_reward_terms_add_up_to_the_step(self):
        st = line_army(attacker=1)
        H = st.N // 2
        before = reward.measure(st)
        st.u["hp_abs"][0, H + 1] *= 0.5
        st.u["men"][0, 0] = 0                                                         # side 1's lord dies
        after = reward.measure(st)
        for fin, win in ((False, 0), (True, 2)):
            args = (before, after, torch.tensor([fin]), torch.tensor([win]), torch.tensor([1]))
            p = reward.parts(*args)
            assert set(p) == {"trade", "lord", "end"} and set(reward.PARTS) >= set(p)
            assert torch.allclose(p["trade"] + p["lord"] + p["end"], reward.step(*args))
            assert float(p["lord"][0, 0]) == pytest.approx(-reward.Weights.lord)
            assert float(p["trade"][0, 0]) < 0                     # its lord (all of it) is worth more than half a unit
        assert reward.parts(*args)["end"][0].tolist() == [-1.5, 1.0]                   # both stand: the time limit

    def test_with_idle_share_one_skirmisher_no_longer_stops_the_idle_cost(self):
        st = line_army(attacker=1)
        st.t[:] = 100.0
        st.u["fire"][0, 3] = True                                                     # side 1's archers shoot
        u = st.u
        cost = u["cost"].clamp(min=1.0)
        mine = u["side"][0] == 1
        idle_share = 1 - float(cost[0, 3] / cost[0][mine].sum())
        m = np.e - 1
        assert reward.idle_cost(st, reward.Weights(idle=0.01, idle_tau_s=100.0))[0].tolist() == [0.0, 0.0]   # the old rule
        w = reward.Weights(idle=0.01, idle_tau_s=100.0, idle_share=1.0)
        assert reward.idle_cost(st, w)[0].tolist() == pytest.approx([0.01 * m * idle_share, 0.0])
        half = dataclasses.replace(w, idle_share=0.5)
        assert reward.idle_cost(st, half)[0].tolist() == pytest.approx([0.01 * m * 0.5 * idle_share, 0.0])
        u["m"][0, :3] = mine[:3]                                                      # the rest of the army in melee
        assert reward.idle_cost(st, w)[0].tolist() == pytest.approx([0.0, 0.0], abs=1e-9)
        st.u["fire"][0] = False
        st.u["m"][0] = False
        assert reward.idle_cost(st, w)[0].tolist() == pytest.approx([0.01 * m, 0.0])   # nobody busy: all of it

    def test_the_damage_rate_is_the_defenders_gold_lost_a_minute_averaged(self):
        st = line_army(attacker=1)
        H = st.N // 2
        before = reward.measure(st)
        st.u["hp_abs"][0, H + 1] *= 0.5                                              # a defender's unit loses half
        after = reward.measure(st)
        share = float(reward.damage_share(before, after, torch.tensor([1]))[0])
        u = st.u
        assert share == pytest.approx(0.5 * float(u["cost"][0, H + 1]) / float(reward.budget(u)[0]), rel=1e-4)
        assert float(reward.damage_share(before, after, torch.tensor([2]))[0]) == 0.0   # attacker 2 took it
        assert float(reward.damage_share(after, before, torch.tensor([1]))[0]) == 0.0   # a rally is not damage
        rate = reward.hit_rate(torch.zeros(1), before, after, torch.tensor([1]), 0.5, 30.0)
        assert float(rate[0]) == pytest.approx((1 - np.exp(-0.5 / 30)) * share * 120, rel=1e-4)
        steady = torch.zeros(1)
        for _ in range(600):                                                          # the same damage every step
            steady = reward.hit_rate(steady, before, after, torch.tensor([1]), 0.5, 30.0)
        assert float(steady[0]) == pytest.approx(share * 120, rel=1e-3)               # a share of the budget a minute

    def test_struck_is_the_defenders_health_falling(self):
        st = line_army(attacker=1)
        H = st.N // 2
        one, two = torch.tensor([1]), torch.tensor([2])
        before = reward.measure(st)
        st.u["hp_abs"][0, 1] *= 0.9                                                   # side 1 loses health
        assert not bool(reward.struck(before, reward.measure(st), one)[0])            # its own loss as the attacker
        assert bool(reward.struck(before, reward.measure(st), two)[0])                # damage dealt by attacker 2
        before = reward.measure(st)
        st.u["hp_abs"][0, H + 1] *= 0.999                                             # side 2 loses a little
        assert bool(reward.struck(before, reward.measure(st), one)[0])
        assert not bool(reward.struck(before, reward.measure(st), two)[0])

    def test_order_changes(self):
        st = two_units()
        u = st.u
        u["order_kind"][0, 0] = O.MOVE
        u["ox"][0, 0], u["oz"][0, 0] = 0.0, 0.0
        o = O.hold(1, 2)
        o.kind[0, 0] = O.MOVE
        o.x[0, 0] = 5.0                                   # the same point within 10 m
        assert not reward.order_changes(u, o)[0, 0]
        o.x[0, 0] = 50.0
        assert reward.order_changes(u, o)[0, 0]
        o.kind[0, 0] = O.KEEP
        assert not reward.order_changes(u, o)[0, 0]
        o.kind[0, 0], o.target[0, 0] = O.ATTACK, 1
        assert reward.order_changes(u, o)[0, 0]
        assert not reward.order_changes(u, o)[0, 1]       # side 2 holds and held

    def test_the_order_cost_is_per_unit_of_the_side(self):
        side = torch.tensor([[1, 1, 2, 0]])
        changes = torch.tensor([[True, True, True, True]])
        cost = reward.order_cost(changes, side, reward.Weights(order_change=0.1))
        assert cost[0].tolist() == pytest.approx([0.1, 0.1])

    def test_a_lords_death_is_paid_by_its_side_and_earned_by_the_other(self):
        w = reward.Weights(win=1.0, hp=0.0, standing=0.0, lord=0.3)
        before = torch.ones(1, 2, 3)
        after = before.clone()
        after[0, 0, 2] = 0.0                                               # side 1's lord died
        r = reward.step(before, after, torch.tensor([False]), torch.tensor([0]), torch.tensor([1]), w)
        assert r[0].tolist() == pytest.approx([-0.3, 0.3])
        st = line_army()
        assert reward.measure(st)[0, :, 2].tolist() == [1.0, 1.0]
        st.u["men"][0, 0] = 0
        assert reward.measure(st)[0, :, 2].tolist() == [0.0, 1.0]

    def test_with_lord_rout_a_shattered_lord_counts_as_dead_and_a_routing_one_in_part(self):
        w = reward.Weights(win=1.0, hp=0.0, standing=0.0, gold=0.0, lord=0.3, lord_rout=0.5)
        st = line_army()
        assert reward.measure(st).shape[-1] == 4                                    # off: the old columns
        no, none = torch.tensor([False]), torch.tensor([0])
        before = reward.measure(st, lord_rout=0.5)
        assert before[0, :, 4].tolist() == [1.0, 1.0]
        st.u["r"][0, 0] = True                                                      # side 1's lord routs
        routing = reward.measure(st, lord_rout=0.5)
        assert routing[0, :, 4].tolist() == [0.5, 1.0] and routing[0, :, 2].tolist() == [1.0, 1.0]
        r = reward.step(before, routing, no, none, torch.tensor([1]), w)
        assert r[0].tolist() == pytest.approx([-0.15, 0.15])
        st.u["r"][0, 0] = False                                                     # it rallies: given back
        rallied = reward.measure(st, lord_rout=0.5)
        assert reward.step(routing, rallied, no, none, torch.tensor([1]), w)[0].tolist() == pytest.approx([0.15, -0.15])
        st.u["r"][0, 0] = st.u["s"][0, 0] = True                                    # it shatters: a whole death
        shattered = reward.measure(st, lord_rout=0.5)
        assert shattered[0, :, 4].tolist() == [0.0, 1.0] and shattered[0, :, 2].tolist() == [1.0, 1.0]
        assert reward.step(rallied, shattered, no, none, torch.tensor([1]), w)[0].tolist() == pytest.approx([-0.3, 0.3])
        st.u["men"][0, 0] = 0                                                       # then dies: nothing more
        assert reward.step(shattered, reward.measure(st, lord_rout=0.5), no, none, torch.tensor([1]),
                           w)[0].tolist() == pytest.approx([0.0, 0.0])

    def test_switching_an_attack_while_the_old_target_stands_is_a_retarget(self):
        st = line_army()
        u = st.u
        H = st.N // 2
        u["order_kind"][0, 1], u["order_target"][0, 1] = O.ATTACK, H + 1
        o = O.hold(1, st.N)
        o.kind[0, 1], o.target[0, 1] = O.ATTACK, H + 2
        assert reward.retargets(u, o)[0, 1] and int(reward.retargets(u, o).sum()) == 1
        o.target[0, 1] = H + 1
        assert not reward.retargets(u, o).any()                           # the same target again
        o.target[0, 1] = H + 2
        u["r"][0, H + 1] = True
        assert not reward.retargets(u, o).any()                           # the old one routs: free to switch
        cost = reward.order_cost(torch.zeros(1, st.N, dtype=torch.bool), u["side"], reward.Weights(retarget=0.4),
                                 torch.tensor([[False, True] + [False] * (st.N - 2)]))
        assert cost[0].tolist() == pytest.approx([0.1, 0.0])               # 0.4 / 4 units

    def test_schedules_go_linearly_from_start_to_end(self):
        from tools.nn.train import run
        assert run.schedule(0.01, 0.0, 0.0) == 0.01 and run.schedule(0.01, 0.0, 0.5) == pytest.approx(0.005)
        assert run.schedule(0.01, 0.0, 2.0) == 0.0

    def test_softening_the_kind_keeps_the_choice_and_raises_the_entropy(self):
        from tools.nn.train import run
        actor, _ = nets()
        with torch.no_grad():
            actor.heads.kind.weight.mul_(50)                                   # a collapsed head
        env = rollout.Battles(league.layout(2, 1, opponent="hold"), MIRROR)
        a, _, _ = env.observe(critic=False)
        x = rollout.rows_of(a, env.rows_learn)
        with torch.no_grad():
            before = actor(x)[0]["kind"]
            run.soften_kind(actor, 4.0)
            after = actor(x)[0]["kind"]
        ctrl = x["ctrl"]
        assert torch.equal(before.argmax(-1)[ctrl], after.argmax(-1)[ctrl])
        assert float(ppo.kind_entropy({"kind": after})[ctrl].mean()) > float(ppo.kind_entropy({"kind": before})[ctrl].mean())

    def test_a_point_turns_into_the_bin_that_points_back_at_it(self):
        env = rollout.Battles(league.layout(2, 1, opponent="hold"), MIRROR, spread=randomise.NONE)
        a, frame, _ = env.observe(critic=False)
        obs_t = rollout.rows_of(a, env.rows_learn)
        fr = rollout.frame_rows(frame, env.rows_learn)
        cfg = config.SMALL
        for want in (5, 37, 100, 127):
            bins = torch.full_like(obs_t["own"], want, dtype=torch.long)
            act = heads.Action(bins * 0, bins, bins * 0 - 1, torch.zeros_like(obs_t["own"]))
            pt = heads.point_world(cfg, act, obs_t, fr, torch.tensor([[-1e4, 1e4, -1e4, 1e4]] * 2))
            got = imitate.point_bin(cfg, pt[..., 0], pt[..., 1], obs_t, fr)
            assert (got[obs_t["own"]] == want).all()

    def test_ai_like_is_in_the_league_in_both_roles(self):
        lay = league.layout(48, 2, {"ai_like": 0.5, "nearest": 0.5}, scene_attacker=[1, 2])
        sel = lay.opponent == league.CODE["ai_like"]
        assert sel.sum() == 24 and set(lay.learner[sel]) == {1, 2}
        assert "ai_like" in evaluate.OPPONENTS

    def test_hold_is_met_only_as_the_defender(self):
        lay = league.layout(48, 2, {"hold": 0.5, "nearest": 0.5}, scene_attacker=[1, 2])
        hold = lay.opponent == league.CODE["hold"]
        assert (lay.learner[hold] == np.array([1, 2])[lay.scene[hold]]).all()
        assert set(lay.learner[~hold]) == {1, 2}

    def test_the_layout_meets_every_opponent_in_every_scene_from_both_sides(self):
        lay = league.layout(240, 6, {"self": 0.25, "nearest": 0.25, "hold": 0.5})
        assert lay.B == 240
        for code in (league.CODE["self"], league.CODE["nearest"], league.CODE["hold"]):
            sel = lay.opponent == code
            combos = {(s, sd) for s, sd in zip(lay.scene[sel], lay.learner[sel])}
            assert len(combos) == 12
        ctrl = lay.controllers()
        both = lay.opponent == league.CODE["self"]
        assert (ctrl[both] == league.LEARNER).all()
        assert ((ctrl[~both] == league.LEARNER).sum(1) == 1).all()

    def test_randomise_moves_numbers_within_the_spread_only_where_asked(self):
        st = scenes.build([0, 0], MIRROR)[0]
        before = {k: st.u[k].clone() for k in ("damage", "run", "leadership", "x")}
        rows = torch.tensor([True, False])
        f = randomise.apply(st, rows, randomise.Spread(0.2, 0.05, 3.0), torch.Generator().manual_seed(0))
        for g in f.values():
            assert ((g[0] >= 0.8 * 0.95 - 1e-6) & (g[0] <= 1.2 * 1.05 + 1e-6)).all() and (g[1] == 1).all()
        for k, v in before.items():
            assert torch.equal(st.u[k][1], v[1])
            assert not torch.equal(st.u[k][0], v[0])
        assert torch.equal(st.u["morale"][0], st.u["leadership"][0])


def pile():
    """Side 1: four spearmen (slots 0-3) and archers (4); side 2: two spearmen (5, 6). Slots 0-2 attack
    enemy 5; enemy 6 strikes unit 3 from behind; nothing else fights."""
    spear, archers = "wh_main_emp_inf_spearmen_0", "wh2_dlc13_emp_inf_archers_0"
    army = {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": spear, "x": -20.0 * i, "z": 0, "b": 90} for i in range(4)]
            + [{"key": archers, "x": -100, "z": 0, "b": 90}]},
        2: {"faction": "wh_main_emp_empire", "units": [{"key": spear, "x": 30, "z": 0, "b": 270},
                                                        {"key": spear, "x": -60, "z": 0, "b": 90}]}}}
    st = scenario.build([army])
    u = st.u
    for i in range(3):
        u["order_kind"][0, i], u["order_target"][0, i] = O.ATTACK, 5
    # enemy 6 at x -60 faces east, unit 3 at x -60 faces east too: 6 strikes 3 in the rear
    u["x"][0, 6] = -66.0
    u["m"][0, 3] = u["m"][0, 6] = True
    u["target"][0, 6], u["target"][0, 3] = 3, 6
    u["flank_hit"][0, 3] = 2.0
    return st


class TestBehaviour:
    def test_facts_see_a_rear_attack_a_flanked_unit_and_a_pile(self):
        from tools.nn.train import behaviour
        st = pile()
        f = behaviour.facts(st, load())
        assert bool(f["flank_attack"][0, 6]) and not bool(f["flank_attack"][0, 3])
        assert bool(f["flanked"][0, 3]) and not bool(f["flanked"][0, 0])
        assert f["crowded"][0, :3].tolist() == pytest.approx([1 / 3] * 3)          # 3 on one enemy: 1 too many
        assert float(f["crowded"][0, 3:].abs().sum()) == 0.0
        st.u["m"][0, 6] = False                                                     # no flanker: no pile
        assert float(behaviour.facts(st, load())["crowded"].sum()) == 0.0

    def test_a_melee_unit_standing_by_while_a_fellow_fights_is_idle(self):
        from tools.nn.train import behaviour
        st = pile()
        assert not bool(behaviour.facts(st, load())["idle_near"].any())   # attack orders, the archers, too far
        st.u["order_kind"][0, 2] = O.HOLD                                  # unit 2 (x -40) by unit 3 (x -60) in melee
        f = behaviour.facts(st, load())
        assert f["idle_near"][0].nonzero().flatten().tolist() == [2]

    def test_liveliness_counts_changes_switches_flips_jitter_and_the_units_own_targets(self):
        from tools.nn.train import behaviour
        st = pile()                          # units 0-2 attack enemy 5; unit 3 fights enemy 6 (target 6)
        u = st.u
        watch = behaviour.Tracker(st, load(), u["side"] == 1)
        live = torch.ones(1, dtype=torch.bool)

        def step(t, orders):
            st.t[:] = t
            for (slot, field), v in orders.items():
                u[field][0, slot] = v
            watch.update(st, live)
        step(0.5, {(0, "order_target"): 6, (1, "order_kind"): O.MOVE, (1, "ox"): 0.0, (1, "oz"): 100.0})
        step(1.0, {(0, "order_target"): 5, (1, "oz"): 103.0})     # 0: back to 5 (a flip); 1: 3 m, not given
        step(1.5, {(1, "oz"): 120.0})                              # 17 m on: a change and a re-point
        u["men"][0, 5] = 0.0                                         # enemy 5 dies: the simulator holds 0 and 2
        step(2.0, {(0, "order_kind"): O.HOLD, (0, "order_target"): -1, (2, "order_kind"): O.HOLD,
                     (2, "order_target"): -1, (3, "target"): 5})
        s = {k: float(v[0]) for k, v in watch.sums.items()}
        assert s["changes"] == 4 and s["switches"] == 2 and s["flips"] == 1     # the deaths' holds are no change
        assert s["repoints"] == 1 and s["repoint_m"] == pytest.approx(17.0)
        assert s["eng_switches"] == 1                               # unit 3: 6 -> 5 between whole seconds
        m = watch.summary(np.array([True]))
        assert m["move_jitter_m"] == pytest.approx(17.0)
        assert m["order_changes_per_min"] == pytest.approx(4 / s["unit_s"] * 60)
        assert m["twitch_share"] is None                             # nobody 30 s out of melee yet

    def test_a_missile_unit_in_melee_is_seen(self):
        from tools.nn.train import behaviour
        st = pile()
        st.u["m"][0, 4] = True
        f = behaviour.facts(st, load())
        assert bool(f["missile_melee"][0, 4]) and not bool(f["missile_melee"][0, 3])

    def test_the_unit_reward_counts_its_own_losses_and_the_shaped_terms(self):
        from tools.nn.train import behaviour
        st = pile()
        p = load()
        before = reward.unit_before(st.u)
        st.u["hp_abs"][0, 3] -= 100.0
        f = behaviour.facts(st, p)
        w = reward.Weights(unit_gold=0.05, flanked=0.01, missile_melee=0.0, crowd=0.02, flank_attack=0.03, neighbour=0.0)
        r = reward.unit_step(before, st, f, p, w)
        lost = 100.0 * float(st.u["cost"][0, 3] / st.u["hp0"][0, 3])                  # gold of 100 HP
        assert float(r[0, 3]) == pytest.approx(-0.05 * 5 * lost / float(reward.budget(st.u)[0]) - 0.01, rel=1e-4)
        assert r[0, :3].tolist() == pytest.approx([-0.02 / 3] * 3)
        # enemy 6 strikes unit 3: the gold unit 3 lost is what 6 destroyed (reward.attributed), + its flank bonus
        assert float(r[0, 6]) == pytest.approx(0.03 + 0.05 * 2 * lost / float(reward.budget(st.u)[0]), rel=1e-4)
        assert float(r[0, 4]) == 0.0
        near = reward.unit_step(before, st, f, p, dataclasses.replace(w, neighbour=1.0, neighbour_m=25.0))
        # unit 1 (x -20) has units 0 (x 0) and 2 (x -40) within 25 m
        assert float(near[0, 1]) == pytest.approx(float(r[0, 1]) + (float(r[0, 0]) + float(r[0, 2])) / 2)

    def test_the_enemys_gold_loss_is_split_among_the_units_that_engage_it(self):
        st = pile()
        p = load()
        u = st.u
        for i in (0, 1):                                                  # 0 and 1 fight enemy 5, 2 only marches
            u["m"][0, i], u["target"][0, i] = True, 5
        before = reward.unit_before(u)
        u["dealt"][0, 0] += 30.0                                          # 0 struck 30 HP, 1 struck 10
        u["dealt"][0, 1] += 10.0
        u["hp_abs"][0, 5] -= 40.0
        u["r"][0, 5] = True                                               # ... and enemy 5 routs
        gold = float((reward.gold_lost(u) - before["gold"])[0, 5])          # its health and the rout's share
        hp_gold = 40.0 * float(u["cost"][0, 5] / u["hp0"][0, 5])
        assert gold > 1.5 * hp_gold
        d = reward.attributed(before, u, p)
        fade = math.exp(-p.dt / p.sim["morale"]["recent_s"])
        w0, w1 = 30.0 + (1 - fade) * float(before["dealt"][0, 0]), 10.0 + (1 - fade) * float(before["dealt"][0, 1])
        assert float(d[0, 0] + d[0, 1]) == pytest.approx(gold, rel=1e-4)       # the whole loss, the rout too
        assert float(d[0, 0]) / float(d[0, 1]) == pytest.approx(w0 / w1, rel=1e-3)
        assert float(d[0, 2]) == 0.0
        # the enemy dies in the step: the simulator drops it from `target`; the target before the step counts
        before = reward.unit_before(u)
        u["men"][0, 5] = 0.0
        u["target"][0, 0] = u["target"][0, 1] = -1
        u["m"][0, 0] = u["m"][0, 1] = False
        d = reward.attributed(before, u, p)
        left = float((reward.gold_lost(u) - before["gold"])[0, 5])
        assert left > 0 and float(d[0, 0] + d[0, 1]) == pytest.approx(left, rel=1e-4)
        w = reward.Weights(unit_gold=0.05, flanked=0.0, missile_melee=0.0, crowd=0.0, flank_attack=0.0, neighbour=0.0)
        from tools.nn.train import behaviour
        f = behaviour.facts(st, p)
        new = reward.unit_step(before, st, f, p, w)
        old = reward.unit_step(before, st, f, p, dataclasses.replace(w, unit_attrib=0.0))
        assert float(new[0, 0]) > 0 and float(old[0, 0]) == 0.0              # the old estimate missed the kill

    def test_a_melee_unit_not_fighting_nor_closing_while_its_side_fights_shirks(self):
        st = pile()                                                       # 3 and 6 fight; 0-2, 5 stand near
        u = st.u
        u["mv"][0, 1] = True                                              # 1 moves (to the fight)
        u["vx"][0, 1] = 3.0                                               # ... east, on its target 5 (not the nearest)
        sh = reward.shirking(u)
        assert sh[0, :7].tolist() == [True, False, True, False, False, True, False] and not bool(sh[0, 7:].any())
        # walking about is not a way out: 0 walks sideways (the old rule, shirk_close 0, excused any move)
        u["mv"][0, 0], u["vz"][0, 0] = True, 3.0
        assert bool(reward.shirking(u)[0, 0]) and not bool(reward.shirking(u, close_mps=0.0)[0, 0])
        u["vx"][0, 0] = 2.0                                               # ... now also towards 5, its nearest enemy
        assert not bool(reward.shirking(u)[0, 0])
        u["vx"][0, 1] = -3.0                                              # 1 turns from its target 5 ...
        assert not bool(reward.shirking(u)[0, 1])                         # ... towards 6, its nearest: still closing
        u["vz"][0, 1], u["vx"][0, 1] = 3.0, 0.0                           # ... sideways: shirks
        assert bool(reward.shirking(u)[0, 1])
        u["vx"][0, 0] = u["vz"][0, 0] = u["vz"][0, 1] = 0.0
        u["mv"][0, 0] = False
        u["vx"][0, 1] = 3.0
        assert not bool(reward.shirking(u, reach_m=5.0)[0, 0])            # no enemy within reach
        u["m"][0, 3] = u["m"][0, 6] = False                               # nobody fights: nobody shirks
        assert not bool(reward.shirking(u).any())
        u["m"][0, 3] = u["m"][0, 6] = True
        p = load()
        before = reward.unit_before(u)
        from tools.nn.train import behaviour
        f = {k: torch.zeros_like(v) for k, v in behaviour.facts(st, p).items()}
        w = reward.Weights(unit_gold=0.0, neighbour=0.0, shirk=0.01)
        r = reward.unit_step(before, st, f, p, w)
        assert r[0, :7].tolist() == pytest.approx([-0.01, 0.0, -0.01, 0.0, 0.0, -0.01, 0.0])
        # the side's twin (shirk_side): either side pays x its shirking share of the standing army, by cost
        c = u["cost"][0]
        cost = reward.idle_cost(st, reward.Weights(idle=0.0, shirk_side=0.1))[0]
        assert cost.tolist() == pytest.approx([0.1 * float((c[0] + c[2]) / c[:5].sum()), 0.1 * float(c[5] / c[5:7].sum())],
                                              rel=1e-4)
        assert reward.idle_cost(st, reward.Weights(idle=0.0))[0].abs().sum() == 0

    def test_friendly_fire_is_the_shooters_loss_not_the_victims(self):
        st = pile()
        p = load()
        before = reward.unit_before(st.u)
        side = st.u["side"][0]
        shooter = next(i for i in range(st.N) if i != 3 and int(side[i]) == int(side[3]))
        st.u["hp_abs"][0, 3] -= 100.0
        gold = 100.0 * float(st.u["cost"][0, 3] / st.u["hp0"][0, 3])
        st.u["ff_dealt"][0, shooter], st.u["ff_taken"][0, 3] = gold, gold
        from tools.nn.train import behaviour
        f = {k: torch.zeros_like(v) for k, v in behaviour.facts(st, p).items()}
        w = reward.Weights(unit_gold=0.05, neighbour=0.0)
        n = float((side == side[3]).sum())
        r = reward.unit_step(before, st, f, p, w)
        charge = -0.05 * n * gold / float(reward.budget(st.u)[0])
        assert float(r[0, 3]) == pytest.approx(0.0, abs=1e-7) and float(r[0, shooter]) == pytest.approx(charge, rel=1e-4)
        off = reward.unit_step(before, st, f, p, dataclasses.replace(w, friendly_fire=0.0))
        assert float(off[0, 3]) == pytest.approx(charge, rel=1e-4) and float(off[0, shooter]) == 0.0

    def test_the_unit_idle_term_charges_the_attackers_standing_units_by_its_idle_multiplier(self):
        from tools.nn.train import behaviour
        st = line_army(attacker=1)
        p = load()
        before = reward.unit_before(st.u)
        st.u["fire"][0, 3] = True                                                     # side 1's archers shoot
        f = behaviour.facts(st, p)
        w = reward.Weights(unit_gold=0.0, flanked=0.0, missile_melee=0.0, crowd=0.0, flank_attack=0.0, neighbour=0.0,
                           unit_idle=0.01)
        off = reward.unit_step(before, st, f, p, w)                                     # no multiplier: no term
        assert off.abs().sum() == 0
        r = reward.unit_step(before, st, f, p, w, idle_m=torch.tensor([2.0]))
        side = st.u["side"][0]
        attacker_units = ((side == 1) & (st.u["men"][0] > 0)).nonzero().flatten().tolist()
        for i in attacker_units:
            assert float(r[0, i]) == pytest.approx(0.0 if i == 3 else -0.02)          # busy: nothing; standing: 0.01 x 2
        assert r[0][side == 2].abs().sum() == 0                                         # the defender never pays

    def test_a_lord_in_melee_with_little_health_pays_lord_exposed(self):
        from tools.nn.train import behaviour
        st = line_army()
        p = load()
        u = st.u
        lords = u["lord"][0].nonzero().flatten().tolist()
        before = reward.unit_before(u)
        w = reward.Weights(unit_gold=0.0, flanked=0.0, missile_melee=0.0, crowd=0.0, flank_attack=0.0, neighbour=0.0,
                           lord_exposed=0.01, lord_exposed_hp=0.5)
        for i in lords:
            u["m"][0, i] = True
        u["hp_abs"][0, lords[0]] = 0.4 * u["hp0"][0, lords[0]]                       # side 1's lord: 40 % health
        before["hp_abs"] = u["hp_abs"].clone()
        r = reward.unit_step(before, st, behaviour.facts(st, p), p, w)
        assert float(r[0, lords[0]]) == pytest.approx(-0.01)
        assert float(r[0, lords[1]]) == 0.0                                            # full health: fights freely
        assert r[0].abs().sum() == pytest.approx(0.01)
        off = reward.unit_step(before, st, behaviour.facts(st, p), p, dataclasses.replace(w, lord_exposed=0.0))
        assert off.abs().sum() == 0

    def test_a_lord_ahead_of_its_line_near_the_enemy_pays_lord_lead(self):
        from tools.nn.train import behaviour
        # side 1's lord 35 m ahead of its spearmen (x -60 -> -25), the enemy's 20 m behind its own
        st = line_army(gap=120.0, lord_ahead=55.0)
        p = load()
        u = st.u
        lords = u["lord"][0].nonzero().flatten().tolist()
        side2_lord = [i for i in lords if int(u["side"][0, i]) == 2][0]
        u["x"][0, side2_lord] = 80.0                                                 # back to 20 m behind
        lead = reward.lord_lead(u, margin_m=10.0, near_m=100.0)
        own = [i for i in lords if int(u["side"][0, i]) == 1][0]
        # the rest's centre (spearmen at -60, archers at -100, by cost) is behind -60: the lead is > 35 m
        assert float(lead[0, own]) == pytest.approx(1.0)
        assert float(lead[0, side2_lord]) == 0.0
        assert lead[0][~u["lord"][0]].abs().sum() == 0                              # only lords
        assert float(reward.lord_lead(u, margin_m=10.0, near_m=5.0).abs().sum()) == 0.0   # no enemy near
        before = reward.unit_before(u)
        w = reward.Weights(unit_gold=0.0, flanked=0.0, missile_melee=0.0, crowd=0.0, flank_attack=0.0, neighbour=0.0,
                           lord_lead=0.01, lord_lead_m=10.0, lord_lead_near=100.0)
        r = reward.unit_step(before, st, behaviour.facts(st, p), p, w)
        assert float(r[0, own]) == pytest.approx(-0.01)
        assert r[0].abs().sum() == pytest.approx(0.01)
        # a lead between the margin and twice it pays a share: put the lord 15 m ahead of the rest's centre
        rest = (u["side"][0] == 1) & ~u["lord"][0]
        c = float((u["x"][0] * u["cost"][0] * rest).sum() / (u["cost"][0] * rest).sum())
        u["x"][0, own] = c + 15.0
        u["z"][0, own] = 0.0
        assert float(reward.lord_lead(u, 10.0, 200.0)[0, own]) == pytest.approx(0.5, abs=1e-3)
        off = reward.unit_step(before, st, behaviour.facts(st, p), p, dataclasses.replace(w, lord_lead=0.0))
        assert off.abs().sum() == 0

    def test_a_lord_that_routs_or_falls_pays_lord_fall_and_a_rally_gives_it_back(self):
        from tools.nn.train import behaviour
        st = line_army()
        p = load()
        u = st.u
        lords = u["lord"][0].nonzero().flatten().tolist()
        own = [i for i in lords if int(u["side"][0, i]) == 1][0]
        w = reward.Weights(unit_gold=0.0, flanked=0.0, missile_melee=0.0, crowd=0.0, flank_attack=0.0, neighbour=0.0,
                           lord_fall=0.2, lord_rout=0.5)
        before = reward.unit_before(u)
        u["r"][0, own] = True                                                        # standing -> routing: 0.5
        r = reward.unit_step(before, st, behaviour.facts(st, p), p, w)
        assert float(r[0, own]) == pytest.approx(-0.1)
        assert r[0].abs().sum() == pytest.approx(0.1)                                # only the lord
        before = reward.unit_before(u)
        u["s"][0, own] = True                                                        # routing -> shattered: 0.5 more
        assert float(reward.unit_step(before, st, behaviour.facts(st, p), p, w)[0, own]) == pytest.approx(-0.1)
        u["s"][0, own] = False
        before = reward.unit_before(u)
        u["r"][0, own] = False                                                       # a rally gives it back
        assert float(reward.unit_step(before, st, behaviour.facts(st, p), p, w)[0, own]) == pytest.approx(0.1)
        before = reward.unit_before(u)
        u["men"][0, own] = 0                                                         # standing -> dead: the whole
        assert float(reward.unit_step(before, st, behaviour.facts(st, p), p, w)[0, own]) == pytest.approx(-0.2)
        off = reward.unit_step(before, st, behaviour.facts(st, p), p, dataclasses.replace(w, lord_fall=0.0))
        assert float(off[0, own]) == 0.0
        spear = [i for i in range(u["side"].shape[1]) if int(u["side"][0, i]) == 1 and not bool(u["lord"][0, i])][0]
        before = reward.unit_before(u)
        u["r"][0, spear] = True                                                      # another unit's rout: not this term
        assert float(reward.unit_step(before, st, behaviour.facts(st, p), p, w)[0, spear]) == 0.0

    def test_gae_per_unit_stops_at_the_end_of_a_battle(self):
        r = torch.tensor([[[1.0, 2.0]], [[0.0, 0.0]], [[2.0, 1.0]]])                # [T, R, N]
        done = torch.tensor([[False], [True], [False]])
        _, ret = ppo.gae(r, torch.zeros(3, 1, 2), done, torch.tensor([[10.0, 0.0]]), gamma=0.5, lam=1.0)
        assert ret[:, 0, 0].tolist() == pytest.approx([1.0, 0.0, 7.0])
        assert ret[:, 0, 1].tolist() == pytest.approx([2.0, 0.0, 1.0])

    def test_per_unit_advantages_reach_the_policy_loss(self):
        loss, _ = ppo.policy_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([[1.0, -1.0]]),
                                  torch.tensor([[True, True]]), 0.2)
        assert float(loss) == pytest.approx(0.0)

    def test_the_critic_loads_a_checkpoint_from_before_the_per_unit_head(self):
        _, c = nets()
        state = {k: v for k, v in c.state_dict().items() if not k.startswith("unit_value.")}
        fresh = critic.Critic(CFG).load(state)
        assert float(fresh.unit_value[2].weight.detach().abs().sum()) == 0.0
        with pytest.raises(RuntimeError):
            critic.Critic(CFG).load({k: v for k, v in state.items() if not k.startswith("value.")})


# --- level 2: properties ---

class TestProperties:
    def test_keep_goes_on_with_the_order_in_force_and_holds_without_one(self):
        st = two_units()
        P = load()
        o = O.hold(1, 2)
        o.kind[0, 0], o.x[0, 0], o.run[0, 0] = O.MOVE, -100.0, True
        battle.step(st, o, P)
        x1 = float(st.u["x"][0, 0])
        keep = O.hold(1, 2)
        keep.kind[:] = O.KEEP
        for _ in range(4):
            battle.step(st, keep, P)
        assert int(st.u["order_kind"][0, 0]) == O.MOVE and float(st.u["ox"][0, 0]) == -100.0
        assert float(st.u["x"][0, 0]) > x1 + 3                          # still going
        assert int(st.u["order_kind"][0, 1]) == O.HOLD                   # never had an order: holds
        assert float(st.u["x"][0, 1]) == pytest.approx(300.0)

    def test_the_memory_through_a_chunk_is_the_same_as_step_by_step(self):
        actor, crit = nets()
        env = rollout.Battles(league.layout(4, 1, opponent="nearest"), MIRROR, params=rollout.params_with_limit(3.0))
        batch = rollout.collect(env, actor, crit, 8)          # a decision a second: battles end at decisions 3, 6
        assert batch["reset"][3].all() and batch["reset"][6].all()
        assert not batch["reset"][[0, 1, 2, 4, 5, 7]].any()
        with torch.no_grad():
            obs = rollout.full_obs(batch["obs"], batch["abil_static"])
            logits, _ = actor.sequence(obs, batch["h0"], batch["reset"])
            lp, _ = heads.log_prob(logits, batch["action"], batch["obs"]["ctrl"])
        assert torch.allclose(lp, batch["lp"], atol=1e-4)

    def test_generated_battles_restart_as_new_armies_and_the_setup_follows(self):
        src = scenes.Generated(range(500, 516), 3, rollout.params_with_limit(1.0), "cpu", seed=1)
        lay = league.layout(8, 1, {"nearest": 0.5, "hold": 0.5}, scene_attacker=[1])
        env = rollout.Battles(lay, MIRROR, params=rollout.params_with_limit(1.0), source=src,
                              spread=randomise.NONE)
        actor, crit = nets()
        before = env.bank_row.clone()
        for _ in range(2):
            env.step(actor, crit)                                        # the 1 s limit: all restart
        rows = env.bank_row
        assert not torch.equal(rows, before)
        bank = src.bank
        assert torch.equal(env.setup.arrays.passport, bank.setup.arrays.passport[rows])
        assert torch.equal(env.setup.char[1], bank.setup.char[1][rows])
        assert torch.equal(env.st.u["side"], bank.state.u["side"][rows])
        hold = torch.as_tensor(lay.opponent == league.CODE["hold"])
        assert torch.equal(env.st.attacker[hold], torch.as_tensor(lay.learner)[hold])   # the learner attacks `hold`

    def test_ai_like_attacker_advances_in_line_and_the_defender_waits_with_its_lord_behind(self):
        st = line_army(attacker=1, gap=600)
        H = st.N // 2
        o = opponents.ai_like(st)
        assert (o.kind[0, 1:3] == O.MOVE).all() and o.run[0, 1:3].all()          # attacker's spearmen advance
        assert (o.x[0, 1:3] > st.u["x"][0, 1:3]).all()                          # towards the enemy (east)
        assert (o.kind[0, H + 1:H + 3] == O.HOLD).all()                          # the defender's line holds
        assert int(o.kind[0, H]) in (O.HOLD, O.MOVE) and int(o.kind[0, 0]) in (O.HOLD, O.MOVE)
        P = load()
        for _ in range(60):                                                       # 30 s
            battle.step(st, opponents.ai_like(st), P)
        lead = st.u["x"][0, 1:3].mean()
        assert float(lead) > -300 + 60                                            # it advanced
        assert float(st.u["x"][0, 0]) < float(lead)                               # the lord behind its line

    def test_ai_like_counter_charges_close_enemies_and_shoots_the_lord_in_range(self):
        st = line_army(attacker=1, gap=80, lord_ahead=30)        # spearmen 80 m apart, the lord 110 m from the archers
        H = st.N // 2
        o = opponents.ai_like(st)
        assert (o.kind[0, H + 1:H + 3] == O.ATTACK).all() and o.run[0, H + 1:H + 3].all()   # counter-charge
        assert int(o.kind[0, H + 3]) == O.ATTACK and int(o.target[0, H + 3]) == 0        # archers: the lord
        assert int(o.kind[0, H]) != O.ATTACK or bool(o.kind[0, H + 1:H + 3].eq(O.ATTACK).any())

    def test_ai_like_lord_does_not_charge_alone(self):
        st = line_army(attacker=1, gap=600, lord_ahead=150)                      # side 1's lord far in front
        o = opponents.ai_like(st)
        assert int(o.kind[0, 0]) == O.MOVE and float(o.x[0, 0]) < float(st.u["x"][0, 0])    # back to the line

    def test_a_bank_keeps_a_share_of_small_armies(self):
        src = scenes.Generated(range(500, 520), 19, rollout.params_with_limit(1.0), "cpu", small=(0.5, 3))
        side = src.bank.state.u["side"]
        per = torch.stack([(side == s).sum(1) for s in (1, 2)], 1)          # units + lord per battle
        assert (per[:10] <= 4).all()
        assert src.bank.N == 40

    def test_the_policy_respects_the_masks(self):
        actor, _ = nets()
        env = rollout.Battles(league.layout(6, 1, opponent="hold"), MIRROR)
        a, frame, _ = env.observe(critic=False)
        rows = env.rows_learn
        for _ in range(5):
            obs_r, h, logits, action, orders, _ = env._act(actor, a, frame, rows, None, False)
            ctrl, ok = obs_r["ctrl"], obs_r["target_ok"]
            assert (action.kind[~ctrl] == O.HOLD).all()
            attack = action.kind == O.ATTACK
            assert ok.gather(1, action.target.clamp(min=0))[attack].all()
            lp, ent = heads.log_prob(logits, action, ctrl)
            assert (lp[~ctrl] == 0).all() and (ent[~ctrl] == 0).all()
            keep = action.kind == O.KEEP
            assert (orders.kind[keep] == O.KEEP).all() and (orders.target[keep] == -1).all()

    def test_rewards_are_symmetric_between_the_sides(self):
        """The mirror arena with the two armies swapped (and the roles): each side's reward
        is the other's of the unswapped battle."""
        env = rollout.Battles(league.layout(2, 1, opponent="nearest"), [("arena", "attack")],
                              spread=randomise.NONE)
        st = env.st
        a = st.clone()
        H = st.N // 2
        swap = torch.cat([torch.arange(H, 2 * H), torch.arange(0, H)])
        b = st.clone()
        for k, v in b.u.items():
            b.u[k] = v[:, swap]
        b.u["side"] = torch.where(b.u["side"] > 0, 3 - b.u["side"], b.u["side"])
        b.attacker = 3 - a.attacker
        P = load()
        for _ in range(400):
            ha, hb = reward.measure(a), reward.measure(b)
            oa = opponents.nearest(a)
            ob_ = O.Orders(*(getattr(oa, k)[:, swap] for k in O.FIELDS))
            ob_.target = torch.where(ob_.target >= 0, (ob_.target + H) % (2 * H), ob_.target)
            battle.step(a, oa, P)
            battle.step(b, ob_, P)
            even = reward.Weights(timeout=1.0)                             # zero-sum at the limit too
            ra = reward.step(ha, reward.measure(a), a.done, a.winner, a.attacker, even)
            rb = reward.step(hb, reward.measure(b), b.done, b.winner, b.attacker, even)
            assert torch.allclose(ra, rb.flip(1), atol=1e-4)
            assert torch.allclose(ra.sum(1), torch.zeros(2), atol=1e-6)
        assert float(reward.health(a)[0, 0]) < 1.0                        # the battle got going

    def test_battles_start_again_when_they_end(self):
        actor, crit = nets()
        env = rollout.Battles(league.layout(4, 1, opponent="nearest"), MIRROR, params=rollout.params_with_limit(1.0),
                              cadence=cad.STEP)                   # step level: a simulator step a call
        out = [env.step(actor, crit) for _ in range(3)]
        assert bool(out[1]["done"].all())                                 # the 1 s limit: 2 steps
        assert sorted(out[1]["reward"].tolist()) == pytest.approx([-1.5, -1.5, 1.0, 1.0], abs=0.01)
        assert float(env.st.t.max()) == pytest.approx(0.5)                # started again
        stats = env.take_stats()
        assert stats["nearest/attack"][0] == 2 and stats["nearest/defend"][0] == 2
        assert stats["nearest/attack"][1] == 0 and stats["nearest/defend"][1] == 2      # the defender wins on time


    def test_with_idle_rate_only_a_steady_damage_rate_counts_as_the_attackers_damage(self):
        actor, crit = nets()
        w = reward.Weights(idle_rate=0.05, idle_window_s=1.0)
        env = rollout.Battles(league.layout(2, 1, opponent="nearest"), MIRROR, weights=w, cadence=cad.STEP)

        def strike(b, share):
            u = env.st.u
            defender = u["side"][b] == 3 - env.st.attacker[b]
            u["hp_abs"][b] = torch.where(defender, u["hp_abs"][b] * (1 - share), u["hp_abs"][b])
        strike(0, 1e-4)                                                   # a scratch (the old rule: struck)
        strike(1, 0.05)                                                   # a real blow
        env.step(actor, crit)
        assert env.last_hit[0] == -1.0 and env.last_hit[1] == pytest.approx(0.5)
        assert float(env.hit_rate[1]) > float(env.hit_rate[0]) > 0

    @pytest.mark.parametrize("cadence", [cad.STEP, cad.Cadence(1.0, 0.0)])
    def test_the_logged_reward_terms_add_up_to_the_reward(self, cadence):
        """The terms a minute of battle (per simulator step) add up to the decisions' rewards (each the sum of
        its env.k steps')."""
        actor, crit = nets()
        env = rollout.Battles(league.layout(4, 1, opponent="nearest"), MIRROR, params=rollout.params_with_limit(1.0),
                              cadence=cadence)
        env.reward_parts()
        out = [env.step(actor, crit) for _ in range(3)]
        att = torch.stack([o["attacks"] for o in out])
        r = torch.stack([o["reward"] for o in out])
        parts = env.reward_parts()
        per_min = 60.0 / env.params.dt
        for role, m in (("attack", att), ("defend", ~att)):            # (a battle that ends starts again at once)
            total = sum(parts[role].values()) * float(m.sum()) * env.k / per_min
            assert total == pytest.approx(float(r[m].sum()), abs=1e-4)
        assert parts["attack"]["end"] < 0 < parts["defend"]["end"]        # the defender wins on time
        assert env.reward_parts() == {}                                   # reset

    def test_the_attackers_last_damage_is_kept_per_battle_and_cleared_when_it_starts_again(self):
        actor, crit = nets()
        env = rollout.Battles(league.layout(4, 1, opponent="nearest"), MIRROR, params=rollout.params_with_limit(1.0),
                              cadence=cad.STEP)                   # step level: a simulator step a call
        assert env.last_hit.tolist() == [-1.0] * env.B
        env.step(actor, crit)                                             # t 0.5: far apart, no damage
        assert env.last_hit.tolist() == [-1.0] * env.B
        u = env.st.u
        defender = u["side"] == (3 - env.st.attacker)[:, None]
        u["hp_abs"][0] = torch.where(defender[0], u["hp_abs"][0] * 0.99, u["hp_abs"][0])   # row 0: the attacker struck
        out = env.step(actor, crit)                                       # t 1.0: the limit, all end
        assert bool(out["done"].all())
        assert env.last_hit.tolist() == [-1.0] * env.B                    # cleared for the new battles
        u = env.st.u
        defender = u["side"] == (3 - env.st.attacker)[:, None]
        u["hp_abs"][1] = torch.where(defender[1], u["hp_abs"][1] * 0.99, u["hp_abs"][1])
        env.step(actor, crit)                                             # t 0.5 of the new battles
        assert env.last_hit.tolist() == [-1.0, 0.5] + [-1.0] * (env.B - 2)   # per row

    def test_each_side_sees_the_attackers_last_damage_as_the_reward_counts_it(self):
        """observation TIMERS: the attacker's row sees `dealt` = rollout.last_hit, the defender's row `taken`;
        per battle row and per side, cleared when a battle starts again."""
        from tools.nn.model import observation as ob
        actor, crit = nets()
        env = rollout.Battles(league.layout(4, 1, opponent="nearest"), MIRROR, params=rollout.params_with_limit(1.5),
                              cadence=cad.STEP)                   # step level: a simulator step a call

        def strike(b, side):          # what a simulator step's damage does to a unit of `side` in battle b
            u = env.st.u
            i = int(((u["side"][b] == side) & (u["men"][b] > 0)).nonzero()[0])
            u["hp_abs"][b, i] *= 0.99
            u["hp"][b, i] = u["hp_abs"][b, i] / u["hp0"][b, i]

        def seen():                   # [B, 2] per battle: (dealt_any, dealt_since, taken_any, taken_since) of sides 1, 2
            ctx = env.cur[0]["ctx"]
            cols = [ob.CTX[n] for n in ("dealt_any", "dealt_since", "taken_any", "taken_since")]
            return ctx[:env.B][:, cols], ctx[env.B:][:, cols]

        def agrees():
            s1, s2 = seen()
            att = env.st.attacker
            dealt = torch.where((att == 1)[:, None], s1, s2)[:, :2]
            taken = torch.where((att == 1)[:, None], s2, s1)[:, 2:]
            hit = env.last_hit >= 0
            since = torch.where(hit, ((env.st.t - env.last_hit) / ob.SINCE).clamp(0, 1), torch.zeros_like(env.last_hit))
            want = torch.stack([hit.float(), since], 1)
            assert torch.allclose(dealt, want, atol=1e-6) and torch.allclose(taken, want, atol=1e-6)
            return s1, s2
        env.step(actor, crit)                                             # t 0.5: no damage
        s1, s2 = agrees()
        assert float(s1.abs().sum() + s2.abs().sum()) == 0
        strike(0, int(3 - env.st.attacker[0]))                            # battle 0: the attacker strikes
        strike(1, int(env.st.attacker[1]))                                # battle 1: the defender strikes
        env.step(actor, crit)                                             # t 1.0
        s1, s2 = agrees()
        a1 = int(env.st.attacker[1])
        assert (s1, s2)[a1 - 1][1].tolist() == [0, 0, 1, 0]               # the defender's damage: seen, no reward clock
        assert float(env.last_hit[1]) == -1 and float(env.last_hit[0]) == 1.0
        env.step(actor, crit)                                             # t 1.5: the limit, all start again
        s1, s2 = agrees()
        assert float(s1.abs().sum() + s2.abs().sum()) == 0                # cleared for the new battles

    def test_both_sides_and_the_companion_see_the_attackers_progress_as_the_reward_clocks_it(self):
        """observation PROGRESS = the rollout's hit_rate / last_hit (idle_rate RATE_MIN) on both sides' rows,
        per battle, cleared when a battle starts again; the companion, given the same states as the game's
        rows (a unit gone from the map: no men), computes the same."""
        from tools.nn.companion import exchange
        from tools.nn.model import observation as ob
        actor, crit = nets()
        w = reward.Weights(idle_rate=ob.RATE_MIN, idle_window_s=ob.RATE_WINDOW, rout_share=ob.ROUT_SHARE)
        env = rollout.Battles(league.layout(4, 1, opponent="nearest"), MIRROR, params=rollout.params_with_limit(6.0),
                              weights=w, cadence=cad.STEP)        # step level: a simulator step a call
        B, cols = env.B, [ob.CTX[n] for n in ob.PROGRESS]

        def want():
            hit = env.last_hit >= 0
            since = torch.where(hit, ((env.st.t - env.last_hit) / ob.SINCE).clamp(0, 1), torch.zeros_like(env.last_hit))
            return torch.stack([(env.hit_rate / ob.RATE_MIN).clamp(0, ob.RATE_CAP), hit.float(), since], 1)

        def unit(b, side, k):         # the k-th unit of `side` in battle b
            return int((env.st.u["side"][b] == side).nonzero()[k])

        def strike(b, side, share, k=-1):
            u = env.st.u
            i = unit(b, side, k)
            u["hp_abs"][b, i] *= 1 - share
            u["hp"][b, i] = u["hp_abs"][b, i] / u["hp0"][b, i]

        def game_doc(b, batch):       # battle b as the game's state document (exchange.py)
            u, st = env.st.u, env.st
            rows = [{"n": f"u{i}", "side": int(u["side"][b, i]), "key": st.keys[b][i], "x": float(u["x"][b, i]),
                     "z": float(u["z"][b, i]), "men": 0.0 if bool(u["gone"][b, i]) else float(u["men"][b, i]),
                     "hp": float(u["hp"][b, i]), "ms": float(u["ms"][b, i]), "r": bool(u["r"][b, i]),
                     "s": bool(u["s"][b, i])} for i in range(st.N) if int(u["side"][b, i]) > 0]
            return {"batch": batch, "move": 0, "t": float(st.t[b]) * 1000, "attacker": int(st.attacker[b]),
                    "units": rows}

        game = {"batch": None, "memory": None, "n": 0}

        def companion():              # battle 0 seen by the companion -> its PROGRESS
            if game["batch"] is None or float(env.st.t[0]) == 0:
                game["n"] += 1
                game["batch"], game["memory"] = f"b{game['n']}", None
                game["battle"] = exchange.battle(game_doc(0, game["batch"]))
            g = game["battle"]
            state = exchange.arrays(game_doc(0, game["batch"]), g.names)
            obs, game["memory"] = ob.observe(state, g.setup, 1, game["memory"])
            return obs.ctx[0, cols]

        env.cur = env.observe(True)                                       # t 0: the first decision's state
        seen_game = companion()
        reached = set()
        for k in range(16):
            att = env.st.attacker
            if 1 <= k <= 4:
                strike(0, int(3 - att[0]), 0.03)                          # battle 0: steady blows
            if k == 6:
                env.st.u["r"][0, unit(0, int(3 - att[0]), -2)] = True    # ... a rout (half its rest lost)
            if k == 8:
                env.st.u["r"][0, unit(0, int(3 - att[0]), -2)] = False   # ... it rallies (no damage)
            if k == 9:
                i = unit(0, int(3 - att[0]), -3)
                env.st.u["r"][0, i], env.st.u["gone"][0, i] = True, True  # ... another leaves the map: whole
            strike(1, int(3 - att[1]), 1e-4)                              # battle 1: scratches only
            if k == 3:
                strike(1, int(att[1]), 0.3)                               # ... and the defender's blow
                strike(2, int(3 - att[2]), 0.3)                           # battle 2: one big blow, then a pause
            if k == 7:
                strike(3, int(3 - att[3]), 0.002, k=0)                    # battle 3: a small blow
            env.step(actor, crit)
            ctx = env.cur[0]["ctx"]
            assert torch.allclose(ctx[:B, cols], want(), atol=1e-5), k
            assert torch.allclose(ctx[B:, cols], want(), atol=1e-5), k
            seen_game = companion()
            assert np.allclose(seen_game, ctx[0, cols].numpy(), atol=1e-5), (k, seen_game, ctx[0, cols])
            reached |= {b for b in range(B) if float(env.last_hit[b]) >= 0}
            if k == 11:                                                   # t 6: the limit, all start again
                assert float(ctx[:, cols].abs().sum()) == 0 and env.last_hit.tolist() == [-1.0] * B
        assert reached == {0, 2}

# --- level 3: the loop ---

class TestAbilities:
    """The networks choose their lords' abilities; only scripted opponents fire by the game-AI rule."""

    def test_only_scripted_sides_fire_by_the_rule_also_after_a_restart(self):
        lay = league.layout(8, 1, {"self": 0.25, "past": 0.25, "nearest": 0.25, "ai_like": 0.25})
        env = rollout.Battles(lay, MIRROR, params=rollout.params_with_limit(1.0), spread=randomise.NONE)
        env.set_past(nets(1)[0])
        side = env.st.u["side"]

        def expected():
            script = env.ctrl >= league.CODE["nearest"]
            return ((side == 1) & script[:, :1]) | ((side == 2) & script[:, 1:])
        network = ((side == 1) & (env.ctrl[:, :1] <= league.CODE["past"])) | (
            (side == 2) & (env.ctrl[:, 1:] <= league.CODE["past"]))
        assert torch.equal(env.st.u["ai"], expected()) and not (env.st.u["ai"] & network).any()
        assert (env.st.u["ai"] & (side == 1)).any()               # a script on side 1 fires by the rule
        actor, crit = nets()
        for _ in range(3):
            env.step(actor, crit)                                       # the 1 s limit: all restart
        assert env.battles >= 8 and torch.equal(env.st.u["ai"], expected())

    def test_the_stored_input_is_small_and_full_obs_gives_back_the_observation(self):
        from tools.nn.model import abilities as mab
        env = rollout.Battles(league.layout(4, 1, opponent="nearest"), MIRROR, params=rollout.params_with_limit(3.0))
        actor, crit = nets()
        a, _, _ = env.observe()
        full = rollout.rows_of(a, env.rows_learn)
        step = env.step(actor, crit)
        assert step["obs"]["abil"].shape[-1] == mab.DYNAMIC and step["action"].ability is not None
        back = rollout.full_obs(step["obs"], env.bank.setup.arrays.abil)
        assert torch.equal(back["abil"], full["abil"]) and "abil_row" not in back
        assert torch.all((step["action"].ability == -1) | full["abil_ok"].gather(
            2, step["action"].ability.clamp(min=0)[..., None])[..., 0])

    def test_a_network_lord_uses_abilities_by_order_and_ppo_trains_the_head(self):
        lay = league.layout(4, 1, opponent="nearest")
        env = rollout.Battles(lay, MIRROR, params=rollout.params_with_limit(20.0), spread=randomise.NONE)
        actor, crit = nets()
        with torch.no_grad():
            actor.heads.ability_none.bias.fill_(-20.0)                 # always use one when ready
        before = actor.heads.ability_q.weight.detach().clone()
        batch = rollout.collect(env, actor, crit, 4)
        assert batch["action"].ability.shape == batch["action"].kind.shape and (batch["action"].ability >= 0).any()
        assert float(env.ability_stats[0]) > 0 and float(env.ability_stats[1]) == 0   # uses; no battle ended yet
        opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
        st = ppo.update(actor, crit, opt, batch, dataclasses.replace(ppo.PPOConfig(), epochs=1, minibatch=8))
        assert all(np.isfinite(v) for v in st.values())
        assert not torch.equal(before, actor.heads.ability_q.weight)


class TestLoop:
    def test_a_tiny_training_step(self):
        actor, crit = nets()
        before = [p.detach().clone() for p in actor.parameters()]
        lay = league.layout(6, 1, {"self": 0.34, "past": 0.33, "hold_shoot": 0.33})
        env = rollout.Battles(lay, MIRROR)
        env.set_past(nets(1)[0], untrained=True)
        batch = rollout.collect(env, actor, crit, 3)
        assert batch["obs"]["tokens"].shape[:2] == (3, env.R)
        assert batch["lp"].shape == (3, env.R, env.N)
        assert batch["unit_reward"].shape == batch["unit_value"].shape == (3, env.R, env.N)
        assert batch["last_unit_value"].shape == (env.R, env.N)
        opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
        st = ppo.update(actor, crit, opt, batch, dataclasses.replace(ppo.PPOConfig(), epochs=2, minibatch=6,
                                                                     unit_credit=0.3))
        assert all(np.isfinite(v) for v in st.values()) and "unit_reward" in st
        assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters()))

    def test_a_huge_critic_loss_does_not_shrink_the_actors_step(self):
        # 02.10: actor and critic were clipped together, the per-unit value's gradient (norm ~1000) scaled
        # the actor's to ~1e-8 a parameter, below Adam's eps: the policy took no step at all
        import copy
        actor, crit = nets()
        env = rollout.Battles(league.layout(6, 1, {"self": 0.5, "nearest": 0.5}), MIRROR)
        batch = rollout.collect(env, actor, crit, 3)
        steps = []
        for weight in (1e-3, 1e6):
            a, c = copy.deepcopy(actor), copy.deepcopy(crit)
            opt = torch.optim.Adam(list(a.parameters()) + list(c.parameters()), lr=1e-3, eps=1e-5)
            torch.manual_seed(0)
            st = ppo.update(a, c, opt, batch, dataclasses.replace(ppo.PPOConfig(), epochs=1, minibatch=6,
                                                                  unit_credit=0.3, unit_value=weight))
            assert "grad_norm" in st and "grad_norm_critic" in st
            steps.append([p.detach() - q.detach() for p, q in zip(a.parameters(), actor.parameters())])
        assert max(float(x.abs().max()) for x in steps[0]) > 1e-5
        assert all(torch.allclose(x, y, atol=1e-7) for x, y in zip(*steps))

    @pytest.mark.skipif(not (checkpoint.DIR / "test5/t0_gold30/m20.pt").exists(), reason="no m20.pt (build/ is not in Git)")
    def test_training_continues_from_a_checkpoint_saved_before_the_damage_timers(self):
        from tools.nn.train import run
        actor, crit = run.networks("small", "cpu", checkpoint.DIR / "test5/t0_gold30/m20.pt")
        assert torch.all(actor.encoder.ctx[0].weight[:, -len(rollout.ob.TIMERS) - len(rollout.ob.PROGRESS):] == 0)
        env = rollout.Battles(league.layout(2, 1, opponent="nearest"), MIRROR)
        batch = rollout.collect(env, actor, crit, 2)
        opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-4)
        st = ppo.update(actor, crit, opt, batch, dataclasses.replace(ppo.PPOConfig(), epochs=1, minibatch=4))
        assert all(np.isfinite(v) for v in st.values())

    def test_a_run_with_its_own_reference_role_advantages_and_the_new_idle_rule(self, tmp_path, monkeypatch):
        from tools.nn.train import run
        actor, crit = nets()
        init = checkpoint.save(tmp_path / "init.pt", actor, crit)
        bare = checkpoint.save(tmp_path / "bare.pt", actor)                    # no critic (a test5 m<minute>.pt)
        monkeypatch.setattr(checkpoint, "DIR", tmp_path)
        monkeypatch.setattr(checkpoint, "RANDOM", init)
        a, c = run.networks("small", "cpu", bare, critic_from=init)
        assert all(torch.equal(x, y) for x, y in zip(c.state_dict().values(), crit.state_dict().values()))
        run.networks("small", "cpu", bare)                                      # a fresh critic, no failure
        args = run.parser().parse_args(["--name", "t", "--init", str(bare), "--critic-init", str(init), "--battles", "4",
                                        "--steps", "2", "--updates", "3", "--minutes", "5", "--device", "cpu", "--no-eval",
                                        "--mix", '{"self": 0.5, "nearest": 0.5}', "--anchor", "0.05", "--reference", "self",
                                        "--reference-every", "2", "--adv-norm", "role", "--idle-share", "1",
                                        "--idle-rate", "0.05", "--print-every", "1"])
        trained, summary, out = run.train(args)
        assert summary["updates"] == 3
        rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert "reward_parts" in rows[-1] and "ev" in rows[-1] and rows[-1]["anchor_weight"] == 0.05
        assert set(rows[-1]["reward_parts"]) <= {"attack", "defend"}

    def test_a_rolling_anchor_moves_the_reference_and_the_log_keeps_the_distance_from_the_start(self, tmp_path,
                                                                                                    monkeypatch):
        from tools.nn.train import run, test5
        actor, crit = nets()
        init = checkpoint.save(tmp_path / "init.pt", actor, crit)
        monkeypatch.setattr(checkpoint, "DIR", tmp_path)
        monkeypatch.setattr(checkpoint, "RANDOM", init)
        args = run.parser().parse_args(["--name", "roll", "--init", str(init), "--battles", "4", "--steps", "2",
                                        "--updates", "3", "--minutes", "5", "--device", "cpu", "--no-eval",
                                        "--mix", '{"nearest": 1.0}', "--anchor", "0.05", "--anchor-roll", "1e-9",
                                        "--lr", "1e-2", "--print-every", "1"])
        trained, summary, out = run.train(args)
        rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["anchor_rolls"] for r in rows] == [1, 2, 3]            # renewed after every update
        assert rows[0]["start_kl"] >= 0 and rows[-1]["start_kl"] > 0     # the actor moved off its start
        assert ppo.distance(trained, trained, rollout.collect(rollout.Battles(league.layout(2, 1, opponent="nearest"),
                                                                              MIRROR), trained, crit, 2), 4) ==             pytest.approx(0.0, abs=1e-6)
        d = test5.distance(out, 1, 3)
        assert d["updates"] == [2, 3] and d["rolls"] == 3 and d["start_kl_last"] == round(rows[-1]["start_kl"], 4)
        assert test5.distance(out, 3, 9) is None and test5.distance(tmp_path / "none", 0, 3) is None
        args = run.parser().parse_args(["--name", "fixed", "--init", str(init), "--battles", "4", "--steps", "2",
                                        "--updates", "2", "--minutes", "5", "--device", "cpu", "--no-eval",
                                        "--mix", '{"nearest": 1.0}', "--anchor", "0.05"])
        _, _, out = run.train(args)
        assert [json.loads(x)["anchor_rolls"] for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()] == [0, 0]

    def test_per_role_normalisation_gives_each_role_mean_0_std_1(self):
        a = torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
        g = torch.tensor([[True, False]] * 3)
        n = ppo.normalise(a, g)
        for m in (g, ~g):
            assert float(n[m].mean()) == pytest.approx(0.0, abs=1e-6) and float(n[m].std()) == pytest.approx(1.0)
        assert torch.allclose(ppo.normalise(a), (a - a.mean()) / (a.std() + 1e-8))

    def test_behaviour_cloning_learns_the_teachers_targets(self):
        actor, _ = nets()
        res = imitate.train(actor, minutes=5, battles=4, device="cpu", lr=3e-3, max_steps=40, scene_list=MIRROR,
                            log=lambda m: None)
        assert res["steps"] == 40 and res["kind_acc"] > 0.9

    def test_behaviour_cloning_of_two_teachers_learns_moves_too(self):
        actor, _ = nets()
        res = imitate.train(actor, minutes=5, battles=4, device="cpu", lr=3e-3, max_steps=6, scene_list=MIRROR,
                            teacher="nearest,ai_like", log=lambda m: None)
        assert res["steps"] == 6 and np.isfinite(res["loss"])

    def test_a_checkpoint_rebuilds_the_same_policy(self, tmp_path):
        actor, crit = nets()
        path = checkpoint.save(tmp_path / "x.pt", actor, crit, "small", {"update": 3})
        again = checkpoint.load_policy(path)
        assert again.cfg == actor.cfg
        data = checkpoint.read(path)
        assert data["kinds"] == list(O.KINDS) and data["meta"]["update"] == 3
        env = rollout.Battles(league.layout(2, 1, opponent="hold"), MIRROR)
        a, _, _ = env.observe(critic=False)
        x = rollout.rows_of(a, env.rows_learn)
        with torch.no_grad():
            assert torch.allclose(actor(x)[0]["kind"], again(x)[0]["kind"])
        assert checkpoint.load_critic(path) is not None

    def test_a_recorded_battle_reads_like_a_game_recording(self, tmp_path):
        actor, _ = nets()
        paths = evaluate.record(actor, tmp_path, opponents=("hold",), limit_s=3.0, scene_list=MIRROR)
        assert len(paths) == 2
        b = gamedata.load(paths[0])
        assert b.f["x"].shape[1] == 14 and len(b.t) >= 3
        assert b.winner in (1, 2) and set(b.side) == {1, 2}

    def test_evaluation_counts_every_battle(self):
        actor, _ = nets()
        res = evaluate.play(actor, opponents=("hold", "past"), per_scene=2, past=nets(1)[0], limit_s=2.0,
                            scene_list=MIRROR)
        hold, past = res["by_opponent"]["hold"], res["by_opponent"]["past"]
        assert hold["games"] == 2 and hold["roles"]["attack"]["games"] == 2      # `hold` only defends
        assert hold["win_rate"] == 0.0                                    # the attacker loses on time
        assert past["roles"]["attack"]["games"] == 1 and past["roles"]["defend"]["wins"] == 1
        assert past["timeouts"] == 1.0
        assert sum(res["kinds"].values()) == pytest.approx(1.0)

    def test_evaluation_of_several_opponents_in_one_batch_counts_them_apart(self):
        actor, _ = nets()
        res = evaluate.play(actor, opponents=("hold", "past"), per_scene=2, past=nets(1)[0], limit_s=2.0,
                            scene_list=MIRROR, together=True)
        hold, past = res["by_opponent"]["hold"], res["by_opponent"]["past"]
        assert hold["games"] == 2 and hold["roles"]["attack"]["games"] == 2
        assert past["roles"]["attack"]["games"] == 1 and past["roles"]["defend"]["games"] == 1
        for o in (hold, past):
            assert sum(o["kinds"].values()) == pytest.approx(1.0)
            assert set(o["behaviour"]) >= {"missile_melee_s", "flanked_share", "crowding_share", "flank_attack_share",
                                           "abilities_per_battle", "order_changes_per_min", "flips_per_min",
                                           "engine_switches_per_min", "opp_engine_switches_per_min"}
            assert "behaviour" in o["roles"]["attack"]
            for x in (o, o["roles"]["attack"]):
                assert x["gold_lost"] >= 0 and x["gold_destroyed"] >= 0 and x["gold_ratio"] >= 0
                assert -1.5 <= x["gold_trade"] <= 1.5


class TestProtocol:
    def test_the_trend_has_a_liveliness_block_per_opponent_and_role(self):
        from tools.nn.train import test5
        point = {"ai_like/attack": {"win": 0.5, "order_changes_per_min": 6.0, "flips_per_min": 0.5},
                 "ai_like/defend": {"win": 0.4, "order_changes_per_min": 4.0}}
        lines = test5.trend([(0, point), ("5", dict(point, **{"ai_like/defend": {"win": 0.4}}))])
        assert "| order changes / unit-min, ai_like | 6.00 / 4.00 | 6.00 / - |" in lines
        assert "| flips A→B→A ≤10 s / unit-min, ai_like | 0.500 / - | 0.500 / - |" in lines
        assert not test5.lively_block([{"ai_like/attack": {"win": 0.5}}], ["min 0"])     # an older evaluation

    def test_the_trend_shows_the_distance_from_the_start_next_to_the_rating(self):
        from tools.nn.train import test5
        sk = (lambda v: {"rating": {"overall": {"value": v, "se": 0.05}}})
        p0 = {"skill": sk(-0.4)}
        p1 = {"skill": sk(-0.1), "distance": {"start_kl": 0.012, "start_kl_last": 0.02, "anchor_kl": 0.008, "rolls": 1}}
        lines = test5.trend([(0, p0), ("10", p1)])
        assert "| rating, overall (logit) | -0.40 | -0.10 |" in lines
        assert any(x.startswith("| distance from start: KL to the init network") and x.endswith("| 0 | 0.012 / 0.020 |")
                   for x in lines)
        assert "| reference renewals (--anchor-roll), so far | - | 1 |" in lines
        assert not test5.distance_block([(0, p0), ("10", {"skill": sk(0.0)})])        # older evaluations: no block
        assert test5.metrics({"by_opponent": {}, "distance": p1["distance"]})["distance"] == p1["distance"]

    def test_the_gpu_lock_is_held_while_the_test_runs_and_freed_after_a_failure(self, tmp_path):
        from tools.nn.train import test5
        lock = tmp_path / "gpu-train.lock"
        with test5.gpu_lock("t", path=lock, poll_s=0.01):
            assert lock.read_text(encoding="utf-8").startswith("t ")
        assert not lock.exists()
        with pytest.raises(ValueError):
            with test5.gpu_lock("t", path=lock, poll_s=0.01):
                raise ValueError
        assert not lock.exists()

    def test_the_gpu_lock_waits_for_the_other_holder(self, tmp_path):
        import threading
        import time
        from tools.nn.train import test5
        lock = tmp_path / "gpu-train.lock"
        lock.write_text("other", encoding="utf-8")
        threading.Timer(0.2, lock.unlink).start()
        t = time.time()
        with test5.gpu_lock("t", path=lock, poll_s=0.02):
            assert time.time() - t >= 0.15

    def test_a_job_started_right_after_a_release_lets_the_queue_go_first(self, tmp_path):
        import time
        from tools.nn.train import test5
        lock = tmp_path / "gpu-train.lock"
        with test5.gpu_lock("first", path=lock, poll_s=0.01, gap_s=0.3):
            pass
        assert lock.with_suffix(".released").read_text(encoding="utf-8").split()[1] == "first"
        t = time.time()
        with test5.gpu_lock("second", path=lock, poll_s=0.01, gap_s=0.3):
            assert time.time() - t >= 0.2

    def test_the_report_table_shows_before_and_after(self):
        from tools.nn.train import test5
        before = {"ai_like/attack": {"win": 0.5, "flanked_share": 0.3}, "ai_like/all": {"kind_attack": 0.5}}
        after = {"ai_like/attack": {"win": 0.6, "flanked_share": 0.2}, "ai_like/all": {"kind_attack": 0.4}}
        text = "\n".join(test5.table(before, after))
        assert "0.500 → 0.600" in text and "0.300 → 0.200" in text and "0.50 → 0.40" in text

    def test_the_trend_table_has_a_column_per_minute(self):
        from tools.nn.train import test5

        def m(w, g):
            return {"ai_like/attack": {"win": w, "gold_ratio": g}, "ai_like/defend": {"win": w + 0.1},
                    "ai_like/all": {"kind_hold": 0.2}}
        text = "\n".join(test5.trend([(0, m(0.5, 1.0)), (10, m(0.6, 1.25))]))
        assert "min 0 | min 10" in text
        assert "| win rate, ai_like | 0.500 / 0.600 | 0.600 / 0.700 |" in text
        assert "| gold exchange ratio, ai_like | 1.00 / - | 1.25 / - |" in text

    def test_the_report_and_the_trend_have_a_block_by_faction(self):
        from tools.nn.train import matchups, test5

        def m(w):
            g = matchups.group([True, w > 0.5], ["wh_main_emp_empire", "wh2_main_skv_skaven"],
                               ["wh2_main_skv_skaven"] * 2, [True, False],
                               gold=(np.array([100.0, 200.0]), np.array([200.0, 100.0]), np.array([1000.0, 1000.0])))
            return {"ai_like/attack": {"win": w}, "ai_like/all": {"kind_hold": 0.2}, "ai_like/factions": g}
        text = "\n".join(test5.trend([(0, m(0.4)), (10, m(0.6))]))
        assert "| EMP win attack / defend (n 1/0), ai_like | 1.000 / - | 1.000 / - |" in text
        assert "| SKV-SKV win / gold ratio (n 1), ai_like | 0.000 / 0.50 | 1.000 / 0.50 |" in text
        table = "\n".join(test5.table(m(0.4), m(0.6)))
        assert "| SKV win attack / defend (n 0/1), ai_like | - / 0.000 → - / 1.000 |" in table
        assert "ai_like/factions" not in table
