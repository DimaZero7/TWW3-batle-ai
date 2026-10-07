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
from tools.nn.train import checkpoint, evaluate, league, opponents, ppo, randomise, reward, rollout, scenes  # noqa: E402
from tools.nn.train import cadence as cad  # noqa: E402

CFG = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)
MIRROR = [("arena", "attack")]
SMALL_RUN = ["--bank", "8", "--max-units", "4"]       # run.train's random armies, few and small (CPU)


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

    def test_the_reward_is_win_plus_gold_and_lord_differences(self):
        # measure() columns: health, standing, lord alive, 1 - gold lost / budget (health and standing: no term)
        w = reward.Weights(win=1.0, gold=1.0, lord=0.5)
        before = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], [[0.5, 0.5, 1.0, 0.6], [0.4, 0.3, 1.0, 0.5]]])
        after = torch.tensor([[[0.9, 1.0, 1.0, 0.9], [0.7, 0.8, 0.0, 0.7]], [[0.5, 0.0, 1.0, 0.6], [0.4, 0.3, 1.0, 0.5]]])
        r = reward.step(before, after, torch.tensor([False, True]), torch.tensor([0, 2]), w)
        shaped = (0.3 - 0.1) + 0.5                                         # gold 0.3 destroyed, 0.1 lost; their lord
        assert r[0].tolist() == pytest.approx([shaped, -shaped])
        assert r[1].tolist() == pytest.approx([-1.0, 1.0])

    def test_at_the_time_limit_the_attacker_loses_as_in_a_lost_fight(self):
        # audit R4 (03.10): the defender wins at the limit (the simulator's winner), the attacker gets -win
        m = torch.ones(2, 2, 4)                                            # both sides still stand
        r = reward.step(m, m, torch.tensor([True, True]), torch.tensor([2, 1]), reward.Weights(win=1.0))
        assert r.tolist() == [[-1.0, 1.0], [1.0, -1.0]]

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

    def test_the_gold_reward_is_zero_sum_and_a_rally_gives_nothing_back(self):
        st = line_army()
        u = st.u
        H = st.N // 2
        w = reward.Weights(win=1.0, gold=1.0, lord=0.0)
        no = torch.tensor([False])
        before = reward.measure(st)
        u["hp_abs"][0, H + 1] = u["hp0"][0, H + 1] * 0.6                              # an enemy spearman
        u["r"][0, 1] = True                                                           # an own spearman routs
        u["lost_worst"] = reward.track(u)                                             # (after every simulator step)
        after = reward.measure(st)
        r = reward.step(before, after, no, torch.tensor([0]), w)
        bud = float(reward.budget(u)[0])
        want = (0.4 * float(u["cost"][0, H + 1]) - 0.5 * float(u["cost"][0, 1])) / bud
        assert r[0].tolist() == pytest.approx([want, -want], rel=1e-4)
        assert float(r.sum()) == pytest.approx(0.0, abs=1e-7)
        u["r"][0, 1] = False                                                          # it rallies
        u["lost_worst"] = reward.track(u)
        back = reward.step(after, reward.measure(st), no, torch.tensor([0]), w)
        assert back[0].tolist() == pytest.approx([0.0, 0.0], abs=1e-7)                # a loss counts once (B3)

    def test_a_rout_a_rally_and_a_rout_again_count_once_in_the_trade_and_the_clock(self):
        # Audit B3 (03.10): a defender unit that routed, rallied and routed again was new damage each
        # time - the attacker's progress clock reset and the trade paid with no real progress.
        st = line_army(attacker=1)
        u = st.u
        H = st.N // 2
        w = reward.Weights(win=1.0, gold=1.0, lord=0.0)
        no, none, att = torch.tensor([False]), torch.tensor([0]), torch.tensor([1])
        bud = float(reward.budget(u)[0])
        foe, own = H + 1, 1                                                          # the defender's, the attacker's
        cf, co = float(u["cost"][0, foe]), float(u["cost"][0, own])

        def turn(set_):
            prev = reward.measure(st)
            for (k, i), v in set_.items():
                u[k][0, i] = v
            u["lost_worst"] = reward.track(u)
            now = reward.measure(st)
            # [side 1's reward, side 2's, the clock's damage share, the defender unit's gold lost]
            return reward.step(prev, now, no, none, w)[0].tolist() + [
                float(reward.damage_share(prev, now, att)[0]), float(reward.gold_lost(u)[0, foe])]

        assert turn({("r", foe): True}) == pytest.approx([0.5 * cf / bud, -0.5 * cf / bud, 0.5 * cf / bud, 0.5 * cf])
        assert turn({("r", foe): False}) == pytest.approx([0, 0, 0, 0.5 * cf], abs=1e-6)          # rallies: nothing
        assert turn({("r", foe): True}) == pytest.approx([0, 0, 0, 0.5 * cf], abs=1e-6)           # routs again: nothing
        # beyond its worst it counts: 60 % health lost while routing -> 0.6 + 0.4 x 0.5 = 0.8 of its cost
        assert turn({("hp_abs", foe): 0.4 * float(u["hp0"][0, foe])}) == pytest.approx(
            [0.3 * cf / bud, -0.3 * cf / bud, 0.3 * cf / bud, 0.8 * cf], rel=1e-4)
        assert turn({("r", foe): False})[2] == 0.0
        assert turn({("s", foe): True, ("r", foe): True}) == pytest.approx(                       # shattered: whole
            [0.2 * cf / bud, -0.2 * cf / bud, 0.2 * cf / bud, cf], rel=1e-4)
        # our side's losses: the same rule (the trade; the clock counts the defender's only)
        assert turn({("r", own): True})[:3] == pytest.approx([-0.5 * co / bud, 0.5 * co / bud, 0], rel=1e-4)
        assert turn({("r", own): False})[:3] == pytest.approx([0, 0, 0], abs=1e-7)
        assert turn({("r", own): True})[:3] == pytest.approx([0, 0, 0], abs=1e-7)

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

    def test_the_idle_cost_is_progress_only_a_busy_unit_does_not_stop_it(self):
        # audit R1 (03.10): only the clock (the attacker's damage) decides, whoever of its units is busy
        st = line_army(attacker=1)
        st.t[:] = 100.0
        w = reward.Weights(idle=0.01, idle_tau_s=100.0)
        idle = reward.idle_cost(st, w)[0].tolist()
        assert idle == pytest.approx([0.01 * (np.e - 1), 0.0])
        st.u["fire"][0, 3] = True                                                     # side 1's archers shoot
        st.u["m"][0, :3] = True                                                       # the rest fights
        assert reward.idle_cost(st, w)[0].tolist() == idle
        assert reward.idle_cost(st, w, torch.tensor([100.0]))[0].tolist() == [0.0, 0.0]  # damage now: nothing

    def test_late_in_the_battle_there_is_no_cost_beside_the_idle_cost(self):
        st = two_units()
        w = reward.Weights(idle=0.01, idle_tau_s=100.0, idle_cap=3.0)
        assert not hasattr(w, "tempo")
        st.t[:] = 1000.0
        assert reward.idle_cost(st, w)[0].tolist() == pytest.approx([0.03, 0.0])          # capped at x3
        assert reward.idle_cost(st, w, torch.tensor([1000.0]))[0].tolist() == [0.0, 0.0]  # damage: nothing
        st.done[:] = True
        assert reward.idle_cost(st, w)[0].tolist() == [0.0, 0.0]                          # the battle is over

    def test_the_reward_terms_add_up_to_the_step(self):
        st = line_army(attacker=1)
        H = st.N // 2
        before = reward.measure(st)
        st.u["hp_abs"][0, H + 1] *= 0.5
        st.u["men"][0, 0] = 0                                                         # side 1's lord dies
        after = reward.measure(st)
        for fin, win in ((False, 0), (True, 2)):
            args = (before, after, torch.tensor([fin]), torch.tensor([win]))
            p = reward.parts(*args)
            assert set(p) == {"trade", "lord", "end"} and set(reward.PARTS) >= set(p)
            assert torch.allclose(p["trade"] + p["lord"] + p["end"], reward.step(*args))
            assert float(p["lord"][0, 0]) == pytest.approx(-reward.Weights.lord)
            assert float(p["trade"][0, 0]) < 0                     # its lord (all of it) is worth more than half a unit
        assert reward.parts(*args)["end"][0].tolist() == [-1.0, 1.0]                   # both stand: the time limit

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
        w = reward.Weights(win=1.0, lord=0.3)
        before = torch.ones(1, 2, 3)
        after = before.clone()
        after[0, 0, 2] = 0.0                                               # side 1's lord died
        r = reward.step(before, after, torch.tensor([False]), torch.tensor([0]), w)
        assert r[0].tolist() == pytest.approx([-0.3, 0.3])
        st = line_army()
        assert reward.measure(st)[0, :, 2].tolist() == [1.0, 1.0]
        st.u["men"][0, 0] = 0
        assert reward.measure(st)[0, :, 2].tolist() == [0.0, 1.0]

    def test_with_lord_rout_a_shattered_lord_counts_as_dead_and_a_routing_one_in_part(self):
        w = reward.Weights(win=1.0, gold=0.0, lord=0.3, lord_rout=0.5)
        st = line_army()
        assert reward.measure(st).shape[-1] == 4                                    # off: the old columns
        no, none = torch.tensor([False]), torch.tensor([0])
        before = reward.measure(st, lord_rout=0.5)
        assert before[0, :, 4].tolist() == [1.0, 1.0]
        st.u["r"][0, 0] = True                                                      # side 1's lord routs
        routing = reward.measure(st, lord_rout=0.5)
        assert routing[0, :, 4].tolist() == [0.5, 1.0] and routing[0, :, 2].tolist() == [1.0, 1.0]
        r = reward.step(before, routing, no, none, w)
        assert r[0].tolist() == pytest.approx([-0.15, 0.15])
        st.u["r"][0, 0] = False                                                     # it rallies: given back
        rallied = reward.measure(st, lord_rout=0.5)
        assert reward.step(routing, rallied, no, none, w)[0].tolist() == pytest.approx([0.15, -0.15])
        st.u["r"][0, 0] = st.u["s"][0, 0] = True                                    # it shatters: a whole death
        shattered = reward.measure(st, lord_rout=0.5)
        assert shattered[0, :, 4].tolist() == [0.0, 1.0] and shattered[0, :, 2].tolist() == [1.0, 1.0]
        assert reward.step(rallied, shattered, no, none, w)[0].tolist() == pytest.approx([-0.3, 0.3])
        st.u["men"][0, 0] = 0                                                       # then dies: nothing more
        assert reward.step(shattered, reward.measure(st, lord_rout=0.5), no, none, w)[0].tolist() == pytest.approx(
            [0.0, 0.0])

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

    def test_defend_only_scripts_have_the_learner_attacking_in_training(self):
        from tools.nn.train import run
        assert league.names_list(run.parser().parse_args([]).defend_only) == league.ATTACK_ONLY   # default as before
        only = league.names_list(run.parser().parse_args(["--defend-only", "hold,hold_shoot"]).defend_only)
        assert only == ("hold", "hold_shoot")
        with pytest.raises(ValueError):
            league.names_list("hold,nope")
        mix = {"hold": 0.25, "hold_shoot": 0.25, "nearest": 0.5}
        lay = league.layout(48, 2, mix, scene_attacker=[1, 2], attack_only=only)
        sel = np.isin(lay.opponent, [league.CODE[n] for n in only])
        assert sel.sum() == 24 and (lay.learner[sel] == np.array([1, 2])[lay.scene[sel]]).all()
        assert set(lay.learner[~sel]) == {1, 2}
        env = rollout.Battles(lay, MIRROR * 2, params=rollout.params_with_limit(1.0), attack_only=only)
        want = env.want.cpu().numpy()
        assert (want[sel] == lay.learner[sel]).all() and (want[sel] != 0).all() and (want[~sel] == 0).all()
        env = rollout.Battles(lay, MIRROR * 2, params=rollout.params_with_limit(1.0))   # the evaluation's default
        shoot = lay.opponent == league.CODE["hold_shoot"]
        assert (env.want.cpu().numpy()[shoot] == 0).all()

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

    def test_the_critic_loads_a_checkpoint_with_the_old_per_unit_head(self):
        _, c = nets()
        state = dict(c.state_dict(), **{"unit_value.0.weight": torch.zeros(CFG.critic_d, 2 * CFG.critic_d),
                                        "unit_value.2.bias": torch.zeros(1)})
        fresh = critic.Critic(CFG).load(state)
        assert all(torch.equal(v, c.state_dict()[k]) for k, v in fresh.state_dict().items())
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

    def test_ai_like_missile_units_step_back_from_close_melee_and_break_off_melee_for_a_while(self):
        """The old trigger (skirmish_targeted False): any enemy melee unit within skirmish_m."""
        old = opponents.Line(skirmish_targeted=False)
        st = line_army(attacker=1, gap=600)
        H = st.N // 2
        u = st.u
        u["x"][0, H + 1] = u["x"][0, 3] + 15.0                  # an enemy spearman 15 m from side 1's archers
        u["z"][0, H + 1] = u["z"][0, 3]
        o = opponents.ai_like(st, old)
        assert int(o.kind[0, 3]) == O.WITHDRAW and bool(o.run[0, 3])
        assert float(o.x[0, 3]) < float(u["x"][0, 3]) - 30                  # away from it (west), not towards
        u["x"][0, H + 1] = u["x"][0, 3] + 30.0                  # 30 m: beyond skirmish_m (20), the archers shoot on
        assert int(opponents.ai_like(st, old).kind[0, 3]) != O.WITHDRAW
        assert int(opponents.ai_like(st, opponents.Line(skirmish_targeted=False, skirmish_m=50.0)).kind[0, 3]) == O.WITHDRAW
        u["x"][0, H + 1], u["z"][0, H + 1] = u["x"][0, 3] - 10.0, u["z"][0, 3] - 11.0   # behind them, to the south
        o = opponents.ai_like(st, old)                                     # away and back: north, a little west
        assert int(o.kind[0, 3]) == O.WITHDRAW and float(o.z[0, 3]) > float(u["z"][0, 3]) + 40
        assert float(o.x[0, 3]) < float(u["x"][0, 3])
        o = opponents.ai_like(st, opponents.Line(skirmish_targeted=False, flee_straight=True))   # straight: north-east
        assert float(o.x[0, 3]) > float(u["x"][0, 3]) + 25 and float(o.z[0, 3]) > float(u["z"][0, 3]) + 25
        u["x"][0, H + 1], u["z"][0, H + 1] = u["x"][0, 3] + 70.0, u["z"][0, 3]
        u["x"][0, H + 1] = u["x"][0, 3] + 8.0                   # caught in melee
        u["m"][0, 3] = u["m"][0, H + 1] = True
        u["target"][0, 3], u["target"][0, H + 1] = H + 1, 3
        u["contact_s"][0, 3] = 2.0
        assert int(opponents.ai_like(st).kind[0, 3]) == O.WITHDRAW          # breaks off
        u["contact_s"][0, 3] = opponents.Line().escape_s + 1.0
        assert int(opponents.ai_like(st).kind[0, 3]) == O.HOLD              # then fights

    def test_ai_like_shooters_hop_back_from_an_enemy_that_attacks_them_and_closes(self):
        """skirmish_targeted (the game AI's hops, build/missile/skirmish): an enemy melee unit ordered to attack the
        archers, closing, within targeted_m -> one WITHDRAW of hop_m at a run, kept until it arrives; the same unit
        attacking someone else, or standing -> none."""
        p = opponents.Line()
        st = line_army(attacker=1, gap=600)
        H = st.N // 2
        u = st.u
        u["x"][0, H + 1], u["z"][0, H + 1] = u["x"][0, 3] + 45.0, u["z"][0, 3]   # 45 m east of side 1's archers
        u["vx"][0, H + 1] = -2.0                                                # running at them
        u["order_kind"][0, H + 1], u["order_target"][0, H + 1] = O.ATTACK, 3
        o = opponents.ai_like(st, p)
        assert int(o.kind[0, 3]) == O.WITHDRAW and bool(o.run[0, 3])
        hop = math.hypot(float(o.x[0, 3] - u["x"][0, 3]), float(o.z[0, 3] - u["z"][0, 3]))
        assert hop == pytest.approx(p.hop_m, abs=0.01) and float(o.x[0, 3]) < float(u["x"][0, 3])   # away (west)
        u["order_target"][0, H + 1] = 2                                         # attacking a neighbour instead
        assert int(opponents.ai_like(st, p).kind[0, 3]) != O.WITHDRAW
        u["order_target"][0, H + 1], u["vx"][0, H + 1] = 3, 0.0                  # attacks them but stands
        assert int(opponents.ai_like(st, p).kind[0, 3]) != O.WITHDRAW
        u["vx"][0, H + 1] = -2.0
        u["x"][0, H + 1] = u["x"][0, 3] + 60.0                                   # beyond targeted_m (50)
        assert int(opponents.ai_like(st, p).kind[0, 3]) != O.WITHDRAW
        # a hop under way goes on to its point although the trigger is gone
        u["order_kind"][0, 3] = O.WITHDRAW
        u["ox"][0, 3], u["oz"][0, 3] = u["x"][0, 3] - 9.0, u["z"][0, 3]
        o = opponents.ai_like(st, p)
        assert int(o.kind[0, 3]) == O.WITHDRAW and float(o.x[0, 3]) == pytest.approx(float(u["ox"][0, 3]))

    def test_ai_like_melee_target_avoids_the_lord_and_its_random_part_is_fixed(self):
        st = line_army(attacker=1, gap=600)
        H = st.N // 2
        u = st.u
        u["x"][0, H + 1], u["z"][0, H + 1] = 0.0, 0.0           # side 2's spearman (defender: counter_m 100)
        u["x"][0, 0], u["z"][0, 0] = -80.0, 0.0                 # side 1's lord 80 m away
        u["x"][0, 1], u["z"][0, 1] = -95.0, 10.0                # a spearman ~95 m away
        u["x"][0, 2], u["z"][0, 2] = -400.0, 300.0              # the rest far
        u["x"][0, 3], u["z"][0, 3] = -400.0, -300.0
        o = opponents.ai_like(st, opponents.Line(pick_noise_m=0.0))
        assert int(o.kind[0, H + 1]) == O.ATTACK and int(o.target[0, H + 1]) == 1     # not the nearer lord
        g = opponents.pick_noise(64, 40, "cpu")
        assert torch.equal(g, opponents.pick_noise(64, 40, "cpu"))                   # the same every step
        assert float(g.mean()) == pytest.approx(0.5772, abs=0.03)                     # standard Gumbel
        assert float(g.std()) == pytest.approx(1.2825, abs=0.03)
        a, b = opponents.ai_like(st), opponents.ai_like(st)
        assert torch.equal(a.kind, b.kind) and torch.equal(a.target, b.target)

    def test_ai_like_line_widens_to_lap_a_wider_enemy_line(self):
        st = line_army(attacker=1, gap=600)
        H = st.N // 2
        u = st.u
        u["z"][0, H + 1], u["z"][0, H + 2] = -150.0, 150.0      # side 2's spearmen 300 m apart, side 1's 80 m
        o = opponents.ai_like(st, opponents.Line(lap_space_m=40.0))
        assert float(o.z[0, 1]) == pytest.approx(-40.0) and float(o.z[0, 2]) == pytest.approx(40.0)   # no room
        p = opponents.Line(lap_space_m=200.0)                   # room 200 - 80: each end out 60 of the 140 wanted
        o = opponents.ai_like(st, p)
        assert (o.kind[0, 1:3] == O.MOVE).all() and (o.x[0, 1:3] > u["x"][0, 1:3]).all()     # still forward
        assert float(o.z[0, 1]) == pytest.approx(-40.0 - p.step_m / 2)                     # aside, half a step
        assert float(o.z[0, 2]) == pytest.approx(40.0 + p.step_m / 2)
        o = opponents.ai_like(line_army(attacker=2, gap=600))                # side 1 defends with archers: holds
        assert (o.kind[0, 1:3] == O.HOLD).all()                              # (no room to widen at 40 m)
        st = line_army(attacker=2, gap=600)
        st.u["z"][0, H + 1], st.u["z"][0, H + 2] = -150.0, 150.0
        o = opponents.ai_like(st, p)
        assert (o.kind[0, 1:3] == O.MOVE).all() and not o.run[0, 1:3].any()             # it sidesteps, walking
        assert float(o.z[0, 1]) == pytest.approx(-48.0) and float(o.x[0, 1]) == pytest.approx(float(st.u["x"][0, 1]))

    def test_ai_like_goes_round_behind_an_enemy_that_fights_its_line(self):
        st = line_army(attacker=2, gap=600)
        H = st.N // 2
        u = st.u
        p = opponents.Line(pick_noise_m=0.0)
        u["x"][0, 1], u["z"][0, 1], u["b"][0, 1] = 0.0, 0.0, 90.0       # side 1's spearman faces east...
        u["x"][0, H + 1], u["z"][0, H + 1] = 10.0, 0.0                  # ... fighting side 2's spearman
        u["m"][0, 1] = u["m"][0, H + 1] = True
        u["target"][0, 1], u["target"][0, H + 1] = H + 1, 1
        u["x"][0, H + 2], u["z"][0, H + 2] = 60.0, 60.0                 # a free spearman of side 2 in front of it
        o = opponents.ai_like(st, p)
        assert int(o.kind[0, H + 2]) == O.MOVE and bool(o.run[0, H + 2])
        assert float(o.x[0, H + 2]) == pytest.approx(-p.wrap_back_m)            # to a point behind it ...
        assert float(o.z[0, H + 2]) == pytest.approx(float(u["width"][0, 1]) / 2 + p.flank_m)   # ... on the near side
        u["x"][0, H + 2], u["z"][0, H + 2] = -40.0, 10.0                # once behind it: attacks
        o = opponents.ai_like(st, p)
        assert int(o.kind[0, H + 2]) == O.ATTACK and int(o.target[0, H + 2]) == 1
        u["x"][0, H + 2], u["z"][0, H + 2] = 60.0, 60.0
        o = opponents.ai_like(st, dataclasses.replace(p, flank_m=0.0))  # flank_m 0: straight at it
        assert int(o.kind[0, H + 2]) == O.ATTACK

    def test_ai_like_guard_cuts_off_an_enemy_closing_on_its_missile_units(self):
        st = line_army(attacker=1, gap=600)                      # side 1: lord (-320, 0), spearmen (-300, +-40),
        H = st.N // 2                                            # archers (-340, 0)
        u = st.u
        p = opponents.Line(pick_noise_m=0.0)
        u["x"][0, H + 1], u["z"][0, H + 1] = -340.0, 60.0        # side 2's spearman 60 m north of the archers ...
        u["vz"][0, H + 1] = -3.0                                 # ... running at them
        u["x"][0, H + 2], u["z"][0, H + 2] = -300.0, -10.0       # another, still, 22 m from side 1's lord
        lord_on = lambda o: int(o.kind[0, 0]) == O.ATTACK and int(o.target[0, 0]) == H + 1
        assert not lord_on(opponents.ai_like(st, p))             # before the first fight: the lord's own target
        u["k"][0, 1] = 5.0                                       # side 1 has fought (a spearman's kills)
        o = opponents.ai_like(st, p)
        assert lord_on(o) and bool(o.run[0, 0])                  # the lord (63 m from it) cuts it off
        assert int(o.kind[0, 2]) == O.ATTACK and int(o.target[0, 2]) == H + 1   # the nearest spearman (45 m) too
        assert not lord_on(opponents.ai_like(st, dataclasses.replace(p, guard_m=0.0)))   # off: the nearer one
        u["vz"][0, H + 1] = 3.0                                  # going away: no guard
        assert not lord_on(opponents.ai_like(st, p))
        u["vz"][0, H + 1] = -3.0
        u["hp"][0, 0] = 0.2                                      # a hurt lord leaves it to the others
        assert not lord_on(opponents.ai_like(st, p))

    def test_ai_like_missile_units_walk_up_only_to_a_post_behind_the_line_after_the_fight(self):
        st = line_army(attacker=1, gap=600)                      # side 1 attacks: its archers walk up to range
        u = st.u
        p = opponents.Line(missile_post_m=55.0)
        assert int(opponents.ai_like(st, p).kind[0, 3]) == O.MOVE             # before the fight: 40 m behind, walks
        u["k"][0, 2] = 5.0                                                    # side 1 has fought
        assert int(opponents.ai_like(st, p).kind[0, 3]) == O.HOLD             # 40 m behind its line: at the post
        u["x"][0, 3] = -400.0                                                 # 100 m behind: walks up
        assert int(opponents.ai_like(st, p).kind[0, 3]) == O.MOVE
        u["x"][0, 3] = -340.0
        assert int(opponents.ai_like(st, dataclasses.replace(p, missile_post_m=0.0)).kind[0, 3]) == O.MOVE   # off

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
            ra = reward.step(ha, reward.measure(a), a.done, a.winner)
            rb = reward.step(hb, reward.measure(b), b.done, b.winner)
            assert torch.allclose(ra, rb.flip(1), atol=1e-4)
            assert torch.allclose(ra.sum(1), torch.zeros(2), atol=1e-6)
        assert float(reward.health(a)[0, 0]) < 1.0                        # the battle got going

    def test_battles_start_again_when_they_end(self):
        actor, crit = nets()
        env = rollout.Battles(league.layout(4, 1, opponent="nearest"), MIRROR, params=rollout.params_with_limit(1.0),
                              cadence=cad.STEP)                   # step level: a simulator step a call
        out = [env.step(actor, crit) for _ in range(3)]
        assert bool(out[1]["done"].all())                                 # the 1 s limit: 2 steps
        assert sorted(out[1]["reward"].tolist()) == pytest.approx([-1.0, -1.0, 1.0, 1.0], abs=0.01)
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
        strike(0, 1e-4)                                                   # a scratch (idle_rate 0: struck)
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
            router = unit(0, int(3 - att[0]), -2)
            low, high = 0.0, float(env.st.u["leadership"][0, router])    # morale points: stays routing / steady
            if k in (6, 10):                                              # ... a rout (half its rest lost), and at
                env.st.u["r"][0, router], env.st.u["morale"][0, router] = True, low   # 10 again: counted once (B3)
            if k == 8:
                env.st.u["r"][0, router], env.st.u["morale"][0, router] = False, high  # ... it rallies (no damage)
            if k == 9:
                i = unit(0, int(3 - att[0]), -3)
                env.st.u["r"][0, i], env.st.u["gone"][0, i] = True, True  # ... another leaves the map: whole
            rate0 = float(env.hit_rate[0])
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
            if k in (6, 9):
                assert float(env.hit_rate[0]) > rate0 and bool(env.st.u["r"][0, router]) == (k == 6), k
            if k in (8, 10):                                              # a rally and a rout again: no damage
                assert float(env.hit_rate[0]) < rate0 and bool(env.st.u["r"][0, router]) == (k == 10), k
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
        assert batch["value"].shape == batch["reward"].shape == (3, env.R) and batch["last_value"].shape == (env.R,)
        opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
        st = ppo.update(actor, crit, opt, batch, dataclasses.replace(ppo.PPOConfig(), epochs=2, minibatch=6))
        assert all(np.isfinite(v) for v in st.values())
        assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters()))

    def test_a_huge_critic_loss_does_not_shrink_the_actors_step(self):
        # 02.10: actor and critic were clipped together, a critic loss's gradient (then the per-unit value's,
        # norm ~1000) scaled the actor's to ~1e-8 a parameter, below Adam's eps: the policy took no step at all
        import copy
        actor, crit = nets()
        env = rollout.Battles(league.layout(6, 1, {"self": 0.5, "nearest": 0.5}), MIRROR)
        batch = rollout.collect(env, actor, crit, 3)
        steps = []
        for weight in (1e-3, 1e6):
            a, c = copy.deepcopy(actor), copy.deepcopy(crit)
            opt = torch.optim.Adam(list(a.parameters()) + list(c.parameters()), lr=1e-3, eps=1e-5)
            torch.manual_seed(0)
            st = ppo.update(a, c, opt, batch, dataclasses.replace(ppo.PPOConfig(), epochs=1, minibatch=6, value=weight))
            assert "grad_norm" in st and "grad_norm_critic" in st
            steps.append([p.detach() - q.detach() for p, q in zip(a.parameters(), actor.parameters())])
        assert max(float(x.abs().max()) for x in steps[0]) > 1e-5
        assert all(torch.allclose(x, y, atol=1e-7) for x, y in zip(*steps))

    def test_a_minibatch_in_parts_takes_the_same_step_as_whole(self):
        # 03.10: the widened network's minibatch was halved to fit the GPU: twice the optimizer steps per update,
        # ~4 times the KL, the run collapsed. Gradient accumulation (PPOConfig.accum) keeps the steps and the step.
        import copy
        actor, crit = nets()
        env = rollout.Battles(league.layout(8, 1, {"self": 0.5, "nearest": 0.5}), MIRROR)
        batch = rollout.collect(env, actor, crit, 3)
        after = []
        for accum in (1, 2, 3):
            a, c = copy.deepcopy(actor), copy.deepcopy(crit)
            opt = torch.optim.Adam(list(a.parameters()) + list(c.parameters()), lr=1e-3, eps=1e-5)
            torch.manual_seed(0)
            st = ppo.update(a, c, opt, batch, dataclasses.replace(ppo.PPOConfig(), epochs=1, minibatch=3 * env.R, accum=accum))
            assert st["minibatches"] == 1 and all(np.isfinite(v) for v in st.values())
            after.append([p.detach().clone() for p in list(a.parameters()) + list(c.parameters())])
        for other in after[1:]:
            assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(after[0], other))

    def test_a_wider_network_keeps_the_minibatch_in_parts_and_its_stream_readers_take_lr_over_width(self):
        from tools.nn.model import widen
        from tools.nn.train import run
        cfg = ppo.PPOConfig(minibatch=4096)
        assert (run.sized(cfg, 22).minibatch, run.sized(cfg, 22).accum) == (4096, 1)
        assert (run.sized(cfg, 44, width=2.0).minibatch, run.sized(cfg, 44, width=2.0).accum) == (2048, 2)
        actor, crit = nets()
        opt = run.optimizer(actor, crit, 1e-3, 2.0)
        readers = widen.stream_readers(actor, widen.ACTOR) | {f"c.{n}" for n in widen.stream_readers(crit, widen.CRITIC)}
        assert "heads.kind.weight" in readers and "memory.cell.weight_hh" in readers and "blocks.0.f1.weight" in readers
        assert "heads.kind.bias" not in readers and "blocks.0.f2.weight" not in readers and "c.value.0.weight" in readers
        (slow, fast) = opt.param_groups
        assert slow["lr"] == pytest.approx(5e-4) and fast["lr"] == pytest.approx(1e-3)
        assert len(slow["params"]) == len(readers)
        assert len(slow["params"]) + len(fast["params"]) == len(list(actor.parameters())) + len(list(crit.parameters()))
        assert len(run.optimizer(actor, crit, 1e-3, 1.0).param_groups) == 1

    def test_the_ema_anchor_follows_the_actor_by_its_half_life(self):
        import copy
        from tools.nn.train import run
        actor, _ = nets()
        ref = copy.deepcopy(actor)
        with torch.no_grad():
            for p in actor.parameters():
                p.add_(1.0)
        before = [p.detach().clone() for p in ref.parameters()]
        run.follow(ref, actor, 0.25)
        assert all(torch.allclose(r, b + 0.25) for r, b in zip(ref.parameters(), before))

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

    def test_a_run_with_the_critic_of_another_checkpoint_role_advantages_and_the_idle_rate(self, tmp_path, monkeypatch):
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
                                        "--mix", '{"self": 0.5, "nearest": 0.5}', "--anchor", "0.05", "--reference", str(init),
                                        "--adv-norm", "role", "--idle-rate", "0.05", "--print-every", "1"] + SMALL_RUN)
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
                                        "--lr", "1e-2", "--print-every", "1"] + SMALL_RUN)
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
                                        "--mix", '{"nearest": 1.0}', "--anchor", "0.05"] + SMALL_RUN)
        _, _, out = run.train(args)
        assert [json.loads(x)["anchor_rolls"] for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()] == [0, 0]

    def test_the_next_bank_is_built_ahead_in_a_thread(self, tmp_path, monkeypatch):
        # building a bank is mostly Python (17-48 s on the 2048 battles): the training no longer waits for it
        import threading
        from tools.nn.train import run
        monkeypatch.setattr(checkpoint, "DIR", tmp_path)
        where = []
        real = scenes.Generated.__init__

        def spy(self, *a, **k):
            where.append(threading.current_thread().name)
            real(self, *a, **k)
        monkeypatch.setattr(scenes.Generated, "__init__", spy)
        base = ["--battles", "4", "--steps", "2", "--updates", "3", "--minutes", "5", "--device", "cpu", "--no-eval",
                "--mix", '{"nearest": 1.0}', "--bank-refresh", "0"] + SMALL_RUN
        run.train(run.parser().parse_args(["--name", "ahead"] + base))
        assert where[0] == "MainThread" and len(where) >= 3 and all(w.startswith("bank") for w in where[1:])
        where.clear()
        run.train(run.parser().parse_args(["--name", "due", "--no-bank-ahead"] + base))
        assert len(where) == 4 and set(where) == {"MainThread"}                 # the first and one per update

    def test_per_role_normalisation_gives_each_role_mean_0_std_1(self):
        a = torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
        g = torch.tensor([[True, False]] * 3)
        n = ppo.normalise(a, g)
        for m in (g, ~g):
            assert float(n[m].mean()) == pytest.approx(0.0, abs=1e-6) and float(n[m].std()) == pytest.approx(1.0)
        assert torch.allclose(ppo.normalise(a), (a - a.mean()) / (a.std() + 1e-8))

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

    @pytest.mark.skipif(not hasattr(__import__("signal"), "SIGKILL"), reason="POSIX signals")
    def test_the_gpu_lock_is_freed_on_sigterm(self, tmp_path):
        # docker stop sends SIGTERM to the container's first process, which ignores it without a handler
        import os
        import signal
        import time
        from tools.nn.train import test5
        lock = tmp_path / "gpu-train.lock"
        before = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit):
            with test5.gpu_lock("t", path=lock, poll_s=0.01):
                assert lock.exists()
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(2.0)                      # (the handler runs at the next bytecode)
        assert not lock.exists() and signal.getsignal(signal.SIGTERM) == before

    def test_sigterm_frees_the_lock_at_once_even_if_the_unwinding_never_gets_there(self, tmp_path):
        # a SystemExit raised inside a finaliser is swallowed; slow inner finally blocks outlast docker's 10 s
        import signal
        from tools.nn.train import test5
        lock = tmp_path / "gpu-train.lock"
        with test5.gpu_lock("t", path=lock, poll_s=0.01, gap_s=0):
            try:
                test5._stop(signal.SIGTERM, None)
            except SystemExit as e:                   # (swallowed here)
                assert e.code == 128 + signal.SIGTERM
            assert not lock.exists() and lock.with_suffix(".released").read_text(encoding="utf-8").endswith(" t\n")
            lock.write_text("test5 other\n", encoding="utf-8")       # another job takes the free lock
        assert lock.read_text(encoding="utf-8") == "test5 other\n"   # ... and keeps it
        assert not test5._HELD

    def test_the_before_evaluation_runs_with_tf32_as_the_training_does(self, tmp_path, monkeypatch):
        # torch.compile's graphs are guarded on the TF32 switch: run.train turned it on after a "before" compiled
        # without it, and the first evaluation after training compiled every graph again
        from types import SimpleNamespace
        from tools.nn.train import test5
        seen = []

        class Stop(Exception):
            pass

        def evaluation(*a, **k):
            seen.append(torch.backends.cuda.matmul.allow_tf32)
            raise Stop
        monkeypatch.setattr(test5, "evaluation", evaluation)
        monkeypatch.setattr(test5, "references", lambda args: None)
        monkeypatch.setattr(test5, "OUT", tmp_path)
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ck = checkpoint.save(tmp_path / "init.pt", *nets())
        args = SimpleNamespace(label="x", init=str(ck), minutes=1.0, updates=1, device="cpu", eval=8, drill_eval=0,
                               profiles=(), before=None, fresh_before=True)
        with pytest.raises(Stop):
            test5.test(args, [])
        assert seen == [True]

    def test_the_before_is_the_previous_step_s_last_evaluation_of_the_same_key(self, tmp_path):
        from types import SimpleNamespace
        from tools.nn.train import test5
        actor, crit = nets()
        ck = checkpoint.save(tmp_path / "prev" / "m20.pt", actor, crit)
        args = SimpleNamespace(eval=8, drill_eval=4, profiles=("behaviour", "drills"))
        key = test5.eval_key(actor, args, cad.GAME)
        assert key == test5.eval_key(checkpoint.load_policy(ck), args, cad.GAME)     # in memory = loaded back
        assert test5.eval_key(nets(seed=1)[0], args, cad.GAME)["actor"] != key["actor"]
        after = {"by_opponent": {}, "key": key, "seconds": 200, "distance": {"start_kl": 0.1}, "teach": {"kiting": {}},
                 "teach_auto": [], "update": 70}
        (tmp_path / "prev" / "after.json").write_text(json.dumps(after), encoding="utf-8")
        (tmp_path / "prev" / "eval_m10.json").write_text(json.dumps(dict(after, key=dict(key, actor="x"))), encoding="utf-8")
        path, before = test5.reusable(ck, key)
        assert path.name == "after.json" and before == {"by_opponent": {}, "key": key, "seconds": 200}
        assert test5.reusable(ck, dict(key, eval=16)) is None                    # other settings: played again
        assert test5.reusable(ck, dict(key, actor="x"))[0].name == "eval_m10.json"
        assert test5.reusable(tmp_path / "none" / "m5.pt", key) is None

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
