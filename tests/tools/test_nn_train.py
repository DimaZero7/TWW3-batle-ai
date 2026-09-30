"""tools.nn.train: PPO training in the simulator (docs/en/training/training.md).

torch is not in the project's .venv: these tests are skipped there and run in the training
container (snake-ai-trainer; its image has no pytest, so put a pure-Python pytest on PYTHONPATH).

Level 1: functions (GAE, clipping, reward, order changes, layout). Level 2: properties (masks are
respected, rewards are symmetric between the sides, KEEP keeps the order). Level 3: a tiny training
step end to end on the CPU, checkpoints, recordings.
"""
import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn import gamedata  # noqa: E402
from tools.nn.model import config, critic, heads, policy  # noqa: E402
from tools.nn.sim import battle, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.train import checkpoint, evaluate, league, opponents, ppo, randomise, reward, rollout, scenes  # noqa: E402

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

    def test_the_reward_is_win_plus_the_health_difference(self):
        w = reward.Weights(win=1.0, hp=0.5)
        before = torch.tensor([[1.0, 1.0], [0.5, 0.4]])
        after = torch.tensor([[0.9, 0.7], [0.5, 0.4]])
        r = reward.step(before, after, torch.tensor([False, True]), torch.tensor([0, 2]), w)
        assert r[0].tolist() == pytest.approx([0.5 * (0.3 - 0.1), -0.5 * (0.3 - 0.1)])
        assert r[1].tolist() == pytest.approx([-1.0, 1.0])

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
            ha, hb = reward.health(a), reward.health(b)
            oa = opponents.nearest(a)
            ob_ = O.Orders(*(getattr(oa, k)[:, swap] for k in O.FIELDS))
            ob_.target = torch.where(ob_.target >= 0, (ob_.target + H) % (2 * H), ob_.target)
            battle.step(a, oa, P)
            battle.step(b, ob_, P)
            ra = reward.step(ha, reward.health(a), a.done, a.winner)
            rb = reward.step(hb, reward.health(b), b.done, b.winner)
            assert torch.allclose(ra, rb.flip(1), atol=1e-4)
            assert torch.allclose(ra.sum(1), torch.zeros(2), atol=1e-6)
        assert float(reward.health(a)[0, 0]) < 1.0                        # the battle got going

    def test_battles_start_again_when_they_end(self):
        actor, crit = nets()
        env = rollout.Battles(league.layout(4, 1, opponent="nearest"), MIRROR, params=rollout.params_with_limit(1.0))
        out = [env.step(actor, crit) for _ in range(3)]
        assert bool(out[1]["done"].all())                                 # the 1 s limit: 2 steps
        assert float(out[1]["reward"].abs().min()) >= 0.99                # the attacker loses, the defender wins
        assert float(env.st.t.max()) == pytest.approx(0.5)                # started again
        stats = env.take_stats()
        assert stats["nearest"][0] == 4


# --- level 3: the loop ---

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
        opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
        st = ppo.update(actor, crit, opt, batch, dataclasses.replace(ppo.PPOConfig(), epochs=2, minibatches=2))
        assert all(np.isfinite(v) for v in st.values())
        assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters()))

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
        for name in ("hold", "past"):
            r = res["by_opponent"][name]
            assert r["games"] == 2 and r["timeouts"] == 1.0
            assert r["win_rate"] == 0.5                                   # the defender wins on time
        assert sum(res["kinds"].values()) == pytest.approx(1.0)
