"""tools.nn.train.cadence and the cadence of rollout.Battles: the networks decide once a second of battle
and their orders land ~0.36 s late, as in the game (docs/en/training/training.md "Decisions").

Level 1 (no torch, runs in the .venv): steps per decision, the latency in steps, the discount per decision.
Level 2/3 (torch, the training container): the decision's simulator steps with spies (where the orders
land, KEEP around them, the reward summed, the per-step hook), the random landing step's mean.
"""
import math

import pytest

from tools.nn.train import cadence as cad


# --- level 1: functions ---

class TestFunctions:
    def test_a_decision_spans_whole_simulator_steps(self):
        assert cad.GAME.steps(0.5) == 2 and cad.STEP.steps(0.5) == 1 and cad.Cadence(1.5, 0).steps(0.5) == 3
        for bad in (0.75, 0.2, 0.0):
            with pytest.raises(ValueError):
                cad.Cadence(bad, 0).steps(0.5)

    def test_the_latency_in_steps_keeps_its_mean(self):
        assert cad.Cadence(1.0, 0.0).delay(0.5) == (0, 0.0)
        assert cad.Cadence(1.0, 0.5).delay(0.5) == (1, 0.0)
        whole, frac = cad.Cadence(1.0, 0.36).delay(0.5)
        assert whole == 0 and frac == pytest.approx(0.72)
        assert (whole + frac) * 0.5 == pytest.approx(0.36)
        whole, frac = cad.Cadence(2.0, 0.8).delay(0.5)
        assert whole == 1 and frac == pytest.approx(0.6)

    def test_orders_must_land_before_the_next_decision(self):
        for c in (cad.Cadence(0.5, 0.3), cad.Cadence(1.0, 0.6), cad.Cadence(1.0, 1.0), cad.Cadence(1.0, -0.1)):
            with pytest.raises(ValueError):
                c.delay(0.5)
        assert cad.STEP.delay(0.5) == (0, 0.0)

    def test_the_discount_keeps_the_horizon_in_seconds(self):
        g = 0.9997
        per_s = cad.GAME.discount(g, 0.5)
        assert per_s == pytest.approx(g ** 2) and cad.STEP.discount(g, 0.5) == pytest.approx(g)
        # the weight of a reward 10 minutes away is the same at either cadence
        assert per_s ** (600 / 1.0) == pytest.approx(g ** (600 / 0.5))
        assert cad.GAME.discount(0.95) == pytest.approx(0.9025)
        # horizon in seconds: decision seconds / (1 - gamma), nearly the same
        assert 1.0 / (1 - per_s) == pytest.approx(0.5 / (1 - g), rel=1e-3)

    def test_the_options_default_to_the_game(self):
        import argparse
        ap = argparse.ArgumentParser()
        cad.add_args(ap)
        assert cad.of_args(ap.parse_args([])) == cad.GAME
        assert cad.of_args(ap.parse_args(["--decide-s", "0.5", "--order-latency", "0"])) == cad.STEP
        assert cad.GAME.meta() == {"decide_s": 1.0, "latency_s": 0.36}
        assert not math.isnan(cad.GAME.discount(0.9997))


# --- level 2/3: the decision's simulator steps (torch) ---

class TestBattles:
    @pytest.fixture(autouse=True)
    def _torch(self):
        pytest.importorskip("torch")

    @staticmethod
    def env(cadence, B=2, limit=60.0, opponent="hold"):
        from tools.nn.train import league, randomise, rollout
        return rollout.Battles(league.layout(B, 1, opponent=opponent), [("arena", "attack")],
                               params=rollout.params_with_limit(limit), spread=randomise.NONE, cadence=cadence,
                               compile=False)

    @staticmethod
    def actor():
        from tools.nn.model import config, policy
        import torch
        torch.manual_seed(0)
        cfg = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)
        return policy.Actor(cfg).eval()

    @staticmethod
    def spy(env):
        """Records the learner rows' order kinds of every simulator step."""
        from tools.nn.sim import orders as O
        seen = []
        orig = env._sim_step

        def wrapped(parts, *a, **k):
            for rows, o in parts:
                if rows is env.rows_learn:
                    seen.append(o.kind.clone())
            return orig(parts, *a, **k)
        env._sim_step = wrapped
        return seen, O

    def test_a_decision_plays_its_steps_of_battle_time(self):
        for c, k in ((cad.GAME, 2), (cad.STEP, 1), (cad.Cadence(1.5, 0.5), 3)):
            env = self.env(c)
            t0 = env.st.t.clone()
            env.step(self.actor())
            assert env.k == k and env.decision_s == pytest.approx(0.5 * k)
            assert (env.st.t - t0).tolist() == pytest.approx([0.5 * k] * env.B)

    def test_the_orders_land_after_the_latency_and_are_kept_around_it(self):
        import torch
        for latency, land in ((0.0, 0), (0.5, 1)):
            env = self.env(cad.Cadence(1.0, latency))
            seen, O = self.spy(env)
            env.step(self.actor())
            assert len(seen) == 2
            other = 1 - land
            assert (seen[other] == O.KEEP).all()
            assert not torch.equal(seen[land], seen[other])          # the decision's own orders (not all KEEP)
            assert seen[land].ne(O.KEEP).any()

    def test_the_scripts_decide_once_a_decision_and_land_with_the_networks(self):
        # (build/audit_orders: in the game ai_like decides a second apart on the state of the decision and its orders
        # land late; before, the simulator asked the scripts every 0.5 s step, the orders at once)
        import torch
        from tools.nn.sim import orders as O
        from tools.nn.train import rollout
        for latency, land in ((0.0, 0), (0.5, 1)):
            env = self.env(cad.Cadence(1.0, latency), opponent="ai_like")
            asked, bases, made, given = [], [], [], []
            orig_scripted, orig_step = env.scripted, env._sim_step

            def scripted():
                asked.append(float(env.st.t[0]))
                made.append(orig_scripted())
                return made[-1]
            env.scripted = scripted

            def wrapped(parts, attacks, rb, rs, was_done, base=None):
                bases.append(O.Orders(*(getattr(base, f).clone() for f in O.FIELDS)))
                given.append(env.assemble(parts, base))                # what the simulator is given
                return orig_step(parts, attacks, rb, rs, was_done, base)
            env._sim_step = wrapped
            t0 = float(env.st.t[0])
            env.step(self.actor())
            env.step(self.actor())
            assert asked == pytest.approx([t0, t0 + 1.0])                 # once a decision, on its state
            assert len(bases) == 4 and all(b is not None for b in bases)
            script_units = rollout.learner_units(env.st.u, env.ctrl) == 0
            for j, b in enumerate(bases):
                k = b.kind[script_units]
                assert bool((k == O.KEEP).all()) is (j % 2 != land)        # KEEP but at the landing step
            assert bool((bases[land].kind[script_units] != O.KEEP).all())
            for j in (land, land + 2):                                    # the script's own orders reach its units
                assert torch.equal(given[j].kind[script_units], made[j // 2].kind[script_units])
                assert torch.equal(given[j].target[script_units], made[j // 2].target[script_units])

    def test_the_old_cadence_gives_every_step_the_decision(self):
        env = self.env(cad.STEP)
        seen, O = self.spy(env)
        for _ in range(3):
            env.step(self.actor())
        assert len(seen) == 3 and all(s.ne(O.KEEP).any() for s in seen)

    def test_a_random_landing_step_has_the_mean_latency(self):
        import torch
        torch.manual_seed(1)
        d = cad.Cadence(1.0, 0.36).delays(0.5, 20000, "cpu")
        assert set(d.unique().tolist()) == {0, 1}
        assert float(d.float().mean()) * 0.5 == pytest.approx(0.36, abs=0.01)
        assert cad.Cadence(1.0, 0.5).delays(0.5, 4, "cpu").tolist() == [1] * 4

    def test_a_hashed_landing_has_the_mean_latency_and_does_not_depend_on_the_batch(self):
        import torch
        c = cad.Cadence(1.0, 0.36)
        ids = torch.arange(20000)
        late = torch.stack([c.lands(0.5, 1, ids, d) for d in range(3)])
        assert bool((late == ~torch.stack([c.lands(0.5, 0, ids, d) for d in range(3)])).all())   # one step or other
        assert float(late.float().mean()) * 0.5 == pytest.approx(0.36, abs=0.01)
        assert torch.equal(c.lands(0.5, 1, ids[5:9], 2), late[2, 5:9])                       # the same in a sub-batch
        assert cad.Cadence(1.0, 0.5).lands(0.5, 1, ids[:3], 7).tolist() == [True] * 3

    def test_the_reward_is_summed_over_the_steps_and_the_hook_sees_each(self):
        import torch
        env = self.env(cad.Cadence(1.5, 0.0))
        orig = env._sim_step
        calls = []

        def spy(parts, attacks, rb, rs, was_done, base=None):
            r, fin = orig(parts, attacks, rb, rs, was_done, base)
            return torch.ones_like(r) * len(calls), fin            # 1, 2, 3 (counted before)
        env._sim_step = lambda *a: (calls.append(1), spy(*a))[1]
        hooks = []
        out = env.step(self.actor(), each=lambda live: hooks.append(live.clone()))
        assert len(calls) == 3 and len(hooks) == 3
        assert out["reward"].tolist() == pytest.approx([1 + 2 + 3] * len(env.rows_learn))
        assert all(h.all() for h in hooks)

    def test_a_battle_that_ends_mid_decision_is_done_once_and_restarts_after(self):
        import torch
        env = self.env(cad.Cadence(1.5, 0.0), limit=1.0)        # ends after the 2nd of 3 steps
        out = env.step(self.actor())
        assert out["done"].all()
        assert (env.st.t == 0).all() and not env.st.done.any()   # restarted at the decision's end
        assert int(env.battles) == env.B                          # counted once
