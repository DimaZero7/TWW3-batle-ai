"""tools.nn.model observation: what one side sees (numpy; with torch also the same on tensors).

Level 1: frame, normalisation, masking. Level 2: properties — the enemy's exact morale never reaches
the input, invisible enemies keep only their last seen position, side symmetry, unit order.
"""
import copy

import numpy as np
import pytest

from tools.nn import gamedata
from tools.nn.model import factions, passport, sources
from tools.nn.model import observation as ob
from tools.nn.model.frame import Frame, edge_distances

I = ob.INDEX


def battle(seed=0, own=5, enemy=6, batch=2):
    return sources.synthetic(batch=batch, own=own, enemy=enemy, seed=seed)


def with_state(state, **changes):
    s = copy.deepcopy(state)
    s.update(changes)
    return s


class TestFrame:
    def test_world_and_frame_are_inverse(self):
        fr = Frame(np.array([10.0]), np.array([-5.0]), np.array([0.6]), np.array([0.8]))
        x, z = np.array([[3.0, -40.0]]), np.array([[7.0, 100.0]])
        f, l = fr.point(x, z)
        bx, bz = fr.world(f, l)
        assert np.allclose(bx, x) and np.allclose(bz, z)

    def test_forward_is_towards_the_enemy_and_right_is_positive_lateral_facing(self):
        fr = Frame(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))   # forward = +x
        c, s = fr.facing(np.array([[90.0, 270.0, 180.0]]))                                # +x, -x, -z
        assert np.allclose(c, [[1, -1, 0]], atol=1e-9) and np.allclose(s, [[0, 0, 1]], atol=1e-9)

    def test_edge_distances_in_a_square_map(self):
        fr = Frame(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))
        bounds = [np.array([v]) for v in (-100.0, 100.0, -100.0, 100.0)]
        e = edge_distances(np.array([[20.0]]), np.array([[10.0]]), fr, bounds)
        assert [float(v[0, 0]) for v in e] == pytest.approx([80, 120, 110, 90])


class TestPassport:
    def test_every_passport_is_scaled_and_padding_is_zero(self):
        keys = list(passport.load())
        t = passport.table(keys + [""])
        assert t.shape == (len(keys) + 1, passport.SIZE)
        assert np.all(t[-1] == 0) and np.all(t[:-1] >= 0) and t.max() <= 1.5

    def test_placeholder_characters_exist_for_empire_and_skaven(self):
        for key in ("wh_main_emp_empire", "wh2_main_skv_skaven"):
            assert factions.character(key).shape == (factions.SIZE,)
        assert factions.adapter("wh_main_emp_empire", True) != factions.adapter("wh_main_emp_empire", False)


class TestObservation:
    def test_shapes_scales_and_padding(self):
        setup, state = battle()
        setup.side[:, -1] = 0            # the last unit is padding
        setup.present[:, -1] = False
        obs, _ = ob.observe(state, setup, 1)
        assert obs.tokens.shape == (2, 11, ob.TOKEN) and obs.ctx.shape == (2, ob.CONTEXT)
        assert np.all(obs.tokens[:, -1] == 0) and not obs.attend[:, -1].any()
        assert np.abs(obs.tokens).max() < 3
        assert np.all(obs.tokens[..., I["men"]] <= 1) and np.all(obs.tokens[..., I["hp"]] <= 1)

    def test_own_only_fields_are_zero_for_enemies(self):
        setup, state = battle()
        obs, _ = ob.observe(state, setup, 1)
        enemy = setup.side == 2
        assert np.all(obs.tokens[enemy][:, list(ob.OWN_ONLY)] == 0)
        assert np.any(obs.tokens[~enemy][:, list(ob.OWN_ONLY)] != 0)

    def test_the_enemy_exact_morale_never_reaches_the_input(self):
        setup, state = battle()
        enemy = setup.side == 2
        base, _ = ob.observe(state, setup, 1)
        rng = np.random.default_rng(1)
        steady = np.where(enemy, rng.integers(1, 5, state["ms"].shape), state["ms"])    # 1-4 all look "steady"
        steady_now = np.where(enemy & (state["ms"] <= 4), steady, state["ms"])
        changed = with_state(state, mp=np.where(enemy, rng.uniform(-2, 2, state["mp"].shape), state["mp"]),
                             ms=steady_now)
        other, _ = ob.observe(changed, setup, 1)
        assert np.array_equal(base.tokens, other.tokens) and np.array_equal(base.ctx, other.ctx)
        full, _ = ob.observe(state, setup, 1, full=True)          # the critic does see it
        full2, _ = ob.observe(changed, setup, 1, full=True)
        assert not np.array_equal(full.tokens, full2.tokens)

    def test_invisible_enemies_keep_only_the_last_seen_position(self):
        setup, state = battle()
        _, memory = ob.observe(state, setup, 1)
        hidden = np.zeros_like(setup.present)
        hidden[:, -1] = True                                         # the last enemy disappears
        later = with_state(state, t=state["t"] + 10, vis=~hidden,
                           x=state["x"] + 50 * hidden, z=state["z"] - 30 * hidden, mp=state["mp"] - 1)
        obs, _ = ob.observe(later, setup, 1, memory)
        tok = obs.tokens[:, -1]
        first, _ = ob.observe(state, setup, 1)
        assert np.allclose(tok[:, [I["fwd"], I["lat"]]], first.tokens[:, -1, [I["fwd"], I["lat"]]])
        assert np.all(tok[:, I["visible"]] == 0) and np.allclose(tok[:, I["age"]], 10 / ob.AGE)
        dyn = [I[n] for n, _ in ob.DYNAMIC if n not in ("fwd", "lat") and not n.startswith("edge")]
        assert np.all(tok[:, dyn] == 0)
        assert not obs.target_ok[:, -1].any() and obs.attend[:, -1].all()
        moved = with_state(later, x=later["x"] + 200 * hidden)      # where it really is does not matter
        again, _ = ob.observe(moved, setup, 1, memory)
        assert np.array_equal(obs.tokens, again.tokens)

    def test_a_never_seen_enemy_has_no_position(self):
        setup, state = battle()
        hidden = np.zeros_like(setup.present)
        hidden[:, -1] = True
        obs, _ = ob.observe(with_state(state, vis=~hidden), setup, 1)
        tok = obs.tokens[:, -1]
        assert np.all(tok[:, : len(ob.DYNAMIC)] == 0) and np.all(tok[:, I["seen"]] == 0)
        assert np.all(obs.pos[:, -1] == 0) and np.any(tok[:, len(ob.DYNAMIC):] != 0)    # passport known

    def test_velocity_from_two_decisions(self):
        setup, state = battle()
        _, memory = ob.observe(state, setup, 1)
        later = with_state(state, t=state["t"] + 2, x=state["x"] + 4)    # 2 m/s towards the enemy
        obs, _ = ob.observe(later, setup, 1, memory)
        assert np.allclose(obs.tokens[..., I["vel_fwd"]], 2 / ob.VEL, atol=1e-3)

    def test_side_symmetry(self):
        setup, state = battle(own=4, enemy=4)
        b = setup.bounds[0]
        cx, cz = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
        mirror = with_state(state, x=2 * cx - state["x"], z=2 * cz - state["z"], b=state["b"] + 180,
                            ox=2 * cx - state["ox"], oz=2 * cz - state["oz"])
        msetup = ob.Setup(keys=setup.keys, side=3 - setup.side, bounds=setup.bounds,
                          factions=[f[::-1] for f in setup.factions], attacker=3 - setup.attacker)
        a, _ = ob.observe(state, setup, 1)
        m, _ = ob.observe(mirror, msetup, 2)
        assert np.allclose(a.tokens, m.tokens, atol=1e-5) and np.allclose(a.ctx, m.ctx)
        assert np.array_equal(a.adapter, m.adapter)

    def test_permuting_units_permutes_the_tokens(self):
        setup, state = battle()
        perm = np.random.default_rng(3).permutation(setup.side.shape[1])
        psetup = ob.Setup(keys=[[k[i] for i in perm] for k in setup.keys], side=setup.side[:, perm],
                          bounds=setup.bounds, factions=setup.factions, attacker=setup.attacker)
        pstate = {k: (v[:, perm] if np.ndim(v) == 2 else v) for k, v in state.items()}
        a, _ = ob.observe(state, setup, 1)
        p, _ = ob.observe(pstate, psetup, 1)
        assert np.allclose(a.tokens[:, perm], p.tokens) and np.allclose(a.ctx, p.ctx)

    def test_routing_and_dead_units_take_no_orders(self):
        setup, state = battle()
        own = setup.side == 1
        obs, _ = ob.observe(state, setup, 1)
        expect = own & (state["men"] > 0) & (state["ms"] < 6)
        assert np.array_equal(obs.ctrl, expect)


RUNS = gamedata.runs(arena="whole_emp_v_skv")


@pytest.mark.skipif(not RUNS, reason="no recorded Empire vs Skaven battles (build/ is not in Git)")
def test_recorded_battles_give_the_same_input_as_the_simulator_would():
    rec = sources.batch([gamedata.load(r) for r in RUNS[:2]])
    for side in (1, 2):
        memory = None
        for ti in range(0, 60, 5):
            obs, memory = ob.observe(rec.state(ti), rec.setup, side, memory)
            assert np.isfinite(obs.tokens).all() and np.isfinite(obs.ctx).all()
            assert obs.ctrl.sum() > 0 and obs.target_ok.sum() > 0
