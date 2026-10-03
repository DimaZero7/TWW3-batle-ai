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
C = ob.CTX
LORDS = slice(ob.CONTEXT_BASE - 4, ob.CONTEXT_BASE)   # own lord slain, how recently, the enemy's


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


class TestEvents:
    """Events a player is told or sees: a lord slain (announced), a unit fought or routed lately."""

    KEYS = ["wh_main_emp_cha_general_0", "wh_main_emp_inf_spearmen_0", "wh2_main_skv_cha_warlord_0",
            "wh2_main_skv_inf_clanrat_spearmen_0"]

    def setup_state(self):
        _, state = sources.synthetic(batch=1, own=2, enemy=2, seed=3)
        setup = ob.Setup(keys=[list(self.KEYS)], side=np.array([[1, 1, 2, 2]]),
                         bounds=np.array([sources.CROSSROADS], np.float32),
                         factions=[(sources.EMPIRE, sources.SKAVEN)], attacker=np.array([1]))
        state = with_state(state, men=np.array([[1.0, 100, 1, 100]]), m=np.zeros((1, 4), bool),
                           r=np.zeros((1, 4), bool), ms=np.full((1, 4), 2.0), vis=np.ones((1, 4), bool))
        return setup, state

    def test_the_general_is_known_by_the_passport(self):
        assert passport.lords(self.KEYS + [""]).tolist() == [True, False, True, False, False]

    def test_a_slain_enemy_lord_is_known_even_unseen_and_fades(self):
        setup, state = self.setup_state()
        obs, mem = ob.observe(with_state(state, t=np.array([29.0])), setup, 1)
        assert obs.ctx[0, LORDS].tolist() == [0, 0, 0, 0]
        dead = with_state(state, t=np.array([30.0]), men=np.array([[1.0, 100, 0, 100]]),
                          vis=np.array([[True, True, False, True]]))
        obs, mem = ob.observe(dead, setup, 1, mem)
        assert obs.ctx[0, LORDS].tolist() == pytest.approx([0, 0, 1, 1])
        obs, mem = ob.observe(with_state(dead, t=np.array([90.0])), setup, 1, mem)
        assert obs.ctx[0, LORDS].tolist() == pytest.approx([0, 0, 1, 0.5])
        other, _ = ob.observe(with_state(dead, t=np.array([90.0])), setup, 2)
        assert other.ctx[0, LORDS].tolist() == pytest.approx([1, 1, 0, 0])      # the Skaven side: own lord

    def test_melee_and_rout_of_an_enemy_count_only_while_seen(self):
        setup, state = self.setup_state()
        fight = with_state(state, t=np.array([30.0]), m=np.array([[False, True, False, True]]),
                           r=np.array([[False, False, False, False]]))
        obs, mem = ob.observe(fight, setup, 1)
        assert obs.tokens[0, [1, 3], I["melee_recent"]].tolist() == [1, 1]
        later = with_state(state, t=np.array([90.0]), r=np.array([[False, False, False, True]]))
        obs, mem = ob.observe(later, setup, 1, mem)
        assert obs.tokens[0, [1, 3], I["melee_recent"]].tolist() == pytest.approx([0.5, 0.5])
        assert obs.tokens[0, 3, I["rout_recent"]] == 1
        hidden = with_state(fight, vis=np.array([[True, True, True, False]]))
        obs, _ = ob.observe(hidden, setup, 1)
        assert obs.tokens[0, 3, I["melee_recent"]] == 0 and obs.tokens[0, 1, I["melee_recent"]] == 1


class TestDamageTimers:
    """When each side last dealt damage (some unit of the other side lost health), per battle row."""

    def states(self, batch=2):
        setup, state = sources.synthetic(batch=batch, own=2, enemy=2, seed=4)
        return setup, with_state(state, hp=np.full((batch, 4), 0.9), vis=np.ones((batch, 4), bool))

    @staticmethod
    def timers(obs):
        return obs.ctx[:, [C["dealt_any"], C["dealt_since"], C["taken_any"], C["taken_since"]]]

    def test_the_timer_restarts_at_each_damage_per_side_and_per_battle(self):
        setup, s = self.states()
        mem = {1: None, 2: None}

        def look(t, hp):
            out = {}
            for side in (1, 2):
                obs, mem[side] = ob.observe(with_state(s, t=np.array([t, t]), hp=np.array(hp)), setup, side, mem[side])
                out[side] = self.timers(obs)
            return out
        o = look(10.0, [[0.9] * 4, [0.9] * 4])
        assert o[1].tolist() == [[0, 0, 0, 0]] * 2 and o[2].tolist() == [[0, 0, 0, 0]] * 2
        o = look(20.0, [[0.9, 0.9, 0.8, 0.9], [0.9] * 4])               # row 0: side 1 strikes an enemy
        assert o[1][0].tolist() == [1, 0, 0, 0] and o[2][0].tolist() == [0, 0, 1, 0]
        assert o[1][1].tolist() == [0, 0, 0, 0] and o[2][1].tolist() == [0, 0, 0, 0]   # row 1: nothing
        o = look(80.0, [[0.9, 0.9, 0.8, 0.9], [0.9] * 4])               # a pause: no new damage
        assert o[1][0].tolist() == pytest.approx([1, 60 / ob.SINCE, 0, 0])
        assert o[2][0].tolist() == pytest.approx([0, 0, 1, 60 / ob.SINCE])
        o = look(90.0, [[0.7, 0.9, 0.7, 0.9], [0.9] * 4])               # both sides strike
        assert o[1][0].tolist() == pytest.approx([1, 0, 1, 0])
        o = look(500.0, [[0.7, 0.9, 0.7, 0.9], [0.9] * 4])
        assert o[1][0].tolist() == pytest.approx([1, 1, 1, 1])          # capped
        o = look(510.0, [[0.7, 0.9, 0.6, 0.9], [0.9] * 4])               # damage sets it back to 0
        assert o[1][0].tolist() == pytest.approx([1, 0, 1, 1]) and o[1][1].tolist() == [0, 0, 0, 0]

    def test_rising_or_unknown_health_is_no_damage_and_a_new_memory_starts_empty(self):
        setup, s = self.states(batch=1)
        _, mem = ob.observe(with_state(s, t=np.array([1.0])), setup, 1)
        obs, mem = ob.observe(with_state(s, t=np.array([2.0]), hp=np.array([[0.9, 0.9, 1.0, np.nan]])), setup, 1, mem)
        assert self.timers(obs).tolist() == [[0, 0, 0, 0]]
        obs, _ = ob.observe(with_state(s, t=np.array([3.0]), hp=np.array([[0.9, 0.9, 1.0, 0.5]])), setup, 1, mem)
        assert self.timers(obs).tolist() == [[0, 0, 0, 0]]             # unknown, then known: nothing
        obs, _ = ob.observe(with_state(s, t=np.array([3.0]), hp=np.array([[0.9, 0.9, 0.1, 0.1]])), setup, 1)
        assert self.timers(obs).tolist() == [[0, 0, 0, 0]]             # no memory: no previous health

    def test_the_context_has_no_time_limit_only_elapsed_time(self):
        setup, s = self.states(batch=1)
        ctx = {t: ob.observe(with_state(s, t=np.array([t])), setup, 1)[0].ctx[0] for t in (0.0, 30.0, 600.0, 3000.0, 7200.0)}
        assert ctx[0.0][C["clock_fine"]] == 0 and ctx[30.0][C["clock_fine"]] == pytest.approx(np.log(2) / np.log(21))
        assert ctx[600.0][C["clock_fine"]] == pytest.approx(1)
        assert np.array_equal(ctx[600.0], ctx[3000.0]) and np.array_equal(ctx[3000.0], ctx[7200.0])
        others = [i for i in range(ob.CONTEXT) if i != C["clock_fine"]]
        assert np.array_equal(ctx[0.0][others], ctx[30.0][others])

    def test_a_recorded_battle_and_the_same_states_live_give_the_same_timers(self):
        """sources.batch (per-second samples of a recording) and observing the states one by one."""
        setup, s = self.states(batch=1)
        hp = [[0.9] * 4, [0.9, 0.9, 0.8, 0.9], [0.9, 0.9, 0.8, 0.9], [0.8, 0.9, 0.8, 0.9], [0.8, 0.9, 0.8, 0.9]]

        f = {k: np.repeat(np.asarray(s[k], float if k in gamedata.FLOAT_FIELDS else bool), 5, 0)
             for k in gamedata.FLOAT_FIELDS + gamedata.BOOL_FIELDS}
        f["hp"] = np.array(hp)
        rec = sources.batch([gamedata.Battle(run="none", own_ai="attack", enemy_role="defend", result={},
                                             t=np.arange(5.0), f=f, target=np.full((5, 4), -1),
                                             names=("a", "b", "c", "d"), keys=tuple(setup.keys[0]),
                                             side=np.array([1, 1, 2, 2]))])
        live = recorded = None
        for ti in range(5):
            a, live = ob.observe(with_state(s, t=np.array([float(ti)]), hp=np.array([hp[ti]])), setup, 1, live)
            b, recorded = ob.observe(rec.state(ti), rec.setup, 1, recorded)
            assert np.allclose(self.timers(a), self.timers(b))
        assert self.timers(b)[0].tolist() == pytest.approx([1, 3 / ob.SINCE, 1, 1 / ob.SINCE])


class TestProgress:
    """The attacker's progress (observation PROGRESS): the reward's idle clock - its damage rate in the
    defender's gold, and when that rate last reached RATE_MIN - seen by both sides."""

    KEYS = TestEvents.KEYS                       # side 1: general, spearmen; side 2: warlord, clanrats

    def setup_state(self, attacker=1):
        _, state = sources.synthetic(batch=1, own=2, enemy=2, seed=3)
        setup = ob.Setup(keys=[list(self.KEYS)], side=np.array([[1, 1, 2, 2]]),
                         bounds=np.array([sources.CROSSROADS], np.float32),
                         factions=[(sources.EMPIRE, sources.SKAVEN)], attacker=np.array([attacker]))
        state = with_state(state, men=np.array([[1.0, 100, 1, 100]]), hp=np.ones((1, 4)), r=np.zeros((1, 4), bool),
                           s=np.zeros((1, 4), bool), ms=np.full((1, 4), 2.0), vis=np.ones((1, 4), bool))
        return setup, state

    @staticmethod
    def progress(obs):
        return obs.ctx[0, [C[n] for n in ob.PROGRESS]]

    def test_the_cost_is_the_passports(self):
        setup, _ = self.setup_state()
        want = [passport.load()[k]["multiplayer_cost"] for k in self.KEYS]
        assert setup.cost[0].tolist() == want and min(want) > 0

    def test_the_rate_is_the_defenders_gold_lost_a_minute_over_the_window_and_both_sides_see_it(self):
        setup, s = self.setup_state()
        cost = setup.cost[0]
        budget = cost.sum() / 2
        mem = {1: None, 2: None}
        rate, last, prev = 0.0, -1.0, None

        def look(t, share, **changes):
            nonlocal rate, last, prev
            if prev is not None:
                a = np.exp(-(t - prev) / ob.RATE_WINDOW)
                rate = a * rate + (1 - a) * share * 60 / (t - prev)
                last = t if rate >= ob.RATE_MIN else last
            prev = t
            want = [min(rate / ob.RATE_MIN, ob.RATE_CAP), float(last >= 0),
                    min((t - last) / ob.SINCE, 1) if last >= 0 else 0]
            for side in (1, 2):
                obs, mem[side] = ob.observe(with_state(s, t=np.array([t]), **changes), setup, side, mem[side])
                assert self.progress(obs).tolist() == pytest.approx(want, abs=1e-6), (t, side)
            return want

        assert look(0.0, 0) == [0, 0, 0]
        hp = np.array([[1, 1, 1, 0.8]])                                    # the attacker strikes the clanrats
        assert look(0.5, cost[3] * 0.2 / budget, hp=hp)[1] == 1            # a real blow: the clock resets
        own = np.array([[1, 0.5, 1, 0.8]])                                 # the defender's blows: not progress
        look(1.0, 0, hp=own)
        r = np.array([[False, False, False, True]])                        # the clanrats rout: half of the rest
        look(1.5, cost[3] * 0.8 * ob.ROUT_SHARE / budget, hp=own, r=r)
        look(2.0, 0, hp=own)                                               # they rally: no damage (not negative)
        look(2.25, 0, hp=own, r=r)                                         # rout again: counted once (audit B3)
        look(2.4, 0, hp=own)                                               # ... and rally again: nothing
        unknown = np.array([[1, 0.5, 1, np.nan]])
        look(2.5, 0, hp=unknown)                                           # not read: nothing
        look(3.0, 0, hp=np.array([[1, 0.5, 1, 0.5]]))                      # read again: the new reference (its worst
        #                                                                    0.6 of the rout stays: 0.5 lost is below it)
        look(3.2, cost[3] * 0.15 / budget, hp=np.array([[1, 0.5, 1, 0.5]]), r=r)   # routs at 0.5: 0.75, 0.15 beyond
        look(3.5, (cost[2] + cost[3] * 0.25) / budget, hp=np.array([[1, 0.5, 1, 0.5]]),
             men=np.array([[1.0, 100, 0, 100]]), s=np.array([[False, False, False, True]]))   # dead, shattered: whole
        for t in (60.0, 200.0, 600.0):                                     # no damage: the rate falls, the time grows
            want = look(t, 0, hp=np.array([[1, 0.5, 1, 0.5]]), men=np.array([[1.0, 100, 0, 100]]),
                        s=np.array([[False, False, False, True]]))
        assert want[0] < 1 and want[2] == 1

    def test_when_side_2_attacks_side_1s_losses_count(self):
        setup, s = self.setup_state(attacker=2)
        _, mem = ob.observe(with_state(s, t=np.array([0.0])), setup, 1)
        obs, mem = ob.observe(with_state(s, t=np.array([0.5]), hp=np.array([[1, 1, 1, 0.5]])), setup, 1, mem)
        assert self.progress(obs).tolist() == [0, 0, 0]
        obs, _ = ob.observe(with_state(s, t=np.array([1.0]), hp=np.array([[1, 0.5, 1, 0.5]])), setup, 1, mem)
        assert self.progress(obs)[0] > 1 and self.progress(obs)[1] == 1


class TestAbilities:
    """Abilities as a player sees them: own lord's bar (ready, timers), the enemy's only while active
    and seen; every ability by its passport (tools/nn/model/abilities.py)."""

    KEYS = TestEvents.KEYS
    A = __import__("tools.nn.model.abilities", fromlist=["x"])

    def setup_state(self, **timers):
        setup, state = TestEvents().setup_state()
        state = with_state(state, t=np.array([30.0]))
        for k in range(3):
            for t in ("on", "cd"):
                state[f"ab{k}_{t}"] = np.array(timers.get(f"ab{k}_{t}", [0.0] * 4), dtype=float)[None]
        return setup, state

    def test_the_own_lords_bar_and_the_passports_of_both_sides(self):
        setup, state = self.setup_state(ab1_on=[10.0, 0, 5.0, 0], ab1_cd=[100.0, 0, 70.0, 0])
        obs, _ = ob.observe(state, setup, 1)
        J = self.A.INDEX
        assert obs.abil.shape == (1, 4, 3, self.A.SIZE)
        lord = obs.abil[0, 0]
        assert lord[:, J["owned"]].tolist() == [1, 1, 1]
        assert lord[:, J["ready"]].tolist() == [1, 0, 0]               # Hold the Line is passive
        assert lord[1, J["recharge_left"]] == pytest.approx(100 / 60) and lord[1, J["active_left"]] == pytest.approx(10 / 30)
        assert obs.abil_ok[0].tolist() == [[True, False, False], [False] * 3, [False] * 3, [False] * 3]
        assert lord[0, J["active_s"]] == pytest.approx(25 / 60)       # Foe Seeker's passport
        enemy = obs.abil[0, 2]                                           # the Warlord: known abilities, no timers
        assert enemy[:, J["owned"]].tolist() == [1, 1, 1] and enemy[:, J["ready"]].tolist() == [0, 0, 0]
        assert enemy[1, J["recharge_left"]] == 0 and enemy[1, J["active_left"]] == 0
        assert enemy[:, J["active_now"]].tolist() == [0, 1, 0]           # seen active
        assert np.all(obs.abil[0, 1] == 0)                               # spearmen own nothing

    def test_an_enemy_ability_is_seen_only_while_the_unit_is(self):
        setup, state = self.setup_state(ab0_on=[0, 0, 5.0, 0], ab0_cd=[0, 0, 65.0, 0])
        hidden = with_state(state, vis=np.array([[True, True, False, True]]))
        obs, _ = ob.observe(hidden, setup, 1)
        assert obs.abil[0, 2, 0, self.A.INDEX["active_now"]] == 0
        full, _ = ob.observe(hidden, setup, 1, full=True)                 # the critic's view sees it
        assert full.abil[0, 2, 0, self.A.INDEX["active_now"]] == 1

    def test_no_timers_known_nothing_is_ready_and_a_routing_lord_uses_nothing(self):
        setup, state = self.setup_state()
        bare = {k: v for k, v in state.items() if not k.startswith("ab")}
        obs, _ = ob.observe(bare, setup, 1)
        assert not obs.abil_ok.any() and obs.abil[0, 0, :, self.A.INDEX["owned"]].tolist() == [1, 1, 1]
        routing = with_state(state, ms=np.array([[6.0, 2, 2, 2]]), r=np.array([[True, False, False, False]]))
        obs, _ = ob.observe(routing, setup, 1)
        assert not obs.abil_ok.any()

    def test_the_warlords_side_sees_its_own_bar(self):
        setup, state = self.setup_state()
        obs, _ = ob.observe(state, setup, 2)
        assert obs.abil_ok[0, 2].tolist() == [True, True, True] and not obs.abil_ok[0, :2].any()

    def test_a_setup_without_abilities_gives_no_ability_input(self):
        setup, state = self.setup_state()
        fields = ("side", "bounds", "rank", "lord_level", "passport", "men0", "ammo0", "present", "attacker", "lord")

        class Bare:                                   # as an older LiveSetup (tools/nn/train/scenes.py)
            def like(self, x):
                return ob._Arrays(**{k: getattr(setup, k) for k in fields})
            character = setup.character
        obs, _ = ob.observe(state, Bare(), 1)
        assert obs.abil is None and obs.abil_ok is None and obs.tokens.shape == (1, 4, ob.TOKEN)
