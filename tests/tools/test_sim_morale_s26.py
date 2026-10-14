"""Morale rules from the drift-free step replay (S26; docs/en/game/mechanics/morale.md, config/nn/sim.json morale):
the attacked side kept from the contact's start (morale.attacked_first), routing enemies counted to the measured reach
(morale.routing_enemies_reach), the charge's morale going with the unit's charging move (morale.charge.moving), the
fight's winning (morale.combat_shooter). ON by default since S26 (the charge's melee_start OFF); each test sets the
switch it tests explicitly in both states. Needs torch (the container)."""
import copy

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, morale, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SPEAR, SLAVE = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0"
TOUCH = min(0.0, P.sim["contact"]["reach_m"])


def army(side1, side2):
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3], "general": False} for r in rows]
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(side1)},
                                      2: {"faction": "wh2_main_skv_skaven", "units": units(side2)}}}


def with_morale(params, key, **kw):
    d = copy.deepcopy(params.sim["morale"].get(key) or {})
    d.update(kw)
    return params.with_cal("morale", **{key: d})


def strong(st, slots):
    for s in slots:                    # nobody routs in the test
        st.u["leadership"][0, s] += 1e4
        st.u["morale"][0, s] += 1e4


class TestAttackedFirst:
    def rear_then_turn(self, params):
        """Spearmen face east; slaves step into their back (contact begins on the rear); then the spearmen are turned
        to face the slaves: flank_hit after the turn."""
        st = scenario.build([army([(SPEAR, 0, 0, 90)], [(SLAVE, -60, 0, 90)])], params)
        H = st.N // 2
        front, depth = geometry.dims(st.u, params.sim["formation"]["spacing_m"])
        st.u["x"][0, H] = -(depth[0, 0] / 2 + depth[0, H] / 2 + TOUCH)
        strong(st, (0, H))
        o = replay.hold(st)
        o.kind[0, H], o.target[0, H] = O.ATTACK, 0
        for _ in range(4):
            battle.step(st, o, params)
        before = float(st.u["flank_hit"][0, 0])
        st.u["b"][0, 0] = 270.0
        for _ in range(2):
            battle.step(st, o, params)
        return before, float(st.u["flank_hit"][0, 0])

    def test_the_rear_contact_is_kept_after_the_turn_only_with_the_rule(self):
        before, after = self.rear_then_turn(with_morale(P, "attacked_first", on=False))
        assert before == 2 and after == 0
        before, after = self.rear_then_turn(with_morale(P, "attacked_first", on=True))
        assert before == 2 and after == 2


class TestRoutingEnemiesReach:
    def counted(self, params, monkeypatch, d):
        st = scenario.build([army([(SPEAR, 0, 0, 90)], [(SLAVE, d, 0, 90)])], params)
        H = st.N // 2
        st.u["r"][0, H] = True
        seen = []
        real = morale.step

        def spy(u, ctx, p, dt):
            seen.append(float(ctx["routing_enemies"][0, 0]))
            return real(u, ctx, p, dt)
        monkeypatch.setattr(morale, "step", spy)
        battle.step(st, replay.hold(st), params)
        return seen[-1]

    def test_a_router_at_150_m_counts_with_the_measured_reach(self, monkeypatch):
        off = with_morale(P, "routing_enemies_reach", on=False)
        assert self.counted(off, monkeypatch, 80) == 1 and self.counted(off, monkeypatch, 150) == 0
        on = with_morale(P, "routing_enemies_reach", on=True)
        m = float(on.sim["morale"]["routing_enemies_reach"]["m"])
        assert self.counted(on, monkeypatch, 150) == 1 and self.counted(on, monkeypatch, m + 30) == 0


class TestCombatShooter:
    def test_a_shooter_weighs_only_its_exchange_with_its_target(self, monkeypatch):
        """Archers shoot spearmen that do not shoot back; fire from others on the archers (shot_taken) makes them
        'losing' by all HP, but not with morale.combat_shooter (their target does not shoot them: +8 once it lost 10 %)."""
        ARCHER = "wh2_dlc13_emp_inf_archers_0"
        out = {}
        for on in (False, True):
            params = with_morale(P, "combat_shooter", on=on)
            st = scenario.build([army([(ARCHER, 0, 0, 90)], [(SPEAR, 100, 0, 270)])], params)
            H = st.N // 2
            strong(st, (0, H))
            o = replay.hold(st)
            o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
            for _ in range(40):
                battle.step(st, o, params)
            assert int(st.u["aim_tgt"][0, 0]) == H
            for s in (0, H):             # both have lost 20 % (the 10 % gates of losing and winning)
                st.u["hp_abs"][0, s] = 0.8 * st.u["hp0"][0, s]
                st.u["hp"][0, s] = 0.8
            st.u["shot_dealt"][0, 0] = 200.0
            st.u["shot_taken"][0, 0] = 2000.0
            seen = []
            real = morale.terms

            def spy(u, ctx, p):
                parts = real(u, ctx, p)
                seen.append(float(parts["combat"][0, 0]))
                return parts
            monkeypatch.setattr(morale, "terms", spy)
            battle.step(st, o, params)
            monkeypatch.setattr(morale, "terms", real)
            out[on] = seen[-1]
        assert out[False] < 0 and out[True] == P.morale["winning_combat_significantly"]


class TestChargeMoving:
    def melee(self, params):
        st = scenario.build([army([(SPEAR, -50, 0, 90)], [(SLAVE, 50, 0, 270)])], params)
        H = st.N // 2
        front, depth = geometry.dims(st.u, params.sim["formation"]["spacing_m"])
        st.u["x"][0, 0] = -(depth[0, 0] / 2 + TOUCH / 2)
        st.u["x"][0, H] = depth[0, H] / 2 + TOUCH / 2
        strong(st, (0, H))
        o = replay.hold(st)
        o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
        o.kind[0, H], o.target[0, H] = O.ATTACK, 0
        for _ in range(30):
            battle.step(st, o, params)
        assert bool(st.u["m"][0, 0])
        return st, o

    def test_a_unit_moving_in_melee_under_attack_gets_a_block(self):
        mc = P.sim["morale"]["charge"]
        for on in (False, True):
            params = P.with_cal("morale", charge={**mc, "moving": {**mc["moving"], "on": on, "melee_start": True}})
            st, o = self.melee(params)
            st.u["chm_s"][0, 0] = 0.0
            st.u["mv"][0, 0] = True
            battle.step(st, o, params)
            assert (float(st.u["chm_s"][0, 0]) == pytest.approx(mc["block_s"])) == on
            st.u["mv"][0, 0] = False
            st.u["chm_s"][0, 0] = 0.0
            battle.step(st, o, params)
            assert float(st.u["chm_s"][0, 0]) == 0

    def test_without_melee_start_no_block_starts_in_melee(self):
        mc = P.sim["morale"]["charge"]
        params = P.with_cal("morale", charge={**mc, "moving": {**mc["moving"], "on": True, "melee_start": False}})
        st, o = self.melee(params)
        st.u["chm_s"][0, 0] = 0.0
        st.u["mv"][0, 0] = True
        battle.step(st, o, params)
        assert float(st.u["chm_s"][0, 0]) == 0

    def test_blocks_go_on_while_the_unit_moves_with_no_cap(self):
        mc = P.sim["morale"]["charge"]
        params = P.with_cal("morale", charge={**mc, "moving": {**mc["moving"], "on": True}})
        st, o = self.melee(params)
        for mv, want in ((True, mc["block_s"]), (False, 0.0)):
            st.u["chm_s"][0, 0] = params.dt * 0.5
            st.u["chm_n"][0, 0] = 5.0
            st.u["mv"][0, 0] = mv
            battle.step(st, o, params)
            assert float(st.u["chm_s"][0, 0]) == pytest.approx(want)


def test_the_documented_defaults():
    # config/nn/sim.json morale: the S26 rules ON, a charge block not started in melee, the strong-enemy balance OFF
    m = P.sim["morale"]
    assert m["attacked_first"]["on"] and m["combat_shooter"]["on"] and m["routing_enemies_reach"]["on"]
    assert m["charge"]["moving"]["on"] and not m["charge"]["moving"]["melee_start"]
    assert m["strong_enemy"]["rule"] == "off"
