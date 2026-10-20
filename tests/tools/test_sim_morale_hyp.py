"""Morale rules checked by the probe plan hyp and the recordings (docs/en/game/mechanics/morale.md, config/nn/sim.json
morale: waver_band, waver_timer.floor, shatter_by, routing_net, combat_even, rally_weak, strong_enemy.speed_ratio).
Each test sets the switch it tests explicitly in both states. Needs torch (the container)."""
import copy

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, morale, replay, scenario  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
R = P.morale
SPEAR, SLAVE = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslaves_0"


def army(side1, side2):
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3], "general": False} for r in rows]
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(side1)},
                                      2: {"faction": "wh2_main_skv_skaven", "units": units(side2)}}}


def with_morale(params, **kw):
    m = copy.deepcopy(params.sim["morale"])
    m.update(kw)
    return params.with_cal("morale", **m)


class TestWaverBand:
    def test_off_wavers_below_16_both_ways(self):
        M = torch.tensor([[11.0, 13.0, 15.0, 17.0]])
        can = torch.ones_like(M, dtype=torch.bool)
        w = morale.wavering(M, torch.zeros_like(can), can, R, False)
        assert w.tolist() == [[True, True, True, False]]

    def test_on_starts_below_12_and_ends_above_16(self):
        M = torch.tensor([[11.0, 12.0, 13.0, 16.0, 17.0]])
        can = torch.ones_like(M, dtype=torch.bool)
        fresh = morale.wavering(M, torch.zeros_like(can), can, R, True)
        assert fresh.tolist() == [[True, False, False, False, False]]
        held = morale.wavering(M, torch.ones_like(can), can, R, True)
        assert held.tolist() == [[True, True, True, True, False]]


class TestWaverFloor:
    def fall(self, params, seconds=4.0):
        """Skavenslaves whose morale target is far below 0 (leadership -150) and an enemy far away: their points each
        step and the step they rout."""
        st = scenario.build([army([(SPEAR, -400, 0, 90)], [(SLAVE, 400, 0, 270)])], params)
        H = st.N // 2
        st.u["leadership"][0, H] = -150.0
        st.u["morale"][0, H] = 20.0
        pts, routed = [], None
        o = replay.hold(st)
        t = 0.0
        while t < seconds:
            battle.step(st, o, params)
            t += float(params.dt)
            pts.append(float(st.u["morale"][0, H]))
            if routed is None and bool(st.u["r"][0, H]):
                routed = t
        return pts, routed

    def test_the_points_stay_at_0_until_the_timer_is_up(self):
        on = with_morale(P, waver_timer=dict(P.sim["morale"]["waver_timer"], on=True, floor={"on": True}))
        pts, routed = self.fall(on)
        # the first step wavers (t 0.5 s), the timer is up 2.5 s later: the rout at 3.0 s, at 0 points until then
        assert routed == pytest.approx(3.0)
        before = pts[:int(round(routed / float(on.dt))) - 1]
        assert min(before) >= 0.0
        off = with_morale(P, waver_timer=dict(P.sim["morale"]["waver_timer"], on=True, floor={"on": False}))
        pts_off, routed_off = self.fall(off)
        # without the floor the points plunge below 0 at once and reach the broken band's -50 before the timer: the
        # unit shatters (morale.shatter_rules) at 1.5 s
        assert min(pts_off[:2]) < 0.0 and routed_off < routed


class TestShatterBy:
    def router(self, params, men_share, hp_share):
        st = scenario.build([army([(SPEAR, -400, 0, 90)], [(SLAVE, 400, 0, 270)])], params)
        H = st.N // 2
        st.u["r"][0, H] = True
        st.u["rout_count"][0, H] = 1.0
        st.u["rout_s"][0, H] = 5.0
        st.u["men"][0, H] = round(float(st.u["men0"][0, H]) * men_share)
        st.u["hp"][0, H] = hp_share
        st.u["hp_abs"][0, H] = hp_share * float(st.u["hp0"][0, H])
        st.u["morale"][0, H] = -10.0
        battle.step(st, replay.hold(st), params)
        return bool(st.u["s"][0, H])

    def test_health_or_men(self):
        hp = with_morale(P, shatter_rules=True, shatter_by="hp")
        men = with_morale(P, shatter_rules=True, shatter_by="men")
        assert self.router(hp, 0.10, 0.04)            # health below 5 %: shattered by health
        assert not self.router(men, 0.10, 0.04)       # 10 % of the men alive: not by men
        assert self.router(men, 0.04, 0.03)


def traced_terms(params, own_routers, foe_routers):
    """Spearmen at the centre; own_routers routing spearmen 40 m behind, foe_routers routing skavenslaves 50 m in front:
    the routing terms of the spearmen after one step."""
    rows1 = [(SPEAR, 0, 0, 90)] + [(SPEAR, -40, 30 * k, 90) for k in range(own_routers)]
    rows2 = [(SLAVE, 50, 30 * k, 270) for k in range(foe_routers)] or [(SLAVE, 900, 0, 270)]
    st = scenario.build([army(rows1, rows2)], params)
    H = st.N // 2
    for k in range(own_routers):
        st.u["r"][0, 1 + k] = True
    for k in range(foe_routers):
        st.u["r"][0, H + k] = True
    trace = []
    morale.TRACE = trace
    try:
        battle.step(st, replay.hold(st), params)
    finally:
        morale.TRACE = None
    return float(trace[-1]["routing_friends"][0, 0]), float(trace[-1]["routing_enemies"][0, 0])


class TestRoutingNet:
    def test_friends_and_enemies_cancel_only_with_the_rule(self):
        off = with_morale(P, routing_net={"on": False})
        on = with_morale(P, routing_net={"on": True})
        assert traced_terms(off, 2, 2) == (-6.0, 5.0)
        assert traced_terms(on, 2, 2) == (0.0, 0.0)
        assert traced_terms(on, 2, 4) == (0.0, 5.0)
        assert traced_terms(on, 2, 0) == (-6.0, 0.0)


class TestCombatEven:
    def test_losing_needs_a_share_balance_below_the_edge(self):
        cal = copy.deepcopy(P.sim["morale"])
        dealt, taken = torch.tensor([[600.0, 600.0]]), torch.tensor([[1000.0, 1000.0]])   # HP balance 0.6: losing
        own = torch.tensor([[0.10, 0.10]])
        foe = torch.tensor([[0.07, 0.05]])                                                 # share balance 0.7 / 0.5
        fight = torch.ones_like(own, dtype=torch.bool)
        cal["combat_even"] = {"on": False, "share": 0.6}
        off = morale.combat_points(dealt, taken, fight, cal, R, own_lost=own, foe_lost=foe)
        cal["combat_even"] = {"on": True, "share": 0.6}
        on = morale.combat_points(dealt, taken, fight, cal, R, own_lost=own, foe_lost=foe)
        assert off.tolist() == [[R["losing_combat"]] * 2]
        assert on.tolist() == [[0.0, R["losing_combat"]]]


class TestRallyWeak:
    def blocked(self, params, foe_men):
        """A routing spearmen unit (past the rally clock, morale above 0) with a skavenslave unit of foe_men men 30 m
        away: is the rally blocked (ctx enemy_near)?"""
        st = scenario.build([army([(SPEAR, 0, 0, 90)], [(SLAVE, 30, 0, 270)])], params)
        H = st.N // 2
        st.u["r"][0, 0] = True
        st.u["rout_s"][0, 0] = 30.0
        st.u["men"][0, H] = float(foe_men)
        st.u["hp"][0, H] = foe_men / float(st.u["men0"][0, H])
        st.u["hp_abs"][0, H] = float(st.u["hp"][0, H]) * float(st.u["hp0"][0, H])
        seen = []
        real = morale.step

        def spy(u, ctx, p, dt):
            seen.append(bool(ctx["enemy_near"][0, 0]))
            return real(u, ctx, p, dt)
        morale.step = spy
        try:
            battle.step(st, replay.hold(st), params)
        finally:
            morale.step = real
        return seen[-1]

    def test_a_weak_enemy_blocks_only_without_the_rule(self):
        off = with_morale(P, rally_weak={"on": False, "ratio": 0.333})
        on = with_morale(P, rally_weak={"on": True, "ratio": 0.333})
        assert self.blocked(off, 6)
        assert not self.blocked(on, 6)
        assert self.blocked(on, 180)


class TestStrongSpeed:
    def test_a_slow_enemy_is_not_counted(self):
        se = dict(P.sim["morale"]["strong_enemy"], rule="balance", speed_ratio=None)
        params = with_morale(P, strong_enemy=se)
        u = {"cp_fixed": torch.tensor([[100.0, 800.0]]), "cp_missile": torch.zeros(1, 2),
             "a": torch.zeros(1, 2), "ammo0": torch.zeros(1, 2), "hp": torch.ones(1, 2),
             "run": torch.tensor([[3.78, 2.8]])}
        d = torch.tensor([[[0.0, 40.0], [40.0, 0.0]]])
        foes = torch.tensor([[[False, True], [True, False]]])
        friends = torch.zeros_like(foes)
        assert float(morale.strong_power(u, d, foes, friends, params)[0, 0]) > 0
        params = with_morale(P, strong_enemy=dict(se, speed_ratio=0.75))
        assert float(morale.strong_power(u, d, foes, friends, params)[0, 0]) == 0.0
