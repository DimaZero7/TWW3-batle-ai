"""A lone man (a lord) and routers in melee by the game's rules (tools/nn/sim/lone_rout.py; config/nn/sim.json
contact.lone_rout, melee.hit_roll_offset): the hit chance (score - 1) / 100, the cut defence rounded half away from
zero, a routing lone man struck from the rear in full, a lone striker on a router at the rule in full and on his routing
attack target with the whole charge (no fatigue), only within his own reach; formations keep the measured pursuit.
The simulator's melee and the fighters lab's share. Needs torch: run in the container."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.fighters import melee as FM  # noqa: E402
from tools.nn.sim import battle, geometry, lone_rout, melee, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
GENERAL, WARLORD = "wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0"
SPEAR, SLAVE = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0"
BLOCK = dict(P.sim["contact"]["lone_rout"])
OLD = P.with_cal("contact", lone_rout=None)
OFF = float(P.sim["melee"]["hit_roll_offset"])


def army(side1, side2):
    def units(rows):
        return [{"key": k, "x": x, "z": z, "b": b} for k, x, z, b in rows]
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(side1)},
                                     2: {"faction": "wh2_main_skv_skaven", "units": units(side2)}}}


def _strike(params, striker, target, routing, back=True, order_on=True, fat=None):
    """(HP/s, HP a hit, state): `striker` (side 1) strikes `target` (side 2) it touches, the target's back to it or its
    face; routing or standing (melee.strikes); the striker's attack order on it (or on nobody)."""
    st = scenario.build([army([(striker, -1.0, 0, 90)], [(target, 1.0, 0, 90 if back else 270)])], params)
    H = st.N // 2
    st.u["r"][0, H] = routing
    st.u["order_kind"][0, 0] = O.ATTACK if order_on else O.HOLD
    st.u["order_target"][0, 0] = H if order_on else -1
    st.u["contact_s"] = torch.full_like(st.u["men"], 100.0)
    if fat is not None:                                     # a fatigue level laid on the striker (fatigue.effects)
        from tools.nn.sim import fatigue
        st.u["fat"][0, 0] = fat
        st.u["charge_bonus"] = st.u["charge_bonus"] * fatigue.stat_mult(st.u, params, "stat_charge_bonus")
    pw = geometry.pairwise(st.u, params.sim["formation"]["spacing_m"])
    contact = pw["enemy"] & torch.ones_like(pw["enemy"])
    rate, hit = melee.strikes(st.u, pw, contact, params, torch.zeros_like(st.u["men"]))[:2]
    return float(rate[0, 0, H]), float(hit[0, 0, H]), st


def test_the_hit_chance_less_one_and_the_rounded_cut():
    a, d = torch.tensor([20.0, 55.0, 0.0]), torch.tensor([34.0, 0.0, 90.0])
    assert OFF == 1.0
    assert melee.hit_chance(a, d, 1.0, offset=OFF).tolist() == pytest.approx([0.20, 0.89, 0.07])
    assert melee.round_half_away(torch.tensor([13.5, 16.5, 12.4, -2.5])).tolist() == [14.0, 17.0, 12.0, -3.0]
    assert melee.directional_defence(torch.tensor([45.0, 45.0, 45.0]), torch.tensor([0.3, 0.6, 1.0])).tolist() == \
        [14.0, 27.0, 45.0]


def test_the_block_holds_the_rules():
    cc = P.sim["contact"]
    assert BLOCK == {"direction": 1.0, "share": 1.0, "charge": 1, "own_reach": 1}
    assert cc["lord_direction"] == 0.0 and cc["pursuit_rate"] == pytest.approx(0.43)
    assert lone_rout.charge(cc, P.battle) == pytest.approx(P.battle["pursuit_charge_bonus_modifier"]) == 1.0
    assert lone_rout.direction(cc, torch.tensor([False, True])).tolist() == [0.0, 1.0]


@pytest.mark.parametrize("striker, target, hp_s", [(WARLORD, GENERAL, 54.0), (GENERAL, WARLORD, 57.0)])
def test_a_lord_on_a_routing_lord_the_rules_check(striker, target, hp_s):
    """The Warlord on a routing General: defence 45 x 0.3 -> 14, 35 + 50 + 35 (the charge) - 14 -> 90, p 0.89, a blow
    ~241 HP every 4 s -> ~54 HP/s; the General on a routing Warlord ~57."""
    rate, hit, st = _strike(P, striker, target, True)
    u = st.u
    dfn = float(melee.round_half_away(torch.tensor(float(u["defence"][0, 1]) *
                                                   P.battle["melee_defence_direction_penalty_coefficient_rear"])))
    score = min(90.0, 35 + float(u["attack"][0, 0]) + float(u["charge_bonus"][0, 0]) - dfn)
    p = (score - OFF) / 100
    assert p == pytest.approx(0.89)
    assert rate == pytest.approx(p / float(u["interval"][0, 0]) * hit, rel=1e-5)
    assert rate == pytest.approx(hp_s, rel=0.03)
    # the old rules: no rear rule on a lone man, the formations' pursuit share, no charge
    old = _strike(OLD, striker, target, True)[0]
    face = _strike(OLD, striker, target, False, back=False)[0]
    assert old == pytest.approx(face * P.sim["contact"]["pursuit_rate"], rel=1e-4)


def test_the_pursuits_charge_without_the_fatigue_and_only_on_the_attack_target():
    fresh = _strike(P, WARLORD, GENERAL, True)[0]
    tired = _strike(P, WARLORD, GENERAL, True, fat=4)[0]           # very tired: the charge bonus x0.75 laid on u
    assert tired == pytest.approx(fresh, rel=1e-5)
    other = _strike(P, WARLORD, GENERAL, True, order_on=False)[0]  # not his attack target: no charge
    no_charge = _strike(P.with_cal("contact", lone_rout={**BLOCK, "charge": 0}), WARLORD, GENERAL, True)[0]
    assert other == pytest.approx(no_charge, rel=1e-5) and other < fresh


def test_a_standing_duel_does_not_change():
    for back in (False, True):
        new = _strike(P, GENERAL, WARLORD, False, back=back)[0]
        old = _strike(OLD, GENERAL, WARLORD, False, back=back)[0]
        assert new == pytest.approx(old, rel=1e-6)
    face = _strike(P, GENERAL, WARLORD, False, back=False)[0]
    assert _strike(P, GENERAL, WARLORD, False, back=True)[0] == pytest.approx(face, rel=1e-6)


def test_infantry_on_a_routing_lord_keep_the_pursuit_share_from_the_rear_without_charge():
    rout, _, st = _strike(P, SPEAR, GENERAL, True)
    standing = _strike(P, SPEAR, GENERAL, False)[0]
    u = st.u
    rear = P.battle["melee_defence_direction_penalty_coefficient_rear"]
    att, dfn = float(u["attack"][0, 0]), float(u["defence"][0, 1])
    p_f = float(melee.hit_chance(torch.tensor(att), torch.tensor(dfn), offset=OFF))
    p_r = float(melee.hit_chance(torch.tensor(att), melee.round_half_away(torch.tensor(dfn * rear)), offset=OFF))
    miss, iv = float(P.sim["melee"]["miss_s"]), float(u["interval"][0, 0])
    per_s = lambda p: p / (p * iv + miss)                                  # a fighting man's blows a second
    assert rout == pytest.approx(standing * P.sim["contact"]["pursuit_rate"] * per_s(p_r) / per_s(p_f), rel=1e-4)


def test_a_lone_striker_on_a_routing_formation_in_full_with_the_charge():
    rout = _strike(P, GENERAL, SLAVE, True)[0]
    old = _strike(OLD, GENERAL, SLAVE, True)[0]
    no_charge = _strike(P.with_cal("contact", lone_rout={**BLOCK, "charge": 0}), GENERAL, SLAVE, True)[0]
    assert no_charge == pytest.approx(old / P.sim["contact"]["pursuit_rate"], rel=1e-5)   # the rule in full
    assert rout > no_charge                                                                 # and the charge
    other = _strike(P, GENERAL, SLAVE, True, order_on=False)[0]           # not his attack target: no charge
    assert other == pytest.approx(no_charge, rel=1e-5)


def test_the_labs_share_lone_on_a_router_in_full_infantry_at_the_pursuit_rate():
    st = scenario.build([army([(GENERAL, -1.0, 0, 90), (SPEAR, -1.0, 8, 90)], [(GENERAL, 1.0, 0, 90)])], P)
    u = st.u
    j = st.N // 2
    u["r"][0, j] = True
    u["order_kind"][0, :2], u["order_target"][0, :2] = O.ATTACK, j
    pw = geometry.pairwise(u, P.sim["formation"]["spacing_m"])
    permit = torch.zeros_like(pw["enemy"])
    permit[0, 1, j] = True                                                 # the spearmen alone (no rival lord)
    g, _ = FM.gates(u, P, permit, pw, torch.zeros_like(u["men"]))
    assert float(g[0, 0, j]) == pytest.approx(BLOCK["share"])
    assert float(g[0, 1, j]) == pytest.approx(P.sim["contact"]["pursuit_rate"])
    g_old, _ = FM.gates(u, OLD, permit, pw, torch.zeros_like(u["men"]))
    assert float(g_old[0, 0, j]) == pytest.approx(P.sim["contact"]["pursuit_rate"])


def _chase_step(params, dx):
    """The enemy General (side 2) routing dx m ahead of our General (centre to centre), his back to him, ours attacking
    him: one simulator step -> (the router's HP lost, his melee flag)."""
    st = scenario.build([army([(GENERAL, 0.0, 0, 90)], [(GENERAL, dx, 0, 90)])], params)
    H = st.N // 2
    st.u["r"][0, H], st.u["morale"][0, H], st.u["rout_count"][0, H] = True, -30.0, 1.0
    hp0 = float(st.u["hp_abs"][0, H])
    o = O.hold(st.B, st.N, st.device)
    o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
    battle.step(st, o, params)
    return hp0 - float(st.u["hp_abs"][0, H]), bool(st.u["m"][0, H])


def test_a_lone_chaser_strikes_a_lone_router_only_within_reach():
    """own_reach: 8 m behind (within the pursuit's 16 m box gap, beyond a lone man's reach) the chasing General strikes
    nothing, the router is still in melee (the chase); off, he strikes him across the gap. Within reach (2 m) he strikes
    him either way."""
    on = P.with_cal("contact", lone_rout={**BLOCK, "own_reach": 1})
    off = P.with_cal("contact", lone_rout={**BLOCK, "own_reach": 0})
    far_on, m_on = _chase_step(on, 8.0)
    far_off, m_off = _chase_step(off, 8.0)
    assert far_on == 0.0 and m_on
    assert far_off > 0.0 and m_off
    assert _chase_step(on, 2.0)[0] > 0.0
