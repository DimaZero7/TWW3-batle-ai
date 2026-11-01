"""effects.mult_add (config/nn/sim.json; R-work contact_effects_module.md 2.4, notes/p4/effects.md): the melee and missile
damage and the charge bonus lay their multipliers as the speed does: with two or more on a unit 1 + their (v - 1) summed."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import abilities, effects, geometry, scenario
from tools.nn.sim.params import load

P = load()
SWORD = "wh_main_emp_inf_swordsmen"
FRENZY, POISON = "wh_main_unit_passive_frenzy", "wh_main_unit_contact_poison"
ONSLAUGHT = "wh_main_character_abilities_deadly_onslaught"


def army():
    return {"name": "t", "attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": SWORD, "x": 0, "z": 0, "b": 0}]},
        2: {"faction": "wh2_main_skv_skaven", "units": [{"key": SWORD, "x": 0, "z": 300, "b": 180}]}}}


def laid(st, on, params):
    """effects.apply and abilities.apply with these effects on the first unit by force: (the fields as the step sees them)."""
    u = st.u
    E = effects.index(params)
    real = effects.delayed
    bits = [E[k] for k in on]

    def forced(u_, p_, T_, E_, on_, dt_):
        on_ = real(u_, p_, T_, E_, on_, dt_).clone()
        for b in bits:
            on_[0, 0, b] = True
        return on_
    effects.delayed = forced
    try:
        pw = geometry.pairwise(u, params.sim["formation"]["spacing_m"])
        same = u["side"][:, :, None] == u["side"][:, None, :]
        alive = (u["side"] > 0) & (u["men"] > 0)
        old = effects.apply(u, params, 0.0, alive & ~u["r"], torch.zeros_like(alive), pw["dist"], same)
        fx = old.pop(effects.SPEED, None)
        base = abilities.apply(u, params, 0.0, alive & ~u["r"], torch.zeros_like(alive), pw["dist"], same, fx_speed=fx)
        after = {k: u[k].clone() for k in ("damage", "ap_damage", "charge_bonus", "m_damage", "run")}
        abilities.restore(u, base)
        effects.restore(u, old)
    finally:
        effects.delayed = real
    assert effects.SPEED not in u
    return after


@pytest.fixture
def st():
    return scenario.build([army()], P)


@pytest.mark.parametrize("field", ["damage", "ap_damage"])
def test_frenzy_and_poison_add(st, field):
    v0 = float(st.u[field][0, 0])
    a = laid(st, [FRENZY, POISON], P)
    assert float(a[field][0, 0]) == pytest.approx(v0 * (1 + 0.1 - 0.15), rel=1e-5)     # 0.95
    b = laid(st, [FRENZY, POISON], P.with_cal("effects", mult_add=0))
    assert float(b[field][0, 0]) == pytest.approx(v0 * 1.1 * 0.85, rel=1e-5)           # the rule off: 0.935


def test_one_alone_as_before(st):
    v0 = float(st.u["damage"][0, 0])
    for p in (P, P.with_cal("effects", mult_add=0)):
        assert float(laid(st, [FRENZY], p)["damage"][0, 0]) == pytest.approx(v0 * 1.1, rel=1e-6)
        assert float(laid(st, [POISON], p)["damage"][0, 0]) == pytest.approx(v0 * 0.85, rel=1e-6)


def test_the_speed_rule_alone_leaves_the_damage_a_product(st):
    v0 = float(st.u["damage"][0, 0])
    p = P.with_cal("effects", mult_add=0, speed_add=1)
    assert float(laid(st, [FRENZY, POISON], p)["damage"][0, 0]) == pytest.approx(v0 * 1.1 * 0.85, rel=1e-5)


def test_the_innate_and_an_ability_add_too(st):
    """Frenzy x1.1 on the unit with Deadly Onslaught x1.25 / charge x1.6 fired by it: 1.35 / 1.7."""
    u = st.u
    a = abilities.keys(P).index(ONSLAUGHT)
    u["ab0"][0, 0] = a
    u["ab0_on"][0, 0] = 5.0
    v0, c0 = float(u["damage"][0, 0]), float(u["charge_bonus"][0, 0])
    x = laid(st, [FRENZY], P)
    assert float(x["damage"][0, 0]) == pytest.approx(v0 * 1.35, rel=1e-5)
    assert float(x["charge_bonus"][0, 0]) == pytest.approx(c0 * 1.7, rel=1e-5)
    y = laid(st, [FRENZY], P.with_cal("effects", mult_add=0))
    assert float(y["damage"][0, 0]) == pytest.approx(v0 * 1.1 * 1.25, rel=1e-5)
    assert float(y["charge_bonus"][0, 0]) == pytest.approx(c0 * 1.1 * 1.6, rel=1e-5)
