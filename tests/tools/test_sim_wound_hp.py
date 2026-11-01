"""kills.wound_hp (melee.fresh_hit / fresh_kill / wound_hp, battle.py): a melee hit takes up to the struck man's health,
so a fresh unit's men take the whole hit until its pool of wounded fills (the game's memory, build/diag_mem/melee)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, melee  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

from tests.tools.test_sim_open_rules import SWORD, CLANRATS, face_off, orders  # noqa: E402

P = load()


def _mc(base, ap, armour, hp, n=400_000):
    """Monte Carlo of the hit X (armour roll U(0.5, 1) x armour % on the base part) and min(X, hp)."""
    g = torch.Generator().manual_seed(0)
    roll = 0.5 + 0.5 * torch.rand(n, generator=g, dtype=torch.float64)
    x = ap + base * (1 - (armour / 100 * roll).clamp(max=1))
    return float(torch.minimum(x, torch.tensor(hp, dtype=torch.float64)).mean()), float((x >= hp).double().mean())


@pytest.mark.parametrize("base, ap, armour, hp", [(21.0, 7.0, 25.0, 60.0),     # swordsmen on clanrats: below the health
                                                   (54.6, 23.4, 25.0, 60.0),   # knights in full charge: always a kill
                                                   (49.9, 21.4, 25.0, 60.0),   # knights at 0.86: straddles the health
                                                   (25.0, 9.0, 30.0, 69.0)])
def test_the_fresh_hit_is_the_hit_up_to_the_mans_health(base, ap, armour, hp):
    t = lambda v: torch.tensor([v], dtype=torch.float64)
    r = float(melee.fresh_hit(t(base), t(ap), t(armour), t(hp)))
    q = float(melee.fresh_kill(t(base), t(ap), t(armour), t(hp)))
    m, k = _mc(base, ap, armour, hp)
    assert r == pytest.approx(m, abs=0.05) and q == pytest.approx(k, abs=0.01)
    # never above per_hit's mean, never below the long fight's HP a hit
    mu = float(melee.per_hit(t(base), t(ap), t(armour), t(1e9)))
    assert r <= mu * (1 + 1e-6) and r >= float(melee.per_hit(t(base), t(ap), t(armour), t(hp))) * (1 - 1e-6)


def test_a_lone_man_takes_the_mean_and_is_never_killed_outright():
    t = lambda v: torch.tensor([v], dtype=torch.float64)
    one = torch.tensor([True])
    assert float(melee.fresh_hit(t(40.0), t(10.0), t(20.0), t(30.0), single=one)) == pytest.approx(
        float(melee.per_hit(t(40.0), t(10.0), t(20.0), t(30.0), single=one)))
    assert float(melee.fresh_kill(t(40.0), t(10.0), t(20.0), t(30.0), single=one)) == 0.0


def _pool_unit(W, men=100.0, hp_man=60.0):
    u = {"hp_man": torch.tensor([[hp_man, hp_man]]), "men": torch.tensor([[men, men]]),
         "men0": torch.tensor([[men, men]]), "hp_abs": torch.tensor([[men * hp_man, men * hp_man - W]])}
    hp = torch.zeros(1, 2, 2)
    hp[0, 0, 1] = 100.0                       # unit 0 strikes unit 1: 5 hits of 20 HP (the long fight's), fresh 25
    hit = torch.full((1, 2, 2), 20.0)
    fresh = torch.full((1, 2, 2), 25.0)
    F = torch.zeros(1, 2, 2)
    F[0, 1, 0] = 10.0                         # unit 1's own men striking back: the pool W* = 10 x (60 - 20) = 400
    return u, hp, hit, fresh, F


def test_wound_hp_fresh_takes_the_whole_hit_full_pool_the_old_rule_between_them_the_pool_algebra():
    for W, want in ((0.0, 125.0), (400.0, 100.0), (1000.0, 100.0), (390.0, 100.0 + 10.0 * (1 - 100 / 125))):
        u, hp, hit, fresh, F = _pool_unit(W)
        hp_n, hit_n = melee.wound_hp(u, hp, hit, fresh, F)
        assert float(hp_n[0, 0, 1]) == pytest.approx(want, rel=1e-5), W
        # the hits' count is kept
        assert float(hp_n[0, 0, 1] / hit_n[0, 0, 1]) == pytest.approx(5.0, rel=1e-5)
    # a lone man (men0 1) is one pool: unchanged
    u, hp, hit, fresh, F = _pool_unit(0.0, men=1.0)
    assert float(melee.wound_hp(u, hp, hit, fresh, F)[0][0, 0, 1]) == pytest.approx(100.0)


def test_a_fresh_unit_loses_more_hp_in_its_first_blows_and_the_switch_off_is_the_old_rule():
    H = None
    lost = {}
    for on in (0, 1):
        st = face_off(SWORD, CLANRATS)
        H = st.N // 2
        p = P.with_cal("kills", wound_hp=on)
        hp0 = float(st.u["hp_abs"][0, H])
        for _ in range(int(6 / p.dt)):
            battle.step(st, orders(st, **{"0": (O.ATTACK, H, False)}), p)
        lost[on] = hp0 - float(st.u["hp_abs"][0, H])
    assert lost[1] > lost[0] * 1.05
    # a unit whose pool is already full: the same HP either way (wounded beyond W*: every hit's surplus lost)
    out = []
    for on in (0, 1):
        st = face_off(SWORD, CLANRATS)
        st.u["hp_abs"][0, H] = float(st.u["hp_abs"][0, H]) - 60 * float(st.u["hp_man"][0, H]) * 0.6
        hp0 = float(st.u["hp_abs"][0, H])
        battle.step(st, orders(st, **{"0": (O.ATTACK, H, False)}), P.with_cal("kills", wound_hp=on))
        battle.step(st, orders(st, **{"0": (O.ATTACK, H, False)}), P.with_cal("kills", wound_hp=on))
        out.append(hp0 - float(st.u["hp_abs"][0, H]))
    assert out[1] == pytest.approx(out[0], rel=1e-5)
