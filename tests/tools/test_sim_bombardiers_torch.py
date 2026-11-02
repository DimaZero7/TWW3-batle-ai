"""The Death Globe Bombardiers in the simulator (02.11.2026; build/diag_roster/wh2_main_skv_inf_death_globe_bombardiers/mechanic.md, build/diag_bombardiers/): the ring at 3.0 m, the knock by mass, the
cycle with no re-aim, the low arc over friends right before the shooters - and the probe's lanes (build/charge-probe/runs/20261102-110856): HP after a volley 0.38 / 0.44, knights 0.61, the cycle 11.6 s."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import artillery, battle, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
B, SP = "wh2_main_skv_inf_death_globe_bombardiers", "wh_main_emp_inf_spearmen_0"
G = "wh2_main_skv_inf_poison_wind_globadiers"
KN = "wh_main_emp_cav_empire_knights"
CL = "wh2_main_skv_inf_clanrat_spearmen_0"
SV = "wh2_main_skv_inf_stormvermin_0"


def army(foe=SP, gap=68.5, key=B, extra=()):
    return {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "ai": False, "units": [{"key": foe, "x": 0, "z": gap, "b": 180, "width": 30}]},
        2: {"faction": "wh2_main_skv_skaven", "ai": False, "units": [{"key": key, "x": 0, "z": 0, "b": 0, "width": 30}, *extra]}}}


def run(arm, secs, params=P, fearless=True):
    st = scenario.build([arm], params)
    H = st.N // 2
    u = st.u
    if fearless:
        u["leadership"][0, 0] += 1e4
        u["morale"][0, 0] += 1e4
    rows = []
    for _ in range(int(secs / params.dt)):
        o = O.hold(1, st.N)
        o.kind[0, H], o.target[0, H] = O.ATTACK, 0
        battle.step(st, o, params, params.dt)
        rows.append((float(st.t[0]), float(u["hp_abs"][0, 0] / u["hp0"][0, 0]), float(u["men"][0, 0]), float(u["a"][0, H]), float(u["lie_p"][0, 0])))
    return st, rows


def at(rows, t, col=1):
    return min(rows, key=lambda r: abs(r[0] - t))[col]


def test_the_ring_ends_at_3_m_and_its_least_hit_is_the_memorys():
    r, v = torch.tensor([4.0]), torch.tensor([15.0])
    assert float(artillery.ring_ticks(r, v)) == 2.0 and float(artillery.ring_reach(r, v)) == pytest.approx(3.0)
    # (the last man the 3 m ring reaches, body 0.65 m: 1 - (d - r_b) / r_e at d^2 = 3^2 + 0.65^2; x 42 HP: the memory's least hit 16 HP on spearmen and on knights)
    d = (3.0 ** 2 + 0.65 ** 2) ** 0.5
    assert (1 - (d - 0.65) / 4.0) * (9 * 0.85 + 33) == pytest.approx(16, abs=1.0)


def test_the_probes_near_lanes():
    """The game: spearmen 65.9-68.5 m, standing: 0.376 and 0.442 after a volley, 0.150 and 0.219 after two; dead 34 and 28 of 120. The simulator's 0.47 / 0.18."""
    st, rows = run(army(), 26.0)
    assert 0.376 <= at(rows, 14.0) <= 0.52
    assert 0.13 <= at(rows, 25.0) <= 0.23
    assert 75 <= at(rows, 14.0, 2) <= 95


def test_the_probes_knights():
    """The game: knights 74 m: 0.606 after a volley, 0.378 after two, 0.195 after three; the force of 65 does not move them."""
    st, rows = run(army(foe=KN, gap=74.0), 38.0)
    assert 0.56 <= at(rows, 14.0) <= 0.68
    assert 0.32 <= at(rows, 25.0) <= 0.44
    assert 0.14 <= at(rows, 37.0) <= 0.24
    assert max(r[4] for r in rows) == 0.0


def test_the_knock_by_mass():
    """Spearmen 90-100 kg and Stormvermin 150 kg are thrown down, knights 1000 kg not: the same rule, the mass (land's dv)."""
    for key in (SP, SV):
        st, rows = run(army(foe=key, gap=60.0), 9.0)
        assert at(rows, 8.0, 4) > 0.5, key
    # (the memory: 62-69 men of 90-120 down in 5.5 s; lie_s 5.8)
    st, rows = run(army(), 14.0)
    assert 0.6 <= at(rows, 8.0, 4) <= 0.85 and at(rows, 13.0, 4) == 0.0


def test_the_cycle_has_no_re_aim_of_a_single_shells_volley():
    """The memory: 11.6 s between the starts while the target's men die, the step's 12.0 s here; without the exception 12.5, 13.0."""
    st, rows = run(army(), 40.0)
    falls = [rows[i][0] for i in range(1, len(rows)) if rows[i][3] < rows[i - 1][3] - 0.5]
    assert falls[0] == pytest.approx(3.0, abs=0.51)
    assert [b - a for a, b in zip(falls, falls[1:])][:3] == [12.0, 12.0, 12.0]
    assert rows[-1][3] == 864 - 4 * 48


def test_a_stream_keeps_its_re_aim():
    """land_w 0 (the Warpfire Throwers; a shooter, a gun): the wait counts as before, the exception is the window's."""
    assert P.static("wh2_main_skv_inf_warpfire_thrower", "wh2_main_skv_skaven")["land_w"] == 0.0
    st, rows = run(army(gap=45.0, key="wh2_main_skv_inf_warpfire_thrower"), 40.0)
    falls = [rows[i][0] for i in range(1, len(rows)) if rows[i][3] < rows[i - 1][3] - 0.5]
    starts = [falls[0]] + [b for a, b in zip(falls, falls[1:]) if b - a > 3.0]
    assert all(18.0 <= b - a <= 19.5 for a, b in zip(starts, starts[1:]))


def test_friends_right_before_the_shooters_leave_the_low_arc():
    """The game: clanrats overlapping the shooters' front ranks: 10-15 deg at 30 m/s, no hit on them. The twin keeps the low arc."""
    st, rows = run(army(extra=[{"key": CL, "x": 0, "z": 9.1, "b": 0, "width": 30}]), 7.0)
    H = st.N // 2
    assert not bool(st.u["sh_lb"][0, H])
    assert float(st.u["hp_abs"][0, H + 1] / st.u["hp0"][0, H + 1]) == 1.0


def test_the_own_unit_at_15_m():
    st, rows = run(army(gap=15.0), 7.0)
    H = st.N // 2
    assert float(st.u["hp_abs"][0, H] / st.u["hp0"][0, H]) == 1.0 and rows[-1][1] < 0.8


def test_the_morale_the_artillerys_and_no_minus_five():
    """The memory: -8, then -10; no -5 (the Globadiers' rule by the category)."""
    st, rows = run(army(), 10.0, fearless=False)
    u = st.u
    assert float(u["art_s"][0, 0]) < 30 and float(u["artd_s"][0, 0]) < 30
    assert float(u["under_fire_s"][0, 0]) > 8


def test_the_rout_within_2_s_of_the_game():
    """The game's spearmen with their morale, 65.9 m: rout at 11.7 s."""
    st = scenario.build([army(gap=65.9)], P)
    H = st.N // 2
    t = None
    for _ in range(int(16 / P.dt)):
        o = O.hold(1, st.N)
        o.kind[0, H], o.target[0, H] = O.ATTACK, 0
        battle.step(st, o, P, P.dt)
        if t is None and bool(st.u["r"][0, 0]):
            t = float(st.t[0])
    assert t is not None and 9.5 <= t <= 13.7


def test_off_there_is_no_class():
    off = P.with_cal("artillery", rounds={"on": 0})
    st, rows = run(army(), 7.0, params=off)
    # the sphere alone, the own hit's 6 + 16 on the few it strikes: no blast
    assert rows[-1][1] > 0.85


def test_the_globadiers_keep_their_numbers():
    st, rows = run(army(gap=66.7, key=G), 30.0)
    assert 0.66 <= at(rows, 15.0) <= 0.78
