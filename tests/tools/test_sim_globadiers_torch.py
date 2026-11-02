"""The Globadiers in the simulator (02.11.2026; build/diag_roster/wh2_main_skv_inf_poison_wind_globadiers/mechanic.md, build/diag_globadiers/): the lob's 30 deg, the landing over a window, the ring's last tick
and its sphere, the poison by contact (direct.from_contact), the friends, the morale - and the probe's lanes: the game's HP of the spearmen after one volley 0.66-0.77, after two 0.37-0.50."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import artillery, battle, direct, effects, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
G, SP = "wh2_main_skv_inf_poison_wind_globadiers", "wh_main_emp_inf_spearmen_0"
W = "wh2_main_skv_inf_warpfire_thrower"
KN = "wh_main_emp_cav_empire_knights"
CL = "wh2_main_skv_inf_clanrat_spearmen_0"
POISON = "wh2_dlc14_unit_contact_poisoned_wind"
DT = P.dt


def army(foe=SP, gap=66.7, key=G, extra=()):
    return {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "ai": False, "units": [{"key": foe, "x": 0, "z": gap, "b": 180, "width": 30}]},
        2: {"faction": "wh2_main_skv_skaven", "ai": False, "units": [{"key": key, "x": 0, "z": 0, "b": 0, "width": 30}, *extra]}}}


def shoot(st, H):
    o = O.hold(1, st.N)
    o.kind[0, H], o.target[0, H] = O.ATTACK, 0
    return o


def run(arm, secs, params=P, fearless=True):
    st = scenario.build([arm], params)
    H = st.N // 2
    u = st.u
    if fearless:
        u["leadership"][0, 0] += 1e4
        u["morale"][0, 0] += 1e4
    rows = []
    for _ in range(int(secs / params.dt)):
        battle.step(st, shoot(st, H), params, params.dt)
        rows.append((float(st.t[0]), float(u["hp_abs"][0, 0] / u["hp0"][0, 0]), float(u["men"][0, 0]), float(u["a"][0, H]), float(u["cdd_sum"][0, 0])))
    return st, rows


def at(rows, t, col=1):
    return min(rows, key=lambda r: abs(r[0] - t))[col]


def slot(key=POISON):
    return next(j for j, _, k in effects.contact_keys(P) if k == key)


def test_the_low_arc_and_the_lob_at_60_m():
    u = {"m_g": torch.tensor([[6.5]]), "muzzle_v": torch.tensor([[30.0]]), "fix_el": torch.tensor([[0.0]]), "lob_el": torch.tensor([[30.0]])}
    d = torch.tensor([[60.0]])
    assert math.degrees(float(artillery.theta_low(u["muzzle_v"], d, u["m_g"]))) == pytest.approx(12.8, abs=0.1)
    assert float(artillery.flight(u, d)) == pytest.approx(2.05, abs=0.1)
    # (the lob: 30 deg, the speed that reaches 60 m: 21.2 m/s, 3.3 s)
    assert float(artillery.flight(u, d, torch.tensor([[True]]))) == pytest.approx(3.26, abs=0.05)
    assert math.sqrt(60.0 * 6.5 / math.sin(math.radians(60))) == pytest.approx(21.2, abs=0.05)


def test_a_closed_line_lobs_30_deg_at_the_speed_for_the_range(monkeypatch):
    # (the line closed: missile.clear_shot's lob; no lane of the probes closed it, in the game or in the twin: friends 21 m before the target and in melee with it, the low arc passes)
    from tools.nn.sim import missile
    monkeypatch.setattr(missile, "friend_rays", lambda u, pw, t, params: (t >= 0) & u["rd"])
    st = scenario.build([army(gap=62.0)], P)
    H = st.N // 2
    u = st.u
    u["leadership"][0, 0] += 1e4
    u["morale"][0, 0] += 1e4
    while float(u["sh_s"][0, H]) < 0:
        battle.step(st, shoot(st, H), P, DT)
    assert bool(u["sh_lb"][0, H])
    assert math.degrees(float(u["sh_el"][0, H])) == pytest.approx(30.0, abs=0.01)
    d = float(u["sh_d"][0, H])
    assert float(u["sh_s"][0, H]) == pytest.approx(math.sqrt(2 * d * math.tan(math.radians(30)) / 6.5), abs=0.51)


def test_friends_on_the_line_leave_the_low_arc():
    # (the probes: friends 21 m before the target: 13.9 deg, 30 m/s in the game)
    st = scenario.build([army(gap=62.0, extra=[{"key": CL, "x": 0, "z": 62.0 - 26.0, "b": 0, "width": 30}])], P)
    H = st.N // 2
    u = st.u
    while float(u["sh_s"][0, H]) < 0:
        battle.step(st, shoot(st, H), P, DT)
    assert not bool(u["sh_lb"][0, H]) and math.degrees(float(u["sh_el"][0, H])) < 15


def test_without_friends_it_stays_low():
    st = scenario.build([army(gap=62.0)], P)
    H = st.N // 2
    u = st.u
    while float(u["sh_s"][0, H]) < 0:
        battle.step(st, shoot(st, H), P, DT)
    assert not bool(u["sh_lb"][0, H]) and math.degrees(float(u["sh_el"][0, H])) < 15


def test_the_arc_of_a_lob_in_the_missile_rule_is_30_deg():
    from tools.nn.sim import missile
    st = scenario.build([army()], P)
    H = st.N // 2
    assert float(missile._lob_tan(st.u)[0, H, 0]) == pytest.approx(math.tan(math.radians(30.0)), abs=1e-4)
    # (a shooter without it: 45 deg, as before)
    assert float(missile._lob_tan(st.u)[0, 0, 0]) == pytest.approx(1.0, abs=1e-6)


def test_the_volley_lands_over_the_window():
    st = scenario.build([army()], P)
    H = st.N // 2
    u = st.u
    u["leadership"][0, 0] += 1e4
    u["morale"][0, 0] += 1e4
    seen = []
    for _ in range(int(10 / DT)):
        battle.step(st, shoot(st, H), P, DT)
        seen.append((float(st.t[0]), float(u["sh_left"][0, H]), float(u["sh_s"][0, H]), float(u["a"][0, H])))
    # all 48 leave at once (-48 ammunition), land after the flight at 2.4 s: 10 shells a step
    launched = [r for r in seen if r[3] < 864]
    assert launched[0][3] == 816 and launched[0][1] == 48
    falling = [r[1] for r in seen if 0 < r[1] < 48]
    assert falling[0] == pytest.approx(38.0, abs=0.01) and falling[1] == pytest.approx(28.0, abs=0.01)
    assert seen[-1][1] == 0 and seen[-1][2] == -1


def test_without_a_window_the_volley_lands_at_once():
    p = P.with_cal("artillery", volley_land_s={})
    st, rows = run(army(), 8.0, p)
    assert float(st.u["sh_w"][0, st.N // 2]) == 0.0
    assert float(st.u["cdd_sum"][0, 0]) > 0


def test_the_ring_ends_at_its_last_whole_tick():
    r = torch.tensor([2.0, 3.0, 4.0, 6.0, 5.0])
    v = torch.tensor([15.0, 14.0, 15.0, 20.0, 11.0])
    assert artillery.ring_ticks(r, v).tolist() == [1.0, 2.0, 2.0, 3.0, 4.0]
    assert artillery.ring_reach(r, v).tolist() == pytest.approx([1.5, 2.8, 3.0, 6.0, 4.4], abs=1e-5)
    # (faster than the ring in one tick: one tick, at most its radius)
    assert float(artillery.ring_reach(torch.tensor([1.0]), torch.tensor([20.0]))) == pytest.approx(1.0)


def exposure_one(h, x=0.0):
    """One burst h m high x m from the middle of a 4 x 4 m cell of men of radius 0.65 (ring 2 m, reach 1.5 m)."""
    P_ = torch.tensor([[[x, 0.0]]])
    Cx = torch.zeros((1, 1, 1, 2))
    saved = artillery.CELLS
    artillery.CELLS = (1, 1)
    try:
        return artillery.exposure_round(P_, torch.tensor([[h]]), Cx, torch.tensor([[4.0]]), torch.tensor([[4.0]]), torch.zeros((1, 1)), torch.tensor([[0.65]]),
                                        torch.tensor([2.0]), torch.tensor([1.5]), 0.5)
    finally:
        artillery.CELLS = saved


def test_the_ring_is_a_sphere_round_the_burst():
    m1, m2, pk, _ = exposure_one(0.0)
    # on the ground: the men within sqrt(1.5^2 + 0.65^2) = 1.63 m of 16 m^2: 8.4 / 16
    assert float(m1 / (m2 / m1)) == pytest.approx(math.pi * (1.5 ** 2 + 0.65 ** 2) / 16, abs=0.01)
    # the closeness at 2 / 3 of the disc: 1 - (1.09 - 0.65) / 2
    assert float(m2 / m1) == pytest.approx(1 - (2 / 3 * 1.635 - 0.65) / 2, abs=0.01)
    assert float(pk) == pytest.approx(float(m1 / (m2 / m1)), abs=0.01)
    low = exposure_one(1.0)
    assert 0 < float(low[0]) < float(m1)
    # at the height of the heads it strikes nobody
    assert float(exposure_one(1.7)[0]) == 0.0
    # half on the cell's edge: half the men
    edge = exposure_one(0.0, x=2.0)
    assert float(edge[0]) == pytest.approx(float(m1) / 2, abs=1e-4)


def test_a_rank_stops_a_shell_or_lets_it_pass_and_a_low_one_comes_down_on_the_ground():
    st = scenario.build([army()], P)
    H = st.N // 2
    u = st.u
    u["leadership"][0, 0] += 1e4
    u["morale"][0, 0] += 1e4
    while float(u["sh_s"][0, H]) < 0:
        battle.step(st, shoot(st, H), P, DT)
    b, i = torch.tensor([0]), torch.tensor([H])
    F, D = artillery.round_box(u, P)
    hs = []
    Pts = artillery.bursts(u, F, D, b, i, hs, 0.49)[0]
    h = hs[0][0]
    H_head = float(u["height"][0, 0])
    # every height between the ground and the heads; the men's ring strikes only the low ones
    assert float(h.min()) == 0.0 and float(h.max()) <= H_head + 1e-3
    # (with a rank that stops every shell, higher bursts on average)
    hs1 = []
    artillery.bursts(u, F, D, b, i, hs1, 0.999)
    assert float(hs1[0][0].mean()) > float(h.mean())
    assert Pts.shape[0] == h.shape[0]


def test_one_contact_is_three_strikes_and_gone_at_12_s():
    st = scenario.build([army()], P)
    u = st.u
    j = slot()
    u[f"cfx{j}_s"][0, 0] = 12.0
    u[f"cdd{j}_by"][0, 0] = st.N // 2
    hp0 = float(u["hp_abs"][0, 0])
    strikes = 0
    t = 0.0
    while t < 13:
        touch = torch.zeros(1, st.N, st.N)
        h, k = direct.from_contact(u, P, DT, touch)
        if float(h.sum()) > 0:
            strikes += 1
        effects.lay(u, P, DT, torch.zeros_like(touch), torch.ones_like(touch), torch.zeros_like(touch), torch.ones_like(touch), torch.ones_like(touch), st.t, blast=torch.zeros_like(touch), touch=touch)
        t += DT
    # (the record's start is the contact's own strike; here the ticks at 4 and 8 s)
    assert strikes == 2 and float(u[f"cfx{j}_s"][0, 0]) == 0
    assert float(u["cdd_sum"][0, 0]) == pytest.approx(2 * 25 * 2.5, abs=1e-3) and float(u["hp_abs"][0, 0]) == hp0


def test_a_strike_is_25_men_at_2_5_hp_and_a_lone_man_2_5():
    st = scenario.build([army()], P)
    H = st.N // 2
    u = st.u
    touch = torch.zeros(1, st.N, st.N)
    touch[0, H, 0] = 1000.0                       # the rings of every tick of the step
    h, _ = direct.from_contact(u, P, DT, touch)
    ticks = DT / direct.TICK_S
    assert float(h[0, H, 0]) == pytest.approx(ticks * 25 * 2.5, abs=1e-3)
    assert float(h.sum()) == float(h[0, H, 0])    # (the shooter's, nobody else's)
    u["men"][0, 0] = 1.0
    u["hp_abs"][0, 0] = 69.0
    h, _ = direct.from_contact(u, P, DT, touch)
    assert float(h[0, H, 0]) == pytest.approx(ticks * 2.5, abs=1e-3)


def test_the_ticks_a_step_are_the_ring_ticks_of_the_shells():
    """6 shells a step, a ring of one tick, 5 ticks a step: 5 (1 - exp(-6 / 5)) strikes; a ring of two ticks: 5 (1 - exp(-12 / 5))."""
    st = scenario.build([army()], P)
    H = st.N // 2
    u = st.u
    touch = torch.zeros(1, st.N, st.N)
    touch[0, H, 0] = 6.0
    h, _ = direct.from_contact(u, P, DT, touch)
    assert float(h[0, H, 0]) == pytest.approx(5 * (1 - math.exp(-6 / 5)) * 25 * 2.5, abs=1e-2)
    u["ex_v"][0, H] = 10.0                        # (2 m at 10 m/s: 2 ticks)
    h, _ = direct.from_contact(u, P, DT, touch)
    assert float(h[0, H, 0]) == pytest.approx(5 * (1 - math.exp(-12 / 5)) * 25 * 2.5, abs=1e-2)


def test_the_magic_resistance_cuts_it_the_armour_does_not():
    st = scenario.build([army()], P)
    H = st.N // 2
    u = st.u
    touch = torch.zeros(1, st.N, st.N)
    touch[0, H, 0] = 1000.0
    base = float(direct.from_contact(u, P, DT, touch)[0][0, H, 0])
    u["armour"][0, 0] = 200.0
    assert float(direct.from_contact(u, P, DT, touch)[0][0, H, 0]) == pytest.approx(base)
    u["resist_magic"][0, 0] = 0.5
    # (round(HP x keep): 2 -> 1, 3 -> 2: a mean of 1.5 of 2.5)
    assert float(direct.from_contact(u, P, DT, touch)[0][0, H, 0]) == pytest.approx(base * 1.5 / 2.5, abs=1e-2)


def test_the_poison_lies_on_friends_but_a_gun_contact_does_not():
    assert effects.row(P, POISON)[effects.COL["ct_allies"]] == 1.0
    assert effects.row(P, "wh_main_unit_contact_poison")[effects.COL["ct_allies"]] == 0.0
    st = scenario.build([army(extra=[{"key": CL, "x": 0, "z": 58, "b": 180, "width": 30}])], P)
    H = st.N // 2
    u = st.u
    fr = H + 1
    assert float(u["side"][0, fr]) == float(u["side"][0, H])
    touch = torch.zeros(1, st.N, st.N)
    touch[0, H, fr] = 3.0
    zero = torch.zeros_like(touch)
    enemy = (u["side"][:, :, None] != u["side"][:, None, :]).float()
    j = slot()
    effects.lay(u, P, DT, zero, torch.ones_like(zero), zero, torch.ones_like(zero), enemy, st.t, blast=zero, touch=touch)
    assert float(u[f"cfx{j}_s"][0, fr]) == 12.0


def test_friends_on_the_way_stop_the_shells_and_their_poison_is_their_own_loss():
    # (friends in contact with the target on the line of fire: both lose health, the friends' poison timer runs)
    extra = [{"key": CL, "x": 0, "z": 66.7 - 10, "b": 0, "width": 30}]
    st, rows = run(army(extra=extra), 12.0)
    u = st.u
    fr = st.N // 2 + 1
    assert float(u["hp_abs"][0, fr]) < float(u["hp0"][0, fr]) and float(u[f"cfx{slot()}_s"][0, fr]) > 0
    assert float(u["cdd_sum"][0, fr]) > 0


def test_the_poison_is_laid_at_the_landing_not_at_the_launch():
    st = scenario.build([army()], P)
    H = st.N // 2
    u = st.u
    u["leadership"][0, 0] += 1e4
    u["morale"][0, 0] += 1e4
    j = slot()
    while float(u["sh_s"][0, H]) < 0:
        battle.step(st, shoot(st, H), P, DT)
    # (the own hit at the launch: HP lost, no poison yet)
    assert float(u["hp_abs"][0, 0]) < float(u["hp0"][0, 0]) and float(u[f"cfx{j}_s"][0, 0]) == 0
    for _ in range(int(4 / DT)):
        battle.step(st, shoot(st, H), P, DT)
    assert float(u[f"cfx{j}_s"][0, 0]) > 0


def test_the_morale_the_artillerys_and_no_minus_five():
    st, rows = run(army(), 10.0, fearless=False)
    u = st.u
    assert float(u["art_s"][0, 0]) < 30 and float(u["artd_s"][0, 0]) < 30
    # (the memory's effects under them: only -8 / -10, like the mortar's: 'under fire' is never set by the sphere's own hit)
    assert float(u["under_fire_s"][0, 0]) > 8


def test_the_warpfire_throwers_still_set_the_minus_five():
    st, rows = run(army(gap=45.0, key=W), 7.0, fearless=False)
    assert float(st.u["under_fire_s"][0, 0]) < 2


def test_the_cycle_and_the_ammunition():
    st, rows = run(army(), 32.0)
    falls = [rows[i][0] for i in range(1, len(rows)) if rows[i][3] < rows[i - 1][3] - 0.5]
    assert len(falls) >= 2 and all(13.5 <= b - a <= 15.0 for a, b in zip(falls, falls[1:]))
    assert rows[-1][3] <= 864 - 96


def test_the_probes_near_lane():
    """The game's spearmen, 66.7 m centre to centre, standing: HP after a volley 0.663 / 0.723 (far 0.742 / 0.766), after two 0.368 / 0.483; poison 980-1420 HP a volley."""
    st, rows = run(army(), 30.0)
    assert 0.66 <= at(rows, 15.0) <= 0.78
    assert 0.37 <= at(rows, 28.0) <= 0.52
    assert 1000 <= at(rows, 15.0, 4) <= 1500


def test_the_probes_knights():
    """The game's knights, 76.8 m: 0.605 after a volley, 0.268 after two; the twin less, the mount's second body is not there (named)."""
    st, rows = run(army(foe=KN, gap=76.8), 30.0)
    assert 0.60 <= at(rows, 15.0) <= 0.76
    assert 0.27 <= at(rows, 28.0) <= 0.45
    assert float(st.u["lie_p"][0, 0]) == 0.0      # (too heavy to lie)


def test_a_battle_without_the_class_keeps_the_new_fields_at_zero():
    st = scenario.build([army(foe=SP, key="wh2_main_skv_inf_gutter_runners_0")], P)
    H = st.N // 2
    for _ in range(int(20 / DT)):
        battle.step(st, shoot(st, H), P, DT)
    u = st.u
    assert float(u["cdd_sum"].abs().sum()) == 0 and float(u["sh_left"].abs().sum()) == 0


def test_the_drills_leave_them_out():
    from tools.nn.unit import drill_hold, drill_shoot
    assert G in drill_hold.NOT_IN_DRILLS
    assert G not in drill_hold.roster()["skv"] and G not in drill_shoot.shooters()["skv"]


def test_off_there_is_no_class():
    off = P.with_cal("artillery", rounds={"on": 0})
    st = scenario.build([army()], off)
    assert not bool(st.u["rd"].any())
