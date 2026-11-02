"""The class 'shells without guns' in the simulator (02.11.2026; artillery.py module doc, build/diag_roster/wh2_main_skv_inf_warpfire_thrower/
mechanic.md): the arc's gravity, the burst of 4 and its ammunition, the flight's wait, the aim, the resistances, the fire clock, the morale, no
holes - and the record's lane: one burst on 120 standing spearmen at 45 m, the game's health 0.3355 at 8 s."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import artillery, battle, geometry, melee, missile, regen, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
W, SP = "wh2_main_skv_inf_warpfire_thrower", "wh_main_emp_inf_spearmen_0"
RG = "wh2_dlc12_skv_inf_ratling_gun_0"
HOR = "wh2_dlc16_skv_mon_brood_horror_0"
MORTAR = "wh_main_emp_art_mortar"
DT = P.dt


def army(foe=SP, gap=45.0, key=W, ai=False, b=180):
    return {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "ai": ai, "units": [{"key": foe, "x": 0, "z": gap, "b": b, "width": 30}]},
        2: {"faction": "wh2_main_skv_skaven", "ai": ai, "units": [{"key": key, "x": 0, "z": 0, "b": 0, "width": 30}]}}}


def shoot(st, H):
    """The shooter of slot H attacks the target of slot 0 (a ranged attack), the rest hold."""
    o = O.hold(1, st.N)
    o.kind[0, H], o.target[0, H] = O.ATTACK, 0
    return o


def run(arm, secs, params=P, fearless=True):
    """The shooter (slot H) fires at the target (slot 0), the target holds. -> (state, [(t, HP share of the target, men, ammo, under_fire_s)])"""
    st = scenario.build([arm], params)
    H = st.N // 2
    u = st.u
    if fearless:
        u["leadership"][0, 0] += 1e4
        u["morale"][0, 0] += 1e4
    rows = []
    for _ in range(int(secs / params.dt)):
        battle.step(st, shoot(st, H), params, params.dt)
        rows.append((float(st.t[0]), float(u["hp_abs"][0, 0] / u["hp0"][0, 0]), float(u["men"][0, 0]), float(u["a"][0, H]), float(u["under_fire_s"][0, 0])))
    return st, rows


def at(rows, t, col=1):
    return min(rows, key=lambda r: abs(r[0] - t))[col]


def test_the_arc_at_80_m_is_the_projectiles_gravity():
    u = {"m_g": torch.tensor([[3.0]]), "muzzle_v": torch.tensor([[30.0]]), "fix_el": torch.tensor([[0.0]])}
    d = torch.tensor([[80.0]])
    th = artillery.theta_low(u["muzzle_v"], d, u["m_g"])
    assert math.degrees(float(th)) == pytest.approx(7.7, abs=0.1)
    assert float(artillery.flight(u, d)) == pytest.approx(2.69, abs=0.03)
    # (with 9.81 it would be 30 deg)
    assert math.degrees(float(artillery.theta_low(u["muzzle_v"], d))) == pytest.approx(30.3, abs=0.3)


def test_only_the_class_is_a_shell_without_guns():
    st = scenario.build([army(key=RG)], P)
    assert not bool(artillery.shell(st.u).any()) and not bool(artillery.stream(st.u).any())
    st = scenario.build([army(key=W)], P)
    H = st.N // 2
    assert bool(artillery.shell(st.u)[0, H]) and bool(artillery.stream(st.u)[0, H]) and float(st.u["eng0"][0, H]) == 0


def test_a_burst_is_four_shells_a_man_and_the_cycle_is_measured():
    st, rows = run(army(gap=45.0), 45.0)
    H = st.N // 2
    a0 = 2304.0
    assert float(st.u["ammo0"][0, H]) == a0
    ammo = [r[3] for r in rows]
    # the first burst: -128 (32 x 4) within the rate's window
    falls = [i for i in range(1, len(ammo)) if ammo[i] < ammo[i - 1] - 1e-6]
    first = [i for i in falls if rows[i][0] < rows[falls[0]][0] + 10]
    assert a0 - ammo[first[-1]] == pytest.approx(128.0, abs=0.5)
    assert (first[-1] - first[0] + 1) * DT == pytest.approx(4 / 1.42, abs=2 * DT)
    # the next burst one cycle later (the reload's clock from the burst's start): the memory's 17.97-18.63
    second = [i for i in falls if rows[i][0] > rows[first[-1]][0] + 3]
    assert (second[0] - first[0]) * DT == pytest.approx(18.3, abs=1.3)     # (17.2 + the wait of a re-aim and the steps)


def test_an_order_does_not_cut_the_burst_but_the_ratling_guns_is():
    st = scenario.build([army(key=W)], P)
    H = st.N // 2
    st.u["bst_s"][0, H] = 2.0
    missile.burst_cut(st.u, torch.ones_like(st.u["men"], dtype=torch.bool))
    assert float(st.u["bst_s"][0, H]) == 2.0
    st = scenario.build([army(key=RG)], P)
    st.u["bst_s"][0, H] = 2.0
    missile.burst_cut(st.u, torch.ones_like(st.u["men"], dtype=torch.bool))
    assert float(st.u["bst_s"][0, H]) == 0.0


def test_the_stream_arrives_after_the_flight():
    st = scenario.build([army(gap=80.0)], P)
    u = st.u
    H = st.N // 2
    pw = geometry.pairwise(u, P.sim["formation"]["spacing_m"])
    tgt = torch.full((1, st.N), -1)
    tgt[0, H] = 0
    dly = artillery.round_delay(u, pw, tgt)
    assert float(dly[0, H]) == pytest.approx(80.0 / 30.0, abs=0.5) and float(dly[0, 0]) == 0
    _, rows = run(army(gap=45.0), 12.0)
    fall = next(r[0] for i, r in enumerate(rows) if r[3] < 2304 - 1e-6)
    hit = next(r[0] for r in rows if r[1] < 0.9999)
    # the ammunition counts with the arrival: aim 3.6 s + the flight; the record's 5 % of health at 5.0 s
    assert hit == pytest.approx(fall, abs=2 * DT) and hit == pytest.approx(3.6 + 1.2, abs=2 * DT + 0.3)


def test_the_aim_is_over_the_front_of_the_shooters():
    st = scenario.build([army(gap=45.0)], P)
    u = st.u
    H = st.N // 2
    pw = geometry.pairwise(u, P.sim["formation"]["spacing_m"])
    shots = torch.zeros((1, st.N))
    shots[0, H] = 5.0
    tgt = torch.full((1, st.N), -1)
    tgt[0, H] = 0
    artillery.launch(u, pw, P, shots, tgt)
    f = float(pw["front"][0, H])
    assert f > 10 and float(u["sh_aa"][0, H]) == pytest.approx(min(f, float(pw["front"][0, 0])) / 2, rel=0.02)
    # no flight of its own (the burst waited it), a burst counts one aim of the zeroing-in
    assert float(u["sh_s"][0, H]) == 0 and float(u["sh_k"][0, H]) == pytest.approx(5.0 / (32 * 4), rel=1e-4)


def test_the_path_ends_at_the_expiry():
    st = scenario.build([army(gap=500.0)], P)     # (no target in the way)
    u = st.u
    H = st.N // 2
    pw = geometry.pairwise(u, P.sim["formation"]["spacing_m"])
    u["sh_d"][0, H], u["sh_el"][0, H] = 84.0, math.radians(8.0)
    u["sh_r"][0, H], u["sh_wz"][0, H], u["sh_wx"][0, H] = 30.0, 1.0, 0.0
    u["sh_z"][0, H] = 84.0
    u["sh_t"][0, H] = -1
    P_ = artillery.bursts(u, pw["front"], pw["depth"], torch.tensor([0]), torch.tensor([H]))
    assert float(P_[..., 1].max()) == pytest.approx(85.0, abs=0.01)


def test_the_resistances_are_the_games_table():
    st = scenario.build([army(foe=HOR)], P)
    u = st.u
    H = st.N // 2
    # the Horror's Regeneration: fire -20 % (the step's effects lay it): x1.2, in the sum; no physical in a magical shell, no spell resistance
    u["resist_flame"][0, 0] = -0.2
    rr = artillery.round_resist(u, u["ex_magic"], u["ex_spell"], u["ex_fire"])
    assert float(rr[0, H, 0]) == pytest.approx(-0.2 + float(u["resist_missile"][0, 0]))
    hit = melee.per_hit(torch.tensor(20.0), torch.tensor(7.0), torch.tensor(0.0), torch.tensor(5000.0), rr[0, H, 0], single=torch.tensor(True))
    assert float(hit) == pytest.approx(27.0 * 1.2, rel=1e-3)
    u["resist_physical"][0, 0], u["resist_magic"][0, 0] = 0.5, 0.5
    rr2 = artillery.round_resist(u, u["ex_magic"], u["ex_spell"], u["ex_fire"])
    assert float(rr2[0, H, 0]) == pytest.approx(float(rr[0, H, 0]))
    # a spell meets the spell resistance, a plain shell the physical; the ward always
    no = torch.zeros_like(u["ex_magic"])
    assert float(artillery.round_resist(u, u["ex_magic"], torch.ones_like(no), no)[0, H, 0]) == pytest.approx(0.5)
    assert float(artillery.round_resist(u, no, no, no)[0, H, 0]) == pytest.approx(0.5)
    u["resist_all"][0, 0] = 0.1
    assert float(artillery.round_resist(u, u["ex_magic"], no, no)[0, H, 0]) == pytest.approx(0.1)


def test_a_flaming_shell_and_its_blast_start_the_fire_clock():
    st = scenario.build([army(foe=HOR)], P)
    u = st.u
    H = st.N // 2
    N = st.N
    zero = torch.zeros((1, N, N))
    enemy = geometry.pairwise(u, P.sim["formation"]["spacing_m"])["enemy"].float()
    regen.burn(u, P, DT, zero, enemy)
    assert float(u["burn_s"][0, 0]) == 0
    for name in ("hp_missile", "hp_art"):
        u["burn_s"][0, 0] = 0
        hit = zero.clone()
        hit[0, H, 0] = 1.0
        regen.burn(u, P, DT, zero, enemy, **{name: hit})
        assert float(u["burn_s"][0, 0]) == P.sim["regen"]["burn_s"], name
    # a shooter without fire does not
    u["p_fire"][0, H] = False
    u["burn_s"][0, 0] = 0
    hit = zero.clone()
    hit[0, H, 0] = 1.0
    regen.burn(u, P, DT, zero, enemy, hp_missile=hit)
    assert float(u["burn_s"][0, 0]) == 0


def test_the_morale_no_artillery_terms_the_minus_five_and_flames():
    st, rows = run(army(gap=45.0), 9.0)
    u = st.u
    assert float(u["art_s"][0, 0]) > 1e5 and float(u["artd_s"][0, 0]) > 1e5        # (never marked)
    assert rows[-1][4] < 1.0                                                          # (the -5's clock runs from the hits)
    from tools.nn.sim import effects as fx
    idx = {k: i for _, i, k in fx.contact_keys(P)}["wh_main_unit_contact_morale"]
    slot = next(j for j, i, k in fx.contact_keys(P) if k == "wh_main_unit_contact_morale")
    assert idx >= 0 and float(u[f"cfx{slot}_s"][0, 0]) > 8.0


def test_a_mortars_morale_is_as_before():
    st = scenario.build([army(gap=150.0, foe=SP, key=MORTAR)], P)
    H = st.N // 2
    assert bool(st.u["cat_art"][0, H]) and not bool(st.u["rd"][0, H])
    st, _ = run(army(gap=150.0, foe=SP, key=MORTAR), 20.0)
    assert float(st.u["art_s"][0, 0]) < 1e5           # (a gun's fire marks its target)


def test_the_drills_leave_them_out():
    from tools.nn.unit import drill_hold, drill_shoot
    assert W in drill_hold.NOT_IN_DRILLS and W not in drill_shoot.shooters()["skv"] and W not in drill_hold.roster()["skv"]
    assert RG in drill_shoot.shooters()["skv"]


def test_the_men_lie_and_do_not_walk_nor_strike():
    st = scenario.build([army(gap=45.0)], P)
    u = st.u
    H = st.N // 2
    N = st.N
    knocks = torch.zeros((1, N, N))
    knocks[0, H, 0] = float(u["men"][0, 0]) * 0.5
    artillery.lie(u, P, DT, knocks)
    assert float(u["lie_p"][0, 0]) == pytest.approx(0.5) and float(u["lie_s"][0, 0]) == pytest.approx(5.8)
    assert float(u["lie_p"][0, H]) == 0
    # the way: x the standing share; a second knock on the rest; the mean of the seconds
    fast = torch.zeros_like(u["men"], dtype=torch.bool)
    assert float(missile.hit_slow_factor(u, P, fast)[0, 0]) == pytest.approx(0.5)
    knocks[0, H, 0] = float(u["men"][0, 0]) * 0.5
    artillery.lie(u, P, DT, knocks)
    assert float(u["lie_p"][0, 0]) == pytest.approx(0.75)
    assert float(u["lie_s"][0, 0]) == pytest.approx((0.5 * (5.8 - DT) + 0.25 * 5.8) / 0.75, rel=1e-4)
    # they rise
    zero = torch.zeros_like(knocks)
    for _ in range(int(6.0 / DT) + 1):
        artillery.lie(u, P, DT, zero)
    assert float(u["lie_p"][0, 0]) == 0
    # a gun's knock lays nobody by this rule
    st = scenario.build([army(gap=150.0, key=MORTAR)], P)
    knocks = torch.zeros((1, N, N))
    knocks[0, H, 0] = 50.0
    artillery.lie(st.u, P, DT, knocks)
    assert float(st.u["lie_p"].abs().sum()) == 0


def test_a_burst_lays_most_men_down_and_the_memorys_lying():
    st, _ = run(army(gap=44.8), 7.5)
    assert float(st.u["lie_p"][0, 0]) > 0.5


def test_the_second_burst_takes_less_with_fewer_men():
    # the memory: a man's hits a second stay, a shell's go with the men left: 0.643 then 0.263 of the health at 56 m
    _, rows = run(army(gap=56.3, foe="wh_main_emp_inf_swordsmen", b=177), 30.0)
    h1 = at(rows, 15.0)
    h2 = at(rows, 30.0)
    assert 1 - h1 == pytest.approx(0.643, rel=0.2)
    assert h1 - h2 < 0.75 * (1 - h1)


def test_the_box_of_these_shells_is_at_full_strength_and_grows_with_the_men_down():
    st = scenario.build([army(gap=45.0)], P)
    u = st.u
    f0, d0 = artillery.round_box(u, P)
    u["men"][0, 0] = 60.0
    f1, d1 = artillery.round_box(u, P)
    assert float(f1[0, 0]) == float(f0[0, 0]) and float(d1[0, 0]) == float(d0[0, 0])
    u["lie_p"][0, 0] = 0.5
    f2, d2 = artillery.round_box(u, P)
    assert float(f2[0, 0]) == pytest.approx(float(f0[0, 0]) * 1.075) and float(d2[0, 0]) == pytest.approx(float(d0[0, 0]) * 1.12)


def test_a_line_meets_the_box_on_its_own_edge():
    th = lambda *a: torch.tensor(a)
    # square on: the near edge at -D / 2 whatever the offset, off the front no meeting
    m, e, l = artillery.line_in_box(th(0.0, 10.0, 16.0), th(0.0, 0.0, 0.0), th(30.0, 30.0, 30.0), th(10.0, 10.0, 10.0))
    assert m.tolist() == [True, True, False] and e[:2].tolist() == [-5.0, -5.0] and l[:2].tolist() == [5.0, 5.0]
    # 30 deg off: the corner nearest is 12.0 m from the centre along the line; the line through the centre meets the box 5.77 m off
    phi = math.radians(30.0)
    _, e, l = artillery.line_in_box(th(0.0), th(phi), th(30.0), th(10.0))
    assert float(e[0]) == pytest.approx(-5 / math.cos(phi), abs=0.01) and float(l[0]) == pytest.approx(5 / math.cos(phi), abs=0.01)
    # the same damage whatever the target's facing within 30 deg (+- 12 %)
    base = at(run(army(gap=56.3, b=180), 12.0)[1], 12.0)
    for b in (170, 150):
        assert 1 - at(run(army(gap=56.3, b=b), 12.0)[1], 12.0) == pytest.approx(1 - base, rel=0.12)


def test_these_shells_start_no_holes():
    st, _ = run(army(gap=45.0), 9.0)
    assert float(st.u["hole_m"][0, 0]) == 0
    assert float(st.u["cellm"].abs().sum()) == 0


def test_it_fires_in_melee_and_its_own_ratling_stays():
    st = scenario.build([army(gap=45.0)], P)
    H = st.N // 2
    assert bool(st.u["melee_fire"][0, H])


def test_off_there_is_no_class():
    off = P.with_cal("artillery", rounds={"on": 0})
    st, rows = run(army(gap=45.0), 12.0, params=off)
    assert not bool(st.u["rd"].any()) and float(st.u["ex_r"].sum()) == 0
    assert at(rows, 12.0) > 0.9                     # (one shot a reload, its own hit alone: the old 3 %)


def test_the_records_lane():
    # the game (morale_probe --plan mem, battle 3, build/morale-probe/runs/20261101-050948): 0.324 from 8.5 s, 67 men; 0.97 at 5.0 s, 0.64 at 6.0 s
    # the health it lost +- 15 % (the twin +13 %; 50.6 m centre to centre, the record's)
    st, rows = run(army(gap=50.6, b=177), 12.0)
    assert 1 - at(rows, 10.0) == pytest.approx(1 - 0.324, rel=0.15)
    assert 5.0 <= next(r[0] for r in rows if r[1] < 0.9) <= 6.5
    # the men by the health left: the record's 0.558 of them at 0.324, the wounds spread over the unit gave 0.78
    assert at(rows, 10.0, 2) / 120 / at(rows, 10.0) == pytest.approx(0.558 / 0.324, rel=0.2)


def test_the_rout_of_the_record():
    # the game's spearmen: wavering at 7.5 s, routing at 10.5 s; +- 2 s
    st = scenario.build([army(gap=50.6, b=177)], P)
    u = st.u
    H = st.N // 2
    routed = None
    for _ in range(int(16 / DT)):
        battle.step(st, shoot(st, H), P, DT)
        if bool(u["r"][0, 0]) and routed is None:
            routed = float(st.t[0])
    assert routed == pytest.approx(10.5, abs=2.0)
