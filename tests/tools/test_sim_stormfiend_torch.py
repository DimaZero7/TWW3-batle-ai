"""The Stormfiends with Grinderfists in the simulator (02.11.2026; build/diag_roster/wh3_dlc29_skv_mon_stormfiend_grinderfists/mechanic.md, the probe 'stormfiend' build/diag_stormfiend): no code of their own -
the rules by data: one press a battle, the Snare on enemies within 35 m, the owner stands and takes no order and strikes on, two owners' Snares side by side, the sparse rule's 2 bodies a place,
the AI presses in melee only. Needs torch (the container)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import abilities, battle, melee, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SF, SWORD, GR = "wh3_dlc29_skv_mon_stormfiend_grinderfists", "wh_main_emp_inf_swordsmen", "wh2_dlc14_skv_inf_warp_grinder_0"
SNARE, GSNARE = "wh3_dlc29_unit_abilities_stormfiend_seismic_snare", "wh2_dlc14_unit_abilities_seismic_snare"


def army(swords, fiends, ai=False, key=SF, grinders=()):
    """side 1: swordsmen at (x, z) facing +z; side 2: Stormfiends at (x, z) facing -z (the Grinders after them)."""
    return {"attacker": 2, "sides": {
        1: {"faction": "wh_main_emp_empire", "ai": False, "units": [{"key": SWORD, "x": x, "z": z, "b": 0} for x, z in swords]},
        2: {"faction": "wh2_main_skv_skaven", "ai": ai, "units": [{"key": key, "x": x, "z": z, "b": 180} for x, z in fiends] + [{"key": GR, "x": x, "z": z, "b": 180} for x, z in grinders]}}}


def slot(key=SNARE, unit=SF):
    return abilities.slot_keys(unit, P.units, P.abilities).index(key)


def orders(st, press=None, kind=None):
    o = O.hold(st.B, st.N, st.device)
    o.kind.fill_(O.KEEP)
    for i, k in (press or {}).items():
        o.ability[0, i] = k
    for i, (k, tg, x, z) in (kind or {}).items():
        o.kind[0, i], o.target[0, i], o.x[0, i], o.z[0, i] = k, tg, x, z
    return o


def step(st, o=None, params=P):
    battle.step(st, o if o is not None else orders(st), params, params.dt)


def steps(st, n, params=P):
    for _ in range(n):
        step(st, params=params)


def test_one_press_a_battle_the_second_does_nothing_and_the_timer_stays_frozen():
    st = scenario.build([army([(300, 0)], [(0, 20)])], P)
    u, H, k = st.u, st.N // 2, slot()
    assert int(u[f"ab{k}_n"][0, H]) == 1
    step(st, orders(st, {H: k}))
    assert int(u[f"ab{k}_n"][0, H]) == 0 and float(u[f"ab{k}_on"][0, H]) == pytest.approx(13, abs=0.3)
    steps(st, int(90 / P.dt))
    assert float(u[f"ab{k}_on"][0, H]) == 0
    assert float(u[f"ab{k}_cd"][0, H]) == pytest.approx(60, abs=1e-3)           # frozen at the recharge: never ready (the probe: the press at 85 s fails)
    step(st, orders(st, {H: k}))
    assert float(u[f"ab{k}_on"][0, H]) == 0 and int(u[f"ab{k}_n"][0, H]) == 0


def test_the_snare_holds_a_walker_within_35_m_and_nobody_beyond():
    st = scenario.build([army([(0, -10), (0, -40)], [(0, 20)])], P)              # 30 m and 60 m from the owner
    u, H, k = st.u, st.N // 2, slot()
    step(st, orders(st, kind={0: (O.MOVE, -1, 0.0, -200.0), 1: (O.MOVE, -1, 0.0, -200.0)}))
    steps(st, 4)
    step(st, orders(st, {H: k}))
    step(st)
    assert int(u["order_kind"][0, 0]) == O.HOLD and int(u["order_kind"][0, 1]) == O.MOVE
    z0, z1 = float(u["z"][0, 0]), float(u["z"][0, 1])
    steps(st, 8)
    assert float(u["z"][0, 0]) == pytest.approx(z0, abs=0.3) and float(u["z"][0, 1]) < z1 - 1.0


def test_the_owner_stands_takes_no_order_and_strikes_on():
    """The probe: 2.44 strikes a second before, 2.38 in the stance, 2.29 after it: the stance does not stop the blows. A move order given in it is refused and not taken up."""
    run = {}
    for press in (False, True):
        st = scenario.build([army([(0, 16)], [(0, 20)])], P)
        u, H, k = st.u, st.N // 2, slot()
        steps(st, 40)
        assert bool(u["m"][0, H])
        hp = float(u["hp_abs"][0, 0])
        step(st, orders(st, {H: k} if press else None))
        steps(st, int(10 / P.dt))                                                # within the 13 s
        run[press] = hp - float(u["hp_abs"][0, 0])
        if press:
            assert float(u[f"ab{k}_on"][0, H]) > 0
    assert run[True] == pytest.approx(run[False], rel=0.02)
    st = scenario.build([army([(300, 0)], [(0, 20)])], P)
    u, H, k = st.u, st.N // 2, slot()
    step(st, orders(st, {H: k}))
    step(st, orders(st, kind={H: (O.MOVE, -1, 0.0, 80.0)}))
    steps(st, 10)
    assert float(u["z"][0, H]) == pytest.approx(20.0, abs=0.3) and int(u["order_kind"][0, H]) == O.HOLD
    steps(st, int(14 / P.dt))
    assert float(u["z"][0, H]) == pytest.approx(20.0, abs=0.3)                  # not taken up after it


def test_two_owners_two_snares_side_by_side():
    """Different keys, two records on one enemy (mechanic.md 2.2): each sets its own bit, the hold ends with the longer."""
    st = scenario.build([army([(0, -5)], [(0, 20)], grinders=[(20, 20)])], P)   # 25 m and 32 m: both within 35 m
    u, H = st.u, st.N // 2
    step(st, orders(st, {H: slot(), H + 1: slot(GSNARE, GR)}))
    names = abilities.keys(P)
    bits = int(u["ab_en"][0, 0])
    assert (bits >> names.index(SNARE)) & 1 and (bits >> names.index(GSNARE)) & 1
    assert bool(abilities.snared(u, P)[0, 0])


def test_the_new_bit_keeps_the_table_order():
    names = abilities.keys(P)
    assert names == sorted(names)
    assert names.index(GSNARE) < names.index(SNARE) < names.index("wh_main_character_abilities_rally") and len(names) <= 62


@pytest.mark.parametrize("clear,k", [(0.25, 2.0), (0.26, 0.0)])
def test_two_bodies_a_place_on_the_edge(clear, k):
    st = scenario.build([army([(0, -30)], [(0, 20)])], P)
    u, H = st.u, st.N // 2
    h = torch.full_like(u["radius"], 6.0)
    cc = {"sparse": {"on": 1, "clear_m": clear, "min_size": 3}}
    assert float(melee.sparse_files(u, h, cc)[0, H, 0]) == k


def test_the_game_ai_presses_only_in_melee():
    st = scenario.build([army([(0, -15)], [(0, 20)], ai=True)], P)                 # within the 'near' 60 m and the 35 m: no melee, no press
    u, H, k = st.u, st.N // 2, slot()
    step(st, orders(st))
    assert float(u[f"ab{k}_on"][0, H]) == 0 and int(u[f"ab{k}_n"][0, H]) == 1
    st = scenario.build([army([(0, 14)], [(0, 20)], ai=True)], P)
    u, H, k = st.u, st.N // 2, slot()
    for _ in range(40):
        step(st, orders(st))
        if float(u[f"ab{k}_on"][0, H]) > 0:
            break
    assert bool(u["m"][0, H]) and float(u[f"ab{k}_on"][0, H]) > 0 and int(u[f"ab{k}_n"][0, H]) == 0


def test_a_battle_without_them_is_the_same_with_the_ability_out_of_the_model():
    a = army([(0, -10)], [(0, 20)], key="wh2_main_skv_inf_clanrats_1")
    off = P.with_cal("abilities", model=[k for k in P.sim["abilities"]["model"] if k != SNARE])
    s1, s2 = scenario.build([a], P), scenario.build([a], off)
    for _ in range(30):
        step(s1)
        step(s2, params=off)
    assert float(s1.u["hp_abs"][0, 0]) == float(s2.u["hp_abs"][0, 0]) and float(s1.u["hp_abs"][0, s1.N // 2]) == float(s2.u["hp_abs"][0, s2.N // 2])
