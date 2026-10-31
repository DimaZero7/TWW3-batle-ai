"""The Warp Grinders' rules in the simulator (31.10.2026; build/diag_roster/wh2_dlc14_skv_inf_warp_grinder_0/mechanic.md,
the probe 'grinder' build/diag_grinder): an active ability on the enemies within 35 m of its owner, held to 40 m, once per
enemy whatever the owners (tools/nn/sim/abilities.py); Warp-Quake's direct damage (direct.py from_abilities) and speed
x 0.75; Seismic Snare's no-move (abilities.snared); the owner standing and taking no order while either is active
(abilities.busy). Needs torch (the container)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import abilities, battle, direct, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
GR, SWORD = "wh2_dlc14_skv_inf_warp_grinder_0", "wh_main_emp_inf_swordsmen"
QUAKE, SNARE = "wh2_dlc14_unit_abilities_warp_quake", "wh2_dlc14_unit_abilities_seismic_snare"


def army(swords, grinders, ai=False):
    """side 1: swordsmen at (x, z) facing +z; side 2: Grinders at (x, z) facing -z."""
    return {"attacker": 2, "sides": {
        1: {"faction": "wh_main_emp_empire", "ai": False, "units": [{"key": SWORD, "x": x, "z": z, "b": 0}
                                                                    for x, z in swords]},
        2: {"faction": "wh2_main_skv_skaven", "ai": ai, "units": [{"key": GR, "x": x, "z": z, "b": 180}
                                                                  for x, z in grinders]}}}


def slot(key):
    return abilities.slot_keys(GR, P.units, P.abilities).index(key)


def orders(st, press=None, kind=None):
    """Hold for all; press {unit index: ability key} fires that slot; kind {unit index: (kind, target, x, z)}."""
    o = O.hold(st.B, st.N, st.device)
    o.kind.fill_(O.KEEP)
    for i, key in (press or {}).items():
        o.ability[0, i] = slot(key)
    for i, (k, tg, x, z) in (kind or {}).items():
        o.kind[0, i], o.target[0, i], o.x[0, i], o.z[0, i] = k, tg, x, z
    return o


def step(st, o=None):
    battle.step(st, o if o is not None else orders(st), P, P.dt)


def test_warp_quake_strikes_at_once_and_every_half_second_for_17_s():
    st = scenario.build([army([(0, -10)], [(0, 20)])], P)
    u, H = st.u, st.N // 2
    hp0 = float(u["hp_abs"][0, 0])
    step(st, orders(st, {H: QUAKE}))
    # 18 men x 3.2 HP (2 / 3 / 4) at once (the record's start), magical: the swordsmen's armour and shield do not cut it
    assert hp0 - float(u["hp_abs"][0, 0]) == pytest.approx(57.6, abs=1e-3)
    for _ in range(40):
        step(st)
    # 34 ticks of 0.5 s in 17 s (1958.4 HP: the probe's game 1952 / 1967), the owners' own count
    assert float(u["dq_sum"][0, 0]) == pytest.approx(34 * 57.6, abs=1e-2)
    assert float(u["dq_sum"][0, H]) == 0.0
    assert int(u["ab_en"][0, 0]) == 0                       # over: the record is gone


def test_the_target_is_held_to_40_m_and_a_new_one_needs_35():
    st = scenario.build([army([(0, -10), (200, 0)], [(0, 20)])], P)
    u, H = st.u, st.N // 2
    step(st, orders(st, {H: QUAKE}))
    got = lambda: float(u["dq_sum"][0, 0])
    u["z"][0, 0] = 20 - 38.0                                # 38 m: held (35 + 5)
    a = got()
    step(st)
    assert got() - a == pytest.approx(57.6, abs=1e-3)
    u["z"][0, 0] = 20 - 41.0                                # beyond 40 m: dropped at once
    a = got()
    step(st)
    assert got() == a and int(u["ab_en"][0, 0]) == 0
    u["z"][0, 0] = 20 - 38.0                                # back to 38 m: no record, not within 35 m - nothing
    step(st)
    assert got() == a
    u["z"][0, 0] = 20 - 34.0                                # within 35 m: a new record, a tick at once
    step(st)
    assert got() - a == pytest.approx(57.6, abs=1e-3)


def test_two_grinders_do_not_stack_and_the_speed_counts_once():
    st = scenario.build([army([(0, -10)], [(-12, 15), (12, 15)])], P)
    u, H = st.u, st.N // 2
    step(st, orders(st, {H: QUAKE, H + 1: QUAKE}))
    assert float(u["dq_sum"][0, 0]) == pytest.approx(57.6, abs=1e-3)
    # the kills' and the HP's owners: shared between the two
    walk = float(P.units[SWORD]["speed"]["walk"])
    from tools.nn.sim import geometry
    pw = geometry.pairwise(u, P.sim["formation"]["spacing_m"])
    same = u["side"][:, :, None] == u["side"][:, None, :]
    alive = (u["side"] > 0) & (u["men"] > 0)
    old = abilities.apply(u, P, P.dt, alive & ~u["r"], torch.zeros_like(alive), pw["dist"], same)
    assert float(u["walk"][0, 0]) == pytest.approx(walk * 0.75, abs=1e-5)       # x 0.75 once, not x 0.5625
    abilities.restore(u, old)


def test_the_magic_resistance_cuts_the_quake():
    one = torch.ones(1, 1)
    assert float(direct.hit_mean(5 * one, 0.5 * one, 0.75 * one, 5)) == pytest.approx(2.4, abs=1e-5)


def test_snared_units_stop_drop_their_move_and_refuse_a_new_one():
    st = scenario.build([army([(0, -15)], [(0, 20)])], P)
    u, H = st.u, st.N // 2
    step(st, orders(st, kind={0: (O.MOVE, -1, 0.0, 60.0)}))
    for _ in range(4):
        step(st)
    step(st, orders(st, {H: SNARE}))
    step(st)
    z = float(u["z"][0, 0])
    assert int(u["order_kind"][0, 0]) == O.HOLD                                  # the move in force is gone
    step(st, orders(st, kind={0: (O.MOVE, -1, 0.0, 60.0)}))                     # a new one refused
    for _ in range(10):
        step(st)
    assert float(u["z"][0, 0]) == pytest.approx(z, abs=0.3) and int(u["order_kind"][0, 0]) == O.HOLD
    for _ in range(20):                                                          # the 13 s over: it stands on
        step(st)
    assert float(u["z"][0, 0]) == pytest.approx(z, abs=0.3)
    step(st, orders(st, kind={0: (O.MOVE, -1, 0.0, 60.0)}))                     # free again: a move is taken
    for _ in range(6):
        step(st)
    assert float(u["z"][0, 0]) > z + 1.0


def test_a_snared_router_stands():
    # (a far-off standing friend keeps the battle going once the swordsmen rout)
    st = scenario.build([army([(0, -15), (400, -400)], [(0, 20)])], P)
    u, H = st.u, st.N // 2
    step(st, orders(st, {H: SNARE}))
    for _ in range(6):
        # (routed as the probe's twin routs by script: below the broken floor with the waver run out)
        if not bool(u["r"][0, 0]):
            u["morale"][0, 0] = float(P.morale["ums_broken_threshold_lower"]) - 10.0
            u["w"][0, 0], u["waver_s"][0, 0] = True, 100.0
        step(st)
    assert bool(u["r"][0, 0])
    x, z = float(u["x"][0, 0]), float(u["z"][0, 0])
    for _ in range(6):
        step(st)
    assert (float(u["x"][0, 0]), float(u["z"][0, 0])) == pytest.approx((x, z), abs=0.3)
    for _ in range(20):                                                          # the 13 s over: it runs on
        step(st)
    assert abs(float(u["z"][0, 0]) - z) + abs(float(u["x"][0, 0]) - x) > 3.0


def test_the_owner_stands_and_takes_no_order_while_its_ability_is_on():
    st = scenario.build([army([(300, 0)], [(0, 20)])], P)
    u, H = st.u, st.N // 2
    step(st, orders(st, {H: QUAKE}))
    step(st, orders(st, kind={H: (O.MOVE, -1, 0.0, 80.0)}))
    for _ in range(10):
        step(st)
    assert float(u["z"][0, H]) == pytest.approx(20.0, abs=0.3) and int(u["order_kind"][0, H]) == O.HOLD
    for _ in range(30):                                                          # 17 s over
        step(st)
    step(st, orders(st, kind={H: (O.MOVE, -1, 0.0, 80.0)}))
    for _ in range(6):
        step(st)
    assert float(u["z"][0, H]) > 21.0


def test_the_game_ai_presses_them_by_the_default_rule():
    st = scenario.build([army([(0, -20)], [(0, 20)], ai=True)], P)
    u, H = st.u, st.N // 2
    step(st, orders(st))
    # an enemy within near_m 60 m: both fire (the stand-in rule: the game's own was not found)
    assert float(u[f"ab{slot(QUAKE)}_on"][0, H]) > 0 and float(u[f"ab{slot(SNARE)}_on"][0, H]) > 0


def test_a_battle_without_grinders_keeps_the_new_fields_at_rest():
    a = army([(0, -10)], [(0, 20)])
    a["sides"][2]["units"][0]["key"] = "wh2_main_skv_inf_clanrats_1"
    st = scenario.build([a], P)
    for _ in range(20):
        step(st)
    assert int(st.u["ab_en"].abs().sum()) == 0 and float(st.u["dq_sum"].abs().sum()) == 0.0


def test_the_unit_drills_leave_the_grinders_out():
    from tools.nn.unit import drill_hold
    assert GR in drill_hold.NOT_IN_DRILLS and GR not in drill_hold.roster()["skv"]
