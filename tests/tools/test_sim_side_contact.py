"""Melee with an enemy's neighbour (the sidecontact probe, tools/nn/charge_probe.py --plan sidecontact; config/nn/sim.json
melee.frontage_edge_m, contact.square_front, contact.push_keep_facing): a formation's men strike whatever enemy their
part of the front touches, so a neighbour of the target that the attacker's front overlaps is struck by the men of that
overlap; the contact's side is the rectangles' least overlap (geometry.front_contact); a formation attacking another
squares its front to the target's, a held one keeps its facing; a walker pushing past keeps the facing of its way.
Needs torch (the container)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SWORD, CLANRATS = "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_clanrats_1"


def box(rows):
    t = lambda k: torch.tensor([[float(r[k]) for r in rows]])     # noqa: E731
    return t("x"), t("z"), t("b"), t("front"), t("depth")


class TestFrontContact:
    def test_head_on_and_offset_fronts_meet_through_the_front(self):
        # i at the origin facing +z (30 x 9); j facing it 9 m ahead (overlap 0.5 m deep): head-on, 20 m and 27 m aside
        for dx in (0.0, 20.0, 27.0):
            x, z, b, f, d = box([dict(x=0, z=0, b=0, front=30, depth=9), dict(x=dx, z=8.5, b=180, front=30, depth=9)])
            sat = geometry.front_contact(x, z, b, f, d)
            assert bool(sat[0, 0, 1]) and bool(sat[0, 1, 0])

    def test_a_unit_on_the_flank_meets_it_through_the_flank(self):
        # j beside i (its front facing i's right flank, 1 m in): i meets it by its flank, j by its front
        x, z, b, f, d = box([dict(x=0, z=0, b=0, front=30, depth=9), dict(x=19.0, z=0, b=270, front=10, depth=9)])
        sat = geometry.front_contact(x, z, b, f, d)
        assert not bool(sat[0, 0, 1]) and bool(sat[0, 1, 0])


def lane(a_dx, bystander=True, a_b=180.0):
    """Swordsmen a_dx m aside of clanrats B, fronts 1 m into each other; clanrats C 1 m beside B (+x), all fearless."""
    units2 = [{"key": CLANRATS, "x": 0.0, "z": 0.0, "b": 0.0}]
    if bystander:
        units2.append({"key": CLANRATS, "x": 31.0, "z": 0.0, "b": 0.0})
    st = scenario.build([{"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": SWORD, "x": a_dx, "z": 0.0, "b": a_b, "width": 30}]},
        2: {"faction": "wh2_main_skv_skaven", "units": units2}}}], P)
    front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
    H = st.N // 2
    st.u["z"][0, 0] = float(depth[0, H] / 2 + depth[0, 0] / 2) - 1.0
    st.u["morale"] = st.u["morale"] + 1e4
    st.u["leadership"] = st.u["leadership"] + 1e4
    return st, H


ON = dict(rect_gap=True, square_front=1, push_keep_facing=1)


def fight(a_dx, bystander=True, seconds=40, on=True, a_b=180.0, b_attacks=False, from_s=0.0):
    """The swordsmen attack B (B held, or attacking them); -> (health share lost by B, by C from from_s on, the
    swordsmen's last bearing, B's last bearing)."""
    params = P.with_cal("contact", **ON).with_cal("melee", frontage_edge_m=0.0) if on else P
    st, H = lane(a_dx, bystander, a_b)
    o = replay.hold(st)
    o.kind[0, 0], o.target[0, 0], o.run[0, 0] = O.ATTACK, H, False
    if b_attacks:
        o.kind[0, H], o.target[0, H], o.run[0, H] = O.ATTACK, 0, False
    hb0 = None
    for k in range(int(seconds / params.dt)):
        if hb0 is None and k * params.dt >= from_s:
            hb0 = [float(st.u["hp"][0, H]), float(st.u["hp"][0, H + 1]) if bystander else 0.0]
        battle.step(st, o, params)
    lost = [hb0[0] - float(st.u["hp"][0, H]), (hb0[1] - float(st.u["hp"][0, H + 1])) if bystander else 0.0]
    return lost[0], lost[1], float(st.u["b"][0, 0]), float(st.u["b"][0, H])


def test_the_neighbour_the_front_overlaps_is_struck_by_that_part_of_the_front():
    b0, c0, _, _ = fight(0.0)                      # square on B: C 1 m beside, untouched
    b12, c12, _, _ = fight(12.0)                   # the front over C by 11 m
    assert c0 == pytest.approx(0.0, abs=0.002)
    assert c12 > 0.1 * b12 and c12 < b12           # the game: C 6.9-10.2 HP/s against B 29.7-36.5
    # after the opening (20-40 s) the old rule (the centre line, the swordsmen turned to B's centre) has lost C
    _, c12_late, _, _ = fight(12.0, from_s=20.0)
    _, c_off_late, _, _ = fight(12.0, on=False, from_s=20.0)
    assert c12_late > 2 * c_off_late


def test_a_square_pair_strikes_as_before():
    # the frontage changes nothing for two formations square on each other (the calibrated pairs)
    b_on, _, _, _ = fight(0.0, bystander=False, seconds=20)
    b_off, _, _, _ = fight(0.0, bystander=False, seconds=20, on=False)
    assert b_on == pytest.approx(b_off, rel=0.02)


def test_the_attacker_squares_its_front_and_the_held_target_keeps_its_facing():
    # the swordsmen in contact turned 20 deg off square: they turn back towards 180 (contact.melee_turn_deg_s), the held
    # clanrats keep theirs (0)
    _, _, a_b, t_b = fight(6.0, bystander=False, seconds=10, a_b=200.0)
    assert abs(a_b - 180.0) < 20.0 - 5.0
    assert t_b == pytest.approx(0.0, abs=1.0)


def test_a_walker_pushing_past_keeps_the_facing_of_its_way():
    params = P.with_cal("contact", **ON).with_cal("melee", frontage_edge_m=0.0)
    st = scenario.build([{"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": SWORD, "x": 26.0, "z": 30.0, "b": 180.0}]},
        2: {"faction": "wh2_main_skv_skaven", "units": [{"key": CLANRATS, "x": 0.0, "z": 0.0, "b": 0.0}]}}}], params)
    st.u["morale"] = st.u["morale"] + 1e4
    st.u["leadership"] = st.u["leadership"] + 1e4
    o = replay.hold(st)
    o.kind[0, 0] = O.MOVE
    o.x[0, 0], o.z[0, 0] = 26.0, -60.0
    o.run[0, 0] = False
    worst, met = 0.0, False
    for _ in range(int(60 / params.dt)):
        battle.step(st, o, params)
        met = met or bool(st.u["m"][0, 0])
        worst = max(worst, abs(float(st.u["b"][0, 0]) - 180.0))
    assert met and worst < 5.0                     # the game: within 15 deg of its way (walkpass2)
    assert float(st.u["z"][0, 0]) < -30.0          # it got past


def test_the_switches_are_off():
    # off: no lever on the whole battles' routs, the frontage lowers their melee rate (config/nn/sim.json *_why)
    assert not P.sim["contact"].get("square_front") and not P.sim["contact"].get("push_keep_facing")
    assert P.sim["melee"].get("frontage_edge_m") is None
