"""apps.logistics: the obstacle band, who goes round, sides, the queue, swaps, the dispatcher, crowding."""
import pytest

from tests.lua_runtime import load, new_runtime

STEP = 3


@pytest.fixture(scope="module")
def lg():
    lua = new_runtime()
    m = load(lua, "apps.logistics.services")
    m.t = lambda x: lua.table_from(x, recursive=True)
    m.lua = lua
    return m


def mask(lg, blocked, rows=60, cols=40, across0=-60.0, along0=-30.0):
    """A stand mask in the battlefield frame: 3 m cells; blocked(along, across) -> True where nobody can stand."""
    stand = {}
    for r in range(rows):
        for c in range(cols):
            along, across = along0 + (r + 0.5) * STEP, across0 + (c + 0.5) * STEP
            stand[r * cols + c + 1] = not blocked(along, across)
    grid = {"step": STEP, "along0": along0, "across0": across0, "rows": rows, "cols": cols,
            "frame": {"origin": {"x": 0, "z": 0}, "bearing": 0}}
    return lg.t({"grid": grid, "stand": stand})


def unit(uid, across, along=0.0, advance=100.0, front=20.0, depth=10.0, key="spear", shapes=None):
    return {"id": uid, "place": uid, "key": key, "start": {"along": along, "across": across},
            "target": {"along": along + advance, "across": across}, "heading0": 0, "heading1": 0,
            "front_m": front, "depth_m": depth, "width": front, "shapes": shapes or []}


def rock(along, across):
    return 30 <= along <= 45 and -10 <= across <= 10


def band_of(lg, m, units, whole=False):
    e = lg.extent(lg.t(units))
    if whole:  # the band over the whole field width (a unit elsewhere runs into the obstacle)
        return lg.band(m, e["from"], e["to"], None, -60, 60)
    return lg.band(m, e["from"], e["to"], None, e["left"], e["right"])


def test_band_grows_the_obstacle_by_5_m_and_finds_the_sides(lg):
    b = band_of(lg, mask(lg, rock), [unit("a", 0)])
    assert b["along0"] == pytest.approx(30 - 5, abs=STEP) and b["along1"] == pytest.approx(45 + 5, abs=STEP)
    assert len(b["intervals"]) == 2
    assert b["intervals"][1][2] == pytest.approx(-15, abs=STEP) and b["intervals"][2][1] == pytest.approx(15, abs=STEP)


def test_nothing_in_the_way_no_band(lg):
    assert band_of(lg, mask(lg, lambda a, c: False), [unit("a", 0)]) is None


def test_a_wall_along_the_march_beside_a_flank_is_not_in_the_way(lg):
    # The map edge (blocked beyond across 45) runs the whole way beside the flank unit.
    b = band_of(lg, mask(lg, lambda a, c: c > 45), [unit("a", 30)])
    assert b is None


def test_who_goes_round_the_obstacle_plus_5_m(lg):
    units = [unit("hit", 0), unit("touch", -22), unit("clear", -40)]
    b = band_of(lg, mask(lg, rock), units)
    plan = lg.plan(lg.t(units), b)
    # "touch": its strip [-32, -12] reaches the rock grown by 5 m (-15).
    assert plan.units["hit"].kind == "detour" and plan.units["touch"].kind == "detour"
    assert plan.units["clear"].kind == "straight"


def test_a_closed_side_sends_everyone_round_the_other(lg):
    # The rock runs to the field's left edge: only the right side is open.
    units = [unit("a", -15), unit("b", 5)]
    b = band_of(lg, mask(lg, lambda a, c: 30 <= a <= 45 and c <= 10), units)
    assert len(b["intervals"]) == 1
    plan = lg.plan(lg.t(units), b)
    assert plan.units["a"].side == plan.units["b"].side == 1
    assert plan.units["a"].slot.across > 10 and plan.units["b"].slot.across > 10


def gap_right(a, c):
    return 30 <= a <= 45 and (c <= 20 or c >= 56)


def test_one_lane_is_a_queue_the_one_nearer_goes_first_nobody_overtakes(lg):
    units = [unit("near", 18), unit("far", -4)]
    plan = lg.plan(lg.t(units), band_of(lg, mask(lg, gap_right), units))
    near, far = plan.units["near"], plan.units["far"]
    assert near.slot.across == pytest.approx(far.slot.across)
    assert far.enter_s >= near.enter_s + (near.slot.depth_m + 2) / 1.5 - 1e-6


def test_the_one_behind_waits_at_its_start_for_the_one_ahead(lg):
    # Same lane, 11 m apart: the rear one would reach the gap too early.
    units = [unit("front", 39, along=0), unit("rear", 39, along=-11)]
    plan = lg.plan(lg.t(units), band_of(lg, mask(lg, gap_right), units, whole=True))
    assert plan.units["front"].kind == plan.units["rear"].kind == "straight"
    rear = plan.units["rear"]
    assert rear.release.at_s > 0 and [a.id for a in rear.release.after.values()] == ["front"]


def test_the_front_row_goes_before_the_rear_row(lg):
    units = [unit("front", 0, along=0), unit("rear", 0, along=-15)]
    plan = lg.plan(lg.t(units), band_of(lg, mask(lg, rock), units))
    assert plan.units["front"].kind == plan.units["rear"].kind == "detour"
    assert plan.units["front"].enter_s < plan.units["rear"].enter_s


def test_narrows_only_when_wider_than_the_gap(lg):
    shapes = [{"ordered_m": 10, "front_m": 9, "depth_m": 22, "reform_s": 8}]
    wide = [unit("a", 0, shapes=shapes)]
    b = band_of(lg, mask(lg, rock), wide)
    assert lg.plan(lg.t(wide), b).units["a"].slot.width == 20
    # A gap of 12 m: only the 9 m width fits; it reforms in place first.
    b2 = band_of(lg, mask(lg, lambda a, c: 30 <= a <= 45 and abs(c) >= 12), wide, whole=True)  # the gap: -7..7
    row = lg.plan(lg.t(wide), b2).units["a"]
    assert row.slot.width == 10 and row.route[1].reform


def test_in_one_queue_the_first_out_takes_the_farthest_place(lg):
    # Both go round the right of a wide rock through one lane; "near" comes out
    # first and must not stand in the way of "far" behind it.
    units = [unit("near", 18), unit("far", -4)]
    plan = lg.plan(lg.t(units), band_of(lg, mask(lg, gap_right), units))
    assert plan.swapped
    assert plan.units["near"].place == "far" and plan.units["far"].place == "near"


def test_dispatcher_releases_legs_and_finishes(lg):
    units = [unit("front", 39, along=0), unit("rear", 39, along=-11)]
    plan = lg.plan(lg.t(units), band_of(lg, mask(lg, gap_right), units, whole=True))
    d = lg.new_dispatch(plan, lg.t({"front": 10, "rear": 10}))
    pos = {"front": {"along": -5, "across": 39}, "rear": {"along": -16, "across": 39}}
    assert [o.id for o in d.update(0, lg.t(pos)).values()] == ["front"]  # "rear" waits
    assert not d.done()
    pos["front"] = {"along": 10, "across": 39}  # "front" has walked on
    orders = list(d.update(5, lg.t(pos)).values())
    assert [o.id for o in orders] == ["rear"] and orders[0].final
    assert d.done()


def test_the_gate_out_is_passed_along_its_own_leg_not_the_next(lg):
    # rock_march_wide, C9 (28.09.2026): the place is behind the rock and to the side. Just left of the
    # gate's line the unit counted the gate out as passed (along the NEXT leg, sideways) and walked
    # straight across the rock to its place.
    route = [{"along": 47.6, "across": 88.3, "width": 20}, {"along": 148.1, "across": 88.3, "width": 20},
             {"along": 148.1, "across": 40.0, "width": 20, "final": True}]
    plan = {"units": {"c9": {"route": route, "route_m": 150.0, "release": {"at_s": 0, "after": []},
                             "slot": {"depth_m": 18}}}}
    d = lg.new_dispatch(lg.t(plan), lg.t({"c9": 18}))
    legs = lambda t, along, across: [o.leg for o in d.update(t, lg.t({"c9": {"along": along, "across": across}})).values()]
    assert legs(0, 0, 85) == [1]
    assert legs(30, 44, 85) == [2]           # 5 m before the gate in: on to the gate out
    assert legs(31, 46, 85) == []            # still in the gap, a little left of its line: keep going
    assert legs(60, 100, 86) == []
    assert legs(90, 142, 88) == [3]          # at the gate out: now sideways into the place


def test_crowding_counts_soldiers_of_two_units_closer_than_1_m(lg):
    c = lg.crowding(lg.t([{"id": "a", "points": [0, 0, 5, 5]}, {"id": "b", "points": [0.5, 0, 20, 20]}]))
    assert c.soldiers == 2 and c.pairs["a|b"] == 2
    apart = lg.crowding(lg.t([{"id": "a", "points": [0, 0]}, {"id": "b", "points": [1.5, 0]}]))
    assert apart.soldiers == 0
