"""apps.alignment: when to realign, where to, and the flank overhang."""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture(scope="module")
def lua():
    runtime = new_runtime()
    runtime.globals().al = load(runtime, "apps.alignment.services")
    return runtime


def field(lua, bearing=0, origin=(0, 0), own=(-20, 20), enemy=(-40, 40)):
    return lua.table_from({"origin": {"x": origin[0], "z": origin[1]}, "bearing": bearing,
                           "own": {"left_m": own[0], "right_m": own[1]},
                           "enemy": {"left_m": enemy[0], "right_m": enemy[1]}}, recursive=True)


def current(lua, x, z, bearing):
    return lua.table_from({"anchor": {"x": x, "z": z}, "bearing": bearing}, recursive=True)


def test_angle_diff(lua):
    d = lua.globals().al.angle_diff
    assert d(350, 10) == pytest.approx(-20) and d(10, 350) == pytest.approx(20) and d(180, 0) == pytest.approx(180)


def enemy(lua, x, z, facing):
    return lua.table_from({"centre": {"x": x, "z": z}, "facing": facing}, recursive=True)


def test_small_errors_do_not_move_the_army(lua):
    c = lua.globals().al.check(field(lua), current(lua, 10, 5, 8))
    assert not c.needed and c.offset_m == pytest.approx(10) and c.angle_off_deg == pytest.approx(8)


def test_turned_or_shifted_army_realigns(lua):
    al = lua.globals().al
    turned = al.check(field(lua), current(lua, 0, 5, 25))
    shifted = al.check(field(lua), current(lua, -40, 5, 0))
    assert turned.needed and list(turned.reasons.values()) == ["angle"]
    assert shifted.needed and list(shifted.reasons.values()) == ["offset"] and shifted.offset_m == pytest.approx(-40)


def test_target_keeps_the_distance_and_sits_on_the_axis(lua):
    al = lua.globals().al
    # Axis looks east (90): along = +X, across = -Z (right of facing east is south).
    t = al.target(field(lua, bearing=90, origin=(0, 0)), current(lua, 30, 25, 60))
    assert t.along_m == pytest.approx(30) and t.anchor.x == pytest.approx(30) and t.anchor.z == pytest.approx(0)
    assert t.bearing == 90


def test_stand_on_the_line_the_enemy_looks_along(lua):
    al = lua.globals().al
    # Their centre (0, 300), they face south (180): we must face north (0) on x = 0.
    them = enemy(lua, 0, 300, 180)
    off = al.check(field(lua, bearing=10), current(lua, 50, 0, 0), them)
    assert off.source == "enemy_facing" and off.needed and off.offset_m == pytest.approx(50)
    t = al.target(field(lua, bearing=10), current(lua, 50, 0, 0), them)
    assert t.anchor.x == pytest.approx(0) and t.anchor.z == pytest.approx(0) and t.bearing == pytest.approx(0)
    # Without their facing the battlefield axis is used.
    assert al.check(field(lua), current(lua, 50, 0, 0), lua.table_from({})).source == "centres"


def test_overhang(lua):
    o = lua.globals().al.overhang(field(lua, own=(-20, 20), enemy=(-40, 25)))
    assert o.left_m == pytest.approx(20) and o.right_m == pytest.approx(5)


def test_current_position_required(lua):
    with pytest.raises(Exception, match="Current position required"):
        lua.globals().al.check(field(lua), lua.table_from({}))


def check(lua, needed, offset=0, angle=0):
    return lua.table_from({"needed": needed, "offset_m": offset, "angle_off_deg": angle})


def test_governor_waits_for_movement_and_cooldown(lua):
    g = lua.globals().al.new_governor()
    off = check(lua, True, offset=50)
    assert g.decide(0, off, True) == "moving"
    assert g.decide(0, off, False) == "align"
    g.record_order(0, off)
    assert g.decide(5000, check(lua, True, offset=20), False) == "cooldown"   # only 5 s later
    assert g.decide(21000, check(lua, True, offset=20), False) == "align"     # 20 s passed, error smaller
    assert g.decide(22000, check(lua, False), False) == "aligned"


def test_governor_gives_up_when_aligning_does_not_help(lua):
    g = lua.globals().al.new_governor({"cooldown_ms": 0, "give_up_after": 3})
    same = check(lua, True, offset=30)
    results = []
    for t in range(0, 5):
        r = g.decide(t * 1000, same, False)
        results.append(r)
        if r == "align":
            g.record_order(t * 1000, same)
    # First try, then two useless ones, then the third useless one gives up for good.
    assert results == ["align", "align", "align", "given_up", "given_up"]
