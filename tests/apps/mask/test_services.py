"""apps.mask: grid over the battlefield, standable cells, fits and lanes (level 1)."""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture(scope="module")
def lua():
    runtime = new_runtime()
    runtime.globals().mask = load(runtime, "apps.mask.services")
    # Battlefield looking north from (0, 0): own lines at along -10..5, theirs 195..210, half width 60.
    runtime.execute("""
        FIELD = {origin = {x = 0, z = 0}, bearing = 0, half_width_m = 60,
            own = {back_m = -10, front_m = 5}, enemy = {front_m = 195, back_m = 210}}
        -- A rock at x -10..10, z 90..110 (not clear, not reachable); an island the
        -- infantry cannot reach in the south-west corner (clear but unreachable).
        function READER(x, z)
            local rock = x > -10 and x < 10 and z > 90 and z < 110
            local island = x < -50 and z < 0
            return not rock, not rock and not island
        end
        function FULL()
            local m = mask.new(mask.grid(FIELD, 3, 0))
            mask.fill(m, READER)
            return m
        end
    """)
    return runtime


def test_grid_covers_the_field(lua):
    g = lua.eval("mask.grid(FIELD, 3, 0)")
    assert g.rows == 74 and g.cols == 40 and g.count == 74 * 40
    first = lua.eval("mask.cell(mask.grid(FIELD, 3, 0), 1)")
    assert first.along == pytest.approx(-8.5) and first.across == pytest.approx(-58.5)
    assert first.x == pytest.approx(-58.5) and first.z == pytest.approx(-8.5)
    assert lua.eval("mask.index(mask.grid(FIELD, 3, 0), -8.5, -58.5)") == 1
    # The default pads 10 m at both ends along the axis.
    assert lua.eval("mask.grid(FIELD, 3)").along0 == pytest.approx(-20)
    assert lua.eval("mask.index(mask.grid(FIELD, 3), 500, 0)") is None


def test_stand_needs_clear_and_reachable(lua):
    s = lua.eval("mask.summary(FULL())")
    assert s.known == s.cells and s.blocked > 0 and s.stand + s.blocked == s.cells
    assert lua.eval("mask.stand_at(FULL(), 100, 0)") is False       # the rock
    assert lua.eval("mask.stand_at(FULL(), 100, 50)") is True        # open ground
    assert lua.eval("mask.stand_at(FULL(), -5, -55)") is False       # clear but unreachable


def test_fits(lua):
    ok = lua.eval("mask.fits(FULL(), {x = 0, z = 50, bearing = 0, front_m = 30, depth_m = 9})")
    rock = lua.eval("mask.fits(FULL(), {x = 0, z = 105, bearing = 0, front_m = 30, depth_m = 9})")
    outside = lua.eval("mask.fits(FULL(), {x = 0, z = 400, bearing = 0, front_m = 10, depth_m = 5})")
    assert ok.ok and ok.samples > 0
    assert not rock.ok and rock.blocked > 0
    assert not outside.ok and outside.unknown == outside.samples


def test_lane(lua):
    straight = lua.eval("mask.lane(FULL(), 0, 28, 5, 195)")
    beside = lua.eval("mask.lane(FULL(), -30, 20, 5, 195)")
    assert not straight.free and 85 <= straight.first_blocked_along <= 92
    assert beside.free


def test_unread_cells_are_unknown_and_block_lanes(lua):
    lua.execute("HALF = mask.new(mask.grid(FIELD, 3, 0)); NEXT, DONE = mask.fill(HALF, READER, 1, 100)")
    g = lua.globals()
    assert g.NEXT == 101 and not g.DONE
    assert lua.eval("mask.summary(HALF)").known == 100
    assert not lua.eval("mask.lane(HALF, 0, 10, 5, 50)").free


def test_encode(lua):
    code = lua.eval("mask.encode(FULL())")
    rows = code.split("/")
    assert len(rows) == 74 and all(len(r) == 40 for r in rows)
    assert "#" in code and "." in code and "?" not in code
