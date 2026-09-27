"""apps.map.services: radar frame from a synthetic radar and grid indexing."""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().map = load(runtime, "apps.map.services")
    # Unrotated radar over x in [-500, 500], z in [-400, 400]; v grows to -z.
    runtime.execute("""
        function radar(x, z) return (x + 500) / 1000, (400 - z) / 800 end
    """)
    return runtime


class TestFrame:
    def test_frame_bounds_from_radar(self, lua):
        lua.execute("""
            local f = map.build_frame(radar)
            assert(math.abs(f.min_x + 500) < 1e-9 and math.abs(f.max_x - 500) < 1e-9)
            assert(math.abs(f.min_z + 400) < 1e-9 and math.abs(f.max_z - 400) < 1e-9)
            assert(#f.corners == 4 and f.movement_boundary_verified == false)
            local u, v = map.world_to_radar(f, 0, 0)
            assert(math.abs(u - 0.5) < 1e-9 and math.abs(v - 0.5) < 1e-9)
        """)

    def test_rotated_radar_is_rejected(self, lua):
        lua.execute("""
            local ok, err = pcall(map.build_frame, function(x, z) return (z + 500) / 1000, (x + 500) / 1000 end)
            assert(not ok and err:find('Untested radar orientation', 1, true))
        """)


class TestGrid:
    def test_grid_cells_and_half_open_indexing(self, lua):
        lua.execute("""
            local grid = map.new_grid(map.build_frame(radar), 3)
            assert(grid.columns == 334 and grid.rows == 267 and grid.count == 334 * 267)
            local x, z = map.cell_center(grid, 0, 0)
            assert(math.abs(x + 498.5) < 1e-9 and math.abs(z + 398.5) < 1e-9)
            local ix, iz = map.cell_index(grid, 335)
            assert(ix == 1 and iz == 1)
            assert(map.world_to_cell(grid, 500.001, 0) == nil)
            assert(map.world_to_cell(grid, -500.001, 0) == nil)
            local cx, cz = map.world_to_cell(grid, -500, -400)
            assert(cx == 0 and cz == 0)
            assert(not pcall(map.cell_center, grid, grid.columns, 0))
        """)


class TestWindow:
    def test_window_grid_is_clipped_and_offset(self, lua):
        lua.execute("""
            local frame = map.build_frame(radar)
            local grid = map.new_grid(frame, 1, {min_x = -160, max_x = -30, min_z = -130, max_z = 20})
            assert(grid.columns == 130 and grid.rows == 150, grid.columns .. 'x' .. grid.rows)
            local x, z = map.cell_center(grid, 0, 0)
            assert(math.abs(x + 159.5) < 1e-9 and math.abs(z + 129.5) < 1e-9)
            assert(math.abs(grid.radar_max_x - 500) < 1e-6)
            assert(not pcall(map.new_grid, frame, 1, {min_x = 600, max_x = 700, min_z = 0, max_z = 10}))
        """)
