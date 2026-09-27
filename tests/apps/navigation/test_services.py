"""apps.navigation.services: end of one ordered move (arrived, stopped, stuck, timeout)."""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().nav = load(runtime, "apps.navigation.services")
    # run(monitor, samples): feeds {t, x, z, moving} rows, returns the first reason and its time.
    runtime.execute("""
        function run(m, samples)
            for _, s in ipairs(samples) do
                local r = m.update(s[1], {x = s[2], z = s[3], moving = s[4]})
                if r then return r, s[1] end
            end
        end
    """)
    return runtime


class TestLegMonitor:
    def test_arrived_after_three_still_samples_near_target(self, lua):
        lua.execute("""
            local m = nav.new_leg_monitor({target = {x = 0, z = 10}, timeout_ms = 60000})
            local r, t = run(m, {{1000, 0, 0, true}, {2000, 0, 5, true}, {3000, 0, 9, false},
                {4000, 0, 9, false}, {5000, 0, 9, false}})
            assert(r == 'arrived' and t == 5000, tostring(r) .. ' ' .. tostring(t))
            assert(math.abs(m.stats.path_m - 9) < 1e-9 and m.stats.samples == 5)
        """)

    def test_stops_right_after_the_order_are_ignored(self, lua):
        lua.execute("""
            local m = nav.new_leg_monitor({target = {x = 0, z = 0}, timeout_ms = 60000})
            local r = run(m, {{500, 0, 0, false}, {1000, 0, 0, false}, {1500, 0, 0, false}})
            assert(r == nil)
            assert(m.update(3000, {x = 0, z = 0, moving = false}) == 'arrived')
        """)

    def test_stopped_far_from_target(self, lua):
        lua.execute("""
            local m = nav.new_leg_monitor({target = {x = 0, z = 100}, timeout_ms = 60000})
            local r = run(m, {{1000, 0, 0, true}, {2000, 0, 20, false}, {3000, 0, 20, false},
                {4000, 0, 20, false}})
            assert(r == 'stopped', tostring(r))
            assert(math.abs(m.stats.min_distance_m - 80) < 1e-9)
        """)

    def test_stuck_when_moving_in_place_for_the_window(self, lua):
        lua.execute("""
            local m = nav.new_leg_monitor({target = {x = 0, z = 100}, timeout_ms = 60000})
            local rows = {}
            for i = 1, 12 do rows[i] = {i * 1000, (i % 2) * 0.5, 30, true} end
            local r, t = run(m, rows)
            assert(r == 'stuck' and t == 11000, tostring(r) .. ' ' .. tostring(t))
        """)

    def test_reform_in_place_is_not_stuck_when_disabled(self, lua):
        lua.execute("""
            local m = nav.new_leg_monitor({target = {x = 0, z = 0}, timeout_ms = 15000, detect_stuck = false})
            local rows = {}
            for i = 1, 16 do rows[i] = {i * 1000, 0, 0, true} end
            local r, t = run(m, rows)
            assert(r == 'timeout' and t == 15000, tostring(r) .. ' ' .. tostring(t))
        """)

    def test_bad_input_is_rejected(self, lua):
        lua.execute("""
            assert(not pcall(nav.new_leg_monitor, {target = {x = 0}, timeout_ms = 1000}))
            local m = nav.new_leg_monitor({target = {x = 0, z = 0}, timeout_ms = 1000})
            assert(not pcall(m.update, 0 / 0, {x = 0, z = 0, moving = true}))
        """)
