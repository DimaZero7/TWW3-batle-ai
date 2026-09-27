"""apps.intel: last-seen memory and the visibility gate on position reads."""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    g = runtime.globals()
    g.intel = load(runtime, "apps.intel.adapter")
    g.memory_service = load(runtime, "apps.intel.services")
    runtime.execute("""
        position_reads = 0
        function enemy(visible)
            local u = {visible = visible}
            function u:is_visible_to_alliance() return self.visible end
            function u:position()
                position_reads = position_reads + 1
                return {get_x=function() return 10 end, get_y=function() return 0 end, get_z=function() return 20 end}
            end
            return u
        end
    """)
    return runtime


class TestIntel:
    def test_visible_then_hidden_keeps_last_seen(self, lua):
        lua.execute("""
            local memory = memory_service.new_memory()
            local u = enemy(true)
            local seen = intel.observe(u, {}, 'enemy-1', 1000, memory)
            assert(seen.visibility == 'visible' and seen.current.x == 10)
            u.visible = false
            local hidden = intel.observe(u, {}, 'enemy-1', 2000, memory)
            assert(hidden.visibility == 'not_visible' and hidden.current == nil)
            assert(hidden.last_seen.seen_ms == 1000 and hidden.last_seen.z == 20)
        """)

    def test_hidden_enemy_position_is_never_read(self, lua):
        lua.execute("""
            position_reads = 0
            intel.observe(enemy(false), {}, 'enemy-2', 0, {})
            assert(position_reads == 0)
        """)

    def test_unknown_visibility(self, lua):
        lua.execute("""
            local u = {is_visible_to_alliance = function() error('no alliance') end}
            local r = intel.observe(u, {}, 'enemy-3', 0, {})
            assert(r.visibility == 'unknown' and r.last_seen == nil)
        """)
