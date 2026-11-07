"""apps.battle.pause_probe: orders off and speed 0 at at_s, readings on real-time callbacks, the speed back after real_ms,
the orders back at at_s + off_s; a missing engine call is reported, not an error."""
from tests.lua_runtime import load, new_runtime


def test_the_probe_pauses_reads_resumes_and_gives_the_orders_back():
    lua = new_runtime()
    lua.globals().probe = load(lua, "apps.battle.pause_probe")
    lua.execute("""
        local calls, real, events = {}, {}, {}
        local speed, now = 10, 0
        local bm = {}
        function bm:disable_orders(v) calls[#calls + 1] = 'orders ' .. tostring(v) end
        function bm:modify_battle_speed(v) speed = v; calls[#calls + 1] = 'speed ' .. v end
        function bm:current_battle_speed() return speed end
        function bm:real_callback(fn, ms, name) real[#real + 1] = {fn = fn, ms = ms, name = name} end
        local h = probe.start({bm = bm, cfg = {at_s = 60, real_ms = 3000, off_s = 20}, speed = 10,
            emit = function(e, f) events[#events + 1] = f end, now_ms = function() return now end,
            written = function() return 7 end})
        now = 59000; h.tick()
        assert(#calls == 0 and h.stage == 'wait')
        now = 60000; h.tick()
        assert(calls[1] == 'orders true' and calls[2] == 'speed 0' and speed == 0, table.concat(calls, ','))
        assert(events[1].stage == 'pause' and events[1].orders_off == 'ok' and events[1].speed == 0)
        assert(#real == 3 and real[1].ms == 1000 and real[2].ms == 2000 and real[3].ms == 3000)
        real[1].fn(); real[2].fn()
        assert(events[2].stage == 'paused' and events[2].written == 7 and events[3].k == 2)
        real[3].fn()
        assert(speed == 10 and h.stage == 'resumed' and events[4].stage == 'resume' and events[4].resume == 'ok')
        now = 79000; h.tick()
        assert(h.stage == 'resumed')
        now = 80000; h.tick()
        assert(h.stage == 'done' and calls[#calls] == 'orders false' and events[#events].stage == 'orders_on')
    """)


def test_a_missing_call_is_reported():
    lua = new_runtime()
    lua.globals().probe = load(lua, "apps.battle.pause_probe")
    lua.execute("""
        local events, speed = {}, 3
        local bm = {}
        function bm:modify_battle_speed(v) speed = v end
        function bm:current_battle_speed() return speed end
        function bm:real_callback(fn, ms, name) end
        local h = probe.start({bm = bm, cfg = {at_s = 1, real_ms = 500, off_s = 5}, speed = 3,
            emit = function(e, f) events[#events + 1] = f end, now_ms = function() return 2000 end,
            written = function() return 0 end})
        h.tick()
        assert(events[1].orders_off == 'missing' and events[1].pause == 'ok' and speed == 0)
    """)
