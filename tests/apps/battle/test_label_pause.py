"""apps.battle.label_pause: the locks, the pause on every operative second (from 0, round(t / 1000)), the real-time poll
until the answer to that move, the speed back; no pause in between."""
from tests.lua_runtime import load, new_runtime


def test_the_battle_pauses_on_the_operatives_seconds_until_the_answer():
    lua = new_runtime()
    lua.globals().label = load(lua, "apps.battle.label_pause")
    lua.execute("""
        local speed, now, written, answered, polls = 3, 0, 0, 0, 0
        local real, events = {}, {}
        local bm = {}
        function bm:modify_battle_speed(v) speed = v end
        function bm:current_battle_speed() return speed end
        function bm:disable_orders(v) end
        function bm:disable_pausing(v) end
        function bm:disable_time_speed_controls(v) error('no ui') end
        function bm:real_callback(fn, ms, name) real[#real + 1] = fn end
        local h = label.start({bm = bm, every_s = 15, speed = 3, now_ms = function() return now end,
            written = function() return written end, answered = function() return answered end,
            poll = function() polls = polls + 1 end, emit = function(e, f) f.event = e; events[#events + 1] = f end})
        local l = h.lock('battle')
        assert(l.disable_orders == 'ok' and l.disable_cycle_battle_speed == 'missing')
        assert(l.disable_time_speed_controls:find('no ui'))
        -- second 0: paused, waits for the answer to move 1
        written = 1; now = 0
        assert(h.after_decide() == true and speed == 0 and #real == 1)
        local fn = table.remove(real, 1); fn()
        assert(speed == 0 and polls == 1 and #real == 1)       -- no answer yet: polled again later
        answered = 1
        fn = table.remove(real, 1); fn()
        assert(speed == 3 and not h.paused and #real == 0)
        assert(events[#events].event == 'label_resume' and events[#events].move == 1)
        -- seconds 1 .. 14: no pause
        for s = 1, 14 do written = s + 1; now = s * 1000; assert(h.after_decide() == false) end
        -- second 15 (its document at 15.05 s): paused
        written = 16; now = 15050
        assert(h.after_decide() == true and speed == 0)
        answered = 16
        fn = table.remove(real, 1); fn()
        assert(speed == 3)
        now = 16000; assert(h.after_decide() == false)
        -- a late document past 30 s still pauses, the next at 45
        now = 31000; written = 32
        assert(h.after_decide() == true and h.next == 45)
    """)
