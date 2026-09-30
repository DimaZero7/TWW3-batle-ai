"""apps.orders.planner_adapter: a unit that routs and rallies goes back to the planner."""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().planner = load(runtime, "apps.orders.planner_adapter")
    runtime.execute("""
        log = {}
        callbacks = {}
        function unit(name)
            local u = {name = name, men = 100, routing = false, shattered = false}
            function u:number_of_men_alive() return self.men end
            function u:is_routing() return self.routing end
            function u:is_shattered() return self.shattered end
            function u:position() return {distance = function() return 10 end} end
            return u
        end
        own = {unit('a'), unit('b')}
        foes = {unit('x')}
        engine = {}
        function engine:add_units(u) log[#log + 1] = 'add ' .. u.name end
        function engine:remove_units(u) log[#log + 1] = 'remove ' .. u.name end
        function engine:attack_unit(u) log[#log + 1] = 'attack ' .. u.name end
        alliance = {create_ai_unit_planner = function() return engine end}
        bm = {callback = function(self, fn, ms, name) callbacks[#callbacks + 1] = fn end}
        handle = planner.hand_over(bm, 'test', alliance, own, foes)
        function run_callbacks() for _, fn in ipairs(callbacks) do fn() end callbacks = {} end
    """)
    return runtime


def log(lua):
    return list(lua.eval("log").values())


def test_a_rallied_unit_rejoins_and_the_last_order_is_repeated(lua):
    lua.execute("handle.attack(); log = {}")
    lua.execute("handle.check_rallies(); own[2].routing = true; handle.check_rallies()")
    assert log(lua) == []
    lua.execute("own[2].routing = false; rallied = handle.check_rallies()")
    assert list(lua.eval("rallied").values()) == [2]
    assert log(lua) == ["remove b", "add b"]
    lua.execute("run_callbacks()")
    assert log(lua)[-1] == "attack x"
    assert lua.eval("handle.rejoined") == 1


def test_a_shattered_or_dead_unit_does_not_rejoin(lua):
    lua.execute("handle.check_rallies(); own[1].routing = true; own[2].routing = true; handle.check_rallies()")
    lua.execute("own[1].routing = false; own[1].shattered = true; own[2].routing = false; own[2].men = 0")
    assert len(list(lua.eval("handle.check_rallies()").values())) == 0


def test_without_an_order_rallied_units_only_rejoin(lua):
    lua.execute("handle.check_rallies(); own[1].routing = true; handle.check_rallies(); own[1].routing = false")
    lua.execute("handle.check_rallies(); run_callbacks()")
    assert log(lua) == ["add a", "add b", "remove a", "add a"]


@pytest.fixture
def idle_lua(lua):
    lua.execute("""
        for _, u in ipairs(own) do
            u.hp, u.moving, u.melee, u.target = 1.0, false, false, nil
            function u:unary_hitpoints() return self.hp end
            function u:is_moving() return self.moving end
            function u:is_in_melee() return self.melee end
            function u:current_target() return self.target end
        end
        solo_planners = 0
        local create = alliance.create_ai_unit_planner
        alliance.create_ai_unit_planner = function()
            solo_planners = solo_planners + 1
            local p = {}
            function p:add_units(u) log[#log + 1] = 'solo add ' .. u.name end
            function p:remove_units(u) log[#log + 1] = 'solo remove ' .. u.name end
            function p:attack_unit(u) log[#log + 1] = 'solo attack ' .. u.name end
            return p
        end
        function shoot(ms) own[1].hp = own[1].hp - 0.01; return handle.check_idle(ms) end
    """)
    return lua


def test_idle_under_fire_is_kicked_back_then_sent_to_attack(idle_lua):
    lua = idle_lua
    lua.execute("handle.attack(); log = {}")
    # losses show from the second reading on: idle under fire from 1 s, the kick at 7 s
    for ms in range(0, 7001, 1000):
        kicked = list(lua.eval(f"shoot({ms})").values())
    assert [dict(k) for k in kicked] == [{"unit": 1, "stage": 1}]
    assert log(lua)[:2] == ["remove a", "add a"]
    lua.execute("run_callbacks()")
    assert log(lua)[-1] == "attack x"
    stages = []
    for ms in range(8000, 30001, 1000):
        stages += [dict(k)["stage"] for k in lua.eval(f"shoot({ms})").values()]
    assert stages == [2]
    assert log(lua)[-3:] == ["remove a", "solo add a", "solo attack x"]


def test_standing_without_losses_or_while_fighting_is_not_idle(idle_lua):
    lua = idle_lua
    for ms in range(0, 20001, 1000):
        assert len(list(lua.eval(f"handle.check_idle({ms})").values())) == 0      # no losses
    lua.execute("own[1].melee = true")
    for ms in range(21000, 40001, 1000):
        assert len(list(lua.eval(f"shoot({ms})").values())) == 0                    # in melee
