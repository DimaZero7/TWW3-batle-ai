"""apps.plan.services (level 3: the chain). The modules are replaced by spies:
only the order of calls and the data passed between them are checked, not
the decisions (those are tested in each module).
"""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().plan = load(runtime, "apps.plan.services")
    runtime.execute("""
        function spies(strategy_key)
            local calls = {}
            local m = {}
            m.assessment = {
                category = function(u) return 'infantry' end,
                assess = function(input)
                    calls[#calls + 1] = 'assess:' .. input.role .. ':' .. #input.own .. ':' .. #input.enemy
                    return {tag = 'features'}
                end}
            m.strategy = {decide = function(features, units, category)
                calls[#calls + 1] = 'decide:' .. features.tag .. ':' .. #units .. ':' .. category({})
                if strategy_key == 'none' then return {strategy = 'none', candidates = {}} end
                return {strategy = strategy_key, candidates = {}, layout = 'line_and_blocks',
                    formation_params = {a = 1}, roles = {u1 = 'wall', u2 = 'arc'}}
            end}
            m.contract = {check_decision = function(d) calls[#calls + 1] = 'check:' .. d.strategy; return d end}
            m.formation = {plan = function(layout, input, params)
                local roles = {}
                for _, u in ipairs(input.units) do roles[#roles + 1] = u.id .. '=' .. u.role end
                calls[#calls + 1] = 'plan:' .. layout .. ':' .. table.concat(roles, ',') .. ':' .. params.a
                    .. ':' .. tostring(params.b) .. ':' .. input.bearing .. ':' .. tostring(input.enemy_lord and input.enemy_lord.x)
                return {status = 'ok', placements = {}}
            end}
            return m, calls
        end
        INPUT = {role = 'attack', bearing = 15, own = {anchor = {x = 0, z = 0}, units = {{id = 'u1'}, {id = 'u2'}}},
            enemy = {units = {{id = 'e1'}}, lord = {x = 7, z = 9}}}
    """)
    return runtime


def test_modules_are_called_in_order_with_their_data(lua):
    lua.execute("""
        local m, calls = spies('wall_and_arc')
        local r = plan.start(INPUT, m)
        assert(r.status == 'ok' and r.decision.strategy == 'wall_and_arc' and r.features.tag == 'features')
        CALLS = table.concat(calls, ' | ')
    """)
    # The bearing reaches the formation on the engine's facing grid: 15 -> 5 steps of 360/128.
    assert lua.globals().CALLS == ("assess:attack:2:1 | decide:features:2:infantry | check:wall_and_arc | "
                                   "plan:line_and_blocks:u1=wall,u2=arc:1:nil:14.0625:7")


def test_no_strategy_stops_before_the_formation(lua):
    lua.execute("""
        local m, calls = spies('none')
        local r = plan.start(INPUT, m)
        assert(r.status == 'no_strategy' and r.formation == nil)
        CALLS = table.concat(calls, ' | ')
    """)
    assert "plan:" not in lua.globals().CALLS


def test_overrides_reach_the_formation_and_input_units_stay_untouched(lua):
    lua.execute("""
        local m, calls = spies('wall_and_arc')
        INPUT.formation_params = {b = 2}
        plan.start(INPUT, m)
        assert(INPUT.own.units[1].role == nil, 'the chain must not change its input')
        CALLS = calls[#calls]
    """)
    assert lua.globals().CALLS.endswith(":1:2:14.0625:7")
