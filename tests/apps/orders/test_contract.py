"""apps.orders.contract: command validation."""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().orders = load(runtime, "apps.orders.contract")
    runtime.execute("""
        own = {
            lord = {alive = true, position = {x=0, z=0}, kind = 'lord'},
            archers = {alive = true, position = {x=0, z=0}, kind = 'archers', ammo = 10},
            spears = {alive = true, position = {x=0, z=0}, kind = 'spearmen'},
            dead = {alive = false, position = {x=0, z=0}, kind = 'spearmen'},
        }
        enemies = {
            seen = {visibility = 'visible', position = {x=1, z=1}},
            hidden = {visibility = 'not_visible'},
        }
        function rejects(commands, message, contract)
            local ok, err = pcall(orders.validate, commands, own, enemies, 14, contract)
            assert(not ok, 'expected rejection: ' .. message)
            assert(err:find(message, 1, true), err)
        end
    """)
    return runtime


class TestValidate:
    def test_valid_mix_passes(self, lua):
        lua.execute("""
            orders.validate({
                {unit_id='lord', action='move', x=1, z=2, run=true},
                {unit_id='lord', action='guard', enabled=true},
                {unit_id='archers', action='attack', target_id='seen', mode='ranged'},
                {unit_id='spears', action='halt'},
            }, own, enemies, 14)
        """)

    def test_rejections(self, lua):
        lua.execute("""
            rejects({{unit_id='dead', action='halt'}}, 'unit ownership/alive required')
            rejects({{unit_id='spears', action='attack', target_id='hidden', mode='melee'}}, 'target not visible')
            rejects({{unit_id='spears', action='attack', target_id='seen', mode='ranged'}}, 'invalid attack mode')
            rejects({{unit_id='spears', action='halt'}, {unit_id='spears', action='halt'}}, 'duplicate motion')
            rejects({{unit_id='spears', action='halt', extra=1}}, 'unknown command field')
            rejects({{unit_id='spears', action='fly'}}, 'unsupported action')
            rejects({[2] = {unit_id='spears', action='halt'}}, 'commands must be array')
        """)

    def test_formation_move_needs_contract_and_width_bounds(self, lua):
        lua.execute("""
            rejects({{unit_id='spears', action='move', x=0, z=0, facing_deg=90, width_m=30}}, 'unknown command field')
            orders.validate({{unit_id='spears', action='move', x=0, z=0, facing_deg=90, width_m=30}},
                own, enemies, 14, 'formation-move-v1')
            rejects({{unit_id='lord', action='move', x=0, z=0, facing_deg=90, width_m=30}},
                'invalid move width', 'formation-move-v1')
        """)
