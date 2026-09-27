from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/python'))
from lupa.lua51 import LuaRuntime


class UnitRangeTests(unittest.TestCase):
    def setUp(self):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.lua.globals().sensor = self.lua.execute((ROOT / 'src/units/range.lua').read_text())
        self.lua.execute('''
            calls=0
            function sensitive(v) calls=calls+1; return v end
            function unit(friendly, visibility)
                local u={friendly=friendly,visibility=visibility}
                function u:is_visible_to_alliance() return self.visibility() end
                function u:missile_range() return sensitive(130) end
                function u:unit_in_range(t) return sensitive(false) end
                function u:unit_distance(t) return sensitive(120) end
                function u:position()
                    sensitive();return {get_x=function()return 1 end,get_z=function()return 2 end}
                end
                return u
            end
            function context()
                return {observer_alliance={},observed_ms=500,
                    is_friendly=function(u)return u.friendly end,
                    cco=function(u,k) return sensitive(nil) end}
            end
        ''')

    def test_every_hidden_or_unknown_endpoint_blocks_all_sensitive_calls(self):
        self.lua.execute('''
            local variants={function()return false end,function()return nil end,
                function()return 0 end,function()return 'true' end,
                function()error('SECRET COORDINATES')end}
            for _,visibility in ipairs(variants) do
                for _,pair in ipairs({'own_enemy','enemy_own','enemy_enemy_source',
                    'enemy_enemy_target','both_enemy','source_only'}) do
                    local own=unit(true,function()error('friendly gate needs no visibility')end)
                    local hidden=unit(false,visibility)
                    local visible=unit(false,function()return true end)
                    local a,b
                    if pair=='own_enemy' then a,b=own,hidden
                    elseif pair=='enemy_own' then a,b=hidden,own
                    elseif pair=='enemy_enemy_source' then a,b=hidden,visible
                    elseif pair=='enemy_enemy_target' then a,b=visible,hidden
                    elseif pair=='both_enemy' then a,b=hidden,hidden
                    else a,b=hidden,nil end
                    calls=0;local r=sensor.observe(a,b,context())
                    assert(r.access=='withheld' and calls==0,pair)
                    assert(r.sensors.missile_range_m.status=='unknown')
                    if b then assert(r.sensors.unit_distance_m.status=='unknown') end
                    assert(not (r.source_gate.reason or ''):find('SECRET'))
                end
            end
        ''')

    def test_relationship_and_observer_fail_closed(self):
        self.lua.execute('''
            local enemy=unit(false,function()return true end)
            for _,bad in ipairs({function()return nil end,function()return 'ally' end,
                function()error('private')end}) do
                local c=context();c.is_friendly=bad;calls=0
                assert(sensor.observe(enemy,nil,c).access=='withheld' and calls==0)
            end
            local c=context();c.observer_alliance=nil;calls=0
            assert(sensor.observe(enemy,nil,c).access=='withheld' and calls==0)
            assert(sensor.observe(enemy,nil,{owned=true,side='ally'}).access=='withheld' and calls==0)
        ''')

    def test_allied_and_visible_sources_and_targets_are_allowed(self):
        self.lua.execute('''
            for _,friendly in ipairs({true,false}) do
                local u=unit(friendly,function()return true end)
                local r=sensor.observe(u,u,context())
                assert(r.access=='allowed' and r.observed_ms==500)
                assert(r.sensors.missile_range_m.value==130)
                assert(r.sensors.unit_in_range.status=='known' and r.sensors.unit_in_range.value==false)
                assert(r.sensors.centre_distance_xz_m.value==0)
                local function plain(t)
                    assert(getmetatable(t)==nil)
                    for _,v in pairs(t) do
                        assert(type(v)~='function' and type(v)~='userdata' and v~=u)
                        if type(v)=='table' then plain(v) end
                    end
                end
                plain(r)
            end
        ''')

    def test_card_key_is_enumerated_zero_is_known_and_work_is_bounded(self):
        self.lua.execute('''
            local u=unit(true,function()return true end)
            for _,index in ipairs({0,3}) do
                local c=context();c.cco=function(_,k)
                    if k=='UnitDetailsContext.StatList.Size' then return 4 end
                    if k=='UnitDetailsContext.StatList.At('..index..').Key' then return 'scalar_missile_range' end
                    if k:find('%.Key$') then return 'other' end
                    if k:find('%.Value$') then return 0 end
                    if k:find('%.DisplayedValue$') then return '130' end
                    error('SECRET')
                end
                local s=sensor.observe(u,nil,c).sensors.card
                assert(s.Value.status=='known' and s.Value.value==0)
                assert(s.DisplayedValue.reason=='invalid_type' and s.ValueBase.reason=='read_error')
            end
            for _,count in ipairs({101,-1,1.5,math.huge}) do
                local c=context();local n=0;c.cco=function(_,k)n=n+1;return count end
                assert(sensor.observe(u,nil,c).sensors.card.Value.status=='unknown' and n==1)
            end
        ''')

    def test_nil_errors_nonfinite_and_invalid_coordinates_are_unknown(self):
        self.lua.execute('''
            local u=unit(true,function()return true end)
            for _,fn in ipairs({function()return nil end,function()return 0/0 end,
                function()return math.huge end,function()error('secret')end}) do
                u.missile_range=fn
                assert(sensor.observe(u,nil,context()).sensors.missile_range_m.status=='unknown')
            end
            u.missile_range=function()return 0 end
            u.unit_distance=function()return 0 end
            u.position=function()return {get_x=function()return math.huge end,get_z=function()return 0 end}end
            local s=sensor.observe(u,u,context()).sensors
            assert(s.missile_range_m.value==0 and s.unit_distance_m.value==0)
            assert(s.centre_distance_xz_m.status=='unknown')
        ''')

    def test_visible_hidden_visible_has_no_last_seen_refresh(self):
        self.lua.execute('''
            local visible=true;local u=unit(false,function()return visible end)
            local c=context();local first=sensor.observe(u,nil,c)
            visible=false;c.observed_ms=600;calls=0
            local hidden=sensor.observe(u,nil,c)
            assert(calls==0 and hidden.sensors.missile_range_m.value==nil)
            assert(first.sensors.missile_range_m.value==130 and first.observed_ms==500)
            visible=true;c.observed_ms=700
            assert(sensor.observe(u,nil,c).sensors.missile_range_m.value==130)
        ''')


if __name__ == '__main__':
    unittest.main()
