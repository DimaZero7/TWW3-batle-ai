from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.tools/python'))
from lupa.lua51 import LuaRuntime
class UnitStateTests(unittest.TestCase):
 def setUp(self):
  self.lua=LuaRuntime(unpack_returned_tuples=True)
  self.lua.globals().sensor=self.lua.execute((ROOT/'src/units/state.lua').read_text())
 def test_zero_false_nil_errors_and_nonfinite(self):
  self.lua.execute('''
   local u={ammo_left=function()return 0 end,is_routing=function()return false end,
    unary_hitpoints=function()return 0/0 end,fatigue_state=function()return nil end}
   local s=sensor.observe(u,{owned=true}).sensors
   assert(s['native.ammo_left'].status=='known' and s['native.ammo_left'].value==0)
   assert(s['native.is_routing'].status=='known' and s['native.is_routing'].value==false)
   assert(s['native.unary_hitpoints'].reason=='nonfinite')
   assert(s['native.fatigue_state'].reason=='nil')
   assert(s['native.initial_number_of_men'].reason=='read_error')
  ''')
 def test_visibility_gate_no_other_enemy_reads(self):
  self.lua.execute('''
   for _,v in ipairs({true,false}) do
    local u=setmetatable({is_visible_to_alliance=function()return v end},{__index=function()error('forbidden sensor')end})
    local s=sensor.observe(u,{owned=false,observer_alliance={},cco=function()error('forbidden CCO')end})
    assert(s.access=='withheld_own_only' and next(s.sensors)==nil and s.visibility.value==v)
   end
   local s=sensor.observe({is_visible_to_alliance=function()error('unknown')end},{observer_alliance={}})
   assert(s.visibility.status=='unknown' and next(s.sensors)==nil)
  ''')
 def test_plain_vector_copy_and_target_gate(self):
  self.lua.execute('''
   local p={get_x=function()return 1 end,get_y=function()return 2 end,get_z=function()return 3 end}
   local u={position=function()return p end,current_target=function()return {is_visible_to_alliance=function()return false end} end}
   local s=sensor.observe(u,{owned=true,observer_alliance={},target_id=function()error('must not map hidden target')end}).sensors
   assert(s['native.position'].value.x==1 and s['native.position'].value.get_x==nil)
   assert(s['native.current_target'].status=='unknown')
  ''')
 def test_cco_wrong_type_and_status_list_bound(self):
  self.lua.execute('''
   local s=sensor.observe({},{owned=true,cco=function(u,k)if k=='IsAlive'then return false elseif k=='StatusList.Size'then return 65 else return 'wrong' end end}).sensors
   assert(s['cco.IsAlive'].status=='known' and s['cco.IsAlive'].value==false)
   assert(s['cco.HealthValue'].reason=='type_string')
   assert(s['cco.StatusList.Keys'].reason=='invalid_status_count')
  ''')
 def test_threat_handles_only_become_visible_opaque_ids(self):
  self.lua.execute('''
   local hidden={is_visible_to_alliance=function()return false end}
   local visible={is_visible_to_alliance=function()return true end}
   local u={left_flank_threat=function()return visible end,right_flank_threat=function()return hidden end,rear_threat=function()return nil end}
   local s=sensor.observe(u,{owned=true,observer_alliance={},target_id=function(t)assert(t==visible);return 'visible-unit' end}).sensors
   assert(s['native.left_flank_threat'].value=='visible-unit')
   assert(s['native.right_flank_threat'].status=='unknown')
   assert(s['native.rear_threat'].value=='')
  ''')
 def test_card_morale_uses_enumerated_key_and_keeps_zero(self):
  self.lua.execute('''
   local s=sensor.observe({},{owned=true,cco=function(u,k)
    if k=='UnitDetailsContext.StatList.Size'then return 1 end
    if k=='UnitDetailsContext.StatList.At(0).Key'then return 'stat_morale' end
    if k=='UnitDetailsContext.StatList.At(0).Value'then return 0 end
   end}).sensors
   assert(s['cco.card.stat_morale.Value'].status=='known' and s['cco.card.stat_morale.Value'].value==0)
   assert(s['cco.card.stat_morale.ValueBase'].status=='unknown')
  ''')
if __name__=='__main__':unittest.main()
