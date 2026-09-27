local unit_state=(function()
-- Trusted read-only sensor module, deliberately NOT an extension of policy API v1.
-- Caller establishes ownership; this does not authenticate an untrusted caller.
local M={version=1}
local function finite(n)return type(n)=='number' and n==n and n-n==0 end
local function unknown(reason)return {status='unknown',reason=reason}end
local function read(fn,kind)
 local ok,value=pcall(fn)
 if not ok then return unknown('read_error') end
 if value==nil then return unknown('nil') end
 if type(value)~=kind then return unknown('type_'..type(value)) end
 if kind=='number' and not finite(value) then return unknown('nonfinite') end
 return {status='known',value=value}
end
local numeric={'unary_hitpoints','number_of_men_alive','initial_number_of_men','unary_of_men_alive',
 'ammo_left','starting_ammo','bearing','ordered_bearing','ordered_width','slow_speed','fast_speed',
 'left_flank_threat','right_flank_threat','rear_threat'}
local booleans={'is_routing','is_shattered','is_leaving_battle','is_moving','is_moving_fast','is_in_melee',
 'is_hidden','is_script_controlled','is_wavering','is_under_missile_attack','is_idle','is_controllable',
 'is_left_flank_threatened','is_right_flank_threatened','is_rear_flank_threatened'}
local cco_numbers={'HealthValue','HealthMax','HealthPercent','PercentHpLostRecently','DamageInflictedRecently','PrimaryAmmoPercent','MoralePercent','MoraleState','FatigueState','NumKills','NumEntities','NumEntitiesInitial'}
local cco_strings={'MoraleName','MoraleGreatestEffect','FatigueName'}
local cco_booleans={'IsFiringMissiles','IsRouting','IsShattered','IsAlive','IsWithdrawing','IsAwaitingOrderAfterRally','IsOutOfControl','IsWavering','IsTakingDamage','IsUnderMissileAttack','IsInLastStand'}
local function vector(fn)
 local ok,p=pcall(function()
  local v=fn();local r={x=v:get_x(),y=v:get_y(),z=v:get_z()}
  assert(finite(r.x) and finite(r.y) and finite(r.z));return r
 end)
 return ok and {status='known',value=p} or unknown('invalid_vector')
end
function M.observe(unit,options)
 options=options or {}
 local result={schema_version=1,sensors={}}
 if options.observer_alliance then
  result.visibility=read(function()return unit:is_visible_to_alliance(options.observer_alliance)end,'boolean')
 else result.visibility=unknown('observer_missing') end
 -- New enemy sensor permissions require a separate reviewed schema. A visible
 -- enemy is not permission to disclose its orders, morale or threat indicators.
 if options.owned~=true then result.access='withheld_own_only';return result end
 result.access='own';local s=result.sensors
 for _,key in ipairs(numeric) do s['native.'..key]=read(function()return unit[key](unit)end,'number') end
 for _,key in ipairs(booleans) do s['native.'..key]=read(function()return unit[key](unit)end,'boolean') end
 s['native.fatigue_state']=read(function()return unit:fatigue_state()end,'string')
 s['native.position']=vector(function()return unit:position()end)
 s['native.ordered_position']=vector(function()return unit:ordered_position()end)
 for _,key in ipairs({'defend','skirmish','fire_at_will','change_formation_spacing'}) do
  s['native.behaviour.'..key]=read(function()return unit:is_behaviour_active(key)end,'boolean')
 end
 s['native.current_target']=read(function()
  local target=unit:current_target()
  if not target then return '' end
  if not options.observer_alliance or target:is_visible_to_alliance(options.observer_alliance)~=true then return nil end
  if not options.target_id then return nil end
  return options.target_id(target)
 end,'string')
 local function cco(key)
  if not options.cco then return nil end
  return options.cco(unit,key)
 end
 for _,key in ipairs(cco_numbers) do s['cco.'..key]=read(function()return cco(key)end,'number') end
 for _,key in ipairs(cco_strings) do s['cco.'..key]=read(function()return cco(key)end,'string') end
 for _,key in ipairs(cco_booleans) do s['cco.'..key]=read(function()return cco(key)end,'boolean') end
 local size=read(function()return cco('StatusList.Size')end,'number')
 s['cco.StatusList.Size']=size
 if size.status=='known' and size.value>=0 and size.value%1==0 and size.value<=64 then
  local values={};local complete=true
  for i=0,size.value-1 do
   local entry=read(function()return cco('StatusList.At('..i..').Key')end,'string')
   if entry.status=='known' then values[#values+1]=entry.value else complete=false end
  end
  s['cco.StatusList.Keys']=complete and {status='known',value=values} or unknown('incomplete_status_list')
 else s['cco.StatusList.Keys']=unknown('invalid_status_count') end
 return result
end
return M

end)()
if not bm or _G.bai_unit_actions then return end
_G.bai_unit_actions=true
local prefix='tww3_bai_map_capture_'
local mode='basic'
local function enc(v)
 local t=type(v)
 if t=='nil' then return 'null' elseif t=='boolean' or t=='number' then return tostring(v) elseif t=='table' then
  local keys,out={},{};for k in pairs(v) do keys[#keys+1]=k end;table.sort(keys,function(a,b)return tostring(a)<tostring(b)end)
  for _,k in ipairs(keys) do out[#out+1]=enc(tostring(k))..':'..enc(v[k]) end;return '{'..table.concat(out,',')..'}'
 else return '"'..tostring(v):gsub('[%z\1-\31\\"]',function(c)local es={['"']='\\"',['\\']='\\\\',['\n']='\\n',['\r']='\\r',['\t']='\\t'};return es[c] or string.format('\\u%04x',string.byte(c))end)..'"' end
end
local function emit(event,r)r=r or {};r.event=event;r.ms=bm:time_elapsed_ms();r.wall=os.date('!%Y-%m-%dT%H:%M:%SZ');local f=assert(io.open(prefix..'events.jsonl','a'));f:write(enc(r)..'\n');f:close()end
local failed=false
local function guard(fn)return function()if failed then return end;local ok,e=xpcall(fn,function(x)return debug.traceback(tostring(x),2)end);if not ok then failed=true;emit('probe_error',{message=e})end end end
local function safe(r,k,fn)local ok,v=pcall(fn);if ok then r[k]=v==nil and 'nil' or v else r[k..'_error']=tostring(v) end end
local U,C={},{},{}
local unit_side,stat_keys={},{}
local own={'spears','archers','general'};local zs={spears=-350,archers=-200,general=250};local target={spears='target1',archers='target2',general='target3'}
local stage='initial';local start_ms=0;local steps={};local step_index=0
local function v(x,z)local p=battle_vector:new();p:set_x(x);p:set_y(bm:get_terrain_height(x,z));p:set_z(z);return p end
local function query(u,key)return common.get_context_value('CcoBattleUnit',tostring(u:unique_ui_id()),key)end
local function sample(n)
 local u=U[n];local side=unit_side[n]
 local function cco(unit,key)return query(unit,key)end
 local options={owned=true,observer_alliance=bm:alliances():item(side),cco=cco,target_id=function(t)return t:name()end}
 local row={stage=stage,name=n,side=side,unit_type=u:type(),elapsed=bm:time_elapsed_ms()-start_ms,
  readings=unit_state.observe(u,options),
  enemy_gate=unit_state.observe(u,{owned=false,observer_alliance=bm:alliances():item(3-side),cco=cco})}
 emit('state_sample',row)
 if not stat_keys[n] then
  stat_keys[n]={};local size=query(u,'UnitDetailsContext.StatList.Size')
  local keys={};if type(size)=='number' and size>=0 and size<=100 then
   for i=0,size-1 do
    local key=query(u,'UnitDetailsContext.StatList.At('..i..').Key');keys[#keys+1]=key
    if type(key)=='string' and (key:lower():find('morale') or key:lower():find('leadership')) then stat_keys[n][#stat_keys[n]+1]={index=i,key=key} end
   end
  end
  emit('card_stat_keys',{name=n,size=size,keys=keys})
 end
 for _,entry in ipairs(stat_keys[n])do
  local values={stage=stage,name=n,key=entry.key,index=entry.index}
  for _,field in ipairs({'Value','DisplayedValue','ValueBase'})do safe(values,field,function()return query(u,'UnitDetailsContext.StatList.At('..entry.index..').'..field)end)end
  emit('card_stat',values)
 end
end
local function all(fn)for _,n in ipairs(own)do fn(n,C[n],U[n])end end
local function add(name,secs,fn)steps[#steps+1]={name=name,ms=secs*1000,fn=fn}end
local function place(n,x,z,b,w)C[n]:halt();C[n]:teleport_to_location(v(x,z),b or 90,w or (n=='general' and 5 or 30));emit('fixture_teleport',{name=n,x=x,z=z,bearing=b or 90,width=w or 30})end
local function geometry(n)
 local u=U[n];local count=query(u,'ManList.Size');local row={stage=stage,name=n,count=count,points={}}
 if type(count)=='number' and count<500 then for i=0,count-1 do local x,y,z=query(u,'ManList.At('..i..').Position');row.points[i]={x=x,y=y,z=z}end end
 emit('formation_geometry',row)
end
local function next_step()
 step_index=step_index+1
 if step_index>#steps then bm:remove_process(prefix..'tick');emit('probe_done',{phase=bm:get_current_phase_name(),speed=bm:current_battle_speed(),mode=mode});return end
 local s=steps[step_index];stage=s.name;start_ms=bm:time_elapsed_ms();emit('step_begin',{stage=stage,duration=s.ms});s.fn()
 bm:callback(guard(function()all(function(n)sample(n); end);emit('step_end',{stage=stage});next_step()end),s.ms,prefix..'step_'..step_index)
end
local function setup_basic()
 add('settle',8,function()end)
 add('walk',65,function()all(function(n,c)c:goto_location(v(230,zs[n]),false)end)end)
 add('run',40,function()all(function(n,c)c:goto_location(v(150,zs[n]),true)end)end)
 add('rotate',25,function()all(function(n,c)c:rotate(90,false)end)end)
 add('narrow_face',30,function()all(function(n,c)c:goto_location_angle_width(v(150,zs[n]),90,n=='general' and 5 or 15,false)end)end)
 add('wide_face',30,function()all(function(n,c)c:goto_location_angle_width(v(150,zs[n]),90,n=='general' and 5 or 50,false)end)end)
 add('move_before_halt',5,function()all(function(n,c)c:goto_location(v(280,zs[n]),true)end)end)
 add('halt',12,function()all(function(n,c)c:halt()end)end)
 add('guard_on',3,function()all(function(n,c,u)if u:can_use_behaviour('defend') then c:change_behaviour_active('defend',true)end end)end)
 add('guard_off',3,function()all(function(n,c,u)if u:can_use_behaviour('defend') then c:change_behaviour_active('defend',false)end end)end)
 add('ranged_fixture',8,function()place('archers',200,-200,90,30);place('target2',300,-200,270,30);C.archers:melee(false);C.archers:fire_at_will(false)end)
 add('ranged_idle_disabled',20,function()end)
 add('ranged_explicit',35,function()C.archers:attack_unit(U.target2,true,false)end)
 add('ranged_toggle_only_off',20,function()C.archers:fire_at_will(false)end)
 add('ranged_halt_off',20,function()C.archers:halt();C.archers:fire_at_will(false)end)
 add('ranged_free_fire',25,function()C.archers:fire_at_will(true)end)
 add('ranged_stop',15,function()C.archers:halt();C.archers:fire_at_will(false)end)
 add('melee_fixture',8,function()all(function(n,c)place(n,200,zs[n],90,30);place(target[n],260,zs[n],270,30);c:fire_at_will(false);c:melee(true)end)end)
 add('melee_charge',40,function()all(function(n,c)c:attack_unit(U[target[n]],false,true)end)end)
 add('disengage',40,function()all(function(n,c)c:goto_location(v(120,zs[n]),true)end)end)
 add('park_targets',5,function()for _,n in ipairs({'target1','target2','target3'})do place(n,360,zs[n=='target1' and 'spears' or n=='target2' and 'archers' or 'general'],270,30)end end)
 for i=1,10 do local x=i%2==1 and 290 or 120;add('fatigue_run_'..i,30,function()all(function(n,c)c:goto_location(v(x,zs[n]),true)end)end) end
 add('rest',240,function()all(function(n,c)c:halt()end)end)
 add('withdraw',160,function()all(function(n,c)c:withdraw(true)end)end)
end
local function setup_passives()
 add('open_ground',20,function()end)
 add('forest_hide',35,function()place('spears',-45.5,-165.5,90,15);place('archers',-45.5,-345.5,90,15);place('general',134.5,389.5,90,5);place('target1',-400,0,90,30);place('target2',-400,150,90,30);place('target3',-400,300,90,30)end)
 add('forest_reveal',15,function()place('target1',-15.5,-165.5,270,15);place('target2',-15.5,-345.5,270,15);place('target3',164.5,389.5,270,15)end)
 add('support_far',20,function()place('spears',180,0,90,30);place('archers',180,-80,90,30);place('general',350,0,90,5);place('target1',-400,0,90,30);place('target2',-400,150,90,30);place('target3',-400,300,90,30)end)
 add('support_near',20,function()place('general',195,0,90,5)end)
 add('support_far_again',20,function()place('general',350,0,90,5)end)
 add('ability_position',10,function()place('general',195,0,90,5)end)
 for _,k in ipairs({'wh_main_character_abilities_stand_your_ground','wh_main_character_abilities_foe_seeker'})do local key=k
 add(key,30,function()local can=U.general:can_perform_special_ability(key);emit('ability_order',{ability=key,can_perform=can});if can then C.general:perform_special_ability(key,U.general)end end)
 end
 add('withdraw_false',160,function()all(function(n,c)c:withdraw(false)end)end)
end
local function setup_charge()
 add('charge_setup',15,function()all(function(n,c)place(n,200,zs[n],90,n=='general' and 5 or 30);place(target[n],310,zs[n],270,30);c:change_behaviour_active('defend',false)end)end)
 add('cavalry_charge',45,function()all(function(n,c)if mode=='charge_moving' then c:goto_location(v(300,zs[n]),false)end;C[target[n]]:attack_unit(U[n],false,true)end)end)
end
local function setup_guard()
 local enabled=mode=='guard_on'
 add('guard_setup',15,function()all(function(n,c)place(n,200,zs[n],90,n=='general' and 5 or 30);place(target[n],260,zs[n],270,30);c:melee(true);c:change_behaviour_active('defend',enabled)end)end)
 add('guard_attack',40,function()all(function(n,c)c:attack_unit(U[target[n]],false,true)end)end)
 add('guard_target_routs',35,function()all(function(n)C[target[n]]:morale_behavior_rout();emit('fixture_forced_rout',{name=target[n]})end)end)
end
local function init()
 bm:modify_battle_speed(0);bm:output_battle_xml(prefix..'ready.xml');local f=assert(io.open(prefix..'grid.csv','w'));f:write('experiment,mode\nunit_actions,'..mode..'\n');f:close()
 for ai=1,bm:alliances():count() do local army=bm:alliances():item(ai):armies():item(1)
  for i=1,army:units():count() do local u=army:units():item(i);local n=u:name();U[n]=u;unit_side[n]=ai;local c=army:create_unit_controller();c:add_units(u);c:take_control();c:halt();c:fire_at_will(false);if u:can_use_behaviour('skirmish')then c:change_behaviour_active('skirmish',false)end;C[n]=c
   local row={name=n,type=u:type(),uid=u:unique_ui_id(),commanding=u:is_commanding_unit(),men=u:initial_number_of_men(),range=u:missile_range()}
   for _,key in ipairs({'defend','skirmish','fire_at_will','change_formation_spacing'})do safe(row,'supports_'..key,function()return u:can_use_behaviour(key)end)end

   for _,key in ipairs({'CharacterRank','HasCharacterRank','ExperienceLevel','StatusList.Size'})do safe(row,'cco_'..key,function()return query(u,key)end)end
   emit('unit_capabilities',row)
  end
 end
 emit('conditions',{version=common.game_version(),mode=mode})
 if mode=='passives' then setup_passives() elseif mode=='charge_still' or mode=='charge_moving' then setup_charge() elseif mode=='guard_on' or mode=='guard_off' then setup_guard() else setup_basic() end
 bm:register_phase_change_callback('Deployed',guard(function()all(function(n)place(n,150,zs[n],90,n=='general' and 5 or 30);place(target[n],360,zs[n],270,30)end);place('reserve',-350,350,90,30);place('enemyreserve',-350,-450,90,5);bm:callback(guard(function()bm:repeat_callback(guard(function()for _,n in ipairs({'spears','archers','general','target1','target2','target3'})do sample(n)end;if stage:find('withdraw') then local gone=true;all(function(n)if not U[n]:is_leaving_battle()then gone=false end end);if gone then bm:remove_process(prefix..'tick');emit('probe_done',{mode=mode,speed=bm:current_battle_speed(),reason='all_test_units_leaving'});failed=true end end end),500,prefix..'tick');next_step()end),1000,prefix..'start')end))
 bm:modify_battle_speed(20);bm:end_current_battle_phase()
end
bm:real_callback(guard(init),3000,prefix..'init')
