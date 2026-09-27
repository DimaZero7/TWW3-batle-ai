local range_sensor=(function()
-- Optional trusted sensor; NOT part of policy API v1 and NOT a sandbox.
-- Only the trusted adapter may call observe or possess its engine arguments.
-- is_friendly must use the adapter's native alliance registry, never policy input.
-- Both endpoints are gated before any range/CCO/pair/position read. No memory.
local M = {version = 1}
local function finite(v)
    return type(v) == 'number' and v == v and v - v == 0
end
local function unknown(reason) return {status = 'unknown', reason = reason} end
local function read(fn, kind)
    local ok, value = pcall(fn)
    if not ok then return unknown('read_error') end
    if value == nil then return unknown('nil') end
    if type(value) ~= kind then return unknown('invalid_type') end
    if kind == 'number' and not finite(value) then return unknown('nonfinite') end
    return {status = 'known', value = value}
end
local function gate(unit, context)
    local relation = read(function() return context.is_friendly(unit) end, 'boolean')
    if relation.status ~= 'known' then return unknown('relationship_unknown') end
    if relation.value then return {status = 'known', value = true} end
    if not context.observer_alliance then return unknown('observer_missing') end
    return read(function() return unit:is_visible_to_alliance(context.observer_alliance) end, 'boolean')
end
local function permitted(g) return g.status == 'known' and g.value == true end
local function card_range(unit, context)
    local function query(key) return context.cco(unit, key) end
    local count = read(function() return query('UnitDetailsContext.StatList.Size') end, 'number')
    local index
    if count.status == 'known' and count.value >= 0 and count.value <= 100 and count.value % 1 == 0 then
        for i = 0, count.value - 1 do
            local key = read(function() return query('UnitDetailsContext.StatList.At(' .. i .. ').Key') end, 'string')
            if key.status == 'known' and key.value == 'scalar_missile_range' then index = i; break end
        end
    end
    local result = {}
    for _, field in ipairs({'Value', 'DisplayedValue', 'ValueBase'}) do
        result[field] = index and read(function()
            return query('UnitDetailsContext.StatList.At(' .. index .. ').' .. field)
        end, 'number') or unknown('stat_key_unavailable')
    end
    return result
end
local function centre_distance(source, target)
    return read(function()
        local a, b = source:position(), target:position()
        local ax, az, bx, bz = a:get_x(), a:get_z(), b:get_x(), b:get_z()
        assert(finite(ax) and finite(az) and finite(bx) and finite(bz))
        return math.sqrt((ax - bx)^2 + (az - bz)^2)
    end, 'number')
end
function M.observe(source, target, context)
    context = context or {}
    local result = {schema_version = M.version, access = 'withheld', sensors = {}}
    if finite(context.observed_ms) then result.observed_ms = context.observed_ms end
    result.source_gate = gate(source, context)
    if target ~= nil then result.target_gate = gate(target, context) end
    if not permitted(result.source_gate) or (target ~= nil and not permitted(result.target_gate)) then
        result.sensors.missile_range_m = unknown('endpoint_unavailable')
        result.sensors.card = {Value = unknown('endpoint_unavailable'),
            DisplayedValue = unknown('endpoint_unavailable'), ValueBase = unknown('endpoint_unavailable')}
        if target ~= nil then
            result.sensors.unit_in_range = unknown('endpoint_unavailable')
            result.sensors.unit_distance_m = unknown('endpoint_unavailable')
            result.sensors.centre_distance_xz_m = unknown('endpoint_unavailable')
        end
        return result
    end
    result.access = 'allowed'
    result.sensors.missile_range_m = read(function() return source:missile_range() end, 'number')
    result.sensors.card = card_range(source, context)
    if target ~= nil then
        result.sensors.unit_in_range = read(function() return source:unit_in_range(target) end, 'boolean')
        result.sensors.unit_distance_m = read(function() return source:unit_distance(target) end, 'number')
        result.sensors.centre_distance_xz_m = centre_distance(source, target)
    end
    return result
end
return M

end)()
if not bm or _G.bai_range_probe then return end
_G.bai_range_probe=true
local mode='range'
local prefix='tww3_bai_map_capture_'
local function enc(v)
 local t=type(v)
 if t=='nil' then return 'null' elseif t=='boolean' then return tostring(v)
 elseif t=='number' then return (v==v and v-v==0) and tostring(v) or 'null'
 elseif t=='table' then
  local keys,out={},{};for k in pairs(v)do keys[#keys+1]=k end;table.sort(keys,function(a,b)return tostring(a)<tostring(b)end)
  for _,k in ipairs(keys)do out[#out+1]=enc(tostring(k))..':'..enc(v[k])end;return '{'..table.concat(out,',')..'}'
 else return '"'..tostring(v):gsub('[%z\1-\31\\"]',function(c)local es={['"']='\\"',['\\']='\\\\',['\n']='\\n',['\r']='\\r',['\t']='\\t'};return es[c] or string.format('\\u%04x',string.byte(c))end)..'"' end
end
local function log(file,row)local f=assert(io.open(prefix..file..'.jsonl','a'));f:write(enc(row)..'\n');f:close()end
local function emit(event,row)row=row or {};row.event=event;row.ms=bm:time_elapsed_ms();log('events',row)end
local failed=false
local function guard(fn)return function()if failed then return end;local ok,e=xpcall(fn,function(x)return debug.traceback(tostring(x),2)end);if not ok then failed=true;emit('probe_error',{message=e})end end end
local U,C,P,meta={},{},{},{}
local unproxy={};local counts={};local all_names={}
local function query(u,key)return common.get_context_value('CcoBattleUnit',tostring(u:unique_ui_id()),key)end
local function mark(k)counts[k]=(counts[k]or 0)+1 end
local function proxy(u)
 local p={};unproxy[p]=u
 function p:is_visible_to_alliance(a)return u:is_visible_to_alliance(a)end
 function p:missile_range()mark('missile_range');return u:missile_range()end
 function p:unit_in_range(t)mark('unit_in_range');return u:unit_in_range(unproxy[t])end
 function p:unit_distance(t)mark('unit_distance');return u:unit_distance(unproxy[t])end
 function p:position()mark('position');return u:position()end
 return p
end
local stage='initial';local started=0;local steps={};local step_index=0
local pairs_to_read={{'archers','target2'},{'hybrid','target1'},{'spears','target3'},
 {'allyarchers','target2'},{'allyhybrid','target1'},{'enemyarchers','archers'},
 {'enemyhybrid','hybrid'},{'archers','enemyarchers'},{'hybrid','enemyhybrid'},
 {'enemyarchers','enemyhybrid'},{'enemyhybrid','enemyarchers'},
 {'enemyarchers'},{'enemyhybrid'},{'allyarchers'},{'allyhybrid'}}
local function safe(fn)local ok,v=pcall(fn);return ok and {status='known',value=v} or {status='unknown',reason='read_error'}end
local function sample()
 for _,pair in ipairs(pairs_to_read)do
  local a,b=pair[1],pair[2];counts={}
  local r=range_sensor.observe(P[a],b and P[b],{observer_alliance=bm:alliances():item(1),
   observed_ms=bm:time_elapsed_ms(),is_friendly=function(p)local m=meta[unproxy[p]];return m and m.alliance==1 end,
   cco=function(p,k)mark('cco');return query(unproxy[p],k)end})
  local total=0;for _,n in pairs(counts)do total=total+n end
  assert(r.access~='withheld' or total==0,'withheld read executed sensitive calls')
  log('permitted',{stage=stage,elapsed=bm:time_elapsed_ms()-started,source=a,target=b,reading=r,sensitive_calls=counts})
  local raw=range_sensor.observe(U[a],b and U[b],{observed_ms=bm:time_elapsed_ms(),is_friendly=function()return true end,cco=query})
  log('raw',{stage=stage,elapsed=bm:time_elapsed_ms()-started,source=a,target=b,reading=raw,
   source_visible=safe(function()return U[a]:is_visible_to_alliance(bm:alliances():item(1))end),
   target_visible=b and safe(function()return U[b]:is_visible_to_alliance(bm:alliances():item(1))end),
   ammo=safe(function()return U[a]:ammo_left()end),firing=safe(function()return query(U[a],'IsFiringMissiles')end),
   in_melee=safe(function()return U[a]:is_in_melee()end),moving=safe(function()return U[a]:is_moving()end),
   bearing=safe(function()return U[a]:bearing()end),width=safe(function()return U[a]:ordered_width()end),
   target_hp=b and safe(function()return U[b]:unary_hitpoints()end)})
 end
end
local function v(x,z)local p=battle_vector:new();p:set_x(x);p:set_y(bm:get_terrain_height(x,z));p:set_z(z);return p end
local function place(n,x,z,b,w)
 C[n]:halt();C[n]:teleport_to_location(v(x,z),b or 90,w or 30)
 emit('fixture_teleport',{stage=stage,name=n,x=x,z=z,bearing=b or 90,width=w or 30})
end
local function add(name,secs,fn)steps[#steps+1]={name=name,ms=secs*1000,fn=fn or function()end}end
local lanes={{'archers','target2',-200,130},{'hybrid','target1',0,90},{'spears','target3',200,0}}
local function geometry(d,b,w,relative)
 for _,l in ipairs(lanes)do place(l[1],200,l[3],b or 90,w or 30);place(l[2],200+d+(relative and l[4]or 0),l[3],270,w or 30)end
end
local function ranged(fn)for _,n in ipairs({'archers','hybrid'})do fn(n,C[n],U[n])end end
local function setup_range()
 add('settle',8)
 for _,d in ipairs({160,145,140,135,130,125,120,110,100,95,90,85,80,70})do
  local distance=d;add('distance_'..d,4,function()geometry(distance,90,30)end)
 end
 for _,w in ipairs({10,30,60})do for _,b in ipairs({0,90,180,270})do
  local width,bearing=w,b;add('width_'..w..'_bearing_'..b..'_range_plus_5',4,function()geometry(5,bearing,width,true)end)
 end end
 add('ranged_idle',12,function()geometry(70,90,30);ranged(function(n,c)c:melee(false);c:fire_at_will(false)end)end)
 add('explicit_shooting',30,function()C.archers:attack_unit(U.target2,true,false);C.hybrid:attack_unit(U.target1,true,false)end)
 add('halt',12,function()ranged(function(n,c)c:halt();c:fire_at_will(false)end)end)
 add('melee_mode_idle',12,function()ranged(function(n,c)c:melee(true)end)end)
 add('ranged_mode_again',12,function()ranged(function(n,c)c:melee(false)end)end)
 add('forced_empty_ammo',12,function()ranged(function(n,c,u)u:set_current_ammo_unary(0);emit('fixture_forced_ammo',{name=n,unary=0})end)end)
 add('forced_empty_attack',12,function()C.archers:attack_unit(U.target2,true,false);C.hybrid:attack_unit(U.target1,true,false)end)
 add('restored_ammo',12,function()ranged(function(n,c,u)c:halt();u:set_current_ammo_unary(1);emit('fixture_forced_ammo',{name=n,unary=1})end);geometry(70,90,30)end)
end
local function setup_fog()
 add('park_all',12,function()
  for i,n in ipairs(all_names)do local m=meta[U[n]];place(n,m.alliance==1 and -400 or 380,-420+i*55,90,15)end
  place('enemyarchers',-45.5,-345.5,90,15);place('enemyhybrid',-45.5,-165.5,90,15)
 end)
 add('visible_before',20,function()place('archers',-15.5,-345.5,270,15);place('hybrid',-15.5,-165.5,270,15)end)
 add('hidden',45,function()place('archers',-400,-345.5,90,15);place('hybrid',-400,-165.5,90,15)end)
 add('visible_after',20,function()place('archers',-15.5,-345.5,270,15);place('hybrid',-15.5,-165.5,270,15)end)
 add('hidden_again',35,function()place('archers',-400,-345.5,90,15);place('hybrid',-400,-165.5,90,15)end)
end
local function next_step()
 step_index=step_index+1
 if step_index>#steps then
  bm:remove_process(prefix..'sample');emit('probe_done',{mode=mode,speed=bm:current_battle_speed()});bm:modify_battle_speed(0);return
 end
 local s=steps[step_index];stage=s.name;started=bm:time_elapsed_ms();emit('step_begin',{stage=stage,duration=s.ms});s.fn()
 bm:callback(guard(function()sample();emit('step_end',{stage=stage});next_step()end),s.ms,prefix..'step_'..step_index)
end
local function init()
 bm:modify_battle_speed(0);bm:output_battle_xml(prefix..'ready.xml')
 local f=assert(io.open(prefix..'grid.csv','w'));f:write('experiment,mode\nmissile_range,'..mode..'\n');f:close()
 for ai=1,bm:alliances():count()do local armies=bm:alliances():item(ai):armies()
  emit('alliance',{alliance=ai,armies=armies:count()})
  for ar=1,armies:count()do local army=armies:item(ar)
   for i=1,army:units():count()do local u=army:units():item(i);local n=u:name()
    U[n]=u;meta[u]={alliance=ai,army=ar};P[n]=proxy(u);all_names[#all_names+1]=n
    local c=army:create_unit_controller();c:add_units(u);c:take_control();c:halt();c:fire_at_will(false)
    if u:can_use_behaviour('skirmish')then c:change_behaviour_active('skirmish',false)end
    C[n]=c;emit('unit',{name=n,type=u:type(),alliance=ai,army=ar,men=u:initial_number_of_men(),uid=u:unique_ui_id()})
   end
  end
 end
 emit('conditions',{version=common.game_version(),mode=mode})
 assert(meta[U.allyarchers].army~=meta[U.archers].army,'not a separate allied army')
 if mode=='fog' then setup_fog() elseif mode=='boundary' then setup_boundary() else setup_range()end
 bm:register_phase_change_callback('Deployed',guard(function()
  bm:callback(guard(function()bm:repeat_callback(guard(sample),1000,prefix..'sample');next_step()end),1000,prefix..'start')
 end))
 bm:modify_battle_speed(20);bm:end_current_battle_phase()
end
bm:real_callback(guard(init),3000,prefix..'init')
