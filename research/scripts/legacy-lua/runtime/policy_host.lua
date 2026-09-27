-- Trusted Lua 5.1 host. No engine objects are passed into this module's policies.
local M = {}
local function finite(n) return type(n)=='number' and n==n and n-n==0 end
function M.copy(value, depth, seen)
    depth=depth or 0; seen=seen or {}
    assert(depth<24,'data depth exceeded')
    local kind=type(value)
    if kind=='nil' or kind=='boolean' or kind=='string' then return value end
    if kind=='number' then assert(finite(value),'nonfinite data'); return value end
    assert(kind=='table' and getmetatable(value)==nil and not seen[value],'plain acyclic data required')
    seen[value]=true
    local result,count={},0
    for k,v in pairs(value) do
        count=count+1; assert(count<=4096,'data size exceeded')
        assert(type(k)=='string' or (finite(k) and k%1==0),'invalid data key')
        result[k]=M.copy(v,depth+1,seen)
    end
    seen[value]=nil; return result
end
-- Trusted capacity selection, never policy-controlled. Field/action schema stays v1.
function M.profile(name,speed)
    name=name or 'empire-7-v1'
    local sizes={['empire-7-v1']={7,14,4,2,20,'Hidden'},['empire-15-v1']={15,30,8,6,7,'Normal'}}
    local s=sizes[name];assert(s,'unknown capacity profile')
    if speed==nil then speed=s[5] end
    assert(speed==20 or (name=='empire-15-v1' and speed==7),'unsupported playback speed')
    return {id=name,units_per_side=s[1],max_command_proposals_per_step=s[2],
        roster_counts={lord=1,spearmen=s[3],archers=s[4]},speed=speed,window_style=s[6],
        decision_interval_ms=1000,instructions_per_call=1000000}
end
local function command_array(commands,limit)
    assert(type(commands)=='table' and #commands<=limit,'command limit exceeded')
    for k in pairs(commands) do assert(type(k)=='number' and k>=1 and k<=#commands and k%1==0,'commands must be array') end
    for i=1,#commands do assert(commands[i]~=nil,'sparse command array')end
end
local function environment()
    local env={pairs=pairs,ipairs=ipairs,next=next,type=type,tonumber=tonumber,
        tostring=tostring,assert=assert,error=error,select=select}
    for lib,names in pairs({math={'abs','ceil','floor','max','min','sqrt','sin','cos','atan','atan2','acos','asin','tan','exp','log','pow','fmod','pi'},
        string={'byte','char','find','format','gmatch','gsub','len','lower','match','sub','upper'},
        table={'concat','insert','remove','sort'}}) do
        env[lib]={}; for _,name in ipairs(names) do env[lib][name]=_G[lib][name] end
    end
    return env
end
local function bounded(fn,...)
    assert(debug and debug.sethook and debug.gethook,'instruction guard unavailable')
    local old,mask,count=debug.gethook()
    local budget=0
    debug.sethook(function() budget=budget+1000; if budget>1000000 then error('policy instruction budget') end end,'',1000)
    local ok,result=pcall(fn,...)
    debug.sethook(old,mask,count)
    assert(ok,result); return result
end
function M.load(source,context,profile_name)
    local capacity=M.profile(profile_name)
    assert(type(source)=='string' and #source<=131072 and source:byte(1)~=27,'source text required')
    local chunk,err=loadstring(source,'policy'); assert(chunk,err)
    setfenv(chunk,environment())
    local module=bounded(chunk)
    assert(type(module)=='table' and module.api_version==1 and type(module.create)=='function'
        and type(module.step)=='function','API v1 module required')
    if context.own_battle_role or module.battle_role_version then
        local role=context.own_battle_role
        assert(module.battle_role_version==1 and type(role)=='table'and role.version==1
            and(role.role=='attacker'or role.role=='defender'),'native own-role v1 context required')
        for key in pairs(role)do assert(key=='version'or key=='role','unknown own-role field')end
    end
    if context.diagnostic_contract or module.diagnostic_contract then
        assert((context.diagnostic_contract=='formation-move-v1'or context.diagnostic_contract=='deployment-march-formation-v1')and module.diagnostic_contract==context.diagnostic_contract,'explicit movement contract required')
    end
    local placement_v2=context.deployment_contract~=nil or module.deployment_version==2 or module.deployment_contract~=nil
    if placement_v2 then
        assert(context.deployment_contract=='deployment-placement-v2'and module.deployment_contract==context.deployment_contract
            and module.deployment_version==2 and type(module.deploy)=='function','explicit deployment-placement-v2 capability required')
    elseif capacity.id=='empire-15-v1' then
        assert(module.deployment_version==1 and type(module.deploy)=='function','deployment-v1 capability required')
    end
    local state=M.copy(bounded(module.create,M.copy(context)))
    local deployed=false
    return {deploy=function(context)
        assert(not deployed and type(module.deploy)=='function','deployment unavailable or already consumed')
        if placement_v2 then assert(context.version==2 and context.contract=='deployment-placement-v2','deployment context version mismatch')
        else assert(module.deployment_version==1,'deployment-v1 capability required')end
        deployed=true
        local plan=M.copy(bounded(module.deploy,state,M.copy(context)))
        command_array(plan,capacity.units_per_side)
        assert(#plan==capacity.units_per_side,'complete deployment required')
        return plan
    end,step=function(observation)
        assert(capacity.id~='empire-15-v1' or deployed,'deployment required before step')
        local result=bounded(module.step,state,M.copy(observation))
        local commands=M.copy(result)
        command_array(commands,capacity.max_command_proposals_per_step)
        return commands
    end}
end
function M.validate(commands,own,enemies,profile_name,contract)
    assert(contract==nil or contract=='formation-move-v1'or contract=='deployment-march-formation-v1','unknown movement contract')
    command_array(commands,M.profile(profile_name).max_command_proposals_per_step)
    local motion,guard={},{}
    for _,c in ipairs(commands) do
        assert(type(c)=='table','command object required')
        local allowed={unit_id=true,action=true}
        local keys=c.action=='move'and{'x','z','run'}or(c.action=='attack'and{'target_id','mode'}or(c.action=='guard'and{'enabled'}or{}))
        for _,key in ipairs(keys)do allowed[key]=true end
        if c.action=='move'and contract then allowed.facing_deg=true;allowed.width_m=true end
        for key in pairs(c)do assert(allowed[key],'unknown command field')end
        local u=own[c.unit_id]
        assert(u and u.alive==true and u.position,'unit ownership/alive required')
        if c.action=='guard' then
            assert(type(c.enabled)=='boolean' and not guard[c.unit_id],'invalid guard')
            guard[c.unit_id]=true
        else
            assert(not motion[c.unit_id],'duplicate motion'); motion[c.unit_id]=true
            if c.action=='move' then
                assert(finite(c.x) and finite(c.z) and (c.run==nil or type(c.run)=='boolean'),'invalid move')
                if contract then
                    assert(finite(c.facing_deg)and c.facing_deg>=0 and c.facing_deg<360,'invalid move facing')
                    local lord=u.kind=='lord'
                    assert(finite(c.width_m)and c.width_m>=(lord and 3 or 20)and c.width_m<=(lord and 8 or 40),'invalid move width')
                end
            elseif c.action=='attack' then
                local target=enemies[c.target_id]
                assert(target and target.visibility=='visible' and target.position,'target not visible')
                assert(c.mode=='melee' or (c.mode=='ranged' and u.kind=='archers' and finite(u.ammo) and u.ammo>0),'invalid attack mode')
            else assert(c.action=='halt','unsupported action') end
        end
    end
    return commands
end
return M
