-- Additional read-only checks, called after the normal 3 m surface capture.
local function extra_probe()
    local function safe(row,key,fn)
        local ok,a,b,c,d=pcall(fn)
        if ok then
            row[key]=a==nil and 'nil' or a
            if b~=nil then row[key..'_2']=b end
            if c~=nil then row[key..'_3']=c end
            if d~=nil then row[key..'_4']=d end
        else row[key..'_error']=tostring(a) end
    end
    local info={}
    safe(info,'version',function()return common.game_version()end)
    for _,key in ipairs({'BuildingsList.Size','IsSiege','BattleTypeState','CapturePointList.Size'}) do
        safe(info,key,function()return common.get_context_value('BattleRoot.'..key)end)
    end
    emit('context_summary',info)
    local context_variants={}
    safe(context_variants,'root_function',function()return common.get_context_value('BattleRoot.BuildingsList().Size')end)
    safe(context_variants,'three_arguments',function()return common.get_context_value('CcoBattleRoot','','BuildingsList.Size')end)
    safe(context_variants,'three_arguments_function',function()return common.get_context_value('CcoBattleRoot','','BuildingsList().Size')end)
    emit('context_variants',context_variants)
    local buildings=bm:buildings()
    emit('buildings_begin',{count=buildings:count()})
    local offset,complete=0,false
    while not complete do
        local rows,next_offset,is_done=objects.read_buildings(bm,offset,256)
        for _,row in ipairs(rows) do emit('objects_module_record',row) end
        offset=next_offset;complete=is_done
    end
    for i=1,buildings:count() do
        local b=buildings:item(i);local row={index=i}
        for _,k in ipairs({'name','category','orientation','health','has_gate','is_fort_wall','is_fort_tower','alliance_owner_id','is_selectable'}) do
            safe(row,k,function()return b[k](b)end)
        end
        safe(row,'x',function()return b:position():get_x()end)
        safe(row,'y',function()return b:position():get_y()end)
        safe(row,'z',function()return b:position():get_z()end)
        safe(row,'center_x',function()return b:central_position():get_x()end)
        safe(row,'center_y',function()return b:central_position():get_y()end)
        safe(row,'center_z',function()return b:central_position():get_z()end)
        emit('building',row)
    end
    local context_rows=objects.read_structure_contexts(common,0,10000)
    for _,row in ipairs(context_rows) do emit('structure_context_module_record',row) end
    local n=info['BuildingsList.Size']
    if type(n)=='number' and n>=0 and n<50000 then
        for i=0,n-1 do
            local row={index=i};local base='BattleRoot.BuildingsList.At('..i..').'
            for _,k in ipairs({'Name','CategoryType','Position','IsBridge','IsDestroyed','IsDestructable','HasMissileWeapon','CanUpdateAbilities','LocalEffectText','GlobalEffectText','SpecialAbilitiesList.Size'}) do
                safe(row,k,function()return common.get_context_value(base..k)end)
            end
            emit('cco_building',row)
        end
    end
    local units={bm:alliances():item(1):armies():item(1):units():item(1),bm:alliances():item(2):armies():item(1):units():item(1)}
    for i,u in ipairs(units) do
        local row={index=i,name=u:name(),unit_type=u:type(),x=u:position():get_x(),y=u:position():get_y(),z=u:position():get_z(),phase=bm:get_current_phase_name()}
        safe(row,'can_reach_self',function()return u:can_reach_position(u:position())end)
        emit('unit_origin',row)
    end
    local selected={};local per_kind={};local f=assert(io.open(prefix..'grid.csv','r'));f:read('*l')
    local index=0
    for line in f:lines() do
        local ix,iz,x,z,h,clear,ground,inside=line:match('^(%d+),(%d+),([^,]+),([^,]+),([^,]+),(%d+),"([^"]+)",(%d+)')
        assert(ix,'CSV parse failed')
        local key=ground..'_'..clear;per_kind[key]=per_kind[key] or 0
        if inside=='1' and (per_kind[key]<8 or (index%997==0)) then
            per_kind[key]=per_kind[key]+1
            selected[#selected+1]={x=tonumber(x),z=tonumber(z),height=tonumber(h),clear=clear,ground=ground}
        end
        index=index+1
    end
    f:close()
    for i,p in ipairs(selected) do
        local v=battle_vector:new();v:set_x(p.x);v:set_y(p.height);v:set_z(p.z)
        local row={index=i,x=p.x,z=p.z,height=p.height,clear_3m=p.clear,ground=p.ground}
        safe(row,'clear_1m',function()return bm:is_area_clear(v,0,1,1,false)end)
        safe(row,'clear_10m',function()return bm:is_area_clear(v,0,10,10,false)end)
        safe(row,'reach_cavalry',function()return units[1]:can_reach_position(v)end)
        safe(row,'reach_infantry',function()return units[2]:can_reach_position(v)end)
        if bm:get_current_phase_name()=='Deployed' then
            local c={ix=i,iz=0,x=p.x,z=p.z,height=p.height,inside_radar=true}
            local result=reachability.read_cells(bm,battle_vector,units,{c})[1]
            assert(result.reachable[1]==row.reach_cavalry and result.reachable[2]==row.reach_infantry,'Reachability module mismatch')
            row.module_match=true
        end
        emit('reach_sample',row)
    end
    emit('extra_done',{sample_count=#selected})
end
