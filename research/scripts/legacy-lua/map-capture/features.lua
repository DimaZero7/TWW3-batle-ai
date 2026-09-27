-- Optional diagnostic capture. Unlike src/map modules, this runner owns battle control.
local M={}
function M.run(manager,common,vector_type,prefix,emit,done)
    local failed=false;local controllers={};local units={}
    local function guarded(fn)
        return function()
            if failed then return end
            local ok,err=xpcall(fn,function(e)return debug.traceback(tostring(e),2)end)
            if not ok then failed=true;emit('probe_error',{stage='features',message=err}) end
        end
    end
    for side=1,2 do
        local army=manager:alliances():item(side):armies():item(1)
        units[side]=army:units():item(1)
        local uc=army:create_unit_controller();uc:add_units(units[side]);uc:take_control();uc:fire_at_will(false);uc:halt()
        controllers[side]=uc
    end
    local function navigation()
        for side,u in ipairs(units) do
            local p=u:position()
            emit('navigation_unit',{side=side,unit_type=u:type(),x=p:get_x(),y=p:get_y(),z=p:get_z(),phase=manager:get_current_phase_name()})
        end
        local f=assert(io.open(prefix..'grid.csv','r'));local header=f:read('*l');local rows={}
        for line in f:lines() do rows[#rows+1]=line end;f:close()
        f=assert(io.open(prefix..'grid.csv','w'));f:write(header..',reach_side_1,reach_side_2\n');f:close()
        local next_i=1;local start=os.clock();local yes={0,0}
        local function batch()
            local cells={};local last=math.min(next_i+1023,#rows)
            for i=next_i,last do
                local ix,iz,x,z,h,clear,ground,inside=rows[i]:match('^(%d+),(%d+),([^,]+),([^,]+),([^,]+),(%d+),"([^"]+)",(%d+)$')
                assert(ix,'Malformed surface CSV')
                cells[#cells+1]={ix=tonumber(ix),iz=tonumber(iz),x=tonumber(x),z=tonumber(z),height=tonumber(h),inside_radar=inside=='1'}
            end
            local result=reachability.read_cells(manager,vector_type,units,cells);local lines={}
            for j,row in ipairs(result) do
                local values={-1,-1}
                if row.inside_radar then
                    for side=1,2 do values[side]=row.reachable[side] and 1 or 0;yes[side]=yes[side]+values[side] end
                end
                lines[#lines+1]=rows[next_i+j-1]..','..values[1]..','..values[2]..'\n'
            end
            local file=assert(io.open(prefix..'grid.csv','a'));file:write(table.concat(lines));file:close()
            next_i=last+1
            if next_i>#rows then
                emit('navigation_done',{cells=#rows,reachable_side_1=yes[1],reachable_side_2=yes[2],clock_elapsed_s=os.clock()-start,phase=manager:get_current_phase_name(),speed=manager:current_battle_speed()})
                done()
            else manager:real_callback(guarded(batch),20,prefix..'navigation_'..next_i) end
        end
        batch()
    end
    local function contexts()
        local next_index=0
        local function batch()
            local rows,following,complete,count=objects.read_structure_contexts(common,next_index,64)
            for _,row in ipairs(rows) do emit('structure_context',row) end
            next_index=following
            if complete then emit('structure_contexts_done',{count=count});navigation()
            else manager:real_callback(guarded(batch),20,prefix..'contexts_'..next_index) end
        end
        batch()
    end
    local function inventory()
        emit('feature_conditions',{version=common.game_version(),phase=manager:get_current_phase_name(),speed=manager:current_battle_speed(),is_siege=common.get_context_value('BattleRoot.IsSiege'),battle_type=common.get_context_value('BattleRoot.BattleTypeState')})
        local next_index=0
        local function batch()
            local rows,following,complete,count=objects.read_buildings(manager,next_index,256)
            for _,row in ipairs(rows) do emit('building',row) end
            next_index=following
            if complete then emit('buildings_done',{count=count});contexts()
            else manager:real_callback(guarded(batch),20,prefix..'objects_'..next_index) end
        end
        batch()
    end
    manager:register_phase_change_callback('Deployed',guarded(function()
        manager:real_callback(guarded(function()
            manager:modify_battle_speed(0)
            -- Speed changes apply on a later engine tick.
            manager:real_callback(guarded(inventory),250,prefix..'features_paused')
        end),250,prefix..'features_deployed')
    end))
    manager:modify_battle_speed(20);manager:end_current_battle_phase()
end
return M
