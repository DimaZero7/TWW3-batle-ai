local function bridge_test(done)
    local units,controllers={},{}
    for side=1,2 do
        local army=bm:alliances():item(side):armies():item(1);units[side]=army:units():item(1)
        local uc=army:create_unit_controller();uc:add_units(units[side]);uc:take_control();uc:fire_at_will(false)
        if units[side]:can_use_behaviour('skirmish') then uc:change_behaviour_active('skirmish',false) end
        uc:halt();controllers[side]=uc
    end
    local bx,bz,by=-274.43798828125,-276.25244140625,150.67486572266
    local function vector(x,z,y)
        local v=battle_vector:new();v:set_x(x);v:set_y(y or bm:get_terrain_height(x,z));v:set_z(z);return v
    end
    local start=os.clock()
    for iz=-30,30 do
        for ix=-30,30 do
            local x,z=bx+3*ix,bz+3*iz;local terrain=bm:get_terrain_height(x,z)
            for level,y in ipairs({terrain,139.6011505127,by}) do
                local v=vector(x,z,y)
                emit('bridge_layer_sample',{ix=ix,iz=iz,x=x,z=z,y=y,terrain=terrain,level=level,ground=bm:ground_type(v),clear=bm:is_area_clear(v,0,3,3,false),reach_cavalry=units[1]:can_reach_position(v),reach_infantry=units[2]:can_reach_position(v)})
            end
        end
    end
    emit('bridge_layers_done',{samples=61*61*3,clock_elapsed_s=os.clock()-start})
    local side,stage,active=1,1,false;local start_ms;local target
    local function issue()
        target=stage==1 and vector(bx,bz,by) or vector(-150,-240)
        start_ms=bm:time_elapsed_ms();controllers[side]:goto_location(target,true);active=true
        emit('bridge_order',{side=side,stage=stage,x=target:get_x(),y=target:get_y(),z=target:get_z(),can_reach=units[side]:can_reach_position(target)})
    end
    local function begin()
        controllers[3-side]:halt();controllers[3-side]:teleport_to_location(vector(150,-50),90,30)
        controllers[side]:halt();controllers[side]:teleport_to_location(vector(-380,-307),74,15)
        stage=1;bm:callback(issue,500,prefix..'bridge_issue')
    end
    local function tick()
        if not active then return end
        local ok,err=pcall(function()
            local u=units[side];local p=u:position();local x,y,z=p:get_x(),p:get_y(),p:get_z()
            local terrain=bm:get_terrain_height(x,z);local dx,dz=x-target:get_x(),z-target:get_z();local distance=math.sqrt(dx*dx+dz*dz)
            emit('bridge_position',{side=side,stage=stage,x=x,y=y,z=z,terrain=terrain,ground_at_unit=bm:ground_type(p),ground_at_terrain=bm:ground_type(vector(x,z)),distance=distance,men=u:number_of_men_alive(),moving=u:is_moving(),elapsed_ms=bm:time_elapsed_ms()-start_ms})
            if distance<20 or bm:time_elapsed_ms()-start_ms>120000 then
                emit('bridge_stage_done',{side=side,stage=stage,reached=distance<20,distance=distance,elapsed_ms=bm:time_elapsed_ms()-start_ms})
                active=false;controllers[side]:halt()
                if stage==1 then stage=2;bm:callback(issue,500,prefix..'bridge_next_stage')
                elseif side==1 then side=2;bm:callback(begin,500,prefix..'bridge_next_side')
                else bm:remove_process(prefix..'bridge_tick');bm:modify_battle_speed(0);done() end
            end
        end)
        if not ok then active=false;emit('probe_error',{stage='bridge_movement',message=tostring(err)}) end
    end
    bm:modify_battle_speed(20);bm:repeat_callback(tick,250,prefix..'bridge_tick');begin()
end
