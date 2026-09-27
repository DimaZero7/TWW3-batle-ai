local function cross_fords(done)
    local cases=FORD_CASES
    local units,controllers={},{}
    for side=1,2 do
        local army=bm:alliances():item(side):armies():item(1);units[side]=army:units():item(1)
        local uc=army:create_unit_controller();uc:add_units(units[side]);uc:take_control();uc:fire_at_will(false)
        if units[side]:can_use_behaviour('skirmish') then uc:change_behaviour_active('skirmish',false) end
        uc:halt();controllers[side]=uc
    end
    local function vector(x,z)
        local v=battle_vector:new();v:set_x(x);v:set_y(bm:get_terrain_height(x,z));v:set_z(z);return v
    end
    local current=0;local side=1;local start_ms;local target;local wet_seen=false;local active=false
    local function next_trial()
        current=current+1
        if current>#cases then
            if side==1 then side=2;current=1 else
                active=false;bm:remove_process(prefix..'ford_tick');bm:modify_battle_speed(0);done();return
            end
        end
        local c=cases[current];local u=units[side];local uc=controllers[side]
        -- Keep the other side away from the crossing; no combat is intended.
        controllers[3-side]:halt();controllers[3-side]:teleport_to_location(vector(-200,200),90,30)
        local from=vector(c.sx,c.sz);target=vector(c.tx,c.tz)
        uc:halt();uc:teleport_to_location(from,c.bearing,30)
        wet_seen=false;start_ms=bm:time_elapsed_ms();active=true
        emit('ford_trial_start',{ford=c.id,side=side,unit_type=u:type(),start_x=c.sx,start_z=c.sz,target_x=c.tx,target_z=c.tz,bearing=c.bearing,can_reach=u:can_reach_position(target)})
        uc:goto_location(target,true)
    end
    local function tick()
        if not active then return end
        local ok,err=pcall(function()
            local u=units[side];local p=u:position();local x,z=p:get_x(),p:get_z();local v=vector(x,z);local ground=bm:ground_type(v)
            local c=cases[current];local dx,dz=x-c.tx,z-c.tz;local dist=math.sqrt(dx*dx+dz*dz)
            if ground=='shallow_water' then wet_seen=true end
            emit('ford_position',{ford=c.id,side=side,x=x,y=p:get_y(),z=z,ground=ground,distance_to_target=dist,elapsed_ms=bm:time_elapsed_ms()-start_ms,men=u:number_of_men_alive(),moving=u:is_moving()})
            if dist<8 or bm:time_elapsed_ms()-start_ms>90000 then
                emit('ford_trial_done',{ford=c.id,side=side,reached=dist<8,shallow_water_seen=wet_seen,elapsed_ms=bm:time_elapsed_ms()-start_ms,distance=dist,men=u:number_of_men_alive()})
                controllers[side]:halt();active=false;bm:callback(next_trial,500,prefix..'next_ford')
            end
        end)
        if not ok then active=false;emit('probe_error',{stage='ford_movement',message=tostring(err)}) end
    end
    bm:modify_battle_speed(20)
    bm:repeat_callback(tick,250,prefix..'ford_tick')
    next_trial()
end
