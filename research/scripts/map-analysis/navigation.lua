local function navigation_scan(done)
    assert(bm:get_current_phase_name()=='Deployed','Navigation requires deployed phase')
    local units={bm:alliances():item(1):armies():item(1):units():item(1),bm:alliances():item(2):armies():item(1):units():item(1)}
    local f=assert(io.open(prefix..'grid.csv','r'));local header=f:read('*l');local rows={}
    for line in f:lines() do rows[#rows+1]=line end;f:close()
    f=assert(io.open(prefix..'grid.csv','w'));f:write(header..',reach_cavalry,reach_infantry\n');f:close()
    local next_i=1;local start=os.clock();local yes1,yes2=0,0
    local function batch()
        local ok,err=pcall(function()
            local output={}
            for i=next_i,math.min(next_i+1023,#rows) do
                local ix,iz,x,z,h,clear,ground,inside=rows[i]:match('^(%d+),(%d+),([^,]+),([^,]+),([^,]+),(%d+),"([^"]+)",(%d+)')
                assert(ix,'Malformed grid')
                local a,b=-1,-1
                if inside=='1' then
                    local v=battle_vector:new();v:set_x(tonumber(x));v:set_y(tonumber(h));v:set_z(tonumber(z))
                    a=units[1]:can_reach_position(v) and 1 or 0;b=units[2]:can_reach_position(v) and 1 or 0
                    yes1=yes1+a;yes2=yes2+b
                end
                output[#output+1]=rows[i]..','..a..','..b..'\n'
            end
            local file=assert(io.open(prefix..'grid.csv','a'));file:write(table.concat(output));file:close()
            next_i=math.min(next_i+1024,#rows+1)
            if next_i>#rows then
                emit('navigation_done',{cells=#rows,reachable_cavalry=yes1,reachable_infantry=yes2,clock_elapsed_s=os.clock()-start,phase=bm:get_current_phase_name(),speed=bm:current_battle_speed()})
                done()
            else bm:real_callback(batch,20,prefix..'nav_'..next_i) end
        end)
        if not ok then emit('probe_error',{message=tostring(err),stage='navigation'}) end
    end
    batch()
end
