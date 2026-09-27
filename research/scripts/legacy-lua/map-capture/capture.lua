-- reader and config are bundled by build.py. This runner is deliberately separate from the AI harness.
if not bm or _G.tww3_bai_map_capture then return end
_G.tww3_bai_map_capture=true
local prefix='tww3_bai_map_capture_'
local function quote(v)
    return '"'..tostring(v):gsub('[%z\1-\31\\"]',function(c)
        local e={['"']='\\"',['\\']='\\\\',['\n']='\\n',['\r']='\\r',['\t']='\\t'}
        return e[c] or string.format('\\u%04x',string.byte(c))
    end)..'"'
end
local function emit(event,row)
    row=row or {};row.event=event;row.wall=os.date('!%Y-%m-%dT%H:%M:%SZ')
    local keys,parts={},{}
    for k in pairs(row) do keys[#keys+1]=k end
    table.sort(keys)
    for _,k in ipairs(keys) do
        local v=row[k];v=(type(v)=='number' or type(v)=='boolean') and tostring(v) or quote(v)
        parts[#parts+1]=quote(k)..':'..v
    end
    local f=assert(io.open(prefix..'events.jsonl','a'));f:write('{'..table.concat(parts,',')..'}\n');f:close()
end
local stopped=false
local function guarded(fn)
    return function()
        if stopped then return end
        local ok,e=xpcall(fn,function(err)return debug.traceback(tostring(err),2)end)
        if not ok then stopped=true;emit('probe_error',{message=e}) end
    end
end
emit('script_loaded')
local function initialize()
    assert(bm:get_current_phase_name()=='Deployment','Expected static Deployment phase')
    bm:modify_battle_speed(0)
    bm:output_battle_xml(prefix..'ready.xml')
    local frame=reader.read_frame(common)
    emit('frame',{min_x=frame.min_x,max_x=frame.max_x,min_z=frame.min_z,max_z=frame.max_z,
        width=frame.width,depth=frame.depth,kind=frame.kind,movement_boundary_verified=false})
    for i,c in ipairs(frame.corners) do emit('corner',{id=i,x=c.x,z=c.z,u=c.u,v=c.v}) end
    local grid=reader.new_grid(frame,config.step)
    emit('grid_begin',grid)
    local filename=prefix..'grid.csv'
    local f=assert(io.open(filename,'w'));f:write('ix,iz,x,z,height,clear,ground,inside_radar\n');f:close()
    local next_id,batches,blocked=0,0,0
    local start=type(os.clock)=='function' and os.clock() or nil
    local function batch()
        local cells,following,done=reader.read_batch(bm,battle_vector,grid,next_id,4096)
        local lines={}
        for _,c in ipairs(cells) do
            if not c.clear then blocked=blocked+1 end
            local ground='"'..c.ground:gsub('"','""')..'"'
            lines[#lines+1]=string.format('%d,%d,%.9f,%.9f,%.6f,%d,%s,%d\n',
                c.ix,c.iz,c.x,c.z,c.height,c.clear and 1 or 0,ground,c.inside_radar and 1 or 0)
        end
        local file=assert(io.open(filename,'a'));file:write(table.concat(lines));file:close()
        next_id=following;batches=batches+1
        if done then
            emit('grid_done',{cells=next_id,blocked=blocked,batches=batches,clock_elapsed_s=start and os.clock()-start or 'unavailable'})
            local function complete()
                emit('probe_done',{phase=bm:get_current_phase_name(),speed=bm:current_battle_speed()})
            end
            if config.features then features.run(bm,common,battle_vector,prefix,emit,complete) else complete() end
        else
            bm:real_callback(guarded(batch),20,prefix..'batch_'..batches)
        end
    end
    batch()
end
bm:real_callback(guarded(initialize),3000,prefix..'init')
