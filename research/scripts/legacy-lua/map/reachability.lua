-- Read-only point reachability for explicit units, after deployment.
-- Positive points are not a proof that all interconnecting straight edges are clear.
local M={}
function M.read_cells(manager,vector_type,units,cells)
    assert(manager:get_current_phase_name()=='Deployed','Reachability capture requires Deployed phase')
    assert(#units>0,'At least one unit is required')
    local rows={}
    for _,cell in ipairs(cells) do
        local row={ix=cell.ix,iz=cell.iz,inside_radar=cell.inside_radar,reachable={}}
        if cell.inside_radar then
            local p=vector_type:new();p:set_x(cell.x);p:set_y(cell.height);p:set_z(cell.z)
            for i,unit in ipairs(units) do
                local value=unit:can_reach_position(p)
                assert(type(value)=='boolean','Reachability did not return a boolean')
                row.reachable[i]=value
            end
        end
        rows[#rows+1]=row
    end
    return rows
end
return M
