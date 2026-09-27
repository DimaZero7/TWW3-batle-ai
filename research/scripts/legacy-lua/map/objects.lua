-- Read-only battle object inventory. Positions are origins/centres, not footprints.
-- Measured scope: docs/map/en/objects.md.
local M={}
function M.read_buildings(manager,start,limit)
    assert(type(start)=='number' and start>=0 and start==math.floor(start),'Invalid start')
    assert(type(limit)=='number' and limit>0 and limit==math.floor(limit),'Invalid limit')
    local all=manager:buildings();local count=all:count()
    assert(start<=count,'Start outside building list')
    local next_index=math.min(start+limit,count);local rows={}
    for index=start+1,next_index do
        local b=all:item(index);local p=b:position();local c=b:central_position()
        rows[#rows+1]={index=index,name=b:name(),category=b:category(),
            x=p:get_x(),y=p:get_y(),z=p:get_z(),center_x=c:get_x(),center_y=c:get_y(),center_z=c:get_z(),
            orientation=b:orientation(),health=b:health(),alliance_owner_id=b:alliance_owner_id(),
            has_gate=b:has_gate(),is_fort_wall=b:is_fort_wall(),is_fort_tower=b:is_fort_tower(),is_selectable=b:is_selectable()}
    end
    return rows,next_index,next_index==count,count
end
-- CCO exposes a separate, smaller list. Do not equate its indices with native indices.
-- Names/effect descriptions are localized UI text, not stable identifiers/formulas.
function M.read_structure_contexts(common,start,limit)
    assert(type(start)=='number' and start>=0 and start==math.floor(start),'Invalid start')
    assert(type(limit)=='number' and limit>0 and limit==math.floor(limit),'Invalid limit')
    local count=common.get_context_value('BattleRoot.BuildingsList.Size')
    assert(type(count)=='number' and count>=0 and count==math.floor(count),'Invalid CCO count')
    assert(start<=count,'Start outside CCO building list')
    local next_index=math.min(start+limit,count);local rows={}
    for index=start,next_index-1 do
        local base='BattleRoot.BuildingsList.At('..index..').'
        local row={index=index}
        for _,key in ipairs({'Name','CategoryType','IsBridge','IsDestroyed','IsDestructable',
            'HasMissileWeapon','CanUpdateAbilities','LocalEffectText','GlobalEffectText','SpecialAbilitiesList.Size'}) do
            local value=common.get_context_value(base..key)
            assert(value~=nil,'Missing CCO field: '..key);row[key]=value
        end
        row.x,row.y,row.z,row.w=common.get_context_value(base..'Position')
        assert(type(row.x)=='number' and type(row.y)=='number' and type(row.z)=='number','Missing CCO position')
        rows[#rows+1]=row
    end
    return rows,next_index,next_index==count,count
end
return M
