-- Trusted-side observation helper, not a Lua sandbox or a complete policy API.
local M = {}
local function finite(n)return type(n)=='number' and n==n and n-n==0 end
local function copy_seen(s)
 if not s then return nil end
 return {x=s.x,y=s.y,z=s.z,seen_ms=s.seen_ms}
end
function M.query(unit, observer_alliance)
 local ok,value=pcall(function()return unit:is_visible_to_alliance(observer_alliance)end)
 if not ok or type(value)~='boolean' then return nil end
 return value
end
-- memory is private to ONE observer and ONE battle; reset it before reuse.
-- public_id comes from a permitted roster mapping, never an enemy Lua handle.
function M.observe(unit, observer_alliance, public_id, now_ms, memory)
 assert(type(public_id)=='string' and public_id~='', 'public_id required')
 assert(finite(now_ms) and now_ms>=0, 'valid time required')
 assert(type(memory)=='table', 'observer memory required')
 local visible=M.query(unit,observer_alliance)
 local out={id=public_id,visibility=visible==nil and 'unknown' or (visible and 'visible' or 'not_visible'),sampled_ms=now_ms}
 if visible then
  local ok,p=pcall(function()
   local pos=unit:position();local x,y,z=pos:get_x(),pos:get_y(),pos:get_z()
   assert(finite(x) and finite(y) and finite(z),'invalid position')
   return {x=x,y=y,z=z,seen_ms=now_ms}
  end)
  if ok then
   memory[public_id]=copy_seen(p);out.current=copy_seen(p)
  else
   out.position_status='unavailable'
  end
 end
 out.last_seen=copy_seen(memory[public_id])
 return out
end
return M
