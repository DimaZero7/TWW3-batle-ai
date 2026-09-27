-- Trusted, own-only diagnostic readouts. Never feeds observations or chooses orders.
local M={}
local function finite(v)return type(v)=='number'and v==v and v-v==0 end
local function unavailable(reason)return {status='unavailable',reason=reason}end
local function read(object,name,kind)
 local ok,value=pcall(function()return object[name](object)end)
 if not ok then return unavailable('call_failed')end
 if type(value)~=kind then return unavailable('invalid_type')end
 if kind=='number'and not finite(value)then return unavailable('nonfinite')end
 return {status='ok',value=value}
end
function M.vector(p)
 local ok,value=pcall(function()return {x=p:get_x(),y=p:get_y(),z=p:get_z()}end)
 if not ok then return unavailable('call_failed')end
 if not(finite(value.x)and finite(value.y)and finite(value.z))then return unavailable('invalid_vector')end
 return {status='ok',value=value}
end
function M.result(value)
 local r={status='ok',return_type=type(value)}
 if type(value)=='boolean'or finite(value)then r.value=value
 elseif value~=nil then r.status='unsupported';r.reason='non_scalar_or_nonfinite'end
 return r
end
local function position(u,name)
 local ok,p=pcall(function()return u[name](u)end)
 if not ok then return unavailable('call_failed')end
 return M.vector(p)
end
function M.state(bm,u)
 return {phase=read(bm,'get_current_phase_name','string'),position=position(u,'position'),
  ordered_position=position(u,'ordered_position'),bearing=read(u,'bearing','number'),
  ordered_bearing=read(u,'ordered_bearing','number'),ordered_width=read(u,'ordered_width','number'),
  moving=read(u,'is_moving','boolean'),script_controlled=read(u,'is_script_controlled','boolean'),
  controllable=read(u,'is_controllable','boolean'),deploying=read(u,'is_deploying','boolean'),
  deployed=read(u,'is_deployed','boolean')}
end
function M.gate(bm,u,p,value)
 return {target=M.vector(p),native_return=M.result(value),state=M.state(bm,u)}
end
local function query(u,p)
 local input=M.vector(p)
 if input.status~='ok'then return {input=input,result=unavailable('input_unavailable')}end
 local ok,value=pcall(function()return u:can_reach_position(p)end)
 return {input=input,result=ok and M.result(value)or unavailable('call_failed')}
end
function M.origins(bm,u,make_vector)
 local ok,p=pcall(function()return u:position()end)
 local input=ok and M.vector(p)or unavailable('call_failed')
 if input.status~='ok'then
  local r={input=input,result=unavailable('input_unavailable')}
  return {native_origin=r,terrain_origin=r}
 end
 -- Use the identical native origin captured above, not a later position sample.
 local native=query(u,p)
 local built,terrain=pcall(make_vector,input.value.x,input.value.z)
 return {native_origin=native,terrain_origin=built and query(u,terrain)or
  {input=unavailable('call_failed'),result=unavailable('input_unavailable')}}
end
return M
