-- Trusted role reader. Native booleans, never alliance-index or spawn inference.
local M={version=1}
function M.read(bm,record)
 local alliances=bm:alliances()
 assert(alliances:count()==2,'native role requires exactly two alliances')
 local roles,raw={},{}
 for i=1,2 do
  local ok,value=pcall(function()return alliances:item(i):is_attacker()end)
  if not(ok and type(value)=='boolean')then
   if record then record({version=1,source='alliance:is_attacker()',failed_alliance=i,return_type=ok and type(value)or'error'})end
   error('native attacker role unavailable')
  end
  raw[i]=value;roles[i]={version=1,role=value and 'attacker'or'defender'}
 end
 local evidence={version=1,source='alliance:is_attacker()',is_attacker=raw}
 if record then record(evidence)end
 assert(raw[1]~=raw[2],'conflicting native attacker roles')
 return roles,evidence
end
function M.verify(roles,expected)
 if expected then
  assert(#expected==2,'two declared roles required')
  for i=1,2 do assert(expected[i]==roles[i].role,'declared/native battle role mismatch')end
 end
 return roles
end
return M
