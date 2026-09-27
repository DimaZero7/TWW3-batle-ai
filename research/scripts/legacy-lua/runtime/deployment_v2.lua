-- Explicit generation-2 placement contract. No footprint is inferred from width.
local M={version=2,contract='deployment-placement-v2'}
local function finite(n)return type(n)=='number'and n==n and n-n==0 end
local function phase(get)assert(get()=='Deployment','deployment phase closed')end
local function distance(a,b)return math.sqrt((a.x-b.x)^2+(a.z-b.z)^2)end
local function contract(c)assert(c.version==2 and c.contract==M.contract,'deployment-placement-v2 context required')end
local function add(list,code,detail)
 local row={code=code};if detail then for k,v in pairs(detail)do row[k]=v end end
 list[#list+1]=row
end
function M.zone(z)
 assert(type(z)=='table'and z.source=='controlled_scenario_xml','trusted scenario zone required')
 for _,p in ipairs({z.center,z.axis_u,z.axis_v})do assert(p and finite(p.x)and finite(p.z),'invalid zone vector')end
 assert(z.center and z.axis_u and z.axis_v,'missing zone vector')
 assert(finite(z.half_u)and finite(z.half_v)and z.half_u>0 and z.half_v>0,'invalid zone dimensions')
 local u,v=z.axis_u,z.axis_v
 assert(math.abs(u.x*u.x+u.z*u.z-1)<.00001 and math.abs(v.x*v.x+v.z*v.z-1)<.00001 and math.abs(u.x*v.x+u.z*v.z)<.00001,'zone axes not orthonormal')
 return z
end
function M.inside(p,z,tolerance)
 local dx,dz=p.x-z.center.x,p.z-z.center.z
 return math.abs(dx*z.axis_u.x+dz*z.axis_u.z)<=z.half_u+(tolerance or 0)and
        math.abs(dx*z.axis_v.x+dz*z.axis_v.z)<=z.half_v+(tolerance or 0)
end
function M.context(roster,zone,copy,selected)
 assert(selected==M.contract,'explicit deployment-placement-v2 selection required');M.zone(zone)
 return {version=2,contract=M.contract,own_roster=copy(roster),zone=copy(zone),limits={
  verification_profile='ordered-anchor-stable-contact-1m-v1',
  infantry={min_width_m=20,max_width_m=40},lord={min_width_m=3,max_width_m=8,width_semantics='single_entity_no_formation_span'},
  width_bounds_kind='engineering_input_bounds_not_measured_game_limits',
  anchor_native_position_semantics='ordered_anchor_exact_actual_reference_stable',reference_boundary_tolerance_m=.01,
  ordered_anchor_tolerance_m=.1,native_position_stability_tolerance_m=.25,
  native_contact_tolerance_m=1,required_stable_samples=2,settle_sample_interval_ms=1000,settle_timeout_ms=10000,
  geometry_scope='ordered_anchor_actual_reference_stability_and_native_pair_distance',full_entity_bounds_verified=false}}
end
function M.validate(plan,c)
 contract(c);local z=M.zone(c.zone);local own={}
 for _,u in ipairs(c.own_roster)do assert(u.id and not own[u.id],'duplicate roster identity');own[u.id]=u end
 assert(type(plan)=='table'and #plan==#c.own_roster,'complete deployment required')
 for k in pairs(plan)do assert(type(k)=='number'and k%1==0 and k>=1 and k<=#plan,'deployment must be array')end
 local seen={};local allowed={unit_id=true,x=true,z=true,facing_deg=true,width_m=true}
 for _,p in ipairs(plan)do
  assert(type(p)=='table','placement object required');for k in pairs(p)do assert(allowed[k],'unknown placement field')end
  local u=own[p.unit_id];assert(u and not seen[p.unit_id],'foreign or duplicate deployment unit');seen[p.unit_id]=true
  for _,k in ipairs({'x','z','facing_deg','width_m'})do assert(finite(p[k]),'nonfinite deployment field')end
  assert(p.facing_deg>=0 and p.facing_deg<360,'invalid facing')
  local lord=u.kind=='lord';assert(p.width_m>=(lord and 3 or 20)and p.width_m<=(lord and 8 or 40),'width outside engineering input bounds')
  assert(M.inside(p,z,0),'requested reference outside own deployment zone')
 end
 return plan
end
function M.prepare(policies,contexts,get_phase,copy)
 phase(get_phase);local plans={}
 for side=1,2 do phase(get_phase);plans[side]=policies[side].deploy(copy(contexts[side]))end
 for side=1,2 do M.validate(plans[side],contexts[side])end
 return plans
end
function M.apply(plans,get_phase,place)
 phase(get_phase)
 for side=1,2 do for _,p in ipairs(plans[side])do phase(get_phase);place(side,p)end end
end
-- Inspect one native sample without changing the submitted plan. Invalid or
-- unsettled samples are retried by the engine until the bounded deadline.
function M.verify(plan,c,measure,pair_distance,get_phase,previous)
 M.validate(plan,c);phase(get_phase)
 local limits=c.limits
 local report={units={},pairs={},errors={},valid=true,stable=previous~=nil,
  geometry_scope=limits.geometry_scope,full_entity_bounds_verified=false}
 local function invalid(target,code,detail)
  add(target,code,detail);add(report.errors,code,detail);report.valid=false
 end
 for i,p in ipairs(plan)do
  phase(get_phase)
  local unit={unit_id=p.unit_id,requested=p,errors={}}
  local ok,m=pcall(measure,p.unit_id)
  if not ok or type(m)~='table'then
   invalid(unit.errors,'native_measurement_unavailable',{unit_id=p.unit_id})
   report.stable=false
  else
   unit.actual=m
   local pos_ok=m.position and finite(m.position.x)and finite(m.position.z)
   local ordered_ok=m.ordered_position and finite(m.ordered_position.x)and finite(m.ordered_position.z)
   if not pos_ok then invalid(unit.errors,'native_position_unavailable',{unit_id=p.unit_id})end
   if not ordered_ok then invalid(unit.errors,'native_ordered_position_unavailable',{unit_id=p.unit_id})end
   if pos_ok then
    unit.requested_native_reference_delta_m=distance(p,m.position)
    if not M.inside(m.position,c.zone,limits.reference_boundary_tolerance_m)then
     invalid(unit.errors,'native_reference_outside_zone',{unit_id=p.unit_id})
    end
   end
   if ordered_ok then
    unit.requested_ordered_anchor_delta_m=distance(p,m.ordered_position)
    if unit.requested_ordered_anchor_delta_m>limits.ordered_anchor_tolerance_m then
     invalid(unit.errors,'ordered_anchor_mismatch',{unit_id=p.unit_id,delta_m=unit.requested_ordered_anchor_delta_m,limit_m=limits.ordered_anchor_tolerance_m})
    end
   end
   if not(finite(m.bearing_deg)and finite(m.ordered_bearing_deg)and finite(m.ordered_width_m))then
    invalid(unit.errors,'native_orientation_or_width_unavailable',{unit_id=p.unit_id})
   else
    unit.requested_native_bearing_delta_deg=math.abs((m.bearing_deg-p.facing_deg+180)%360-180)
    unit.requested_ordered_width_delta_m=math.abs(m.ordered_width_m-p.width_m)
   end
   if previous then
    local old=previous[p.unit_id]
    if pos_ok and old and old.position and finite(old.position.x)and finite(old.position.z)then
     unit.native_position_delta_since_previous_m=distance(m.position,old.position)
     unit.stable=unit.native_position_delta_since_previous_m<=limits.native_position_stability_tolerance_m
    else unit.stable=false end
    if not unit.stable then
     report.stable=false
     add(unit.errors,'native_position_unstable',{unit_id=p.unit_id,delta_m=unit.native_position_delta_since_previous_m,limit_m=limits.native_position_stability_tolerance_m})
    end
   else
    unit.stable=false;add(unit.errors,'previous_sample_required',{unit_id=p.unit_id})
   end
  end
  for j=1,i-1 do
   phase(get_phase);local other=plan[j].unit_id
   local pair={unit_a=p.unit_id,unit_b=other,errors={}}
   local good,d=pcall(pair_distance,p.unit_id,other)
   if good and finite(d)then
    pair.distance_m=d
    report.min_pair_distance=report.min_pair_distance and math.min(report.min_pair_distance,d)or d
    if d<=limits.native_contact_tolerance_m then
     invalid(pair.errors,'native_contact_unresolved',{unit_a=p.unit_id,unit_b=other,distance_m=d,limit_m=limits.native_contact_tolerance_m})
    end
   else invalid(pair.errors,'native_pair_distance_unavailable',{unit_a=p.unit_id,unit_b=other})end
   report.pairs[#report.pairs+1]=pair
  end
  report.units[#report.units+1]=unit
 end
 return report
end
return M
