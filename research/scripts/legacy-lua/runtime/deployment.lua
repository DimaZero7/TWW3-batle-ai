-- Trusted deployment-v1 transaction. No engine handle is passed to policies.
-- Reservations are restrictive engineering limits, not universal unit geometry.
local M={version=1}
local function finite(n)return type(n)=='number' and n==n and n-n==0 end
local function phase(get_phase)assert(get_phase()=='Deployment','deployment phase closed')end
local function distance(a,b)return math.sqrt((a.x-b.x)^2+(a.z-b.z)^2)end
local function radius(kind)return kind=='lord' and 8 or 35 end
local function geometry(context)
 if context.version==1 then assert(context.contract==nil,'unexpected deployment contract');return false end
 assert(context.version==2 and (context.contract=='deployment-geometry-v1'or context.contract=='deployment-march-v1'or context.contract=='deployment-march-formation-v1'),'unknown deployment contract');return true
end
function M.zone(zone)
 assert(type(zone)=='table' and zone.source=='controlled_scenario_xml','trusted scenario zone required')
 for _,p in ipairs({zone.center,zone.axis_u,zone.axis_v})do assert(p and finite(p.x) and finite(p.z),'invalid zone vector')end
 assert(finite(zone.half_u) and finite(zone.half_v) and zone.half_u>0 and zone.half_v>0,'invalid zone dimensions')
 local u,v=zone.axis_u,zone.axis_v
 assert(math.abs(u.x*u.x+u.z*u.z-1)<.00001 and math.abs(v.x*v.x+v.z*v.z-1)<.00001 and math.abs(u.x*v.x+u.z*v.z)<.00001,'zone axes not orthonormal')
 return zone
end
function M.inside(p,zone,margin)
 local dx,dz=p.x-zone.center.x,p.z-zone.center.z
 return math.abs(dx*zone.axis_u.x+dz*zone.axis_u.z)+margin<=zone.half_u and
        math.abs(dx*zone.axis_v.x+dz*zone.axis_v.z)+margin<=zone.half_v
end
function M.context(roster,zone,copy,contract)
 M.zone(zone)
 assert(contract==nil or contract=='deployment-geometry-v1'or contract=='deployment-march-v1'or contract=='deployment-march-formation-v1','unknown deployment contract')
 local context={version=1,own_roster=copy(roster),zone=copy(zone),limits={
  infantry={min_width_m=20,max_width_m=40,reservation_radius_m=35},
  lord={min_width_m=3,max_width_m=8,reservation_radius_m=8,width_semantics='single_entity_no_formation_span'},min_gap_m=2,
  geometry_scope='native_reference_reserved_area_and_native_pair_boxes',full_entity_bounds_verified=false}}
 if contract then
  context.version=2;context.contract=contract;context.limits.pair_circle_check=false
  context.limits.native_pair_min_exclusive_m=1;context.limits.max_anchor_error_m={infantry=20,lord=8}
  context.limits.termination_phase='Deployment'
  if contract=='deployment-march-v1'or contract=='deployment-march-formation-v1'then context.limits.termination_phase='combat_window';context.limits.combat_window_ms=120000 end
 end
 return context
end
function M.validate(plan,context)
 local probe=geometry(context)
 local zone=M.zone(context.zone);local own={};for _,u in ipairs(context.own_roster)do own[u.id]=u end
 assert(type(plan)=='table' and #plan==#context.own_roster,'complete deployment required')
 for k in pairs(plan)do assert(type(k)=='number' and k%1==0 and k>=1 and k<=#plan,'deployment must be array')end
 local seen={};local allowed={unit_id=true,x=true,z=true,facing_deg=true,width_m=true}
 for i,p in ipairs(plan)do
  assert(type(p)=='table','placement object required');for k in pairs(p)do assert(allowed[k],'unknown placement field')end
  local unit=own[p.unit_id];assert(unit and not seen[p.unit_id],'foreign or duplicate deployment unit');seen[p.unit_id]=true
  for _,k in ipairs({'x','z','facing_deg','width_m'})do assert(finite(p[k]),'nonfinite deployment field')end
  assert(p.facing_deg>=0 and p.facing_deg<360,'invalid facing')
  local lord=unit.kind=='lord';assert(p.width_m>=(lord and 3 or 20) and p.width_m<=(lord and 8 or 40),'invalid width')
  assert(M.inside(p,zone,radius(unit.kind)),'reservation outside own deployment zone')
  if not probe then for j=1,i-1 do assert(distance(p,plan[j])>=radius(unit.kind)+radius(own[plan[j].unit_id].kind)+2,'deployment reservations overlap')end end
 end
 for id in pairs(own)do assert(seen[id],'missing deployment unit')end
 return plan
end
function M.prepare(policies,contexts,get_phase,copy)
 phase(get_phase);local plans={}
 -- No placement callback exists in this stage. Both plans precede any apply.
 for side=1,2 do phase(get_phase);plans[side]=policies[side].deploy(copy(contexts[side]))end
 for side=1,2 do M.validate(plans[side],contexts[side])end
 return plans
end
function M.apply(plans,get_phase,place)
 phase(get_phase)
 for side=1,2 do for _,p in ipairs(plans[side])do phase(get_phase);place(side,p)end end
end
function M.verify(plan,context,measure,pair_distance,get_phase)
 local probe=geometry(context)
 phase(get_phase);local report={units={},min_pair_distance=nil,geometry_scope='native_reference_reserved_area_and_native_pair_boxes',full_entity_bounds_verified=false};local own={}
 for _,u in ipairs(context.own_roster)do own[u.id]=u end
 for i,p in ipairs(plan)do
  phase(get_phase);local m=measure(p.unit_id);local r=radius(own[p.unit_id].kind)
  assert(m.position and finite(m.position.x) and finite(m.position.z),'actual position unavailable')
  assert(distance(p,m.position)<=(probe and math.min(r,20)or r),'actual centre outside allowed residual')
  assert(M.inside(m.position,context.zone,0),'native reference outside own zone')
  assert(finite(m.bearing_deg) and finite(m.ordered_width_m),'actual orientation/width unavailable')
  if probe then assert(finite(m.ordered_bearing_deg),'ordered bearing unavailable')end
  local angle=math.abs((m.bearing_deg-p.facing_deg+180)%360-180)
  -- facing_deg is the native teleport parameter; bearing() is the resulting
  -- main-squad orientation. Record their difference, do not assert an invented
  -- equality/tolerance between them after engine formation adjustment.
  -- Ordered width and entity span are distinct readouts, not required to equal
  -- the command parameter. CCO entity freshness is unverified for the other
  -- army in this XML setup; never transform/cache-refresh those coordinates.
  local low,high;local angle_rad=m.bearing_deg*math.pi/180
  local valid_points=0;local outside_reservation=0;local outside_zone=0
  for _,point in ipairs(m.points or {})do
   if finite(point.x) and finite(point.z)then
    valid_points=valid_points+1
    if not M.inside(point,context.zone,2)then outside_zone=outside_zone+1 end
    if distance(point,p)>r then outside_reservation=outside_reservation+1 end
    local lateral=(point.x-p.x)*math.cos(angle_rad)-(point.z-p.z)*math.sin(angle_rad)
    low=low and math.min(low,lateral)or lateral;high=high and math.max(high,lateral)or lateral
   end
  end
  local entity_span=high and high-low or nil
  for j=1,i-1 do
   local d=pair_distance(p.unit_id,plan[j].unit_id)
   assert(finite(d) and d>1,'actual bounding boxes overlap or touch')
   report.min_pair_distance=report.min_pair_distance and math.min(report.min_pair_distance,d) or d
  end
  report.units[#report.units+1]={unit_id=p.unit_id,requested=p,actual=m,position_error_m=distance(p,m.position),bearing_error_deg=angle,
   ordered_width_error_m=math.abs(m.ordered_width_m-p.width_m),entity_centre_width_m=entity_span,
   width_check='separate_readouts_not_equivalent',cco_geometry={freshness='unverified',valid_points=valid_points,outside_reserved_area=outside_reservation,outside_zone=outside_zone}}
 end
 return report
end
return M
