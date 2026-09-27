from pathlib import Path
import json,hashlib,collections
H=Path('tmp/map-research/field-features');R=Path('build/map-capture/run-20260925-234830');es=[json.loads(l) for l in (R/'tww3_bai_map_capture_events.jsonl').read_text().splitlines()]
cs=[e for e in es if e['event']=='structure_context_module_record'];ds=[e for e in es if e['event']=='cco_building'];assert len(cs)==len(ds)==1
for c,d in zip(cs,ds):
 for k,v in c.items():
  if k in ('event','wall'):continue
  other={'x':'Position','y':'Position_2','z':'Position_3','w':'Position_4'}.get(k,k)
  assert d[other]==v,(k,v,d.get(other))
native=[e for e in es if e['event']=='building'];module=[e for e in es if e['event']=='objects_module_record'];assert len(native)==len(module)==2696
for a,b in zip(native,module):assert {k:v for k,v in a.items() if k not in ('wall','event')}=={k:v for k,v in b.items() if k not in ('wall','event')}
ls=[e for e in es if e['event']=='bridge_layer_sample'];lookup={(e['ix'],e['iz'],e['level']):e for e in ls};assert len(lookup)==61*61*3
keys=['clear','ground','reach_cavalry','reach_infantry'];changes={k:0 for k in keys}
for i in range(-30,31):
 for j in range(-30,31):
  group=[lookup[i,j,l] for l in (1,2,3)]
  for k in keys:changes[k]+=int(len({r[k] for r in group})>1)
summary={'run':R.name,'bridge_native':next(e for e in native if e['category']=='bridge'),'bridge_cco':ds[0],'validation':{'native_module_matching_records':len(native),'cco_module_matching_records':len(cs),'objects_lua_sha256':hashlib.sha256(Path('src/map/objects.lua').read_bytes()).hexdigest()},'layer_queries':{'points':61*61,'layers':3,'values_changed_by_y':changes,'centre':[lookup[0,0,l] for l in (1,2,3)]},'movement_orders':[e for e in es if e['event']=='bridge_order'],'movement_results':[e for e in es if e['event']=='bridge_stage_done'],'crossing_verified':False,'caveats':['Native object enumeration and CCO IsBridge succeed; no traversal was demonstrated.','Changing query Y to object origin or central height changed none of the sampled properties.','Object origin/central height is not a proven deck surface height.','Four movement stages timed out; target point reachability did not prove a usable route from the teleported formation.']}
O=H/R.name;O.mkdir(exist_ok=True);(O/'bridge-validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Bridge native/CCO wrappers validated; layer differences',changes)
