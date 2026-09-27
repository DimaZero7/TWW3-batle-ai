import csv,json,hashlib
from pathlib import Path
H=Path('tmp/map-research/field-features');R=Path('build/map-capture/run-20260925-235449');B=Path('build/map-capture/run-20260925-234152')
a=list(csv.DictReader((R/'tww3_bai_map_capture_grid.csv').open()));b=list(csv.DictReader((B/'tww3_bai_map_capture_grid.csv').open()));assert len(a)==len(b)==116964
for new,old in zip(a,b):
 for k,v in old.items():
  nk={'reach_cavalry':'reach_side_1','reach_infantry':'reach_side_2'}.get(k,k)
  assert new[nk]==v,(k,new[nk],v)
e=lambda p:[json.loads(l) for l in (p/'tww3_bai_map_capture_events.jsonl').read_text().splitlines()]
es,oldes=e(R),e(B)
newbs=[x for x in es if x['event']=='building'];oldbs=[x for x in oldes if x['event']=='building'];assert len(newbs)==len(oldbs)==2696
clean=lambda x:{k:v for k,v in x.items() if k not in ('wall','event')}
for x,y in zip(newbs,oldbs):assert clean(x)==clean(y)
assert next(x for x in es if x['event']=='structure_contexts_done')['count']==1
cc=next(x for x in es if x['event']=='structure_context');oc=next(x for x in oldes if x['event']=='cco_building')
for k,v in clean(cc).items():assert v==oc[{'x':'Position','y':'Position_2','z':'Position_3','w':'Position_4'}.get(k,k)]
result={'run':R.name,'baseline':B.name,'identical_grid_rows':len(a),'identical_native_objects':len(newbs),'identical_cco_objects':1,'events':[x for x in es if x['event'] in ('feature_conditions','navigation_done','probe_done')],'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('tools/map-capture/build.py'),Path('tools/map-capture/capture.lua'),Path('tools/map-capture/features.lua'),Path('src/map/objects.lua'),Path('src/map/reachability.lua')]}}
out=H/R.name;out.mkdir(exist_ok=True);(out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
