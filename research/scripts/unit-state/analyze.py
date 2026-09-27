"""Summarize diagnostic readouts, retaining unknowns and fixture boundaries."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,shutil,sys,math
HERE=Path(__file__).resolve().parent
source=Path(sys.argv[1]).resolve();name=sys.argv[2]
out=HERE/name
assert not out.exists(),'retain previous analysis unchanged'
shutil.copytree(source,out)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
overall={};stages={};previous={};examples=[];stat_rows=[];keys=[];events=Counter();unknown=Counter();samples=Counter();last_sample={};fixture_events=[]
def collect(bucket,k,sensor,row):
    s=bucket.setdefault(k,{'availability':{},'observed_values':[],'transitions':[]})
    status=sensor['status'];reason=sensor.get('reason',status)
    s['availability'][reason]=s['availability'].get(reason,0)+1
    if status!='known':return
    value=sensor['value']
    if isinstance(value,(float,int)) and not isinstance(value,bool):
        s['min']=min(s.get('min',value),value);s['max']=max(s.get('max',value),value)
    if not isinstance(value,dict) and value not in s['observed_values'] and len(s['observed_values'])<24:s['observed_values'].append(value)
    if s.get('_previous')!=value and len(s['transitions'])<16:s['transitions'].append({'ms':row['ms'],'stage':row['stage'],'value':value})
    s['_previous']=value
def val(row,key):
    r=row['readings']['sensors'].get(key,{})
    return r.get('value') if r.get('status')=='known' else None
comparison=defaultdict(lambda:{'samples':0,'max_fraction_error':0,'max_native_cco_error':0,'max_ammo_fraction_error':0})
gate=Counter();consistency=Counter();card_pairs=defaultdict(list)
for line in (out/'tww3_bai_map_capture_events.jsonl').open(encoding='utf-8'):
    row=json.loads(line);events[row['event']]+=1
    if row['event'].startswith('fixture_'):fixture_events.append(row)
    if row['event']=='card_stat_keys':keys.append(row)
    if row['event']=='card_stat' and row['name'] in ('spears','archers','general'):
        stat_rows.append(row)
        sample=last_sample.get(row['name'])
        if sample and sample['ms']==row['ms']:
            card_pairs[row['name']].append({'ms':row['ms'],'stage':row['stage'],'key':row['key'],'Value':row.get('Value'),'DisplayedValue':row.get('DisplayedValue'),'ValueBase':row.get('ValueBase'),'MoralePercent':val(sample,'cco.MoralePercent'),'MoraleState':val(sample,'cco.MoraleState'),'IsRouting':val(sample,'cco.IsRouting')})
    if row['event']!='state_sample' or row['name'] not in ('spears','archers','general'):continue
    n=row['name'];samples[n]+=1;last_sample[n]=row
    assert row['enemy_gate']['access']=='withheld_own_only' and not row['enemy_gate']['sensors']
    gate[str(row['enemy_gate']['visibility'])]+=1
    for k,s in row['readings']['sensors'].items():
        collect(overall.setdefault(n,{}),k,s,row)
        collect(stages.setdefault(n,{}).setdefault(row['stage'],{}),k,s,row)
        if s['status']!='known':unknown[k+':'+s['reason']]+=1
    hp,hmax,hpct,native=[val(row,k) for k in ('cco.HealthValue','cco.HealthMax','cco.HealthPercent','native.unary_hitpoints')]
    ammo,ammo0,apct=[val(row,k) for k in ('native.ammo_left','native.starting_ammo','cco.PrimaryAmmoPercent')]
    c=comparison[n]
    if all(isinstance(x,(int,float)) for x in (hp,hmax,hpct,native)) and hmax>0:
        c['samples']+=1;c['max_fraction_error']=max(c['max_fraction_error'],abs(hp/hmax-hpct));c['max_native_cco_error']=max(c['max_native_cco_error'],abs(native-hpct))
    if all(isinstance(x,(int,float)) for x in (ammo,ammo0,apct)) and ammo0>0:c['max_ammo_fraction_error']=max(c['max_ammo_fraction_error'],abs(ammo/ammo0-apct))
    old=previous.get(n)
    if old:
        oldhp=val(old,'cco.HealthValue');oldmen=val(old,'native.number_of_men_alive');men=val(row,'native.number_of_men_alive')
        if isinstance(hp,(int,float)) and isinstance(oldhp,(int,float)) and hp<oldhp and men==oldmen:
            consistency['damage_without_death_'+n]+=1
            if len([e for e in examples if e['name']==n])<3:examples.append({'name':n,'stage':row['stage'],'from_ms':old['ms'],'to_ms':row['ms'],'hp_before':oldhp,'hp_after':hp,'men':men})
    previous[n]=row
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k!='_previous'}
    if isinstance(v,list):return [clean(x) for x in v]
    return v
def write(file,value):
    (out/file).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
write('fields.json',overall);write('stages.json',stages);write('card-morale-pairs.json',dict(card_pairs));write('card-stat-keys.json',keys)
summary={'run_name':name,'source_run':str(source),'events':dict(events),'samples_by_type':dict(samples),'field_count':{k:len(v) for k,v in overall.items()},'unknown_counts':dict(unknown),'comparisons':dict(comparison),'damage_without_death_counts':dict(consistency),'damage_without_death_examples':examples,'enemy_gate_counts':dict(gate),'fixture_events':fixture_events,'module_sha256':sha(out/'reader.lua'),'manifest':json.loads((out/'manifest.json').read_text()),'cleanup':json.loads((out/'cleanup.json').read_text(encoding='utf-8-sig')),'status':json.loads((out/'status.json').read_text(encoding='utf-8-sig')),'limits':['Diagnostic fixtures include teleports; movements, damage and morale otherwise follow the selected suite unless a fixture event explicitly says forced.', 'Observed values/transitions do not enumerate every possible state.','0/false is preserved; unknown is not interpreted as an absent mechanic.','CCO morale card values and percent require interpretation; no assumed conversion.','Sampled damage indicators do not establish complete MATCH-IDLE-001 telemetry.','This module is own-unit readout only and not admitted in policy API v1.']}
write('summary.json',summary)
write('hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='hashes.json'})
print(json.dumps({k:summary[k] for k in ('samples_by_type','unknown_counts','comparisons','damage_without_death_counts')},ensure_ascii=False,indent=2))
