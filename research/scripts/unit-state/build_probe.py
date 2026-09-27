from pathlib import Path
import hashlib,importlib.util,json,re,sys
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
mode=sys.argv[1] if len(sys.argv)>1 else 'basic'
assert mode in ('basic','charge_still','charge_extended','guard_off')
module=(ROOT/'src/units/state.lua').read_text(encoding='utf-8')
suite=(ROOT/'docs/units/evidence/empire-20260926/suite.lua').read_text(encoding='utf-8-sig').replace('SUITE_MODE','charge_still' if mode=='charge_extended' else mode)
if mode=='charge_extended':suite=suite.replace("add('cavalry_charge',45,", "add('cavalry_charge',120,")
suite=suite.replace('local U,C={},{},{}','local U,C={},{},{}\nlocal unit_side,stat_keys={},{}')
sample='''local function sample(n)
 local u=U[n];local side=unit_side[n]
 local function cco(unit,key)return query(unit,key)end
 local options={owned=true,observer_alliance=bm:alliances():item(side),cco=cco,target_id=function(t)return t:name()end}
 local row={stage=stage,name=n,side=side,unit_type=u:type(),elapsed=bm:time_elapsed_ms()-start_ms,
  readings=unit_state.observe(u,options),
  enemy_gate=unit_state.observe(u,{owned=false,observer_alliance=bm:alliances():item(3-side),cco=cco})}
 emit('state_sample',row)
 if not stat_keys[n] then
  stat_keys[n]={};local size=query(u,'UnitDetailsContext.StatList.Size')
  local keys={};if type(size)=='number' and size>=0 and size<=100 then
   for i=0,size-1 do
    local key=query(u,'UnitDetailsContext.StatList.At('..i..').Key');keys[#keys+1]=key
    if type(key)=='string' and (key:lower():find('morale') or key:lower():find('leadership')) then stat_keys[n][#stat_keys[n]+1]={index=i,key=key} end
   end
  end
  emit('card_stat_keys',{name=n,size=size,keys=keys})
 end
 for _,entry in ipairs(stat_keys[n])do
  local values={stage=stage,name=n,key=entry.key,index=entry.index}
  for _,field in ipairs({'Value','DisplayedValue','ValueBase'})do safe(values,field,function()return query(u,'UnitDetailsContext.StatList.At('..entry.index..').'..field)end)end
  emit('card_stat',values)
 end
end
'''
suite=re.sub(r'local function sample\(n\).*?\nlocal function all',sample+'local function all',suite,flags=re.S)
suite=suite.replace('U[n]=u;local c=', 'U[n]=u;unit_side[n]=ai;local c=')
suite=suite.replace(',active=u:owned_non_passive_special_abilities(),passive=u:owned_passive_special_abilities()', '')
suite=re.sub(r"   for _,key in ipairs\(\{'charge_defense_vs_large'.*?end\)end",'',suite)
# The original suite's geometry readout is not needed for state sensors.
suite=suite.replace("if mode=='withdraw' then geometry(n)end",'')
script='local unit_state=(function()\n'+module+'\nend)()\n'+suite
xml=(ROOT/'docs/units/evidence/empire-20260926/basic-rank1/map_probe.xml').read_bytes()
if mode in ('charge_still','charge_extended'):
 import xml.etree.ElementTree as ET
 root=ET.fromstring(xml)
 for u in root.findall('alliance')[1].find('army').findall('unit'):
  if u.attrib['script_name'].startswith('target'):
   u.find('unit_type').set('type','wh_main_emp_cav_empire_knights');u.set('num_soldiers','60')
 xml=ET.tostring(root,encoding='utf-8',xml_declaration=True)
sys.path.insert(0,str(ROOT/'.tools/python'))
from lupa.lua51 import LuaRuntime
LuaRuntime().eval('function(s)local f,e=loadstring(s);assert(f,e)end')(script)
spec=importlib.util.spec_from_file_location('pack',ROOT/'tools/build.py');pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
files={'script\\battle\\mod\\tww3_bai_map_capture.lua':script.encode(),
 'script\\battle\\tww3_bai_map_capture\\scenario.lua':b'load_script_libraries()\n',
 'script\\battle\\tww3_bai_map_capture\\map_probe.xml':xml}
blob=pack.pack_files(files);assert pack.read_pack(blob)==files
out=ROOT/'build/map-capture';out.mkdir(exist_ok=True)
sha=lambda data:hashlib.sha256(data).hexdigest()
manifest={'mode':mode,'step':0,'sha256':sha(blob),'capture_sha256':sha(script.encode()),'module_sha256':sha(module.encode()),'scenario_sha256':sha(xml),'suite_sha256':sha((ROOT/'docs/units/evidence/empire-20260926/suite.lua').read_bytes()),'classification':'state_sensor_diagnostic','field':'chokepoint_badlands_river'}
for name,data in [('capture.lua',script.encode()),('reader.lua',module.encode()),('map_probe.xml',xml),('tww3_bai_map_capture.pack',blob),('manifest.json',json.dumps(manifest,indent=2).encode())]:(out/name).write_bytes(data)
print(json.dumps(manifest,indent=2))
