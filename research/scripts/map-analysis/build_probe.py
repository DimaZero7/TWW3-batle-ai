from pathlib import Path
import argparse,sys,json,hashlib,importlib.util,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
ap=argparse.ArgumentParser();ap.add_argument('--map',default='kislev');ap.add_argument('--combat',action='store_true');ap.add_argument('--catchment-only',action='store_true');ap.add_argument('--navigation',action='store_true');ap.add_argument('--cross-fords',action='store_true');ap.add_argument('--position',type=float,nargs=2);ap.add_argument('--bridge-test',action='store_true');args=ap.parse_args()
xml=ET.fromstring((ROOT/'scenarios/map_capture.xml').read_bytes())
if args.map!='kislev':
    row=next(r for r in json.loads((HERE/'battles.json').read_text()) if r['key']==args.map)
    assert row['type'] in ('classic','river_crossing_battle') and not row['is_large_settlement'] and not row['has_15m_walls']
    xml.find('./battle_map_definition/name').text=row['specification']
    xml.find('./battle_map_definition/catchment_area').text=row['catchment_name']
    (HERE/'selected-map.json').write_text(json.dumps(row,indent=2))
if args.catchment_only:
    md=xml.find('./battle_map_definition');md.remove(md.find('tile_map_position'))
if args.position:
    md=xml.find('./battle_map_definition');md.remove(md.find('catchment_area'))
    node=md.find('tile_map_position')
    if node is None:node=ET.SubElement(md,'tile_map_position')
    node.set('x',str(args.position[0]));node.set('y',str(args.position[1]))
reader=(ROOT/'src/map/reader.lua').read_bytes();capture=(ROOT/'tools/map-capture/capture.lua').read_text()
module_preamble=''
for module in ('objects','reachability'):
    module_preamble+='local '+module+'=(function()\n'+(ROOT/f'src/map/{module}.lua').read_text()+'\nend)()\n'
capture=module_preamble+capture
extra=(HERE/'probe.lua').read_text()
if args.navigation:
    assert args.combat
    extra+='\n'+(HERE/'navigation.lua').read_text()
if args.cross_fords:
    assert args.combat and not args.navigation
    cases=json.loads((HERE/'ford-movement-cases.json').read_text())
    cases_lua='{'+','.join('{'+','.join(k+'='+str(v) for k,v in c.items())+'}' for c in cases)+'}'
    extra+='\n'+(HERE/'cross_fords.lua').read_text().replace('FORD_CASES',cases_lua)
if args.bridge_test:
    assert args.combat and not args.navigation and not args.cross_fords
    extra+='\n'+(HERE/'bridge_test.lua').read_text()
capture=capture.replace('local stopped=false',extra+'\nlocal stopped=false')
if args.combat:
    capture=capture.replace("            emit('probe_done',{phase=bm:get_current_phase_name(),speed=bm:current_battle_speed()})",'''            local controllers={}
            for ai=1,2 do
                local army=bm:alliances():item(ai):armies():item(1)
                local uc=army:create_unit_controller();uc:add_units(army:units():item(1));uc:take_control();uc:fire_at_will(false);uc:halt();controllers[#controllers+1]=uc
            end
            bm:register_phase_change_callback('Deployed',guarded(function()
                bm:real_callback(guarded(function()
                    bm:modify_battle_speed(0)
                    bm:real_callback(guarded(function()
                        extra_probe()
                        emit('probe_done',{phase=bm:get_current_phase_name(),speed=bm:current_battle_speed()})
                    end),250,prefix..'paused_scan')
                end),250,prefix..'combat_scan')
            end))
            bm:modify_battle_speed(20)
            bm:end_current_battle_phase()''')
else:
    capture=capture.replace("            emit('probe_done'","            extra_probe()\n            emit('probe_done'")
if args.navigation:
    capture=capture.replace("                        emit('probe_done',{phase=bm:get_current_phase_name(),speed=bm:current_battle_speed()})","                        navigation_scan(function() emit('probe_done',{phase=bm:get_current_phase_name(),speed=bm:current_battle_speed()}) end)")
if args.cross_fords:
    capture=capture.replace("                        emit('probe_done',{phase=bm:get_current_phase_name(),speed=bm:current_battle_speed()})","                        cross_fords(function() emit('probe_done',{phase=bm:get_current_phase_name(),speed=bm:current_battle_speed()}) end)")
if args.bridge_test:
    capture=capture.replace("                        emit('probe_done',{phase=bm:get_current_phase_name(),speed=bm:current_battle_speed()})","                        bridge_test(function() emit('probe_done',{phase=bm:get_current_phase_name(),speed=bm:current_battle_speed()}) end)")
capture=capture.encode();scenario=ET.tostring(xml,encoding='utf-8',xml_declaration=True)
script=b'local reader=(function()\n'+reader+b'\nend)()\nlocal config={step=3}\n'+capture
sys.path.insert(0,str(ROOT/'.tools/python'))
from lupa.lua51 import LuaRuntime
LuaRuntime().eval('function(s)local f,e=loadstring(s);assert(f,e)end')(script.decode())
spec=importlib.util.spec_from_file_location('pack_builder',ROOT/'tools/build.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
files={'script\\battle\\mod\\tww3_bai_map_capture.lua':script,'script\\battle\\tww3_bai_map_capture\\scenario.lua':b'load_script_libraries()\n','script\\battle\\tww3_bai_map_capture\\map_probe.xml':scenario}
blob=builder.pack_files(files);assert builder.read_pack(blob)==files
out=ROOT/'build/map-capture';out.mkdir(exist_ok=True,parents=True)
for name,data in [('reader.lua',reader),('capture.lua',capture),('map_probe.xml',scenario),('tww3_bai_map_capture.pack',blob)]: (out/name).write_bytes(data)
manifest=dict(sha256=hashlib.sha256(blob).hexdigest(),step=3,experiment='field-features',map_key=args.map,combat=args.combat,catchment_only=args.catchment_only,entries=list(files),reader_sha256=hashlib.sha256(reader).hexdigest(),capture_sha256=hashlib.sha256(capture).hexdigest(),scenario_sha256=hashlib.sha256(scenario).hexdigest())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
