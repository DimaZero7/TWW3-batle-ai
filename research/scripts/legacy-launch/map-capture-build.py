"""Build the isolated map-capture scenario; never replace the duel harness build."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('pack_builder',ROOT/'tools/build.py')
pack_builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack_builder)

def build(step,scenario=None,features=False):
    assert step in (1,2,3,5)
    reader=(ROOT/'src/map/reader.lua').read_bytes()
    capture=(ROOT/'tools/map-capture/capture.lua').read_bytes()
    xml=(scenario or ROOT/'scenarios/map_capture.xml').read_bytes()
    if features:
        preamble=b''
        for name,path in [('objects','src/map/objects.lua'),('reachability','src/map/reachability.lua'),('features','tools/map-capture/features.lua')]:
            preamble+=b'local '+name.encode()+b'=(function()\n'+(ROOT/path).read_bytes()+b'\nend)()\n'
        capture=preamble+capture
    script=b'local reader=(function()\n'+reader+b'\nend)()\n'+f'local config={{step={step},features={str(features).lower()}}}\n'.encode()+capture
    sys.path.insert(0,str(ROOT/'.tools/python'))
    from lupa.lua51 import LuaRuntime
    LuaRuntime().eval('function(s) local f,e=loadstring(s);assert(f,e) end')(script.decode())
    files={'script\\battle\\mod\\tww3_bai_map_capture.lua':script,
           'script\\battle\\tww3_bai_map_capture\\scenario.lua':b'load_script_libraries()\n',
           'script\\battle\\tww3_bai_map_capture\\map_probe.xml':xml}
    blob=pack_builder.pack_files(files);assert pack_builder.read_pack(blob)==files
    out=ROOT/'build/map-capture';out.mkdir(parents=True,exist_ok=True)
    for name,data in [('reader.lua',reader),('capture.lua',capture),('map_probe.xml',xml),('tww3_bai_map_capture.pack',blob)]:
        (out/name).write_bytes(data)
    manifest={'sha256':hashlib.sha256(blob).hexdigest(),'step':step,'features':features,'entries':list(files),
              'reader_sha256':hashlib.sha256(reader).hexdigest(),'capture_sha256':hashlib.sha256(capture).hexdigest(),
              'scenario_sha256':hashlib.sha256(xml).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step',type=int,choices=(1,2,3,5),default=5)
    parser.add_argument('--scenario',type=Path,help='Battle XML to capture; default is the fixed Kislev scenario')
    parser.add_argument('--features',action='store_true',help='Also read native/CCO objects and both units\' reachability after deployment')
    args=parser.parse_args()
    print(json.dumps(build(args.step,args.scenario,args.features),indent=2))
