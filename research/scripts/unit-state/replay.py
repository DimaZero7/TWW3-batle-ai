"""Rebuild exact archived diagnostic bytes; does not launch a game."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
name=sys.argv[1];assert name in ('basic','charge','charge-extended')
source=HERE/name;manifest=json.loads((source/'manifest.json').read_text())
spec=importlib.util.spec_from_file_location('pack',ROOT/'tools/build.py');pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
files={'script\\battle\\mod\\tww3_bai_map_capture.lua':(source/'capture.lua').read_bytes(),
 'script\\battle\\tww3_bai_map_capture\\scenario.lua':b'load_script_libraries()\n',
 'script\\battle\\tww3_bai_map_capture\\map_probe.xml':(source/'map_probe.xml').read_bytes()}
blob=pack.pack_files(files)
assert hashlib.sha256(blob).hexdigest()==manifest['sha256'],'archived build does not reproduce original hash'
out=ROOT/'build/map-capture'
for file in ('capture.lua','reader.lua','map_probe.xml','manifest.json'):shutil.copyfile(source/file,out/file)
(out/'tww3_bai_map_capture.pack').write_bytes(blob)
print(manifest['sha256'])
