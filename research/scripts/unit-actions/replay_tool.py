"""Rebuild an exact, verified unit-action probe. Does not launch or install the game."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/units/evidence/empire-20260926'


def main():
    choices = [r['label'] for r in json.loads((EVIDENCE / 'runs.json').read_text())]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=choices)
    args = parser.parse_args()
    source = EVIDENCE / args.experiment
    hashes = json.loads((EVIDENCE / 'hashes.json').read_text())
    def read(name):
        data = (source / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == hashes[f'{args.experiment}/{name}'], name
        return data
    capture, scenario, reader = read('capture.lua'), read('map_probe.xml'), read('reader.lua')
    manifest = json.loads(read('manifest.json'))
    sys.path.insert(0, str(ROOT / '.tools/python'))
    from lupa.lua51 import LuaRuntime
    LuaRuntime().eval('function(s) local f,e=loadstring(s);assert(f,e) end')(capture.decode('utf-8'))
    spec = importlib.util.spec_from_file_location('pack_builder', ROOT / 'tools/build.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    files = {
        'script\\battle\\mod\\tww3_bai_map_capture.lua': capture,
        'script\\battle\\tww3_bai_map_capture\\scenario.lua': b'load_script_libraries()\n',
        'script\\battle\\tww3_bai_map_capture\\map_probe.xml': scenario,
    }
    pack = builder.pack_files(files)
    assert builder.read_pack(pack) == files
    assert hashlib.sha256(pack).hexdigest() == manifest['sha256'], 'Different pack bytes'
    out = ROOT / 'build/map-capture'
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [('capture.lua', capture), ('map_probe.xml', scenario), ('reader.lua', reader),
                       ('manifest.json', read('manifest.json')), ('tww3_bai_map_capture.pack', pack)]:
        (out / name).write_bytes(data)
    print(json.dumps({'experiment': args.experiment, 'exact_pack_sha256': manifest['sha256']}))


if __name__ == '__main__':
    main()
