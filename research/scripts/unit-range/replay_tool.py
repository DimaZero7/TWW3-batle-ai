"""Verify the missile-range archive and reconstruct an exact diagnostic pack; no launch."""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/units/evidence/missile-range-20260926'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def verify(run):
    archive = json.loads((EVIDENCE / 'archive.json').read_text(encoding='utf-8'))
    for name, expected in archive['files'].items():
        assert sha((EVIDENCE / name).read_bytes()) == expected, name
    for entry in archive['compressed_logs']:
        assert sha(gzip.decompress((EVIDENCE / entry['stored']).read_bytes())) == entry['uncompressed_sha256'], entry['stored']
    source = EVIDENCE / run
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8-sig'))
    spec = importlib.util.spec_from_file_location('range_pack_builder', ROOT / 'tools/build.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    entries = {
        'script\\battle\\mod\\tww3_bai_map_capture.lua': (source / 'capture.lua').read_bytes(),
        'script\\battle\\tww3_bai_map_capture\\scenario.lua': b'load_script_libraries()\n',
        'script\\battle\\tww3_bai_map_capture\\map_probe.xml': (source / 'map_probe.xml').read_bytes(),
    }
    pack = builder.pack_files(entries)
    assert builder.read_pack(pack) == entries
    assert sha(pack) == manifest['sha256'], 'Exact pack mismatch'
    assert sha((source / 'capture.lua').read_bytes()) == manifest['capture_sha256']
    assert sha((source / 'reader.lua').read_bytes()) == manifest['module_sha256']
    assert sha((source / 'map_probe.xml').read_bytes()) == manifest['scenario_sha256']
    return pack, len(archive['files'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', choices=('range', 'fog', 'boundary'))
    parser.add_argument('--output', type=Path, help='Optional new pack file; refuses to overwrite')
    args = parser.parse_args()
    pack, count = verify(args.run)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            stream.write(pack)
    print(json.dumps({'run': args.run, 'verified_files': count, 'pack_sha256': sha(pack), 'game_launched': False}))

if __name__ == '__main__':
    main()
