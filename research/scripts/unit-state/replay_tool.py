"""Verify the public unit-state archive and rebuild one exact probe; never launch."""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/units/evidence/unit-state-20260926'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', choices=('basic', 'charge', 'charge-extended'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    archive = json.loads((EVIDENCE / 'archive.json').read_text())
    for name, expected in archive['files'].items():
        assert digest((EVIDENCE / name).read_bytes()) == expected, name
    for entry in archive['compressed_logs']:
        raw = gzip.decompress((EVIDENCE / entry['stored']).read_bytes())
        assert digest(raw) == entry['uncompressed_sha256'], entry['stored']
    source = EVIDENCE / args.run
    manifest = json.loads((source / 'manifest.json').read_text())
    spec = importlib.util.spec_from_file_location('pack_builder', ROOT / 'tools/build.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    files = {
        'script\\battle\\mod\\tww3_bai_map_capture.lua': (source / 'capture.lua').read_bytes(),
        'script\\battle\\tww3_bai_map_capture\\scenario.lua': b'load_script_libraries()\n',
        'script\\battle\\tww3_bai_map_capture\\map_probe.xml': (source / 'map_probe.xml').read_bytes(),
    }
    pack = builder.pack_files(files)
    assert builder.read_pack(pack) == files
    assert digest(pack) == manifest['sha256'], 'Exact probe hash mismatch'
    if not args.verify_only:
        out = ROOT / 'build/map-capture'
        out.mkdir(parents=True, exist_ok=True)
        for name in ('capture.lua', 'reader.lua', 'map_probe.xml', 'manifest.json'):
            shutil.copyfile(source / name, out / name)
        (out / 'tww3_bai_map_capture.pack').write_bytes(pack)
    print(json.dumps({'run': args.run, 'verified_files': len(archive['files']),
                      'pack_sha256': digest(pack), 'game_launched': False,
                      'build_written': not args.verify_only}))


if __name__ == '__main__':
    main()
