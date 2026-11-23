"""A build's pack for one launcher SLOT (bridge/docs/launch/slots.md): several game processes at once, each with its own
files in the game folder. From a build (manifest.json + pack) it writes <out>/manifest.json and <out>/<stem>_s<N>.pack:
the same pack with a shim put first into every Lua file, which renames the game-folder files our Lua opens -
any name 'tww3_bai_<x>.<ext>' without a folder (also './tww3_bai_<x>', any case) becomes 'tww3_bai_<x>_s<N>.<ext>' in
io.open, io.lines, io.input, io.output, os.rename and os.remove (tww3_bai_events.jsonl -> tww3_bai_events_s2.jsonl).
A build whose Lua names a tww3_bai_ file inside a folder (the shim would not rename it) is refused; so is a pack whose
bytes are not the manifest's (pack_sha256, and --expect: the hash the launcher read). Nothing in bridge/lua changes;
slot 0 is not a slot (the build as it is).

    python -m bridge.pack.slot <build_dir> <slot 1..3> <out_dir> [--expect <pack sha256>]
    python -m bridge.pack.slot --name tww3_bai_events.jsonl 2      # the slot's name of one file (the launcher's rule)
"""
import hashlib
import json
import re
import sys
from pathlib import Path

from bridge.pack import pfh5

SLOTS = (1, 2, 3)
SHIM = """-- bridge/pack/slot.py: slot {n} - our game-folder files get the suffix _s{n} (several game processes at once)
if TWW3_BAI_SLOT and TWW3_BAI_SLOT ~= {n} then error('bridge/pack/slot.py: this Lua state is slot ' .. tostring(TWW3_BAI_SLOT) .. ', not {n}') end
if not TWW3_BAI_SLOT then
    TWW3_BAI_SLOT = {n}
    local function slot_name(p)
        if type(p) ~= 'string' then return p end
        local q = p:gsub('^%.[/\\\\]', '')
        if q:sub(1, 9):lower() ~= 'tww3_bai_' or q:find('[/\\\\:]') then return p end
        local stem, rest = q:match('^([^.]*)(.*)$')
        return stem .. '_s{n}' .. rest
    end
    local open, lines, input, output, rename, remove = io.open, io.lines, io.input, io.output, os.rename, os.remove
    io.open = function(p, ...) return open(slot_name(p), ...) end
    io.lines = function(p, ...) if p == nil then return lines() end return lines(slot_name(p), ...) end
    io.input = function(p) if p == nil then return input() end return input(slot_name(p)) end
    io.output = function(p) if p == nil then return output() end return output(slot_name(p)) end
    os.rename = function(a, b) return rename(slot_name(a), slot_name(b)) end
    os.remove = function(p) return remove(slot_name(p)) end
end
"""
FOLDER_NAME = re.compile(rb"""[/\\]tww3_bai_""", re.IGNORECASE)


def name(file, n):
    """The slot's name of a game-folder file (the shim's rule; the launcher uses the same)."""
    if n == 0 or not file.startswith("tww3_bai_") or re.search(r"[/\\:]", file):
        return file
    stem, rest = re.match(r"^([^.]*)(.*)$", file).groups()
    return f"{stem}_s{n}{rest}"


def make(build_dir, n, out_dir, expect=None):
    if n not in SLOTS:
        raise ValueError(f"slot must be one of {SLOTS}")
    build_dir, out_dir = Path(build_dir).resolve(), Path(out_dir).resolve()
    if out_dir == build_dir:
        raise ValueError(f"the slot's output {out_dir} is the build itself")
    man = json.loads((build_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    if man.get("slot"):
        raise ValueError(f"{build_dir} is already a slot build (slot {man['slot']})")
    if expect is not None and man.get("pack_sha256") != expect:
        raise ValueError(f"{build_dir}/manifest.json changed: pack_sha256 {man.get('pack_sha256')} is not {expect}")
    raw = (build_dir / man["pack"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != man.get("pack_sha256"):
        raise ValueError(f"{build_dir / man['pack']} is not the pack of its manifest (pack_sha256)")
    files, deps = pfh5.read_pack(raw)
    shim = SHIM.replace("{n}", str(n)).encode("utf-8")
    for key in list(files):
        if key.lower().endswith(".lua"):
            if FOLDER_NAME.search(files[key]):
                raise ValueError(f"{key} names a tww3_bai_ file inside a folder: the slot's shim would not rename it")
            files[key] = shim + files[key]
    blob = pfh5.pack_files(files, deps)
    pack = Path(man["pack"]).stem + f"_s{n}.pack"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / pack).write_bytes(blob)
    man.update(pack=pack, pack_sha256=hashlib.sha256(blob).hexdigest(), slot=n, slot_of=str(build_dir),
               source_pack_sha256=hashlib.sha256(raw).hexdigest())
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return out_dir / pack


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a and a[0] == "--name":
        print(name(a[1], int(a[2])))
        return 0
    expect = None
    if "--expect" in a:
        i = a.index("--expect")
        expect = a[i + 1]
        del a[i:i + 2]
    if len(a) != 3:
        print(__doc__)
        return 2
    print(make(a[0], int(a[1]), a[2], expect))
    return 0


if __name__ == "__main__":
    sys.exit(main())
