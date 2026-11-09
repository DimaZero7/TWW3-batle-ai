"""Read-only: find the memory sites' byte patterns of a sites file in a Warhammer3.exe and print each match's RVA.
After a game patch: every site must have exactly 1 hit; write the new rva and exe_sha256 into a new
reverse/mem/sites_exe<mmdd>.json (the layout offsets stay; the probe's own checks decide).
usage (from the project root): python reverse/mem/scan.py [sites.json] [Warhammer3.exe]
defaults: reverse/mem/sites_exe1103.json and the Steam exe."""
import json, re, struct, sys
SITES = sys.argv[1] if len(sys.argv) > 1 else "reverse/mem/sites_exe1103.json"
EXE = sys.argv[2] if len(sys.argv) > 2 else r"C:\Program Files (x86)\Steam\steamapps\common\Total War WARHAMMER III\Warhammer3.exe"
d = json.load(open(SITES, encoding="utf-8"))
b = open(EXE, "rb").read()
pe = struct.unpack_from("<I", b, 0x3c)[0]
nsec = struct.unpack_from("<H", b, pe + 6)[0]
opt = struct.unpack_from("<H", b, pe + 20)[0]
secs = []
for i in range(nsec):
    o = pe + 24 + opt + 40 * i
    name = b[o:o + 8].rstrip(b"\0").decode()
    vsize, va, rsize, raw = struct.unpack_from("<IIII", b, o + 8)
    secs.append((name, va, vsize, raw, rsize))
def rva(off):
    for name, va, vs, raw, rs in secs:
        if raw <= off < raw + rs:
            return va + off - raw, name
    return None, None
def rx(p):
    return re.compile(b"".join(b"." if t == "??" else re.escape(bytes([int(t, 16)])) for t in p.split()), re.S)
for k, s in d["sites"].items():
    p = s.get("pattern") or s.get("expect")
    if not p:
        print(k, "no pattern; old rva", s["rva"]); continue
    hits = [m.start() for m in rx(p).finditer(b)]
    print(k, "old", s["rva"], "hits", len(hits), [(hex(rva(h)[0]), rva(h)[1]) for h in hits[:6]])
# battle_ref: a mov rax,[rip+X]; mov rcx,[rax+0x440] - the same code with any displacement
p = "48 8b 05 ?? ?? ?? ?? 48 8b 88 40 04 00 00"
hits = [m.start() for m in rx(p).finditer(b)]
out = []
for h in hits:
    r, _ = rva(h)
    disp = struct.unpack_from("<i", b, h + 3)[0]
    out.append((hex(r), hex(r + 7 + disp)))
from collections import Counter
print("battle_ref-like", len(hits), Counter(g for _, g in out).most_common(5))
