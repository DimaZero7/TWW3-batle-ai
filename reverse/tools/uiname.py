import sys, re, struct
sys.path.insert(0, r'reverse/tools')
from tdis import data, pe, IB, dis
from xref import xrefs
s=[x for x in pe.sections if x.Name.startswith(b'.udata')][0]
base=IB+s.VirtualAddress-s.PointerToRawData
for a in sys.argv[1:]:
    off=int(a,16)
    pat=b'\x48\x8b\x41\x08\x0f\xb6\x80'+struct.pack('<I',off)+b'\xc3'
    for m in re.finditer(re.escape(pat), data):
        g=base+m.start()
        if not (s.PointerToRawData<=m.start()<s.PointerToRawData+s.SizeOfRawData): continue
        refs=xrefs([g])[g]
        names=[]
        for r in refs[:3]:
            L=dis(r-0x60,0x64)
            nm=[re.search(r'"([^"]+)"',l).group(1) for l in L if '"' in l]
            names.append(nm[-1] if nm else '?')
        print(a, hex(g), names)
