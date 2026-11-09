import re, sys, pefile
import os
# the exe the tables (pdata.pkl, idx_battle.pkl), the Ghidra project and the notes' addresses belong to - 15.10.2026;
# another exe (the live Steam one): RE_EXE=<path>
p = os.environ.get('RE_EXE') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ghidra', 'exe1015', 'Warhammer3.exe')
data = open(p,'rb').read()
pe = pefile.PE(p, fast_load=True)
def sec_of(off):
    for s in pe.sections:
        if s.PointerToRawData <= off < s.PointerToRawData+s.SizeOfRawData:
            return s.Name.rstrip(b'\0').decode(), s.VirtualAddress + off - s.PointerToRawData
    return None, None
pats = sys.argv[1:]
for pat in pats:
    for m in re.finditer(pat.encode(), data):
        # expand to full ascii string
        a=m.start(); b=m.end()
        while a>0 and 32<=data[a-1]<127: a-=1
        while b<len(data) and 32<=data[b]<127: b+=1
        n,rva = sec_of(a)
        print(n, hex(0x140000000+rva) if rva is not None else None, data[a:b][:200].decode())
