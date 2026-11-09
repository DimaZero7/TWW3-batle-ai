import pefile, math, sys, collections
import os
# the exe the tables (pdata.pkl, idx_battle.pkl), the Ghidra project and the notes' addresses belong to - 15.10.2026;
# another exe (the live Steam one): RE_EXE=<path>
p = os.environ.get('RE_EXE') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ghidra', 'exe1015', 'Warhammer3.exe')
pe = pefile.PE(p, fast_load=True)
pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'], pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT'], pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_TLS']])
print("ImageBase", hex(pe.OPTIONAL_HEADER.ImageBase), "EP", hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint))
def ent(b):
    if not b: return 0
    c = collections.Counter(b); n=len(b)
    return -sum(v/n*math.log2(v/n) for v in c.values())
for s in pe.sections:
    d = s.get_data()
    print(s.Name.rstrip(b'\0'), hex(s.VirtualAddress), hex(s.Misc_VirtualSize), hex(s.SizeOfRawData), "ent=%.3f"%ent(d[:50_000_000]), hex(s.Characteristics))
ep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
for s in pe.sections:
    if s.VirtualAddress <= ep < s.VirtualAddress+s.Misc_VirtualSize: print("EP in", s.Name)
tot=0
for imp in getattr(pe,'DIRECTORY_ENTRY_IMPORT',[]):
    print(imp.dll, len(imp.imports)); tot+=len(imp.imports)
print("imports total", tot)
print("TLS", hasattr(pe,'DIRECTORY_ENTRY_TLS'))
