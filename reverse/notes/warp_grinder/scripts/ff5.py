import sys,re
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB, dis
from fn import bounds
s=[x for x in pe.sections if x.Name.startswith(b'.udata')][0]
lo=s.PointerToRawData; buf=data[lo:lo+s.SizeOfRawData]
seen={}
for m in re.finditer(rb'(?:\xb8|\xbb|\xbf|\xbe|\xba|\xb9)\xf5\x00\x00\x00',buf,re.S):
    va=IB+s.VirtualAddress+m.start()
    if not (0x142c00000<va<0x143300000): continue
    f=bounds(va)[0]
    seen.setdefault(f,[]).append(va)
for f,v in sorted(seen.items()): print('%x'%f, len(v))
