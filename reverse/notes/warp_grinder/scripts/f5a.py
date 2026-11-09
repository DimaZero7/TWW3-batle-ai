import sys,re
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB, dis
from fn import bounds
s=[x for x in pe.sections if x.Name.startswith(b'.udata')][0]
lo=s.PointerToRawData; buf=data[lo:lo+s.SizeOfRawData]
seen={}
for m in re.finditer(rb'(?:\x80[\x78-\x7f]\x5a\x00|\x38[\x40-\x7f]\x5a|\x0f\xb6[\x40-\x7f]\x5a|\x80[\x78-\x7f]\x59\x00|\x38[\x40-\x7f]\x59|\x0f\xb6[\x40-\x7f]\x59)',buf,re.S):
    va=IB+s.VirtualAddress+m.start()
    if not (0x1430c0000<va<0x1430f0000 or 0x142d80000<va<0x142de0000): continue
    f=bounds(va)[0]
    seen.setdefault(f,[]).append(va)
for f,v in sorted(seen.items()): print('%x'%f, ' '.join('%x'%x for x in v))
