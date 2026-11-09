# find indirect calls/jmps via [reg+disp] for given vtable slot offsets; print containing function
import sys, os, re, struct
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB
from fn import bounds
s=[x for x in pe.sections if x.Name.startswith(b'.udata')][0]
code=data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData]; base=IB+s.VirtualAddress
res={}
for a in sys.argv[1:]:
    d=int(a,16)
    pat=re.compile(b'\xff[\x90-\x97\xa0-\xa7]'+re.escape(struct.pack('<i',d)))
    for m in pat.finditer(code):
        va=base+m.start()
        if va>0x143400000: continue
        f=bounds(va)[0]
        res.setdefault(f,[]).append(a)
for f in sorted(res): print('%x'%f, ' '.join(res[f]))
