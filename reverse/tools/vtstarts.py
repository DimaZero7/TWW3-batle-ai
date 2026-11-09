# vtable starts = rip-relative targets in code that fall within [lo,hi), 8-aligned
import sys, numpy as np
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16)
s=[x for x in pe.sections if x.Name.startswith(b'.udata')][0]
buf=np.frombuffer(data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData],dtype=np.uint8)
n=len(buf)-4
d=(buf[:n].astype(np.int64)|(buf[1:n+1].astype(np.int64)<<8)|(buf[2:n+2].astype(np.int64)<<16)|(buf[3:n+3].astype(np.int64)<<24))
d=np.where(d>=2**31,d-2**32,d)
base=IB+s.VirtualAddress
t=base+np.arange(n)+4+d
m=(t>=lo)&(t<hi)&(t%8==0)
idx=np.nonzero(m)[0]
# require lea opcode: 48/4c 8d xx before disp
res={}
for i in idx:
    if i>=3 and buf[i-2]==0x8d and buf[i-3] in (0x48,0x4c):
        res.setdefault(int(t[i]),[]).append(base+i-3)
for k in sorted(res): print('%x'%k, ' '.join('%x'%x for x in res[k][:4]))
