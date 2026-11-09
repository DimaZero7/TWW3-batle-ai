# for slot addresses: nearest lea-referenced vtable start below, offset
import sys, numpy as np, bisect
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB
x=[s for s in pe.sections if s.Name.startswith(b'.xcode')][0]
lo=IB+x.VirtualAddress; hi=lo+x.Misc_VirtualSize
s=[s for s in pe.sections if s.Name.startswith(b'.udata')][0]
buf=np.frombuffer(data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData],dtype=np.uint8)
n=len(buf)-4
d=(buf[:n].astype(np.int64)|(buf[1:n+1].astype(np.int64)<<8)|(buf[2:n+2].astype(np.int64)<<16)|(buf[3:n+3].astype(np.int64)<<24))
d=np.where(d>=2**31,d-2**32,d)
base=IB+s.VirtualAddress
t=base+np.arange(n)+4+d
m=(t>=lo)&(t<hi)&(t%8==0)
idx=np.nonzero(m)[0]
idx=[i for i in idx if i>=3 and buf[i-2]==0x8d and buf[i-3] in (0x48,0x4c)]
vts=sorted(set(int(t[i]) for i in idx))
import pickle; pickle.dump(vts, open('vts.pkl','wb'))
for a in sys.argv[1:]:
    p=int(a,16); k=bisect.bisect_right(vts,p)-1
    print('%x  vt %x  +%x'%(p,vts[k],p-vts[k]))
