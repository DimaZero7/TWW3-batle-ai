# vtables (lea-referenced, in .xcode) having >= N code slots; print slots given
import sys, struct, numpy as np
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB, read, dis
N=int(sys.argv[1],16); show=[int(a,16) for a in sys.argv[2].split(',')]
x=[s for s in pe.sections if s.Name.startswith(b'.xcode')][0]
lo=IB+x.VirtualAddress; hi=lo+x.Misc_VirtualSize
s=[s for s in pe.sections if s.Name.startswith(b'.udata')][0]
cl=IB+s.VirtualAddress; ch=cl+s.Misc_VirtualSize
buf=np.frombuffer(data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData],dtype=np.uint8)
n=len(buf)-4
d=(buf[:n].astype(np.int64)|(buf[1:n+1].astype(np.int64)<<8)|(buf[2:n+2].astype(np.int64)<<16)|(buf[3:n+3].astype(np.int64)<<24))
d=np.where(d>=2**31,d-2**32,d)
t=cl+np.arange(n)+4+d
m=(t>=lo)&(t<hi)&(t%8==0)
idx=np.nonzero(m)[0]
idx=[i for i in idx if i>=3 and buf[i-2]==0x8d and buf[i-3] in (0x48,0x4c)]
vts=sorted(set(int(t[i]) for i in idx))
def q(v,o):
    b=read(v+o,8); return struct.unpack('<Q',b)[0] if b and len(b)==8 else 0
for v in vts:
    if all(cl<=q(v,o)<ch for o in range(0,N+8,8)):
        print('%x'%v, ' '.join('+%x=%x'%(o,q(v,o)) for o in show))
