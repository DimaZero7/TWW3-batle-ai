# for every lea-referenced vtable in .xcode with slot0 in [rlo,rhi): print slots given if slot A matches pattern
import sys, struct, numpy as np, re
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB, read, dis
rlo=int(sys.argv[1],16); rhi=int(sys.argv[2],16)
need=[(int(a.split('=')[0],16), a.split('=')[1]) for a in sys.argv[3].split(',')]
show=[int(a,16) for a in sys.argv[4].split(',')]
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
def q(v,o):
    b=read(v+o,8)
    return struct.unpack('<Q',b)[0] if b and len(b)==8 else 0
def txt(p):
    b=read(p,0x18)
    if not b: return '?'
    L=dis(p,0x18)[:2]; return ' | '.join(l.split(':',1)[1].strip() for l in L)
for v in vts:
    p0=q(v,0)
    if not (rlo<=p0<rhi): continue
    ok=True
    for o,pat in need:
        p=q(v,o)
        if not (IB<p<IB+0x10000000) or not re.search(pat, txt(p)): ok=False;break
    if not ok: continue
    print('%x'%v, ' ; '.join('+%x %x %s'%(o,q(v,o),txt(q(v,o))[:50]) for o in show))
