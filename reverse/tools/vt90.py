import sys, struct, numpy as np
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB, read, dis
rlo=int(sys.argv[1],16); rhi=int(sys.argv[2],16); slot=int(sys.argv[3],16)
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
for v in vts:
    try:
        p0=struct.unpack('<Q',read(v,8))[0]; ps=struct.unpack('<Q',read(v+slot,8))[0]
    except Exception: continue
    if not (rlo<=p0<rhi): continue
    try: L=dis(ps,0x18)[:2]
    except Exception: continue
    tx=' | '.join(l.split(':',1)[1].strip() for l in L)
    print('%x slot0 %x  +%x %x %s'%(v,p0,slot,ps,tx[:80]))
