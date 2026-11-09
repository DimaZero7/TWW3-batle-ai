import sys,struct
sys.path.insert(0,r'reverse/tools')
from tdis import read
def sl(v): return [struct.unpack_from('<Q',read(v,0x130),j*8)[0] for j in range(38)]
bv=sl(0x143919b28)
for l in open(sys.argv[1]):
    v=int(l.split()[0],16); s=sl(v)
    same=sum(1 for k in range(3,38) if s[k]==bv[k])
    spec=sum(1 for k in (26,27,36,25,34,35,37) if s[k]==bv[k])
    if spec>=2: print('%x same=%d spec=%d s28=%x s60=%x s78=%x s0=%x s8=%x'%(v,same,spec,s[5],s[12],s[15],s[0],s[1]))
