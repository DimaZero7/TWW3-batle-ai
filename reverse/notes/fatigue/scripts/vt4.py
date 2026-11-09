import sys,struct
sys.path.insert(0,r'reverse/tools')
from tdis import read
def q(a): return struct.unpack('<Q',read(a,8))[0]
T={0x1408d3d10:'F',0x1408d3d30:'T',0x1408d3ca0:'0',0x1408d3c60:'c60'}
for a in sys.argv[1:]:
    v=int(a,16)
    print('%x: '%v+' '.join('%d:%s'%(i,T.get(q(v+i*8),'%x'%q(v+i*8))) for i in range(0,38)))
