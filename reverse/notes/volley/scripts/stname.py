import sys,struct
sys.path.insert(0, r'reverse/tools')
from tdis import read, dis
def name(vt):
    f=struct.unpack('<Q',read(vt+0x20,8))[0]
    L=dis(f,8)
    return L[0] if L else '?'
for a in sys.argv[1:]:
    a=int(a,16)
    vt=struct.unpack('<Q',read(a,8))[0]
    sl=[struct.unpack('<Q',read(vt+o,8))[0] for o in (0x28,0x30,0x38)]
    print('%x vt %x'%(a,vt), name(vt), ' '.join('%x'%s for s in sl))
