import sys,struct
sys.path.insert(0,r'reverse/tools')
from tdis import read
def q(a): return struct.unpack('<Q',read(a,8))[0]
for a in sys.argv[1:]:
    a=int(a,16)
    # walk back to find vtable start: previous qword not a code pointer in .udata range
    v=a
    while 0x140266000 <= q(v-8) < 0x143400000: v-=8
    print('vt %x (col %x): '%(v,q(v-8)) + ' '.join('%d:%x'%(i,q(v+i*8)) for i in range(0,40)))
