# dump vtable entries and short disasm of each
import sys, struct
sys.path.insert(0, r'reverse/tools')
from tdis import read, dis
vt=int(sys.argv[1],16); n=int(sys.argv[2]); m=int(sys.argv[3]) if len(sys.argv)>3 else 0x40
for i in range(n):
    f=struct.unpack('<Q',read(vt+8*i,8))[0]
    print('== [%d] +0x%x -> %x'%(i,8*i,f))
    L=dis(f,m)
    for l in L:
        print('   ',l)
        if ' ret' in l or 'jmp ' in l: break
