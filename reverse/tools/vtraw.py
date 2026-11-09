# print first instrs of each vtable slot: vtraw.py vt count
import sys, struct
sys.path.insert(0, r'reverse/tools')
from tdis import read, dis
vt=int(sys.argv[1],16); n=int(sys.argv[2]); k=int(sys.argv[3]) if len(sys.argv)>3 else 4
for i in range(n):
    p=struct.unpack('<Q',read(vt+8*i,8))[0]
    L=dis(p,0x40)
    out=[]
    for l in L[:k]:
        t=l.split(':',1)[1].strip(); out.append(t)
        if t.startswith('ret') or t.startswith('jmp'): break
    print('slot %3x  %x  %s'%(i*8,p,' | '.join(out)[:160]))
