# map vtable slots to kv names: vtmap.py vtable_start count
import sys, struct, json, re
sys.path.insert(0, r'reverse/tools')
from tdis import read, dis
kv=json.load(open(r'reverse/tools/kv_value_offsets.json'))
vt=int(sys.argv[1],16); n=int(sys.argv[2])
for i in range(n):
    p=struct.unpack('<Q',read(vt+8*i,8))[0]
    L=dis(p,0x30)[:4]
    txt=' | '.join(l.split(':',1)[1].strip() for l in L)
    nm=''
    m=re.search(r'\[rax \+ (0x[0-9a-f]+)\]', txt)
    if m and 'rcx + 8' in txt: nm=kv.get('%x'%int(m.group(1),16),'?')
    print('slot %3x  %x  %-40s %s'%(i*8,p,nm,txt[:120] if not nm else ''))
