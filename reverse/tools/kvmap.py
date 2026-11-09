import sys, os, json; D=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,D)
from tdis import dis
import re
lines = dis(0x140552968, 0x1405544b6-0x140552968)
m={}
cur=None
for l in lines:
    r=re.search(r'lea\s+rcx, \[rsi \+ (0x[0-9a-f]+)\]', l)
    if r: cur=int(r.group(1),16)
    if 'add      rcx, 0x18' in l: cur=0x18
    r=re.search(r'"([^"]+)"', l)
    if r and cur is not None and 'lea      rdx' in l:
        m[cur]=r.group(1); cur=None
json.dump({'%x'%(k+0x60):v for k,v in m.items()}, open(os.path.join(D,'kv_value_offsets.json'),'w'), indent=0)
print(len(m))
for k in sys.argv[1:]:
    k=int(k,16); print(hex(k), m.get(k-0x60))
