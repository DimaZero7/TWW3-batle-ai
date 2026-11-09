import sys,collections
sys.path.insert(0,r'reverse/tools')
from dispfind import find_disp
from fn import bounds
off=[]
for l in open(r'reverse/notes/morale/kv_offsets.txt'):
    o,n,c,v=l.split(); off.append((int(o,16),n,v))
lo,hi=0x142d00000,0x143300000
for o,n,v in off:
    fs=set()
    for a,mn,op in find_disp(o):
        if 'rsp' in op or 'rbp' in op: continue
        if lo<=a<hi: fs.add(bounds(a)[0])
    print('%-5x %-55s %-8s %s'%(o,n,v[:6],' '.join('%x'%f for f in sorted(fs))))
