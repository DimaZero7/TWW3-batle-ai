import sys,os,collections
sys.path.insert(0,r'reverse/tools')
from dispfind import find_disp
from fn import bounds
off={}
for l in open(r'reverse/notes/morale/kv_offsets.txt'):
    o,n,c,v=l.split(); off[int(o,16)]=n
cnt=collections.defaultdict(set)
for o,n in off.items():
    if o<0x600: continue  # small offsets too common
    for a,mn,op in find_disp(o):
        if 'rsp' in op or 'rbp' in op: continue
        cnt[bounds(a)[0]].add(n)
res=sorted(cnt.items(), key=lambda x:-len(x[1]))
for f,s in res[:60]:
    if len(s)<2: break
    print('%x %d %s'%(f,len(s),' '.join(sorted(s))))
