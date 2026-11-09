import sys, os, re; D=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,D)
from dispfind import find_disp
from fn import bounds
from tdis import dis
d=int(sys.argv[1],16); flt=sys.argv[2] if len(sys.argv)>2 else None
seen=set()
for a,mn,op in find_disp(d):
    if flt and not re.search(flt, mn+' '+op): continue
    s,e=bounds(a)
    if a>=e: s,e=a,a+0x80
    key=s
    strs=[]
    if key not in seen and e-s<0x3000:
        for l in dis(s,e-s):
            m=re.search(r'"([^"]+)"',l)
            if m: strs.append(m.group(1))
    seen.add(key)
    print('%x %s %s | f=%x size=%x | %s'%(a,mn,op,s,e-s,'; '.join(strs[:6])))
