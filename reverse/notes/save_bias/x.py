import sys
sys.path.insert(0,'reverse/tools')
from xref import xrefs
from fn import bounds
base=int(sys.argv[1],16); span=int(sys.argv[2],0)
ts=[base+i for i in range(0,span,4)]
for tail in (0,1,2,4):
    r=xrefs(ts,tail=tail)
    for t,v in r.items():
        if v:
            print('tail',tail,hex(t),[ (hex(x),hex(bounds(x)[0])) for x in v][:30])
