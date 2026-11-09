# registration index of a pref: the function that loads the name string also loads r9d = index
import sys, re
sys.path.insert(0,'reverse/tools')
from tdis import dis,IB,data,pe
from xref import xrefs
from fn import bounds
def va_of(s):
    out=[]
    for m in re.finditer(re.escape(s.encode())+b'\0',data):
        off=m.start()
        for sec in pe.sections:
            if sec.PointerToRawData<=off<sec.PointerToRawData+sec.SizeOfRawData and (off==0 or data[off-1]==0):
                out.append(IB+sec.VirtualAddress+off-sec.PointerToRawData)
    return out
for name in sys.argv[1:]:
    for v in va_of(name):
        for x in xrefs([v])[v]:
            s,e=bounds(x)
            if e-s>300: continue
            ls=dis(s,e-s)
            t=[l for l in ls if 'r9d' in l and 'mov' in l]
            print(name,hex(v),hex(x),hex(s),[l.split(': ',1)[1] for l in t][:2])
