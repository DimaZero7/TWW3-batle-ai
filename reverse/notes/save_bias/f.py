import sys, re, struct
sys.path.insert(0,'reverse/tools')
from tdis import data,pe,IB,dis
from fn import bounds
import numpy as np
secs={s.Name.rstrip(b'\0').decode():s for s in pe.sections}
s=secs['.udata']
buf=data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData]
base=IB+s.VirtualAddress
arr=np.frombuffer(buf,dtype=np.uint8)
idx=np.nonzero((arr==0xE8)|(arr==0xE9))[0]
idx=idx[idx<len(buf)-5]
rel=(arr[idx+1].astype(np.int64)|arr[idx+2].astype(np.int64)<<8|arr[idx+3].astype(np.int64)<<16|arr[idx+4].astype(np.int64)<<24)
rel=np.where(rel>=2**31,rel-2**32,rel)
tgt=base+idx+5+rel
for a in sys.argv[1:]:
    t=int(a,16)
    cs=[int(base+i) for i in idx[tgt==t]]
    print(hex(t),'bounds',[hex(x) for x in bounds(t)],'callers',[(hex(c),hex(bounds(c)[0])) for c in cs][:40])
# xrefs across all sections to the pref's value
from xref import xrefs
for name in secs:
    if name in ('.xcode','.didata'): continue
    try:
        r=xrefs([0x143f9b930,0x143f9b934,0x143f9b938,0x143f9b8d0+0x48],secname=name)
    except Exception as e:
        print(name,'err',e); continue
    print(name,{hex(k):[hex(x) for x in v][:8] for k,v in r.items() if v})
