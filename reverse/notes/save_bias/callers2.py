import sys, re, struct
sys.path.insert(0,'reverse/tools')
from tdis import data,pe,IB,dis
from fn import bounds
secs={s.Name.rstrip(b'\0').decode():s for s in pe.sections}
s=secs['.udata']
buf=data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData]
base=IB+s.VirtualAddress
targets={0x1405aec88,0x1405aec80}
# all E8 / E9 calls
import numpy as np
arr=np.frombuffer(buf,dtype=np.uint8)
idx=np.nonzero((arr==0xE8)|(arr==0xE9))[0]
idx=idx[idx<len(buf)-5]
rel=(arr[idx+1].astype(np.int64)|arr[idx+2].astype(np.int64)<<8|arr[idx+3].astype(np.int64)<<16|arr[idx+4].astype(np.int64)<<24)
rel=np.where(rel>=2**31,rel-2**32,rel)
tgt=base+idx+5+rel
res={}
for t in targets:
    res[t]=[int(base+i) for i in idx[tgt==t]]
    print(hex(t),len(res[t]))
# look at the operand of each call
seen=0
for t,cs in res.items():
    for c in cs:
        pre=dis(c-24,24+5)
        # find last line that sets ecx/edx with an immediate
        tx=' | '.join(x.split(': ',1)[1] for x in pre[-5:])
        if '0x5e' in tx or True:
            m=re.findall(r'mov\s+e[cd]x, (0x[0-9a-f]+)',tx)
            print(hex(c),hex(bounds(c)[0]),m[-1:] ,tx[:150] if '0x5e' in tx else '')
