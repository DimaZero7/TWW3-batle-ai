import sys, re, struct
sys.path.insert(0,'reverse/tools')
from tdis import data,pe,IB,dis
from fn import bounds
print('\n'.join(l[:100] for l in dis(0x1405aec70,48)))
secs={s.Name.rstrip(b'\0').decode():s for s in pe.sections}
s=secs['.udata']
buf=data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData]
base=IB+s.VirtualAddress
hits=[]
for pat in (b'\xb9\x5e\x00\x00\x00', b'\xba\x5e\x00\x00\x00', b'\x41\xb8\x5e\x00\x00\x00', b'\x8d\x4a\x5e', b'\x6a\x5e'):
    for m in re.finditer(re.escape(pat),buf):
        hits.append((m.start(),pat))
print(len(hits))
out=[]
for off,pat in hits:
    va=base+off
    # look for a call within the next 24 bytes
    tail=buf[off+len(pat):off+len(pat)+30]
    for i in range(len(tail)-5):
        if tail[i]==0xe8:
            rel=struct.unpack('<i',tail[i+1:i+5])[0]
            tgt=va+len(pat)+i+5+rel
            out.append((tgt,va))
import collections
c=collections.Counter(t for t,v in out)
print([ (hex(k),n) for k,n in c.most_common(25)])
import pickle
pickle.dump(out,open('reverse/notes/save_bias/calls5e.pkl','wb'))
