import sys, re, struct
sys.path.insert(0,'reverse/tools')
from tdis import data,pe,IB,dis
from fn import bounds
secs={s.Name.rstrip(b'\0').decode():s for s in pe.sections}
s=secs['.udata']
buf=data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData]
base=IB+s.VirtualAddress
hits=[]
for pat in (b'\xba\x5e\x00\x00\x00', b'\x8d\x51\x5e', b'\x8d\x50\x5e', b'\x8d\x53\x5e', b'\x8d\x56\x5e', b'\x8d\x57\x5e', b'\x8d\x55\x5e', b'\x41\x8d\x51\x5e', b'\x41\x8d\x56\x5e', b'\x41\x8d\x57\x5e', b'\x41\x8d\x55\x5e', b'\x41\x8d\x54\x24\x5e', b'\x6a\x5e\x5a'):
    for m in re.finditer(re.escape(pat),buf):
        off=m.start()
        tail=buf[off+len(pat):off+len(pat)+24]
        # an indirect call with displacement 0x18 / 0x20 / 0x10 / 0x28 etc. soon after
        if re.search(rb'\xff[\x50-\x57\x10-\x17][\x08\x10\x18\x20\x28\x30\x38\x40\x48\x50]', tail) or re.search(rb'\xff\x15', tail):
            va=base+off
            hits.append(va)
print(len(hits))
seen=set()
for va in hits:
    f=bounds(va)[0]
    ls=dis(va,len(b'12345')+24)
    tx=' | '.join(x.split(': ',1)[1] for x in ls[:5])
    print(hex(va),'func',hex(f),tx[:160])
