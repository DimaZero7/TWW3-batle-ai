import sys,re
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB, dis
s=[x for x in pe.sections if x.Name.startswith(b'.udata')][0]
lo=s.PointerToRawData; hi=lo+s.SizeOfRawData
buf=data[lo:hi]
pat=re.compile(rb'(?:\xf6[\x80-\x87\x40-\x47]|\xf7[\x80-\x87]|\x8b[\x80-\xbf]|\x0f\xb6[\x80-\xbf])\x74\x01\x00\x00', re.S)
hits=[]
for m in pat.finditer(buf):
    o=m.start()
    win=buf[max(0,o-60):o]
    if b'\x48\x3d\x00\x00' in win:
        va=IB+s.VirtualAddress+o
        hits.append(va)
print(len(hits))
for va in hits:
    ls=dis(va-60,90)
    # print only from instruction that contains 3d48
    out=[l for l in ls]
    print('---- %x'%va)
    for l in out:
        a=int(l.split(':')[0],16)
        if va-40<=a<=va+12: print(l)
