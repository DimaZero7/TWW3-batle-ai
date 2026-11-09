import sys, struct
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB, dis
for a in sys.argv[1:]:
    va=int(a,16); print('==',a, ' | '.join(dis(va,16)[:3]))
    d=struct.pack('<Q',va)
    for s in pe.sections:
        raw=data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData]
        i=raw.find(d)
        while i>=0:
            print('  ptr at %x %s'%(IB+s.VirtualAddress+i, s.Name.strip(b'\0').decode())); i=raw.find(d,i+1)
