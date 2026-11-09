# find 8-byte pointers equal to given addresses in .xcode/.srdata etc.; print location and nearest preceding lea-referenced vtable start offset
import sys, struct, numpy as np
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB
for s in pe.sections:
    nm=s.Name.rstrip(b'\0').decode()
    if nm in ('.udata','.didata','.data1'): continue
    raw=data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData//8*8]
    buf=np.frombuffer(raw,dtype='<u8')
    for a in sys.argv[1:]:
        v=int(a,16)
        for i in np.nonzero(buf==v)[0]:
            print(a, nm, '%x'%(IB+s.VirtualAddress+8*i))
