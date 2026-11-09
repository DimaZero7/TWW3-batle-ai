# find raw byte patterns of [reg+disp32] loads for given disps in executable sections; print VA + disasm
import sys, struct, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB, dis
for a in sys.argv[1:]:
    d=struct.pack('<I',int(a,16))
    for s in pe.sections:
        if not (s.Characteristics & 0x20000000): continue
        raw=data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData]
        i=raw.find(d)
        while i>=0:
            va=IB+s.VirtualAddress+i
            for back in range(2,6):
                try:
                    L=dis(va-back,back+4+8)
                    if L and L[0].startswith('%x'%(va-back)) and (a.lower() in L[0]):
                        print(a, s.Name.strip(b'\0').decode(), L[0], '|', L[1] if len(L)>1 else ''); break
                except Exception: pass
            i=raw.find(d,i+1)
