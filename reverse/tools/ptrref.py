import sys, os, struct, re; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdis import pe, data, IB
for a in sys.argv[1:]:
    pat=struct.pack('<Q', int(a,16))
    for m in re.finditer(re.escape(pat), data):
        off=m.start()
        for s in pe.sections:
            if s.PointerToRawData<=off<s.PointerToRawData+s.SizeOfRawData:
                print(a, s.Name.rstrip(b'\0').decode(), hex(IB+s.VirtualAddress+off-s.PointerToRawData))
