import sys,re,struct
sys.path.insert(0,'reverse/tools')
from tdis import data,pe,IB
for s in sys.argv[1:]:
    for m in re.finditer(re.escape(s.encode())+b'\0',data):
        off=m.start()
        for sec in pe.sections:
            if sec.PointerToRawData<=off<sec.PointerToRawData+sec.SizeOfRawData:
                va=IB+sec.VirtualAddress+off-sec.PointerToRawData
                print(s,sec.Name.rstrip(b'\0').decode(),hex(va))
                pat=struct.pack('<Q',va)
                for m2 in re.finditer(re.escape(pat),data):
                    o2=m2.start()
                    for s2 in pe.sections:
                        if s2.PointerToRawData<=o2<s2.PointerToRawData+s2.SizeOfRawData:
                            print('   ptr at',s2.Name.rstrip(b'\0').decode(),hex(IB+s2.VirtualAddress+o2-s2.PointerToRawData))
