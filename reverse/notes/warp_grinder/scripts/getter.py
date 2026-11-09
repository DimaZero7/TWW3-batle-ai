import sys,re
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB, dis
s=[x for x in pe.sections if x.Name.startswith(b'.udata')][0]
lo=s.PointerToRawData; buf=data[lo:lo+s.SizeOfRawData]
for pat in [rb'\x8b\x81\x74\x01\x00\x00\xc3', rb'\xf6\x81\x74\x01\x00\x00\x01\x0f\x95\xc0\xc3', rb'\x8b\x81\x74\x01\x00\x00[\x83\x24]', rb'\x0f\xb6\x81\x74\x01\x00\x00']:
    for m in re.finditer(pat,buf,re.S):
        va=IB+s.VirtualAddress+m.start()
        print('%x'%va, ' | '.join(l.split(':',1)[1].strip() for l in dis(va-8,30)))
