import sys,re
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB, dis
from fn import bounds
s=[x for x in pe.sections if x.Name.startswith(b'.udata')][0]
lo=s.PointerToRawData; buf=data[lo:lo+s.SizeOfRawData]
for pat in [rb'\xf6[\x80-\x87]\x74\x01\x00\x00[\x01\x11\x03]', rb'\x41\xf6[\x80-\x87]\x74\x01\x00\x00[\x01\x11\x03]']:
    for m in re.finditer(pat,buf,re.S):
        va=IB+s.VirtualAddress+m.start()
        f=bounds(va)[0]
        if 0x142c00000<f<0x143400000:
            print('%x f=%x'%(va,f), ' | '.join(l.split(':',1)[1].strip() for l in dis(va-14,30)))
