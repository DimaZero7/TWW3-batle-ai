import sys,re,struct
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB, read, dis
x=[s for s in pe.sections if s.Name.startswith(b'.xcode')][0]
raw=data[x.PointerToRawData:x.PointerToRawData+x.SizeOfRawData]; xb=IB+x.VirtualAddress
names={}
for m in re.finditer(rb'[A-Z_]{3,}_STATE\0', raw):
    names[xb+m.start()]=m.group()[:-1].decode()
u=[s for s in pe.sections if s.Name.startswith(b'.udata')][0]
code=data[u.PointerToRawData:u.PointerToRawData+u.SizeOfRawData]; ub=IB+u.VirtualAddress
# pattern: 48 8d 05 rel32 c3
fn={}
for m in re.finditer(rb'\x48\x8d\x05(....)\xc3', code, re.S):
    a=ub+m.start(); t=a+7+struct.unpack('<i',m.group(1))[0]
    if t in names: fn[a]=names[t]
# find vtables containing these fns at slot0
xa=__import__('numpy').frombuffer(raw[:len(raw)//8*8],dtype='<u8')
import numpy as np
for a,n in sorted(fn.items(), key=lambda z:z[1]):
    for i in np.nonzero(xa==a)[0]:
        vt=xb+8*i
        sl=[struct.unpack('<Q',read(vt+8*k,8))[0] for k in range(1,4)]
        print(n, '%x'%vt, ' '.join('%x'%s for s in sl))
