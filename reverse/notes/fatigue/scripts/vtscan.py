import sys,struct
sys.path.insert(0,r'reverse/tools')
from tdis import read, pe, data, IB, dis
base=0x143919b28
N=48
bv=[struct.unpack_from('<Q',read(base,N*8),i*8)[0] for i in range(N)]
for i,v in enumerate(bv): print(i, hex(i*8), hex(v))
# scan for derived vtables: qword-aligned arrays sharing slots 3..37 mostly with base
import numpy as np
res=[]
for s in pe.sections:
    nm=s.Name.rstrip(b'\0').decode()
    if nm not in ('.srdata','.xcode','.rdata'): continue
    raw=data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData]
    n=len(raw)//8
    arr=np.frombuffer(raw[:n*8],dtype='<u8')
    va0=IB+s.VirtualAddress
    for i in range(n-38):
        # quick filter: slot 3 and 33 identical to base often
        m=0
        for k in (3,5,6,10,11,13,14,16,17,18,22,24,25):
            if arr[i+k]==bv[k]: m+=1
        if m>=8:
            res.append(va0+i*8)
print('derived', len(res))
out=open(sys.argv[1],'w') if len(sys.argv)>1 else sys.stdout
for v in res:
    sl=[struct.unpack_from('<Q',read(v,0x130),j*8)[0] for j in range(38)]
    out.write('%x s60=%x s78=%x s28=%x s0=%x\n'%(v,sl[12],sl[15],sl[5],sl[0]))
