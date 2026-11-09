# find vtable-like runs in .xcode with pointers to code; print runs having any ptr into [lo,hi)
import sys, struct, numpy as np
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16)
s=[x for x in pe.sections if x.Name.startswith(b'.xcode')][0]
u=[x for x in pe.sections if x.Name.startswith(b'.udata')][0]
cl=IB+u.VirtualAddress; ch=cl+u.Misc_VirtualSize
buf=np.frombuffer(data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData//8*8],dtype='<u8')
base=IB+s.VirtualAddress
isc=(buf>=cl)&(buf<ch)
runs=[];i=0;n=len(buf)
while i<n:
    if isc[i]:
        j=i
        while j<n and isc[j]: j+=1
        if j-i>=8:
            seg=buf[i:j]
            if ((seg>=lo)&(seg<hi)).sum()>=2: runs.append((i,j))
        i=j
    else: i+=1
for i,j in runs:
    print('%x len %d'%(base+8*i, j-i))
