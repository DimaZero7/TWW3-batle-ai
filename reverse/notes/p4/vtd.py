import sys,os,struct
sys.path.insert(0,'reverse/tools')
from tdis import read, cstr
va=int(sys.argv[1],16); n=int(sys.argv[2]) if len(sys.argv)>2 else 120
ptrs=struct.unpack('<%dQ'%n, read(va,8*n))
for i,p in enumerate(ptrs):
    print('%03x:%x'%(i*8,p), end='  ')
    if i%6==5: print()
print()
