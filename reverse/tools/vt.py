import sys, os, struct; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdis import read
for a in sys.argv[1:]:
    va=int(a,16); n=12
    ptrs=struct.unpack('<%dQ'%n, read(va, 8*n))
    print(a, ' '.join('%x:%x'%(i*8,p) for i,p in enumerate(ptrs)))
