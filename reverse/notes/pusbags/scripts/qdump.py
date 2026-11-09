import sys, struct
sys.path.insert(0, r'reverse/tools')
from tdis import read
a=int(sys.argv[1],16); n=int(sys.argv[2])
b=read(a,8*n)
for i in range(n): print('%x: %x'%(a+8*i, struct.unpack_from('<Q',b,8*i)[0]))
