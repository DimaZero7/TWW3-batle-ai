# list file names of a PFH5 pack (index only); usage: packls.py pack regex
import struct, sys, re
p=sys.argv[1]; rx=re.compile(sys.argv[2]) if len(sys.argv)>2 else None
with open(p,'rb') as f:
    magic,kind,pc,ps,fc,fs,ts=struct.unpack('<4s6I',f.read(28))
    ext=20 if kind&0x100 else 0
    f.read(ext); f.read(ps); idx=f.read(fs)
pos=0; n=0; out=[]
for _ in range(fc):
    size=struct.unpack_from('<I',idx,pos)[0]; pos+=4+(4 if kind&0x40 else 0)
    comp=idx[pos]; pos+=1
    e=idx.index(0,pos); name=idx[pos:e].decode('latin1'); pos=e+1
    n+=1
    if rx is None or rx.search(name): out.append((name,size,comp))
print(magic,hex(kind),'files',n,'match',len(out))
for x in out[:int(sys.argv[3]) if len(sys.argv)>3 else 40]: print(*x)
