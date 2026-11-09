# dump loc strings from a pack whose key matches regex
import sys, struct, re, io, zstandard
pack=sys.argv[1]; pat=re.compile(sys.argv[2])
with open(pack,'rb') as f:
    magic,kind,pc,ps,fc,fs,ts=struct.unpack('<4s6I',f.read(28))
    ext=20 if kind&0x100 else 0
    f.read(ext); f.read(ps); idx=f.read(fs); pos=0; off=28+ext+ps+fs
    ents=[]
    for _ in range(fc):
        size=struct.unpack_from('<I',idx,pos)[0]; pos+=4+(4 if kind&0x40 else 0)
        comp=idx[pos]; pos+=1
        e=idx.index(0,pos); name=idx[pos:e].decode('latin1'); pos=e+1
        ents.append((name,off,size,comp)); off+=size
    for name,o,size,comp in ents:
        if not name.endswith('.loc'): continue
        f.seek(o); b=f.read(size)
        if comp: b=zstandard.ZstdDecompressor().stream_reader(io.BytesIO(b[4:])).read()
        p=2+4+4; n=struct.unpack_from('<I',b,p)[0]; p+=4
        def s(p):
            l=struct.unpack_from('<H',b,p)[0]; return b[p+2:p+2+2*l].decode('utf-16-le','replace'), p+2+2*l
        for _ in range(n):
            k,p=s(p); v,p=s(p); p+=1
            if pat.search(k): print(k,'=',v.replace('\n',' | '))
