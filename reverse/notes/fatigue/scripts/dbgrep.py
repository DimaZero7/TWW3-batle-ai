import sys,struct,io,zstandard
sys.path.insert(0,r'reverse/tools')
import db
pats=[p.encode() for p in sys.argv[1:]]+[p.encode('utf-16-le') for p in sys.argv[1:]]
with open(db.PACK,'rb') as f:
    magic,kind,pc,ps,fc,fs,ts=struct.unpack('<4s6I',f.read(28))
    ext=20 if kind&0x100 else 0
    f.read(ext); f.read(ps); idx=f.read(fs)
    pos=0; off=28+ext+ps+fs
    for _ in range(fc):
        size=struct.unpack_from('<I',idx,pos)[0]; pos+=4+(4 if kind&0x40 else 0)
        comp=idx[pos]; pos+=1
        e=idx.index(0,pos); name=idx[pos:e].decode('latin1'); pos=e+1
        f.seek(off); data=f.read(size)
        if comp: data=zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data[4:])).read()
        for p in pats:
            c=data.count(p)
            if c: print(name, p[:40], c)
        off+=size
