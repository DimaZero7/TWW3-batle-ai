"""Minimal reader of WH3 db.pack tables by the RPFM schema (own copy in tools/db)."""
import re, struct, sys, os, json, io
import zstandard
D=os.path.dirname(os.path.abspath(__file__))
PACK=r"C:/Program Files (x86)/Steam/steamapps/common/Total War WARHAMMER III/data/db.pack"
RON=os.path.join(D,'..','ghidra','db','schema_wh3.ron')
_schema=None
def schema():
    global _schema
    if _schema: return _schema
    t=open(RON,encoding='utf8').read()
    out={}
    for m in re.finditer(r'\n        "([a-z0-9_]+)": \[', t):
        name=m.group(1); start=m.end()
        end=t.find('\n        ],', start)
        body=t[start:end]
        vers=[]
        for vm in re.finditer(r'version: (\d+),\s*fields: \[(.*?)\n                \],', body, re.S):
            fields=re.findall(r'name: "([^"]+)",\s*field_type: ([A-Za-z0-9]+)', vm.group(2))
            vers.append((int(vm.group(1)), fields))
        out[name]=vers
    _schema=out; return out
def entries(names):
    got={}
    with open(PACK,'rb') as f:
        magic,kind,pc,ps,fc,fs,ts=struct.unpack('<4s6I',f.read(28))
        ext=20 if kind&0x100 else 0
        f.read(ext); f.read(ps); idx=f.read(fs)
        pos=0; off=28+ext+ps+fs
        for _ in range(fc):
            size=struct.unpack_from('<I',idx,pos)[0]; pos+=4+(4 if kind&0x40 else 0)
            comp=idx[pos]; pos+=1
            e=idx.index(0,pos); name=idx[pos:e].decode('latin1'); pos=e+1
            if name in names:
                f.seek(off); data=f.read(size)
                if comp:
                    data=zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data[4:])).read()
                got[name]=data
            off+=size
    return got
NUM={"I16":("<h",2),"I32":("<i",4),"I64":("<q",8),"F32":("<f",4),"F64":("<d",8),"ColourRGB":("<I",4)}
def rf(b,p,t):
    if t in NUM:
        fm,n=NUM[t]; return struct.unpack_from(fm,b,p)[0], p+n
    if t=='Boolean': return b[p], p+1
    if t=='StringU8':
        n=struct.unpack_from('<H',b,p)[0]; return b[p+2:p+2+n].decode('utf8','replace'), p+2+n
    if t=='StringU16':
        n=struct.unpack_from('<H',b,p)[0]*2; return b[p+2:p+2+n].decode('utf-16-le','replace'), p+2+n
    if t.startswith('Optional'):
        fl=b[p]; p+=1
        if not fl: return None,p
        return rf(b,p,t[len('Optional'):])
    raise ValueError(t)
def table(name):
    key='db\%s\data__'%name
    pk=[k for k in [key]]
    b=entries(set(pk))[key]
    p=0; ver=None
    if b[:4]==b'\xfd\xfe\xfc\xff': p=6+2*struct.unpack_from('<H',b,4)[0]
    if b[p:p+4]==b'\xfc\xfd\xfe\xff': ver=struct.unpack_from('<i',b,p+4)[0]; p+=8
    p+=1
    n=struct.unpack_from('<I',b,p)[0]; p+=4
    cand=[f for v,f in schema()[name] if v==(ver or 0)] or [schema()[name][0][1]]
    fields=cand[0]
    rows=[]
    for _ in range(n):
        r={}
        for fname,t in fields:
            r[fname],p=rf(b,p,t)
        rows.append(r)
    assert p==len(b), (p,len(b))
    return ver, fields, rows
if __name__=='__main__':
    v,f,rows=table(sys.argv[1])
    print('version',v,'rows',len(rows)); print([x[0] for x in f])
    flt=sys.argv[2] if len(sys.argv)>2 else None
    for r in rows:
        if flt is None or re.search(flt, json.dumps(r)): print(json.dumps(r, ensure_ascii=False))
