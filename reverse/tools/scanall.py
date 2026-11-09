# scan every db table in db.pack for rows containing any of the patterns (regex)
import sys, re, json, struct, io
sys.path.insert(0, r'reverse/tools')
import db, zstandard
pat=re.compile(sys.argv[1]); out=open(sys.argv[2],'w',encoding='utf8')
# list all entries
names=[]
with open(db.PACK,'rb') as f:
    magic,kind,pc,ps,fc,fs,ts=struct.unpack('<4s6I',f.read(28))
    ext=20 if kind&0x100 else 0
    f.read(ext); f.read(ps); idx=f.read(fs); pos=0
    for _ in range(fc):
        size=struct.unpack_from('<I',idx,pos)[0]; pos+=4+(4 if kind&0x40 else 0)
        comp=idx[pos]; pos+=1
        e=idx.index(0,pos); name=idx[pos:e].decode('latin1'); pos=e+1
        names.append(name)
tabs=sorted(set(n.split(chr(92))[1] for n in names if n.startswith('db'+chr(92)) and n.endswith('data__')))
print(len(tabs))
for t in tabs:
    try: v,f,rows=db.table(t)
    except Exception as ex:
        print('ERR',t,str(ex)[:80]); continue
    for r in rows:
        s=json.dumps(r,ensure_ascii=False)
        if pat.search(s): out.write(t+'\t'+s+'\n')
out.close()
