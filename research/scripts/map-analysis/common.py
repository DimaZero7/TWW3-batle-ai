from pathlib import Path
import sys,json,re,struct,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'crossroads'))
from inspect_files import index,extract,GAME
SCHEMA=(HERE.parent/'crossroads/schema_wh3.ron').read_text(encoding='utf-8')
def decode_db(e):
    raw=extract(e);pos=0
    def read(fmt):
        nonlocal pos
        v=struct.unpack_from('<'+fmt,raw,pos)[0];pos+=struct.calcsize('<'+fmt);return v
    def string(enc='utf-8'):
        nonlocal pos
        n=read('H')*(2 if enc=='utf-16-le' else 1);v=raw[pos:pos+n].decode(enc);pos+=n;return v
    def boolean():
        v=read('B');assert v in (0,1);return bool(v)
    assert raw[:4]==bytes.fromhex('fd fe fc ff');pos=4;guid=string('utf-16-le')
    version=0
    if raw[pos:pos+4]==bytes.fromhex('fc fd fe ff'):pos+=4;version=read('I')
    flag=boolean();count=read('I');header=raw[:pos]
    table=e['path'].split('/')[1];start=SCHEMA.index('"'+table+'"');end=SCHEMA.find('\n        "',start+1)
    section=SCHEMA[start:end];a=section.index('version: '+str(version)+',');b=section.index('localised_fields:',a)
    fields=re.findall(r'name: "([^"]+)",\s+field_type: ([^,]+),',section[a:b]);rows=[]
    fmts={'Boolean':'B','I32':'i','F32':'f','I64':'q','F64':'d'}
    for _ in range(count):
        row={}
        for name,kind in fields:
            if kind=='StringU8':v=string()
            elif kind=='OptionalStringU8':v=string() if boolean() else None
            elif kind=='Boolean':v=boolean()
            else:v=read(fmts[kind])
            row[name]=v
        rows.append(row)
    assert pos==len(raw),(table,pos,len(raw))
    encoded=bytearray(header)
    for row in rows:
        for name,kind in fields:
            v=row[name]
            if kind=='OptionalStringU8':
                encoded+=struct.pack('<B',v is not None)
                if v is None:continue
                kind='StringU8'
            if kind=='StringU8':
                bs=v.encode();encoded+=struct.pack('<H',len(bs))+bs
            else:encoded+=struct.pack('<'+fmts[kind],v)
    assert bytes(encoded)==raw
    return dict(source=e,version=version,fields=fields,rows=rows,sha256=hashlib.sha256(raw).hexdigest(),roundtrip=True)
if __name__=='__main__':
    entries=[]
    for pack in GAME.glob('*.pack'):
        if pack.name.startswith(('audio','movies','local','models','variants','variant','vfx','terrain_textures','texture','tile','terrain_camp')):continue
        try:items=index(pack)
        except AssertionError:continue
        entries+=items
    (HERE/'index.json').write_text(json.dumps(entries))
    db=[e for e in entries if e['path'].startswith('db/battles_tables/')]
    rows=[]
    for e in db:
        d=decode_db(e);rows+=d['rows']
    (HERE/'battles.json').write_text(json.dumps(rows,indent=2))
    picks=[r for r in rows if re.search('river|bridge|ford|marsh|swamp',str(r),re.I)]
    (HERE/'water-map-candidates.json').write_text(json.dumps(picks,indent=2))
    print('entries',len(entries),'battles',len(rows),'candidates',len(picks))
    for r in picks[:35]:print(json.dumps(r))
