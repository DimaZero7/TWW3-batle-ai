import re,sys
off={}
for l in open(r'reverse/notes/morale/kv_offsets.txt'):
    o,n,c,v=l.split(); off[int(o,16)]=(n,v)
src=open(sys.argv[1]).read()
funcs=re.split(r'(?=// ===== )',src)
for f in funcs:
    if not f.strip(): continue
    head=f.splitlines()[0]
    hits=set()
    for m in re.finditer(r'0x([0-9a-f]+)\)',f):
        o=int(m.group(1),16)
        if o in off: hits.add(o)
    print(head, len(f.splitlines()), ', '.join('%x=%s(%s)'%(o,off[o][0],off[o][1]) for o in sorted(hits)))
