import sys, json
sys.path.insert(0, r'reverse/tools')
import db
t=sys.argv[1]; pat=sys.argv[2] if len(sys.argv)>2 else None
r=db.table(t)
print(type(r), len(r))
hdr=r[1] if isinstance(r,tuple) else None
rows=r[2] if isinstance(r,tuple) else r
print('fields:', [f for f in (rows[0].keys() if rows else [])])
for x in rows:
    if pat is None or any(pat in str(v) for v in x.values()):
        print(json.dumps(x,ensure_ascii=False))
