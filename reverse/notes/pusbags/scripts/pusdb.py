import sys,json;sys.path.insert(0,r'reverse/tools');import db
pat=sys.argv[1].split(',')
tabs=[t for t in db.schema() if any(k in t for k in ("special_abilit","unit_abilit","phase","vortex","_kv_"))]
for t in sorted(tabs):
    t=t[:-7] if t.endswith("_tables") else t
    try: r=db.table(t+'_tables')
    except Exception as e: continue
    rows=r[2] if isinstance(r,tuple) else r
    for x in rows:
        s=json.dumps(x,ensure_ascii=False)
        if any(p in s for p in pat): print(t, s)
