import sys, json
L=[l.rstrip('\n').split('\t',1) for l in open(sys.argv[1],encoding='utf8')]
tab=sys.argv[2]; keys=sys.argv[3].split(',') if len(sys.argv)>3 else None
for t,s in L:
    if t!=tab: continue
    r=json.loads(s)
    if keys and not any(any(k==str(v) for v in r.values()) for k in keys): continue
    print(json.dumps(r,ensure_ascii=False))
