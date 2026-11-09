import sys, pickle, json
idx,calls=pickle.load(open(sys.argv[1],'rb'))
kv=json.load(open(r'reverse/tools/kv_value_offsets.json'))
for a in sys.argv[2:]:
    d=int(a,16)
    for ins,f,t in idx.get(d,[]):
        if 'lea' in t: continue
        print('%-6s %-45s %x f=%x %s'%(a,kv.get(a,''),ins,f,t))
