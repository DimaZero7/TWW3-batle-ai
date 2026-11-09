import sys,json,os;sys.path.insert(0,r'reverse/tools');import db
D=os.path.dirname(os.path.abspath(__file__))
def rows(t):
    r=db.table(t+'_tables'); return r[2] if isinstance(r,tuple) else r
units=set(open(D+'/pool_keys.txt').read().split())|set(open(D+'/queue_keys.txt').read().split())
mu={x['unit']:x['land_unit'] for x in rows('main_units')}
lus={x['key']:x for x in rows('land_units')}
g=rows('unit_attributes_to_groups_junctions')
print(list(g[0].keys()), list(next(iter(lus.values())).keys())[:80])
ga={}
for x in g: ga.setdefault(x['attribute_group'],set()).add(x['attribute'])
for u in sorted(units):
    l=lus.get(mu.get(u,u)); 
    if not l: continue
    a=ga.get(l['attribute_group'],set())
    hit=[z for z in a if 'invulnerable' in z or 'ignore_imbue' in z or 'flying' in z]
    res={k:l[k] for k in ['damage_mod_magic','damage_mod_all'] if l.get(k)}
    if hit or res: print(u, hit, res)
