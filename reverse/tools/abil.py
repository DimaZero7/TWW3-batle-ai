import sys, json
sys.path.insert(0, r'reverse/tools')
import db
def rows(t):
    r=db.table(t+'_tables'); return r[2] if isinstance(r,tuple) else r
A=['wh_main_unit_passive_regeneration','wh3_main_unit_passive_single_entity','wh_main_unit_passive_frenzy','wh2_main_unit_passive_scurry_away']
out=open(sys.argv[1],'w',encoding='utf8')
def P(*a): print(*a,file=out)
usa=rows('unit_special_abilities'); ua=rows('unit_abilities')
j=rows('special_ability_to_special_ability_phase_junctions')
ph={x['id']:x for x in rows('special_ability_phases')}
se=rows('special_ability_phase_stat_effects'); ae=rows('special_ability_phase_attribute_effects')
PH=set(['wh_main_unit_contact_poison'])
for a in A:
    P('=====',a)
    for x in ua:
        if x['key']==a: P('UA',json.dumps(x,ensure_ascii=False))
    for x in usa:
        if x['key']==a: P('USA',json.dumps(x,ensure_ascii=False))
    for x in j:
        if x['special_ability']==a: P('J',x); PH.add(x['phase'])
for p in sorted(PH):
    P('===== PHASE',p, json.dumps(ph.get(p),ensure_ascii=False))
    for x in se:
        if x['phase']==p: P('  STAT',x)
    for x in ae:
        if x['phase']==p: P('  ATTR',x)
for t in ['_kv_rules','_kv_morale','_kv_fatigue']:
    for x in rows(t):
        s=json.dumps(x)
        if any(w in s for w in ['regen','heal','replen','fear','terror','single','entity','large','frenz','poison','magic','ether','lone','vigour','mass','collision','splash','monster']):
            P(t,s)
