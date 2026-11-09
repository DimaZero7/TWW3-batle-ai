import sys, json
sys.path.insert(0, r'reverse/tools')
import db
def rows(t):
    r=db.table(t+'_tables'); return r[2] if isinstance(r,tuple) else r
A=['wh2_dlc14_unit_abilities_seismic_snare','wh2_dlc14_unit_abilities_warp_quake']
usa=rows('unit_special_abilities'); ua=rows('unit_abilities')
j=rows('special_ability_to_special_ability_phase_junctions')
ph={x['id']:x for x in rows('special_ability_phases')}
se=rows('special_ability_phase_stat_effects'); ae=rows('special_ability_phase_attribute_effects')
for a in A:
    print('=====',a)
    for x in ua:
        if x['key']==a: print('UA',json.dumps(x,ensure_ascii=False))
    for x in usa:
        if x['key']==a: print('USA',json.dumps(x,ensure_ascii=False))
    for x in j:
        if x['special_ability']==a:
            print('J',x); p=x['phase']; print('PHASE',json.dumps(ph.get(p),ensure_ascii=False))
            for y in se:
                if y['phase']==p: print('  STAT',y)
            for y in ae:
                if y['phase']==p: print('  ATTR',y)
for t in ['_kv_unit_ability_scaling_rules']:
    for x in rows(t): print(t, x)
