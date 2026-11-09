import sys, json
sys.path.insert(0, r'reverse/tools')
import db
def rows(t):
    r=db.table(t+'_tables'); return r[2] if isinstance(r,tuple) else r
out=open(sys.argv[1],'w',encoding='utf8')
def P(*a): print(*a,file=out)
U=['wh2_main_skv_mon_rat_ogres','wh2_dlc16_skv_mon_brood_horror_0','wh2_main_skv_mon_hell_pit_abomination','wh2_dlc16_skv_mon_rat_ogre_mutant']
lu={x['key']:x for x in rows('land_units')}
mu=[x for x in rows('main_units') if x.get('land_unit') in U]
be={x['key']:x for x in rows('battle_entities')}
mw={x['key']:x for x in rows('melee_weapons')}
attj=rows('unit_attributes_to_groups_junctions')
abj=rows('land_units_to_unit_abilites_junctions')
for k in U:
    l=lu[k]; P('=== land_units',k); P(json.dumps(l,ensure_ascii=False))
    P('--- battle_entity'); P(json.dumps(be.get(l['man_entity']),ensure_ascii=False))
    P('--- melee_weapon'); P(json.dumps(mw.get(l['primary_melee_weapon']),ensure_ascii=False))
    P('--- attributes', [a for a in attj if l['attribute_group'] in json.dumps(a)])
    P('--- abilities', [a for a in abj if k in json.dumps(a)])
for m in mu: P('=== main_units', json.dumps(m,ensure_ascii=False))
