import sys,json,os;sys.path.insert(0,r'reverse/tools');import db
D=os.path.dirname(os.path.abspath(__file__))
def rows(t):
    r=db.table(t+'_tables'); return r[2] if isinstance(r,tuple) else r
pool=set(open(D+'/pool_keys.txt').read().split()); queue=set(open(D+'/queue_keys.txt').read().split())
lu=rows('land_units_to_unit_abilites_junctions')
usa={x['key']:x for x in rows('unit_special_abilities')}
ua={x['key']:x for x in rows('unit_abilities')}
j=rows('special_ability_to_special_ability_phase_junctions')
ph={x['id']:x for x in rows('special_ability_phases')}
inv={}
for x in rows('special_ability_to_invalid_target_flags'): inv.setdefault(x['special_ability'],[]).append(x['invalid_target'])
mu={x['unit']:x['land_unit'] for x in rows('main_units')}
dmgph={k for k,p in ph.items() if (p.get('damage_amount') or 0)>0}
for u in sorted(pool|queue):
    for a in [x['ability'] for x in lu if x['land_unit']==mu.get(u,u)]:
        s=usa.get(a); 
        if not s: continue
        for jj in j:
            if jj['special_ability']==a and jj['phase'] in dmgph:
                p=ph[jj['phase']]
                print('%-5s %s | %s type=%s src=%s | dmg %s x%s every %ss dur %s | self=%s fr=%s en=%s | range=%s nEn=%s every_frame=%s | magical=%s | inv=%s ai=%s'%(
                  'POOL' if u in pool else 'QUEUE',u,a,ua.get(a,{}).get('type'),ua.get(a,{}).get('source_type'),p['damage_amount'],p['max_damaged_entities'],p['hp_change_frequency'],p['duration'],
                  jj['target_self'],jj['target_friends'],jj['target_enemies'],s['effect_range'],s['num_effected_enemy_units'],s['update_targets_every_frame'],p['imbue_magical'],inv.get(a),s['ai_usage']))
# whole DB: count of enemy-target direct damage phases by type
from collections import Counter
c=Counter(); 
for jj in j:
    if jj['phase'] in dmgph: c[(ua.get(jj['special_ability'],{}).get('type'), 'en' if jj['target_enemies'] else ('self' if jj['target_self'] else 'fr'))]+=1
for k,v in c.most_common(20): print('ALL',k,v)
# who has invulnerable_to_effects_enemy among pool/queue
ua_attr=rows('unit_attributes_to_groups_junctions') if 'unit_attributes_to_groups_junctions_tables' in db.schema() else []
