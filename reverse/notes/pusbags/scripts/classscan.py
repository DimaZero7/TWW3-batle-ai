import sys,json;sys.path.insert(0,r'reverse/tools');import db
def rows(t):
    r=db.table(t+'_tables'); return r[2] if isinstance(r,tuple) else r
pool=set(open('pool_keys.txt').read().split()); queue=set(open('queue_keys.txt').read().split())
units=pool|queue
lu=rows('land_units_to_unit_abilites_junctions')
usa={x['key']:x for x in rows('unit_special_abilities')}
j=rows('special_ability_to_special_ability_phase_junctions')
ph={x['id']:x for x in rows('special_ability_phases')}
se=rows('special_ability_phase_stat_effects')
deact={}
for x in rows('special_ability_to_auto_deactivate_flags'): deact.setdefault(x['special_ability'],[]).append(x['deactivate_flag'])
# main_units -> land_unit
mu={x['unit']:x['land_unit'] for x in rows('main_units')}
out=[]
for u in sorted(units):
    lu_key=mu.get(u,u)
    abil=[x['ability'] for x in lu if x['land_unit']==lu_key]
    for a in abil:
        s=usa.get(a)
        if not s: continue
        tags=[]
        if s.get('behaviour') and 'vortex' in str(s['behaviour']): tags.append('behaviour=%s vortex=%s'%(s['behaviour'],s['vortex']))
        elif s.get('vortex'): tags.append('vortex=%s'%s['vortex'])
        if s.get('passive_aoe') or s.get('active_aoe') or (s.get('effect_range') or 0)>0: tags.append('aura range=%s passive_aoe=%s'%(s.get('effect_range'),s.get('passive_aoe')))
        for jj in j:
            if jj['special_ability']!=a: continue
            p=ph.get(jj['phase'])
            if not p: continue
            if (p.get('damage_amount') or 0)>0: tags.append('phase damage %s x%s every %ss self=%s friends=%s enemies=%s'%(p['damage_amount'],p['max_damaged_entities'],p['hp_change_frequency'],jj['target_self'],jj['target_friends'],jj['target_enemies']))
            mor=[x for x in se if x['phase']==jj['phase'] and x['stat']=='stat_morale']
            if mor and (jj['target_enemies']): tags.append('enemy morale %s'%[m['value'] for m in mor])
        if tags: out.append('%s %s %s | %s | deact=%s passive=%s'%('POOL' if u in pool else 'QUEUE',u,a,'; '.join(tags),deact.get(a),s.get('passive')))
print('\n'.join(out))
