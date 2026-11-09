import sys, json
sys.path.insert(0, r'reverse/tools')
import db
s=db.schema()
for t in ['battle_context_unit_ability_junctions','special_ability_to_auto_deactivate_flags','special_ability_to_invalid_usage_flags','unit_special_abilities','special_ability_phases']:
    try:
        r=db.table(t+'_tables'); rows=r[2] if isinstance(r,tuple) else r
    except Exception as e: print(t,'ERR',e); continue
    for x in rows:
        st=json.dumps(x)
        if 'single_entity' in st or 'regeneration' in st and t!='special_ability_phases':
            print(t, st[:3000])
