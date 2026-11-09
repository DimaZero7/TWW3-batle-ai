import sys, json
sys.path.insert(0, r'reverse/tools')
import db
def rows(t):
    r=db.table(t+'_tables'); return r[2] if isinstance(r,tuple) else r
A=['wh2_dlc14_unit_abilities_seismic_snare','wh2_dlc14_unit_abilities_warp_quake','wh3_dlc29_unit_abilities_stormfiend_seismic_snare']
for t in ['special_ability_to_invalid_target_flags','special_ability_to_invalid_usage_flags','special_ability_to_auto_deactivate_flags','special_ability_behaviour_to_ability_junctions','special_ability_groups_to_unit_abilities_junctions']:
    try: R=rows(t)
    except Exception as e: print(t,'ERR',e); continue
    for x in R:
        if any(a in json.dumps(x) for a in A): print(t,x)
# ai usage
for t in ['battle_ai_abilities_usage_groups_to_templates','battle_ai_abilities_usage_templates_to_params','battle_ai_abilities_usage_params','battle_ai_abilities_usage_group_names','battle_ai_abilities_usage_template_names']:
    try: R=rows(t)
    except Exception as e: print(t,'ERR',e); continue
    print('##',t,len(R), R[:2])
