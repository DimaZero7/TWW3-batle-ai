import sys, json
sys.path.insert(0, r'reverse/tools')
import db
units=sys.argv[1].split(',')
_,_,lu=db.table('land_units_tables'); lu={r['key']:r for r in lu}
_,_,be=db.table('battle_entities_tables'); be={r['key']:r for r in be}
cols=['run_speed','charge_speed','charge_distance_commence_run','charge_distance_adopt_charge_pose','charge_distance_pick_target','mass','size','hit_reactions_ignore_chance','knock_interrupts_ignore_chance','dealt_collision_knocked_down_threshold_multiplier','dealt_collision_knocked_back_threshold_multiplier','radius']
print('unit', 'charge_bonus', 'can_brace', 'rank_depth', *cols)
for u in units:
    r=lu.get(u)
    if not r: print(u,'?'); continue
    e=be.get(r['man_entity'],{})
    print(u.replace('wh2_main_','').replace('wh_main_','').replace('wh2_dlc16_','').replace('wh_dlc04_',''), r['charge_bonus'], r['can_brace'], r['rank_depth'], *[e.get(c) for c in cols])
