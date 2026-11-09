import sys, collections
sys.path.insert(0, r'reverse/tools')
import db
_,_,rows=db.table('projectiles_tables')
c=collections.Counter((r['category'],r['trajectory_sight'],r['can_damage_allies']) for r in rows)
for k,v in sorted(c.items(), key=lambda x:str(x)): print(k,v)
for pat in sys.argv[1:]:
    for r in rows:
        if pat in r['key']: print(r['key'],r['category'],r['shot_type'],r['trajectory_sight'],r['can_damage_allies'],r['fixed_elevation'],r['max_elevation'],r['muzzle_velocity'],r['collision_radius'],r['shots_per_volley'],r['projectile_penetration'])
