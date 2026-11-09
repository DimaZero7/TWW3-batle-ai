import sys, json
sys.path.insert(0, r'reverse/tools')
import db
def rows(t):
    r=db.table(t+'_tables'); return r[2] if isinstance(r,tuple) else r
for x in rows('_kv_battle_ai_ability_usage_variables'): print(x)
