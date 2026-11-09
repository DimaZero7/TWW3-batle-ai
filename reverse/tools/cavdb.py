# dump DB rows for the Empire cavalry pool (run from the project dir with its venv)
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from sim.passports.dbtables import read_tables
PACK = r"C:/Program Files (x86)/Steam/steamapps/common/Total War WARHAMMER III/data/db.pack"
T = ["main_units", "land_units", "battle_entities", "mounts", "melee_weapons", "unit_attributes_to_groups_junctions",
     "_kv_rules", "unit_spacings", "land_units_to_unit_abilites_junctions", "special_ability_phases",
     "special_ability_phase_stat_effects", "unit_armour_types"]
t = read_tables(PACK, T)
json.dump(t, open(sys.argv[1], 'w', encoding='utf-8'), ensure_ascii=False, default=str)
print({k: len(v) for k, v in t.items()})
