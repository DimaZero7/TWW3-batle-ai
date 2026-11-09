"""The campaign's army templates from the game's database -> sim/config/army_templates.json.

CA's army generator (cdir_military_generator_*, the "campaign director") describes an army of a
faction as a set of templates; a template gives ratios by unit group (melee infantry frontline
spears, ranged infantry backline, artillery, ...). A unit belongs to a group
(unit_qualities), a group may have a parent (unit_group_overrides). The same templates serve
the multiplayer army generator (config WH_mp_Empire_land -> WH_Empire_land, ...).

    py -3.14 -m sim.armies.templates      # reads data/db.pack, writes sim/config/army_templates.json

Every table is read whole to the last byte (sim/passports/dbtables.decode), its columns by the RPFM
schema's names (sim/passports/dbschema.py).

Used by sim/armies/generate.py (family_shares): a template's ratios are summed by unit
family - the root of a group's parent chain - over the families the faction's pool has.
"""
import argparse
import json
import sys
from pathlib import Path

from sim import CONFIG
from sim.passports import game_settings
from sim.armies import pools as P

OUT = CONFIG / "army_templates.json"
TABLES = ("cdir_military_generator_template_priorities", "cdir_military_generator_template_ratios",
          "cdir_military_generator_unit_group_overrides", "cdir_military_generator_unit_qualities")


def read(pack):
    """{table: rows} of the generator's tables (Python 3.14+)."""
    from sim.passports import dbtables
    return dbtables.read_tables(pack, TABLES)


def document(tables, pools, source):
    """The file's content: per pool faction its templates, the groups of its pool's units, the parents."""
    ratios, priorities = {}, {}
    for r in tables["cdir_military_generator_template_ratios"]:
        ratios.setdefault(r["template_key"], {})[r["unit_group_key"]] = r["ratio"]
    for r in tables["cdir_military_generator_template_priorities"]:
        priorities.setdefault(r["config_key"], {})[r["template_key"]] = r["priority"]
    unit_groups = {}
    for r in tables["cdir_military_generator_unit_qualities"]:
        unit_groups.setdefault(r["unit_key"], []).append({"group": r["group_key"], "quality": r["quality"]})
    factions = {}
    for faction, pool in pools.items():
        config = pool.generator
        assert config in priorities, f"{faction}: no generator config {config!r} in the database"
        templates = {t: {"priority": p, "ratios": ratios.get(t, {})} for t, p in sorted(priorities[config].items())}
        units = {u.key: unit_groups.get(u.key, []) for u in (pool.lord,) + pool.units}
        missing = [k for k, g in units.items() if not g and k != pool.lord.key]
        assert not missing, f"{faction}: units in no generator group: {missing}"
        factions[faction] = {"config": config, "templates": templates, "units": units}
    return {"_description": "The campaign's army templates (cdir_military_generator_*, CA's army generator) for "
                            "the factions of sim/config/pools.json; random armies follow them "
                            "(docs/en/training/armies.md). Written by py -3.14 -m sim.armies.templates.",
            "_source": str(source),
            "parents": {r["group"]: r["parent"] for r in tables["cdir_military_generator_unit_group_overrides"]},
            "factions": factions}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    pack = Path(game_settings()["game_dir"]) / "data" / "db.pack"
    pools = P.load(templates_path=False, scheme="v3")      # the v3 pools hold every unit (units_v2, units_v3 too)
    doc = document(read(pack), pools, pack)
    args.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    for faction, f in doc["factions"].items():
        print(faction, f["config"], ", ".join(f["templates"]))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
