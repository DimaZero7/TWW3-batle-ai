"""The campaign's army templates from the game's database -> config/nn/army_templates.json.

CA's army generator (cdir_military_generator_*, the "campaign director") describes an army of a
faction as a set of templates; a template gives ratios by unit group (melee infantry frontline
spears, ranged infantry backline, artillery, ...). A unit belongs to a group
(unit_qualities), a group may have a parent (unit_group_overrides). The same templates serve
the multiplayer army generator (config WH_mp_Empire_land -> WH_Empire_land, ...).

    py -3.14 -m tools.nn.armies.templates      # reads data/db.pack, writes config/nn/army_templates.json

Every table is read whole to the last byte (tools/nn/dbtables.decode). The field names are
ours, from the values: ratio, template, group, parent, config, priority, quality.

Used by tools/nn/armies/generate.py (family_shares): a template's ratios are summed by unit
family - the root of a group's parent chain - over the families the faction's pool has.
"""
import argparse
import json
import sys
from pathlib import Path

from tools import config as project
from tools.nn.armies import pools as P

OUT = project.CONFIG_DIR / "nn" / "army_templates.json"
LAYOUTS = {
    "cdir_military_generator_template_priorities": (1, "config:s priority:i template:s f03:b"),
    "cdir_military_generator_template_ratios": (None, "ratio:i template:s group:s"),
    "cdir_military_generator_unit_group_overrides": (None, "group:s parent:s"),
    "cdir_military_generator_unit_qualities": (2, "group:s unit:s quality:s"),
}


def read(pack):
    """{table: rows} of the generator's tables (Python 3.14+)."""
    from tools.nn import dbtables, gamedb
    names = {dbtables.entry_name(t): t for t in LAYOUTS}
    raw = gamedb.read_entries(pack, set(names))
    assert set(raw) == set(names), f"not in {pack}: {sorted(set(names) - set(raw))}"
    return {names[n]: dbtables.decode(b, names[n], (LAYOUTS[names[n]][0], dbtables._layout(LAYOUTS[names[n]][1])))
            for n, b in raw.items()}


def document(tables, pools, source):
    """The file's content: per pool faction its templates, the groups of its pool's units, the parents."""
    ratios, priorities = {}, {}
    for r in tables["cdir_military_generator_template_ratios"]:
        ratios.setdefault(r["template"], {})[r["group"]] = r["ratio"]
    for r in tables["cdir_military_generator_template_priorities"]:
        priorities.setdefault(r["config"], {})[r["template"]] = r["priority"]
    unit_groups = {}
    for r in tables["cdir_military_generator_unit_qualities"]:
        unit_groups.setdefault(r["unit"], []).append({"group": r["group"], "quality": r["quality"]})
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
                            "the factions of config/nn/pools.json; random armies follow them "
                            "(docs/en/training/armies.md). Written by py -3.14 -m tools.nn.armies.templates.",
            "_source": str(source),
            "parents": {r["group"]: r["parent"] for r in tables["cdir_military_generator_unit_group_overrides"]},
            "factions": factions}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    pack = Path(project.load()["game_dir"]) / "data" / "db.pack"
    pools = P.load(templates_path=False)
    doc = document(read(pack), pools, pack)
    args.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    for faction, f in doc["factions"].items():
        print(faction, f["config"], ", ".join(f["templates"]))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
