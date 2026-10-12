"""The game's own battle rules, read from its database (data/db.pack):

  _kv_morale_tables                 morale rules: the lord's aura, his death,
                                    casualties, flanks, arrows, combat, thresholds;
  _kv_fatigue_tables                fatigue rules;
  _kv_rules_tables                  battle rules: melee hit chance, attack interval, armour roll,
                                    defence in the flank and rear, charge decay;
  land_units_tables                 base melee attack, defence and leadership of the arena's units
                                    (all unit numbers: tools/nn/units.py -> config/nn/units.json);
  unit_experience_bonuses_tables    what a rank adds (leadership, attack, defence, accuracy);
  unit_experience_thresholds_tables experience needed for each rank.
Every battle of the project runs with the required mods (config/mod-dependencies.json: True Sight); their own
_kv_*_tables rows replace the vanilla rows of the same key, as the game loads them (mods after db.pack), and
game_rules.json "_mods" lists each replaced key with its vanilla and modded value.
Writes config/nn/game_rules.json (data for training the network). The game's
files are zstd-compressed: needs Python 3.14+ (compression.zstd).

    py -3.14 -m tools.nn.gamedb
"""
import json
import struct
import sys
from pathlib import Path

from tools import config as project
from tools.nn import dbtables

OUT = project.CONFIG_DIR / "nn" / "game_rules.json"
TABLES = {"morale": r"db\_kv_morale_tables\data__", "fatigue": r"db\_kv_fatigue_tables\data__",
          "battle": r"db\_kv_rules_tables\data__",
          "units": r"db\land_units_tables\data__", "exp_bonus": r"db\unit_experience_bonuses_tables\data__",
          "exp_levels": r"db\unit_experience_thresholds_tables\data__"}
UNIT_KEYS = {"lord": "wh_main_emp_cha_general_0", "spear": "wh_main_emp_inf_spearmen_0",
             "archer": "wh2_dlc13_emp_inf_archers_0"}


def read_entries(pack, names):
    from compression import zstd
    got = {}
    with open(pack, "rb") as f:
        magic, kind, pc, ps, fc, fs, ts = struct.unpack("<4s6I", f.read(28))
        assert magic == b"PFH5"
        ext = 20 if kind & 0x100 else 0
        f.read(ext)
        f.read(ps)
        idx = f.read(fs)
        pos, off = 0, 28 + ext + ps + fs
        for _ in range(fc):
            size = struct.unpack_from("<I", idx, pos)[0]
            pos += 4 + (4 if kind & 0x40 else 0)
            comp = idx[pos]
            pos += 1
            end = idx.index(0, pos)
            name = idx[pos:end].decode("latin1")
            pos = end + 1
            if name in names:
                f.seek(off)
                data = f.read(size)
                got[name] = zstd.decompress(data[4:]) if comp else data
            off += size
    return got


def read_prefixed(pack, prefixes):
    """{entry name: bytes} of the pack's entries whose name starts with one of prefixes (a mod's own tables)."""
    from compression import zstd
    got = {}
    with open(pack, "rb") as f:
        magic, kind, pc, ps, fc, fs, ts = struct.unpack("<4s6I", f.read(28))
        assert magic == b"PFH5"
        ext = 20 if kind & 0x100 else 0
        f.read(ext)
        f.read(ps)
        idx = f.read(fs)
        pos, off = 0, 28 + ext + ps + fs
        for _ in range(fc):
            size = struct.unpack_from("<I", idx, pos)[0]
            pos += 4 + (4 if kind & 0x40 else 0)
            comp = idx[pos]
            pos += 1
            end = idx.index(0, pos)
            name = idx[pos:end].decode("latin1")
            pos = end + 1
            if name.startswith(tuple(prefixes)):
                f.seek(off)
                data = f.read(size)
                got[name] = zstd.decompress(data[4:]) if comp else data
            off += size
    return got


KV_TABLES = {"morale": "_kv_morale", "fatigue": "_kv_fatigue", "battle": "_kv_rules"}    # section -> table
MOD_TABLES = {section: f"db\\{t}_tables\\" for section, t in KV_TABLES.items()}


def mod_rules(settings=None):
    """[(pack, {section: {key: value}})] of the required mods (config/mod-dependencies.json), in load order."""
    settings = settings or project.load()
    deps = json.loads((project.ROOT / "config" / "mod-dependencies.json").read_text(encoding="utf-8"))
    out = []
    for mod in deps["mods"]:
        pack = Path(settings["workshop_dir"]) / mod["workshop_id"] / mod["pack_name"]
        raw = read_prefixed(pack, MOD_TABLES.values())
        rows = {}
        for section, prefix in MOD_TABLES.items():
            for name, b in sorted(raw.items()):
                if name.startswith(prefix):
                    rows.setdefault(section, {}).update(kv_table(b, KV_TABLES[section]))
        out.append((mod["pack_name"], rows))
    return out


def kv_table(b, table="_kv_morale"):
    """Key-value table (the _kv_*_tables: key, value; the RPFM schema's names): {key: value rounded to 4 digits}."""
    return {r["key"]: round(r["value"], 4) for r in dbtables.decode(b, table, digits=None)}


def exp_bonuses(b):
    """unit_experience_bonuses (the RPFM schema's names: stat, value, growth_rate, growth_scalar) per stat, written as
    {"flag": value, "a": growth_rate, "b": growth_scalar} (the keys config/nn/game_rules.json has always had).

    How the engine applies them is not verified (leadership has growth_rate 0, growth_scalar 1.06: it looks
    like a flat +1.06 a rank; melee attack/defence 0.6 and 0.12)."""
    return {r["stat"]: {"flag": r["value"], "a": round(r["growth_rate"], 4), "b": round(r["growth_scalar"], 4)}
            for r in dbtables.decode(b, "unit_experience_bonuses", digits=None)}


def exp_levels(b):
    """unit_experience_thresholds (key, value): the experience a rank needs, the land units' rows."""
    rows = dbtables.decode(b, "unit_experience_thresholds")
    return {r["key"]: r["value"] for r in rows if r["key"].startswith("land_")}


def unit_row(rows, key):
    """Melee attack, melee defence and leadership of a unit: land_units decoded whole (tools/nn/dbtables.py)."""
    row = next(r for r in rows if r["key"] == key)
    return {"melee_attack": row["melee_attack"], "melee_defence": row["melee_defence"], "leadership": row["morale"]}


def card_melee(key):
    """Base melee attack and defence on the unit's card (data/roster/<key>.json)."""
    stats = json.loads((project.ROOT / "data" / "roster" / f"{key}.json").read_text(encoding="utf-8"))["card"]["stats"]
    return stats["stat_melee_attack"]["base"], stats["stat_melee_defence"]["base"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    game = Path(project.load()["game_dir"]) / "data" / "db.pack"
    raw = read_entries(game, set(TABLES.values()))
    land_units = dbtables.decode(raw[TABLES["units"]], "land_units")
    rules = {"_source": str(game), **{s: kv_table(raw[TABLES[s]], t) for s, t in KV_TABLES.items()},
             "experience_bonus": exp_bonuses(raw[TABLES["exp_bonus"]]),
             "experience_levels": exp_levels(raw[TABLES["exp_levels"]]),
             "units": {t: dict(unit_row(land_units, k), key=k) for t, k in UNIT_KEYS.items()}}
    for t, u in rules["units"].items():
        assert (u["melee_attack"], u["melee_defence"]) == card_melee(u["key"]), (t, u)
    # the required mods' rows over the vanilla ones (True Sight: the line-of-fire checks and the melee thresholds)
    rules["_mods"] = {}
    for pack, sections in mod_rules():
        for section, rows in sections.items():
            for key, value in rows.items():
                rules["_mods"].setdefault(pack, {}).setdefault(section, {})[key] = {
                    "vanilla": rules[section].get(key), "modded": value}
                rules[section][key] = value
    OUT.write_text(json.dumps(rules, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(rules["units"], ensure_ascii=False), json.dumps(rules["experience_bonus"]))
    print("written", OUT)


if __name__ == "__main__":
    main()
