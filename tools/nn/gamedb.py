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


def _skip_header(b):
    p = 0
    if b[p:p + 4] == b"\xfd\xfe\xfc\xff":
        n = struct.unpack_from("<H", b, p + 4)[0]
        p += 6 + 2 * n
    if b[p:p + 4] == b"\xfc\xfd\xfe\xff":
        p += 8
    return p + 1


def kv_table(b):
    """Key-value table: rows of (string, float32)."""
    p = _skip_header(b)
    rows = struct.unpack_from("<I", b, p)[0]
    p += 4
    out = {}
    for _ in range(rows):
        n = struct.unpack_from("<H", b, p)[0]
        key = b[p + 2:p + 2 + n].decode("utf-8")
        p += 2 + n
        out[key] = round(struct.unpack_from("<f", b, p)[0], 4)
        p += 4
    return out


def exp_bonuses(b):
    """unit_experience_bonuses: stat, (i32 flag), float a, float b per rank.

    How the engine applies a and b is not verified (leadership has a = 0, b = 1.06: it looks
    like a flat +1.06 a rank; melee attack/defence a = 0.6, b = 0.12)."""
    p = _skip_header(b)
    rows = struct.unpack_from("<I", b, p)[0]
    p += 4
    out = {}
    for _ in range(rows):
        n = struct.unpack_from("<H", b, p)[0]
        key = b[p + 2:p + 2 + n].decode("utf-8")
        p += 2 + n
        flag, a, c = struct.unpack_from("<iff", b, p)
        p += 12
        out[key] = {"flag": flag, "a": round(a, 4), "b": round(c, 4)}
    return out


def exp_levels(b):
    p = _skip_header(b)
    rows = struct.unpack_from("<I", b, p)[0]
    p += 4
    out = {}
    for _ in range(rows):
        n = struct.unpack_from("<H", b, p)[0]
        key = b[p + 2:p + 2 + n].decode("utf-8")
        p += 2 + n
        out[key] = struct.unpack_from("<i", b, p)[0]
        p += 4
    return {k: v for k, v in out.items() if k.startswith("land_")}


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
    rules = {"_source": str(game), "morale": kv_table(raw[TABLES["morale"]]),
             "fatigue": kv_table(raw[TABLES["fatigue"]]), "battle": kv_table(raw[TABLES["battle"]]),
             "experience_bonus": exp_bonuses(raw[TABLES["exp_bonus"]]),
             "experience_levels": exp_levels(raw[TABLES["exp_levels"]]),
             "units": {t: dict(unit_row(land_units, k), key=k) for t, k in UNIT_KEYS.items()}}
    for t, u in rules["units"].items():
        assert (u["melee_attack"], u["melee_defence"]) == card_melee(u["key"]), (t, u)
    OUT.write_text(json.dumps(rules, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rules["units"], ensure_ascii=False), json.dumps(rules["experience_bonus"]))
    print("written", OUT)


if __name__ == "__main__":
    main()
