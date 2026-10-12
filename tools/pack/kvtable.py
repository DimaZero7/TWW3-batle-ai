"""Key-value rule tables of the game's database (_kv_morale_tables, _kv_rules_tables, _kv_fatigue_tables) for a
probe pack that overrides some of their rows: the way the game's mods change a rule (the required True Sight
mod ships db\\_kv_rules_tables\\!true_sight with only its 9 rows). A mod's table of the same table folder whose
name sorts before the game's data__ ('!' < 'd') is read first and its rows win; the other rows stay the game's.

Format (the game's own tables and RPFM's, tools/nn/gamedb.kv_table reads it): the GUID marker FD FE FC FF, the
table's GUID (u16 length, UTF-16LE), no version marker (kv tables are version 0), a byte 01, the row count (u32),
then each row: its key (u16 length, UTF-8) and its value (f32).

Only for probes (tools/build.py --kv): an ordinary build never carries one.
"""
import json
import struct
import uuid

from tools import config as project

TABLES = {"morale": "_kv_morale_tables", "battle": "_kv_rules_tables", "fatigue": "_kv_fatigue_tables"}
TABLE_NAME = "!tww3_bai_probe_kv"     # sorts before data__; no digit at the end (modders: that can crash)
GUID_MARKER = b"\xfd\xfe\xfc\xff"
GAME_RULES = project.CONFIG_DIR / "nn" / "game_rules.json"


def table_bytes(rows, guid):
    """{key: value} -> the table file's bytes (rows sorted by key)."""
    g = str(guid).encode("utf-16-le")
    out = bytearray(GUID_MARKER + struct.pack("<H", len(g) // 2) + g + b"\x01" + struct.pack("<I", len(rows)))
    for key, value in sorted(rows.items()):
        k = key.encode("utf-8")
        out += struct.pack("<H", len(k)) + k + struct.pack("<f", float(value))
    return bytes(out)


def entries(overrides):
    """{section: {key: value}} -> {pack path: bytes}; the GUID is derived from the rows (equal inputs, equal bytes)."""
    out = {}
    for section, rows in sorted(overrides.items()):
        if not rows:
            continue
        guid = uuid.uuid5(uuid.NAMESPACE_URL, "tww3_bai/" + TABLES[section] + json.dumps(rows, sort_keys=True))
        out[f"db\\{TABLES[section]}\\{TABLE_NAME}"] = table_bytes(rows, guid)
    return out


def known_keys(path=GAME_RULES):
    """{section: {key: value}} of the game's rows (config/nn/game_rules.json, written by tools/nn/gamedb.py)."""
    rules = json.loads(path.read_text(encoding="utf-8"))
    return {section: rules[section] for section in TABLES}


def parse(specs, known=None):
    """['[section.]key=value' or 'key=xN' (N times the game's value)] -> {section: {key: float}}. A key must be one
    of the game's (no typos: a wrong key would only add a row the game ignores); without a section it is looked
    up, and must be in exactly one table."""
    known = known if known is not None else known_keys()
    out = {}
    for spec in specs or ():
        if "=" not in spec:
            raise ValueError(f"--kv {spec!r}: [section.]key=value or key=xN")
        name, value = (x.strip() for x in spec.split("=", 1))
        section, _, key = name.rpartition(".")
        if section:
            if section not in TABLES or key not in known.get(section, {}):
                raise ValueError(f"--kv {spec!r}: no {key!r} in the game's {section} table")
        else:
            found = [s for s in TABLES if key in known.get(s, {})]
            if len(found) != 1:
                raise ValueError(f"--kv {spec!r}: {key!r} is in {found or 'no table'}; give section.key")
            section = found[0]
        if value.lower().startswith("x"):
            v = float(value[1:]) * float(known[section][key])
        else:
            v = float(value)
        out.setdefault(section, {})[key] = round(v, 4)
    return out


def label(overrides):
    """A short name of a variant: 'normal' or 'waver_base_timeout=250' (keys sorted)."""
    rows = sorted((k, v) for rows in (overrides or {}).values() for k, v in rows.items())
    return ",".join(f"{k}={v:g}" for k, v in rows) or "normal"
