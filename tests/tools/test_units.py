"""tools.nn.dbtables and tools.nn.units: reading DB tables, unit passports, the check against cards."""
import copy
import json
import struct
from pathlib import Path

import pytest

from tools import config as project
from tools.nn import dbtables, units

PASSPORTS = units.OUT


def table(version, rows):
    """Bytes of a DB table: version marker, one byte, row count, rows."""
    head = (b"\xfc\xfd\xfe\xff" + struct.pack("<i", version) if version is not None else b"") + b"\x01"
    return head + struct.pack("<I", len(rows)) + b"".join(rows)


def text(s):
    raw = s.encode()
    return struct.pack("<H", len(raw)) + raw


LAYOUT = (3, dbtables._layout("key:s n:i x:f flag:b extra:o"))


def row(key, n, x, flag, extra):
    return (text(key) + struct.pack("<i", n) + struct.pack("<f", x) + bytes([flag])
            + (b"\x01" + text(extra) if extra else b"\x00"))


def test_decode_reads_every_type():
    b = table(3, [row("a", -5, 1.5, 1, "wood"), row("b", 7, 0.25, 0, None)])
    assert dbtables.decode(b, "t", LAYOUT) == [
        {"key": "a", "n": -5, "x": 1.5, "flag": True, "extra": "wood"},
        {"key": "b", "n": 7, "x": 0.25, "flag": False, "extra": None}]


def test_decode_refuses_a_layout_that_does_not_fit():
    good = [row("a", 1, 1.0, 0, None)]
    with pytest.raises(ValueError, match="version"):
        dbtables.decode(table(4, good), "t", LAYOUT)
    with pytest.raises(ValueError, match="left"):
        dbtables.decode(table(3, good) + b"\x00", "t", LAYOUT)
    with pytest.raises(ValueError, match="flag"):
        dbtables.decode(table(3, [row("a", 1, 1.0, 2, None)]), "t", LAYOUT)
    with pytest.raises(ValueError, match="not text"):
        dbtables.decode(table(3, [text("a\x01") + row("", 1, 1.0, 0, None)[2:]]), "t", LAYOUT)


def test_header_without_version():
    assert dbtables.header(b"\x01\x00\x00\x00\x00") == (None, 1)
    guid = b"\xfd\xfe\xfc\xff" + struct.pack("<H", 2) + "ab".encode("utf-16-le")
    assert dbtables.header(guid + b"\xfc\xfd\xfe\xff" + struct.pack("<i", 9) + b"\x01") == (9, len(guid) + 9)


def test_layouts_are_consistent():
    for name, (version, fields) in dbtables.LAYOUTS.items():
        names = [n for n, _ in fields]
        assert len(names) == len(set(names)), name
        assert dbtables.INFERRED.get(name, set()) <= set(names), name


def saved():
    return json.loads(PASSPORTS.read_text(encoding="utf-8"))


def test_passports_hold_the_v1_units_and_equal_their_cards():
    doc = saved()
    assert list(doc["units"]) == list(units.UNITS)
    checked = 0
    for key, p in doc["units"].items():
        c = units.card(key)
        if c is not None:
            assert units.check_card(p, c) == [], key
            assert p["card_check"] == "equal"
            checked += 1
    assert checked == len(units.UNITS) == 7


def test_v1_spears_have_no_shield_and_missile_units_have_missiles():
    u = saved()["units"]
    for key in ("wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0"):
        assert u[key]["shield"] == {"key": "none", "missile_block_chance": 0}
    for key in ("wh2_dlc13_emp_inf_archers_0", "wh2_main_skv_inf_skavenslave_slingers_0"):
        assert u[key]["missile"]["range_m"] > 0 and u[key]["missile"]["ammo"] > 0
    assert all(u[k]["missile"] is None for k in u if "archers" not in k and "slingers" not in k)


def test_every_passport_field_names_its_source():
    doc = saved()
    for p in doc["units"].values():
        keys = {k for k, v in p.items() if not isinstance(v, dict) or k == "damage_resist"}
        keys |= {f"{k}.{s}" for k, v in p.items() if isinstance(v, dict) and k != "damage_resist" for s in v}
        keys -= {"card_check", "missile"}
        assert keys <= set(doc["_fields"]), sorted(keys - set(doc["_fields"]))


def test_check_card_names_a_difference():
    doc = saved()
    key = "wh_main_emp_inf_spearmen_0"
    p = copy.deepcopy(doc["units"][key])
    p["melee"]["damage"] += 1
    p["armour"] = 31
    diff = units.check_card(p, units.card(key))
    assert len(diff) == 2 and diff[0].startswith("melee damage") and diff[1].startswith("armour")


def game_db():
    try:
        import compression.zstd  # noqa: F401  (Python 3.14+)
    except ImportError:
        return None
    pack = Path(project.load().get("game_dir", "")) / "data" / "db.pack"
    return pack if pack.is_file() else None


@pytest.mark.skipif(game_db() is None, reason="needs the installed game and Python 3.14+ (compression.zstd)")
def test_passports_are_rebuilt_from_the_game_the_same():
    tables = dbtables.read_tables(game_db(), units.TABLES)
    assert units.build(tables) == saved()["units"]
