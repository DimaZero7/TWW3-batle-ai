"""tools.nn.dbtables and tools.nn.units: reading DB tables by the schema's names, unit passports, the check against
cards."""
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


LAYOUT = (3, [("key", "StringU8"), ("n", "I32"), ("x", "F32"), ("flag", "Boolean"), ("extra", "OptionalStringU8")])


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


def test_decode_reads_the_schemas_layout_of_the_tables_version():
    """No layout given: the table's version picks its fields from the schema copy (config/db_schema.json)."""
    rows = [text("waver_base_timeout") + struct.pack("<f", 25.0), text("x") + struct.pack("<f", 0.1)]
    assert dbtables.decode(table(None, rows), "_kv_morale") == [{"key": "waver_base_timeout", "value": 25.0},
                                                                 {"key": "x", "value": 0.1}]
    assert dbtables.decode(table(None, rows[1:]), "_kv_morale", digits=None)[0]["value"] == struct.unpack(
        "<f", struct.pack("<f", 0.1))[0]
    with pytest.raises(ValueError, match="version 99 is not in"):
        dbtables.decode(table(99, rows), "_kv_morale")
    with pytest.raises(KeyError, match="not in"):
        dbtables.decode(table(None, rows), "no_such_table")


def test_other_schema_types():
    wide = "ab".encode("utf-16-le")
    b = table(1, [struct.pack("<h", -2) + struct.pack("<q", 2 ** 40) + struct.pack("<d", 0.5) + struct.pack("<I", 7)
                  + struct.pack("<H", 2) + wide + b"\x01" + struct.pack("<i", 9) + b"\x00"])
    layout = (1, [("a", "I16"), ("b", "I64"), ("c", "F64"), ("d", "ColourRGB"), ("e", "StringU16"),
                  ("f", "OptionalI32"), ("g", "OptionalI32")])
    assert dbtables.decode(b, "t", layout) == [{"a": -2, "b": 2 ** 40, "c": 0.5, "d": 7, "e": "ab", "f": 9, "g": None}]


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
    # The v1 units, the shielded spearmen and the swordsmen (02.10.2026) have cards from battle; the
    # probe's halberds and stormvermin (lord swarm, 01.10.2026) do not, nor (yet: list
    # config/roster/capture_emp_wave2.json) the flagellants, greatswords and militia (02.10.2026), nor the
    # third wave (07.10.2026).
    assert checked >= 9 and len(units.UNITS) == 28           # + the 5 new units of 11.10.2026 (no cards yet)


def test_v1_spears_have_no_shield_and_missile_units_have_missiles():
    u = saved()["units"]
    for key in ("wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0"):
        assert u[key]["shield"] == {"key": "none", "missile_block_chance": 0}
    # The same spearmen with shields: a 35% metal shield, a better spear (defence 42, a faster blow), dearer.
    assert u["wh_main_emp_inf_spearmen_1"]["shield"] == {"key": "wh_missile_block_35_metal", "missile_block_chance": 35}
    assert u["wh_main_emp_inf_spearmen_1"]["multiplayer_cost"] > u["wh_main_emp_inf_spearmen_0"]["multiplayer_cost"]
    shooters = ("wh2_dlc13_emp_inf_archers_0", "wh2_main_skv_inf_skavenslave_slingers_0",
                "wh2_main_skv_inf_night_runners_1",
                "wh_dlc04_emp_inf_free_company_militia_0",
                # the third wave (07.10.2026)
                "wh_main_emp_inf_handgunners", "wh_main_emp_inf_crossbowmen", "wh2_main_skv_inf_night_runners_0",
                # the Ratling Gun (11.10.2026)
                "wh2_dlc12_skv_inf_ratling_gun_0")
    for key in shooters:
        assert u[key]["missile"]["range_m"] > 0 and u[key]["missile"]["ammo"] > 0
    assert all(u[k]["missile"] is None for k in u if k not in shooters)


def test_the_second_empire_wave():
    """Flagellants never rout and frenzy; greatswords are armoured and armour-piercing; the militia's
    pistols are the first direct (flat) fire: trajectory low, unlike the arcing arrow and sling."""
    u = saved()["units"]
    flag, great, mil = (u[k] for k in ("wh_dlc04_emp_inf_flagellants_0", "wh_main_emp_inf_greatswords",
                                       "wh_dlc04_emp_inf_free_company_militia_0"))
    assert "unbreakable" in flag["attributes"] and flag["armour"] == 0
    assert set(flag["abilities"]) == {"wh_dlc04_unit_passive_strength_of_the_penitent", "wh_main_unit_passive_frenzy"}
    assert great["armour"] == 95 and great["melee"]["ap_damage"] > great["melee"]["damage"]
    assert great["melee"]["bonus_v_infantry"] > 0
    assert mil["missile"]["trajectory"] == "low" and mil["missile"]["direct"] is True
    assert "mounted_fire_move" in mil["attributes"]
    assert not any(u[k]["missile"]["direct"] for k in ("wh2_dlc13_emp_inf_archers_0",
                                                        "wh2_main_skv_inf_skavenslave_slingers_0"))


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
    built = units.build(tables)
    assert built == saved()["units"]
    assert units.contact_phases(built, tables) == saved()["contact_phases"]


@pytest.mark.skipif(game_db() is None, reason="needs the installed game and Python 3.14+ (compression.zstd)")
def test_every_table_we_read_decodes_whole_by_the_schema():
    from tools.nn import dbschema
    got = dbtables.read_tables(game_db(), [t for t in dbschema.TABLES])
    assert all(rows for rows in got.values())


def test_the_skaven_wave():
    """Plain skavenslaves (cheapest, expendable), clanrats with shields, Night Runners with slings (fast,
    vanguard, arcing sling): every Skaven infantry unit has Strength in Numbers and Scurry Away!."""
    u = saved()["units"]
    slaves, clanrats, runners = (u[k] for k in ("wh2_main_skv_inf_skavenslaves_0", "wh2_main_skv_inf_clanrats_1",
                                                "wh2_main_skv_inf_night_runners_1"))
    assert slaves["multiplayer_cost"] == 125 and "expendable" in slaves["attributes"] and slaves["armour"] == 0
    assert clanrats["shield"]["missile_block_chance"] == 35 and clanrats["melee"]["defence"] == 22
    assert runners["speed"]["run"] > u["wh2_main_skv_inf_clanrat_spearmen_0"]["speed"]["run"]
    assert "guerrilla_deploy" in runners["attributes"] and "stalk" not in runners["attributes"]
    assert runners["missile"]["category"] == "sling" and not runners["missile"]["direct"]
    for p in (slaves, clanrats, runners):
        assert set(p["abilities"]) == {"wh2_main_unit_passive_scurry_away", "wh2_main_unit_passive_strength_in_numbers"}


def test_the_third_wave():
    """Handgunners (the game's "Пистольеры"): flat (direct) armour-piercing fire at 145 m, standing only;
    crossbowmen: arcing bolts at 160 m; Night Runners with throwing stars: direct fire at 70 m all round
    (a 360 deg fire arc) and whilst moving; clanrats without shields, clanrat spearmen and stormvermin with them."""
    u = saved()["units"]
    hg, xb, stars = (u[k] for k in ("wh_main_emp_inf_handgunners", "wh_main_emp_inf_crossbowmen",
                                    "wh2_main_skv_inf_night_runners_0"))
    assert hg["men"] == 90 and hg["multiplayer_cost"] == 600
    assert hg["missile"]["direct"] and hg["missile"]["range_m"] == 145
    assert hg["missile"]["ap_damage"] > hg["missile"]["damage"] and "mounted_fire_move" not in hg["attributes"]
    assert xb["missile"]["category"] == "arrow" and not xb["missile"]["direct"] and xb["missile"]["range_m"] == 160
    assert stars["missile"]["direct"] and stars["missile"]["range_m"] == 70 and stars["missile"]["fire_arc_deg"] == 180
    assert {"mounted_fire_move", "guerrilla_deploy"} <= set(stars["attributes"]) and "stalk" not in stars["attributes"]
    assert u["wh2_main_skv_inf_clanrats_0"]["shield"]["missile_block_chance"] == 0
    for k in ("wh2_main_skv_inf_clanrat_spearmen_1", "wh2_main_skv_inf_stormvermin_1"):
        assert u[k]["shield"]["missile_block_chance"] == 35
    assert "charge_reflection" in u["wh2_main_skv_inf_clanrat_spearmen_1"]["attributes"]
    assert u["wh2_main_skv_inf_stormvermin_1"]["armour"] == 90


def test_the_new_units_and_their_contact_phases():
    """The Ratling Gun (32 men, a volley of 18 bullets a man, suppressing fire), the Plague Monks and Censer Bearers
    (the censer's contagion, magical), the Giant Rats (war beasts; _1 with poison), and the contact phases' numbers."""
    doc = saved()
    u, ph = doc["units"], doc["contact_phases"]
    gun = u["wh2_dlc12_skv_inf_ratling_gun_0"]
    assert gun["men"] == 32 and gun["missile"]["shots_per_volley"] == 18 and gun["missile"]["direct"]
    assert gun["missile"]["contact_phase"] == "wh2_dlc12_unit_contact_suppressive_fire"
    censer = u["wh2_main_skv_inf_plague_monk_censer_bearer"]
    assert censer["melee"]["magic"] and censer["melee"]["contact_phase"] == "wh2_main_unit_contact_plague_monk_censer_morale"
    rats, poison = u["wh2_dlc16_skv_mon_wolf_rats_0"], u["wh2_dlc16_skv_mon_wolf_rats_1"]
    assert rats["caste"] == poison["caste"] == "war_beast" and rats["size"] == "small"
    assert poison["melee"]["contact_phase"] == "wh_main_unit_contact_poison" and not poison["melee"]["magic"]
    assert "wh_main_unit_passive_frenzy" in u["wh2_main_skv_inf_plague_monks"]["abilities"]
    assert set(ph) == {"wh2_dlc12_unit_contact_suppressive_fire", "wh2_main_unit_contact_plague_monk_censer_morale",
                       "wh_main_unit_contact_poison"}
    assert all(p["duration_s"] == 10.0 and p["effect"] == "negative" for p in ph.values())
    assert ph["wh2_main_unit_contact_plague_monk_censer_morale"]["stats"] == [
        {"stat": "stat_morale", "how": "add", "value": -10.0}]
    assert units.phase_keys(u) == sorted(ph)


def test_the_card_counts_every_shot_of_a_volley():
    p = copy.deepcopy(saved()["units"]["wh2_dlc12_skv_inf_ratling_gun_0"])
    card = {"stats": {"scalar_speed": {"base": 38}, "stat_melee_attack": {"base": 16}, "stat_melee_defence": {"base": 16},
                      "stat_charge_bonus": {"base": 2}, "stat_weapon_damage": {"base": 26}, "stat_armour": {"base": 70},
                      "stat_ammo": {"base": 270}, "scalar_missile_range": {"base": 145},
                      "stat_missile_damage_over_time": {"base": 288.0}},
            "details": {"NumEntitiesInitial": 32, "HealthMax": 3936, "Mass": 150.0},
            "profile": {"unit_class": "inf_mis", "slow_speed": 1.4, "fast_speed": 3.8,
                        "owned_non_passive_special_abilities": [],
                        "owned_passive_special_abilities": list(p["abilities"])}}
    assert units.check_card(p, card) == []                     # twdatabase: 288 = (2 + 6) x 18 x 10 / 5


def test_contact_phases_take_the_phases_duration_effect_type_and_stats():
    u = {"gun": {"melee": {"contact_phase": None}, "missile": {"contact_phase": "a_contact"}}}
    t = {"special_ability_phases": [{"id": "b_passive", "duration": -1.0, "effect_type": "positive"},
                                    {"id": "a_contact", "duration": 10.0, "effect_type": "negative"}],
         "special_ability_phase_stat_effects": [{"phase": "a_contact", "stat": "scalar_speed", "how": "mult",
                                                 "value": 0.7}]}
    assert units.contact_phases(u, t) == {"a_contact": {"duration_s": 10.0, "effect": "negative", "stats": [
        {"stat": "scalar_speed", "how": "mult", "value": 0.7}]}}
