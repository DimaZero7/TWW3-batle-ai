"""The probe pack's own rule tables (tools/pack/kvtable.py): the game's format, read back by tools/nn/gamedb.kv_table."""
import pytest

from tools.nn import gamedb
from tools.pack import kvtable

KNOWN = {"morale": {"waver_base_timeout": 25.0, "post_rally_no_rout_timer": 10.0},
         "battle": {"charge_timeout": 1.0}, "fatigue": {"charge_timeout": 2.0}}


def test_a_table_reads_back_as_the_game_s_tables_do():
    rows = {"waver_base_timeout": 250.0, "post_rally_no_rout_timer": 0.0}
    b = kvtable.table_bytes(rows, "00000000-0000-0000-0000-000000000001")
    assert b.startswith(kvtable.GUID_MARKER)
    assert gamedb.kv_table(b) == rows


def test_entries_sort_before_the_game_s_table_and_are_deterministic():
    over = {"morale": {"waver_base_timeout": 250.0}}
    a, b = kvtable.entries(over), kvtable.entries(over)
    assert a == b and list(a) == ["db\\_kv_morale_tables\\!tww3_bai_probe_kv"]
    assert kvtable.TABLE_NAME < "data__" and not kvtable.TABLE_NAME[-1].isdigit()
    assert kvtable.entries({}) == {} and kvtable.entries({"morale": {}}) == {}
    assert kvtable.entries({"morale": {"waver_base_timeout": 0.0}}) != a      # other rows, another table


def test_parse_takes_values_multiples_and_sections():
    assert kvtable.parse(["waver_base_timeout=x10"], KNOWN) == {"morale": {"waver_base_timeout": 250.0}}
    assert kvtable.parse(["waver_base_timeout=0"], KNOWN) == {"morale": {"waver_base_timeout": 0.0}}
    assert kvtable.parse(["battle.charge_timeout=3"], KNOWN) == {"battle": {"charge_timeout": 3.0}}
    for bad in (["waver_base_timout=1"], ["charge_timeout=1"], ["morale.charge_timeout=1"], ["waver_base_timeout"]):
        with pytest.raises(ValueError):
            kvtable.parse(bad, KNOWN)


def test_the_real_game_rules_know_the_morale_timers():
    assert kvtable.parse(["waver_base_timeout=x10"]) == {"morale": {"waver_base_timeout": 250.0}}


def test_label_names_a_variant():
    assert kvtable.label(None) == "normal"
    assert kvtable.label({"morale": {"waver_base_timeout": 250.0}}) == "waver_base_timeout=250"
