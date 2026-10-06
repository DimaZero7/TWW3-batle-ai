"""Ability passports: reading the database's ability tables (tools/nn/dbtables.py decode_prefix),
building passports (tools/nn/abilities.py), the saved config/nn/abilities.json, and the network's
ability features and slots (tools/nn/model/abilities.py). Numpy only."""
import json
import struct

import numpy as np
import pytest

from tools.nn import abilities, dbtables
from tools.nn.model import abilities as mab
from tools.nn.model import passport
from tools.nn.sim.abilities import SLOTS, slot_keys

GENERAL, WARLORD = "wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0"
FS, SYG, HTL = ("wh_main_character_abilities_foe_seeker", "wh_main_character_abilities_stand_your_ground",
                "wh_main_lord_passive_hold_the_line")
VV, DO, RALLY = ("wh2_main_character_abilities_verminous_valour", "wh_main_character_abilities_deadly_onslaught",
                 "wh_main_character_abilities_rally")


def text(s):
    b = s.encode()
    return struct.pack("<H", len(b)) + b


def prefix_row(key, active, recharge, uses, rng, own, friendly, enemy, tail=b"\x00" * 9):
    return (text(key) + struct.pack("<ffif", active, recharge, uses, rng) + bytes([own])
            + struct.pack("<ii", friendly, enemy) + b"\x00" + struct.pack("<f", 0.0) + tail)


def table(version, rows):
    return b"\xfc\xfd\xfe\xff" + struct.pack("<i", version) + b"\x01" + struct.pack("<I", len(rows)) + b"".join(rows)


class TestPrefix:
    def test_a_row_is_found_by_its_key_and_read_up_to_the_prefix(self):
        b = table(74, [prefix_row("a_buff", 25.0, 60.0, -1, 0.0, 1, 0, 0),
                       prefix_row("b_aura", 18.0, 90.0, -1, 35.0, 1, -1, 0)])
        got = dbtables.decode_prefix(b, "unit_special_abilities", ["b_aura", "a_buff"])
        assert got["b_aura"]["effect_range"] == 35 and got["b_aura"]["num_effected_friendly_units"] == -1
        assert got["a_buff"]["active_time"] == 25 and got["a_buff"]["targets_own"] is True
        assert got["a_buff"]["initial_recharge"] == 0

    def test_a_key_written_inside_another_row_is_not_taken_and_a_missing_key_raises(self):
        inside = text("a_buff") + b"\xff" * 30          # the key as a reference: no sane numbers after it
        b = table(74, [prefix_row("a_buff", 25.0, 60.0, -1, 0.0, 1, 0, 0, tail=inside)])
        assert dbtables.decode_prefix(b, "unit_special_abilities", ["a_buff"])["a_buff"]["recharge_time"] == 60
        with pytest.raises(ValueError):
            dbtables.decode_prefix(b, "unit_special_abilities", ["nothing"])
        with pytest.raises(ValueError):
            dbtables.decode_prefix(table(75, []), "unit_special_abilities", [])


def tables():
    j = [{"order": 1, "special_ability": "buff", "phase": "buff", "target_self": True, "target_friends": True,
          "target_enemies": False},
         {"order": 1, "special_ability": "buff", "phase": "dlc_variant", "target_self": True,
          "target_friends": False, "target_enemies": False},
         {"order": 1, "special_ability": "hex", "phase": "hex_phase", "target_self": False,
          "target_friends": False, "target_enemies": True}]
    s = [{"phase": "buff", "stat": "stat_morale", "how": "add", "value": 16.0},
         {"phase": "dlc_variant", "stat": "stat_morale", "how": "add", "value": 99.0},
         {"phase": "hex_phase", "stat": "stat_melee_attack", "how": "add", "value": -10.0}]
    a = [{"attribute": "unbreakable", "phase": "buff", "effect": "positive"}]
    return {"special_ability_to_special_ability_phase_junctions": j, "special_ability_phase_stat_effects": s,
            "special_ability_phase_attribute_effects": a}


class TestPassport:
    PREFIX = {"buff": {"active_time": 14.0, "recharge_time": 60.0, "num_uses": -1, "effect_range": 35.0,
                       "targets_own": True, "num_effected_friendly_units": -1, "num_effected_enemy_units": 0,
                       "update_targets": False},
              "hex": {"active_time": 13.0, "recharge_time": 0.0, "num_uses": 1, "effect_range": 35.0,
                      "targets_own": False, "num_effected_friendly_units": 0, "num_effected_enemy_units": -1,
                      "update_targets": False},
              "aura": {"active_time": -1.0, "recharge_time": -1.0, "num_uses": -1, "effect_range": 35.0,
                       "targets_own": True, "num_effected_friendly_units": -1, "num_effected_enemy_units": 0,
                       "update_targets": True}}

    def test_the_phase_named_as_the_ability_wins_over_variants(self):
        p = abilities.passport("buff", self.PREFIX, tables())
        assert p["phases"] == ["buff"] and p["targets"] == {"self": True, "friends": True, "enemies": False}
        assert [(e["stat"], e["value"], e["on"]) for e in p["effects"]] == [("stat_morale", 16.0, ["self", "friends"])]
        assert p["attributes"][0]["attribute"] == "unbreakable"
        assert p["self_cast"] and not p["passive"]

    def test_a_hex_needs_a_target_and_a_passive_has_no_times(self):
        hex_ = abilities.passport("hex", self.PREFIX, tables())
        assert hex_["phases"] == ["hex_phase"] and hex_["targets"]["enemies"] and not hex_["self_cast"]
        aura = abilities.passport("aura", self.PREFIX, tables())
        assert aura["passive"] and not aura["self_cast"] and aura["effects"] == []

    def test_cards_check_which_abilities_are_passive(self, tmp_path):
        card = {"card": {"profile": {"owned_non_passive_special_abilities": ["buff"],
                                     "owned_passive_special_abilities": ["aura"]}}}
        (tmp_path / "lord.json").write_text(json.dumps(card), encoding="utf-8")
        good = {k: abilities.passport(k, self.PREFIX, tables()) for k in ("buff", "aura")}
        assert abilities.check_cards(good, {"lord": {}}, tmp_path) == []
        bad = dict(good, aura=dict(good["aura"], passive=False))
        assert len(abilities.check_cards(bad, {"lord": {}}, tmp_path)) == 1


class TestSaved:
    def test_every_owned_ability_has_a_passport_and_the_cards_agree(self):
        saved = mab.load()
        units = passport.load()
        assert set(abilities.owned(units)) <= set(saved)
        assert abilities.check_cards(saved, units) == []

    def test_the_lords_numbers_are_the_databases(self):
        s = mab.load()
        assert (s[FS]["active_s"], s[FS]["recharge_s"], s[FS]["range_m"]) == (25, 60, 0)
        assert (s[SYG]["active_s"], s[SYG]["recharge_s"], s[SYG]["range_m"]) == (18, 90, 35)
        assert (s[DO]["active_s"], s[DO]["recharge_s"]) == (31, 90)
        assert (s[VV]["active_s"], s[RALLY]["active_s"]) == (17, 14)
        assert s[HTL]["passive"] and not s[HTL]["self_cast"]
        assert all(s[k]["self_cast"] for k in (FS, SYG, DO, VV, RALLY))
        eff = {(e["stat"], e["how"]): e["value"] for e in s[SYG]["effects"]}
        assert eff == {("stat_melee_defence", "add"): 24, ("stat_morale", "add"): 16}
        assert s[SYG]["targets"] == {"self": True, "friends": True, "enemies": False}
        assert s[FS]["targets"] == {"self": True, "friends": False, "enemies": False}

    def test_update_targets_from_the_database(self):
        # unit_special_abilities field 8 (update_targets_every_frame): Rally and Hold the Line pick their targets every
        # frame (an aura that follows the lord), Stand Your Ground is laid once at the cast (build/effects/spec.md P2)
        s = mab.load()
        assert s[RALLY]["update_targets"] and s[HTL]["update_targets"] and not s[SYG]["update_targets"]
        assert s["wh_dlc04_unit_passive_strength_of_the_penitent"]["recharge_when"] == ["losing_melee_combat"]
        assert s["wh3_main_unit_passive_single_entity"]["recharge_when"] == ["health_below_25%"]

    def test_initial_recharge_from_the_database(self):
        # unit_special_abilities.initial_recharge: our lords' actives start ready, Strength of the Penitent after 3 s,
        # Single Entity (Wounds) after 5 s (build/melee2/spec.md 5).
        s = mab.load()
        assert all(s[k]["initial_s"] == 0 for k in (FS, SYG, DO, VV, RALLY))
        assert s["wh_dlc04_unit_passive_strength_of_the_penitent"]["initial_s"] == 3
        assert s["wh3_main_unit_passive_single_entity"]["initial_s"] == 5


class TestFeatures:
    def test_size_scale_and_padding(self):
        feat, owned, usable = mab.slots([GENERAL, "", WARLORD, "wh_main_emp_inf_spearmen_0"])
        assert feat.shape == (4, SLOTS, mab.STATIC) and np.abs(feat).max() <= 1.5
        assert owned.tolist() == [[True] * 3, [False] * 3, [True] * 3, [False] * 3]
        assert usable.tolist()[0] == [True, True, False] and usable.tolist()[2] == [True, True, True]
        assert np.all(feat[1] == 0) and np.all(feat[3] == 0)

    def test_actives_first_then_passives_that_reach_others(self):
        units, saved = passport.load(), mab.load()
        assert slot_keys(GENERAL, units, saved) == [FS, SYG, HTL]          # single entity left out
        assert slot_keys(WARLORD, units, saved) == [VV, DO, RALLY]
        assert slot_keys("unknown_unit", units, saved) == [""] * SLOTS

    def test_an_ability_is_described_by_what_it_does(self):
        f = dict(zip(mab.STATIC_NAMES, mab.features(mab.load()[SYG])))
        assert f["active_s"] == pytest.approx(18 / 60) and f["range_m"] == pytest.approx(0.35)
        assert f["allies_stat_melee_defence_add"] == pytest.approx(24 / 50)
        assert f["allies_stat_morale_add"] == pytest.approx(16 / 50) and f["enemies_stat_morale_add"] == 0
        hexed = {"effects": [{"stat": "stat_melee_attack", "how": "add", "value": -10, "on": ["enemies"]},
                             {"stat": "unknown_stat", "how": "add", "value": 5, "on": ["enemies"]}],
                 "targets": {"enemies": True}, "active_s": 10, "recharge_s": 30}
        g = dict(zip(mab.STATIC_NAMES, mab.features(hexed)))
        assert g["enemies_stat_melee_attack_add"] == pytest.approx(-0.2) and g["allies_stat_melee_attack_add"] == 0
        assert len(mab.features(hexed)) == mab.STATIC
