"""Ability passports: building them from the database's ability tables read by the schema's names
(tools/nn/abilities.py), the saved config/nn/abilities.json, and the network's ability features and slots
(tools/nn/model/abilities.py). Numpy only."""
import json

import numpy as np
import pytest

from tools.nn import abilities
from tools.nn.model import abilities as mab
from tools.nn.model import passport
from tools.nn.sim.abilities import SLOTS, slot_keys

GENERAL, WARLORD = "wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0"
FS, SYG, HTL = ("wh_main_character_abilities_foe_seeker", "wh_main_character_abilities_stand_your_ground",
                "wh_main_lord_passive_hold_the_line")
VV, DO, RALLY = ("wh2_main_character_abilities_verminous_valour", "wh_main_character_abilities_deadly_onslaught",
                 "wh_main_character_abilities_rally")


def tables():
    # the junction's columns by the schema: order, phase, special_ability; "dlc_variant" is another ability that
    # reuses the phase "buff" (it is not a phase of "buff")
    j = [{"order": 1, "phase": "buff", "special_ability": "buff", "target_self": True, "target_friends": True,
          "target_enemies": False},
         {"order": 1, "phase": "buff", "special_ability": "dlc_variant", "target_self": True,
          "target_friends": False, "target_enemies": False},
         {"order": 1, "phase": "buff_extra", "special_ability": "buff", "target_self": True,
          "target_friends": False, "target_enemies": False},
         {"order": 1, "special_ability": "hex", "phase": "hex_phase", "target_self": False,
          "target_friends": False, "target_enemies": True}]
    s = [{"phase": "buff", "stat": "stat_morale", "how": "add", "value": 16.0},
         {"phase": "buff_extra", "stat": "stat_morale", "how": "add", "value": 99.0},
         {"phase": "hex_phase", "stat": "stat_melee_attack", "how": "add", "value": -10.0}]
    a = [{"attribute": "unbreakable", "phase": "buff", "attribute_type": "positive"}]
    return {"special_ability_to_special_ability_phase_junctions": j, "special_ability_phase_stat_effects": s,
            "special_ability_phase_attribute_effects": a}


def ability(key, active, recharge, uses, rng, own, friendly, enemy, every_frame, cp=(0.0, 0.0)):
    """A unit_special_abilities row by the schema's names (only the columns a passport reads)."""
    return {"key": key, "active_time": active, "recharge_time": recharge, "num_uses": uses, "effect_range": rng,
            "affect_self": own, "num_effected_friendly_units": friendly, "num_effected_enemy_units": enemy,
            "update_targets_every_frame": every_frame, "initial_recharge": 0.0,
            "additional_melee_cp": cp[0], "additional_missile_cp": cp[1]}


class TestPassport:
    PREFIX = {"buff": ability("buff", 14.0, 60.0, -1, 35.0, True, -1, 0, False, (50.0, 25.5)),
              "hex": ability("hex", 13.0, 0.0, 1, 35.0, False, 0, -1, False),
              "aura": ability("aura", -1.0, -1.0, -1, 35.0, True, -1, 0, True)}

    def test_the_phase_named_as_the_ability_wins_over_its_other_phases(self):
        p = abilities.passport("buff", self.PREFIX, tables())
        assert p["phases"] == ["buff"] and p["targets"] == {"self": True, "friends": True, "enemies": False}
        assert p["cp"] == [50, 25.5] and isinstance(p["cp"][0], int)
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

    def test_build_indexes_the_rows_and_reads_the_phases_fatigue(self):
        t = dict(tables(), unit_special_abilities=list(self.PREFIX.values()),
                 special_ability_to_recharge_contexts=[{"recharge_context": "losing_melee_combat",
                                                        "special_ability": "buff"}],
                 special_ability_to_auto_deactivate_flags=[{"deactivate_flag": "out_of_melee",
                                                            "special_ability": "buff"}],
                 special_ability_phases=[{"id": "buff", "fatigue_change_ratio": -0.009999999776482582},
                                         {"id": "hex_phase", "fatigue_change_ratio": 0.0}])
        got = abilities.build(t, ["buff", "hex"], {})
        assert got["buff"]["vigour_per_s"] == -0.01 and got["hex"]["vigour_per_s"] == 0
        assert got["buff"]["recharge_when"] == ["losing_melee_combat"] and got["buff"]["off_when"] == ["out_of_melee"]
        with pytest.raises(ValueError, match="no row"):
            abilities.build(t, ["nothing"], {})


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
