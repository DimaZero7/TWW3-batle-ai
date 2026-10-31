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


def ability(key, active, recharge, uses, rng, own, friendly, enemy, every_frame, cp=(0.0, 0.0), aim=()):
    """A unit_special_abilities row by the schema's names (only the columns a passport reads); aim: the targets a player
    chooses among (target_friends / target_enemies / target_ground)."""
    return {"key": key, "active_time": active, "recharge_time": recharge, "num_uses": uses, "effect_range": rng,
            "affect_self": own, "num_effected_friendly_units": friendly, "num_effected_enemy_units": enemy,
            "update_targets_every_frame": every_frame, "initial_recharge": 0.0,
            "additional_melee_cp": cp[0], "additional_missile_cp": cp[1],
            **{f"target_{k}": k in aim for k in ("friends", "enemies", "ground")}}


class TestPassport:
    PREFIX = {"buff": ability("buff", 14.0, 60.0, -1, 35.0, True, -1, 0, False, (50.0, 25.5)),
              "hex": ability("hex", 13.0, 0.0, 1, 35.0, False, 0, -1, False, aim=("enemies",)),
              # (31.10.2026, the Warp Grinders' Warp-Quake) on the enemies around its owner, no target to choose
              "quake": ability("quake", 17.0, 90.0, -1, 35.0, False, 0, -1, True),
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

    def test_an_ability_around_its_owner_has_no_target_to_choose(self):
        t = tables()
        t["special_ability_to_special_ability_phase_junctions"] = t["special_ability_to_special_ability_phase_junctions"] + [
            {"order": 1, "special_ability": "quake", "phase": "quake", "target_self": False, "target_friends": False,
             "target_enemies": True}]
        quake = abilities.passport("quake", self.PREFIX, t)
        assert quake["self_cast"] and quake["targets"]["enemies"] and not quake["targets_own"]

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
        units = passport.load(passport.PASSPORTS)          # (config/nn/units.json: the casters' cards are the probe's)
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
        idx, usable, fx = mab.slots([GENERAL, "", WARLORD, "wh_main_emp_inf_spearmen_0"])
        feat = mab.table()[idx]
        assert feat.shape == (4, mab.SLOTS, mab.TABLE) and np.abs(feat).max() <= 1.5
        owned = idx >= 0
        A = mab.ACTIVE
        # the General: 2 actives, 2 passives (Hold the Line, Wounded); the Warlord: 3 actives, 2 passives
        assert owned[0].nonzero()[0].tolist() == [0, 1, A, A + 1]
        assert owned[2].nonzero()[0].tolist() == [0, 1, 2, A, A + 1]
        assert not owned[1].any() and not owned[3].any()
        assert usable[0].nonzero()[0].tolist() == [0, 1] and usable[2].nonzero()[0].tolist() == [0, 1, 2]
        assert np.all(feat[1] == 0) and np.all(feat[3] == 0)
        names = mab.keys()
        assert [names[i] for i in idx[0][owned[0]]] == [FS, SYG, HTL, "wh3_main_unit_passive_single_entity"]
        # a passive's innate effect (its on now): config/nn/effects.json order
        from tools.nn.model import effects
        assert effects.keys()[fx[0, A]] == HTL and fx[0, 0] == -1

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


class TestSpells:
    """The spell passports (tools/nn/abilities.py spell_passport; config/nn/abilities.json spells)."""

    TABLES = {
        "unit_special_abilities": [{"key": "sd", "mana_cost": 6.0, "wind_up_time": 1.0, "target_intercept_range": 200.0,
                                    "active_time": 8.0, "recharge_time": 30.0, "initial_recharge": -1.0,
                                    "target_enemies": True, "target_friends": False, "target_self": False,
                                    "target_ground": True, "ai_usage": "aoe_bombardment", "bombardment": "sd_b"}],
        "unit_abilities": [{"key": "sd", "type": "wh_type_bombardment", "overpower_option": "sd_up"}],
        "projectile_bombardments": [{"bombardment_key": "sd_b", "num_projectiles": 40, "start_time": 2.0,
                                     "arrival_window": 3.0, "radius_spread": 10.0, "launch_height": 150,
                                     "launch_source": "above_target", "projectile_type": "sd_p"}],
        "projectiles": [{"key": "sd_p", "damage": 24, "ap_damage": 12, "collision_radius": 0.4,
                         "projectile_penetration": "very_low_magic", "fixed_elevation": 45, "muzzle_velocity": 30.0,
                         "gravity": 20.0, "is_magical": True, "can_damage_allies": True, "explosion_type": "sd_e",
                         "mass": 50.0}],
        "projectiles_explosions": [{"key": "sd_e", "detonation_radius": 3.0, "detonation_damage": 0.0,
                                    "detonation_damage_ap": 30.0, "affects_allies": True,
                                    "detonation_force": 400.0}],
        "projectile_penetration_junctions": [{"key": "very_low_magic", "max_penetration": 2,
                                              "entity_size_cap": "very_small"}],
    }

    def test_a_bombardment_spell_from_its_tables(self):
        p = abilities.spell_passport("sd", self.TABLES)
        assert (p["kind"], p["mana"], p["wind_up_s"], p["range_m"], p["initial_s"]) == \
            ("wh_type_bombardment", 6.0, 1.0, 200.0, 0.0)
        b = p["bombardment"]
        assert (b["projectiles"], b["start_s"], b["window_s"], b["spread_m"]) == (40, 2.0, 3.0, 10.0)
        assert (b["projectile"]["penetration"], b["explosion"]["ap"], b["explosion"]["radius_m"]) == (2, 30.0, 3.0)
        # the fall from 150 m at 30 m/s, 45 deg down, gravity 20: 2.955 s (the probe's first hit 6.0 s after the order)
        assert b["flight_s"] == pytest.approx(2.955, abs=0.002)
        # the knock (sim.json magic.knock): the projectile's mass and speed at the end of that fall, the shockwave's force
        assert (b["projectile"]["mass"], b["explosion"]["force"]) == (50.0, 400.0)
        assert b["projectile"]["impact_v"] == pytest.approx(83.07, abs=0.01)

    def test_the_fall_of_a_projectile(self):
        assert abilities.flight_s(100, 120.0, 45, 0.8) == pytest.approx(1.172, abs=0.002)    # Warp Lightning
        assert abilities.flight_s(0, 30.0, 45, 20.0) == 0.0
        # its speed at the end: 120 m/s at 45 deg down, gravity 0.8 for 1.17 s - 120.7 m/s
        assert abilities.impact_v(100, 120.0, 45, 0.8) == pytest.approx(120.66, abs=0.01)

    def test_the_casters_spells_are_in_the_file(self):
        doc = json.loads(abilities.OUT.read_text(encoding="utf-8"))
        probe = json.loads(abilities.PROBE_UNITS.read_text(encoding="utf-8"))["units"]
        assert abilities.spells_of(probe) == sorted(doc["spells"])
        assert set(doc["spells"]).isdisjoint(doc["abilities"])        # the network's ability table is unchanged
