"""tools/nn/general/army_input.py: the general's army input (design/general/index.html, "Армии"; "Проверки анализа
армий", group 1 "При сборке": the input's part - the reader's in test_general_army_reader_torch.py)."""
import copy
import math

import numpy as np
import pytest

from tools.nn.general import army_input as ai
from tools.nn.model import passport as pp

LORD = "wh_main_emp_cha_general_0"
SPEAR = "wh_main_emp_inf_spearmen_0"
ARCHERS = "wh2_dlc13_emp_inf_archers_0"
GUNS = "wh_main_emp_inf_handgunners"
RATLING = "wh2_dlc12_skv_inf_ratling_gun_0"
SLAVES = "wh2_main_skv_inf_skavenslaves_0"
WOLVES = "wh2_dlc16_skv_mon_wolf_rats_0"
OURS = [SPEAR, LORD, ARCHERS, GUNS, "wh_dlc04_emp_inf_flagellants_0"]
THEIRS = [SLAVES, RATLING, "wh2_main_skv_cha_warlord_0", WOLVES, "wh2_main_skv_inf_night_runners_0", SLAVES]


@pytest.fixture(scope="module")
def passports():
    return pp.load()


def test_the_slot_is_the_tables_groups():
    groups = {}
    for _, g, _ in ai.NUMBERS:
        groups[g] = groups.get(g, 0) + 1
    assert groups == {"side": 1, "cost": 1, "size": 5, "body": 4, "melee": 10, "defence": 8, "missile": 12,
                      "properties": 4, "state": 4}
    assert list(groups) == ["side", "cost", "size", "body", "melee", "defence", "missile", "properties", "state"]
    assert ai.SIZE == 49 and ai.OLD_SIZE == 45 and ai.NAMES[ai.OLD_SIZE:] == ai.STATE and ai.LABELS == ("trajectory", "caste", "size")
    assert len(ai.CASTES) == 12 and len(ai.SIZES) == 4 and ai.TRAJECTORIES[1:] == ("direct", "high", "dual")
    assert "other" not in ai.CASTES + ai.SIZES + ai.TRAJECTORIES
    a = ai.build({1: OURS, 2: THEIRS})
    assert a.numbers.shape == (80, 49) and a.labels.shape == (80, 3) and a.empty.shape == (80,)
    assert a.numbers.dtype == np.float32 and a.labels.dtype == np.int64


def test_own_first_then_enemy_and_empty_slots():
    a = ai.build({1: OURS, 2: THEIRS})
    assert (~a.empty[:5]).all() and a.empty[5:40].all()
    assert (~a.empty[40:46]).all() and a.empty[46:].all()
    assert (a.numbers[:5, ai.SIDE] == 1).all() and (a.numbers[40:46, ai.SIDE] == 0).all()
    assert not a.numbers[a.empty].any() and not a.labels[a.empty].any()


def test_slot_order_means_nothing():
    """Permuted slots: the same numbers, permuted with them (the lord anywhere)."""
    a = ai.build({1: OURS, 2: THEIRS})
    perm = [3, 0, 4, 1, 2]
    b = ai.build({1: [OURS[i] for i in perm], 2: THEIRS})
    assert np.array_equal(b.numbers[:5], a.numbers[perm]) and np.array_equal(b.labels[:5], a.labels[perm])
    assert np.array_equal(b.numbers[5:], a.numbers[5:])


def test_empty_slots_change_nothing_else():
    a = ai.build({1: OURS, 2: THEIRS})
    gappy = [None, OURS[0], "", OURS[1], OURS[2], None, None, OURS[3], OURS[4]]
    b = ai.build({1: gappy, 2: THEIRS + [None]})
    full = [i for i, e in enumerate(gappy) if e]
    assert np.array_equal(b.numbers[full], a.numbers[:5]) and np.array_equal(b.labels[full], a.labels[:5])
    assert b.empty.sum() == a.empty.sum() and np.array_equal(b.numbers[40:], a.numbers[40:])


def test_own_and_enemy_differ_only_by_the_side_flag():
    a = ai.build({1: [ARCHERS], 2: [ARCHERS]})
    diff = np.nonzero(a.numbers[0] != a.numbers[40])[0]
    assert diff.tolist() == [ai.SIDE] and np.array_equal(a.labels[0], a.labels[40])


def test_side_2_is_the_mirror():
    """Side 2 sees its army as its own: the same input as side 1 in the mirrored battle."""
    a = ai.build({1: OURS, 2: THEIRS}, side=2)
    b = ai.build({1: THEIRS, 2: OURS}, side=1)
    for x, y in ((a.numbers, b.numbers), (a.labels, b.labels), (a.empty, b.empty)):
        assert np.array_equal(x, y)
    assert (a.numbers[:6, ai.SIDE] == 1).all()                    # the skaven are "own" for side 2


def test_at_most_40_a_side():
    ai.build({1: [SPEAR] * 40, 2: []})
    with pytest.raises(ValueError):
        ai.build({1: [SPEAR] * 41, 2: []})


@pytest.mark.parametrize("field,value", [("caste", "other"), ("caste", "wizard"), ("size", "huge")])
def test_unknown_caste_or_size_is_an_error(passports, field, value):
    p = copy.deepcopy(passports[SPEAR])
    p[field] = value
    with pytest.raises(ai.UnknownVariant):
        ai.build({1: ["x"], 2: []}, passports={"x": p})


@pytest.mark.parametrize("value", ["high", "dual_high", None])
def test_unknown_trajectory_is_an_error(passports, value):
    p = copy.deepcopy(passports[ARCHERS])
    p["missile"]["trajectory"] = value
    with pytest.raises(ai.UnknownVariant):
        ai.build({1: ["x"], 2: []}, passports={"x": p})


def test_labels(passports):
    a = ai.build({1: [LORD, SPEAR, ARCHERS, GUNS, WOLVES], 2: []})
    traj = [ai.TRAJECTORIES[i] for i in a.labels[:5, 0]]
    caste = [ai.CASTES[i] for i in a.labels[:5, 1]]
    assert traj == ["none", "none", "dual", "direct", "none"]
    assert caste == ["lord", "melee_infantry", "missile_infantry", "missile_infantry", "war_beast"]
    assert (a.labels[:5, 2] == ai.SIZES.index("small")).all()
    for k, p in passports.items():                        # every unit of ours has a known trajectory, caste, size
        ai.slot(ai.final_stats(p, unit=k), True, k)


def test_numbers_span(passports):
    """Every unit's numbers stay within SPAN and are finite; no column is a constant of all (but the side and the
    come shares, 1 in our battles, the fresh unit's state but the ammunition, and what none of ours has)."""
    rows = np.array([ai.slot(ai.final_stats(p, unit=k), True, k)[0] for k, p in passports.items()])
    lo, hi = ai.SPAN
    assert np.isfinite(rows).all() and rows.min() >= lo and rows.max() <= hi
    const = {ai.NAMES[i] for i in range(ai.SIZE) if np.ptp(rows[:, i]) == 0}
    # (resist_magic: the Knights of the Blazing Sun's 25 % since the cavalry joined the pool, 29.10.2026)
    assert const == {"side", "men_come", "hp_come", "resist_all", "resist_flame", "morale", "wavering", "fatigue"}


def test_the_scales_are_the_passports(passports):
    """Where the unit network's passport (passport.features) has the field, the slot's number is the same."""
    same = {"men_full": 0, "hp_full": 1, "hp_per_man": 2, "mass": 3, "walk": 4, "run": 5, "charge": 6,
            "attack": 7, "defence": 8, "charge_bonus": 9, "damage": 10, "ap_damage": 11, "bonus_v_large": 12,
            "bonus_v_infantry": 13, "attack_interval": 14, "splash": 15, "armour": 16, "shield": 17, "leadership": 18,
            "resist_all": 19, "resist_physical": 20, "resist_missile": 21, "resist_magic": 22, "resist_flame": 23,
            "cost": 27, "range": 29, "ammo": 30, "missile_damage": 31, "missile_ap": 32, "reload": 33,
            "projectiles": 35}
    vel = pp.SIZE - 1
    for k, p in passports.items():
        f = np.array(pp.features(p), np.float32)
        nums = ai.slot(ai.final_stats(p, unit=k), True, k)[0]
        for name, col in same.items():
            assert nums[ai.INDEX[name]] == pytest.approx(f[col], abs=1e-6), (k, name)
        assert nums[ai.INDEX["muzzle_velocity"]] == pytest.approx(f[vel], abs=1e-6)
        assert nums[ai.INDEX["fire_arc"]] == pytest.approx(pp.fire_arc([k])[0], abs=1e-6)
        assert nums[ai.INDEX["volley_shots"]] == pytest.approx(pp.weapon([k])[0, 0], abs=1e-6)
        assert nums[ai.INDEX["magic"]] == pp.weapon([k])[0, 1]
        assert nums[[ai.INDEX["aim"], ai.INDEX["silence"]]] == pytest.approx(pp.aim([k])[0], abs=1e-6)


def test_properties(passports):
    a = ai.build({1: [SPEAR, SLAVES, "wh_dlc04_emp_inf_flagellants_0", "wh_dlc04_emp_inf_free_company_militia_0"]})
    cols = [ai.INDEX[n] for n, _ in ai.PROPERTIES]
    assert a.numbers[:4][:, cols].tolist() == [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


SQUARE = {"spread": "square"}


def test_spread_is_the_simulators_square():
    """At the calibration distance, accuracy 100 and a flat shot the square's area is the calibration area; it grows
    as 100 / accuracy and as d^2; nearer than the calibration distance x sqrt(accuracy / 100) it stays."""
    flat = 1e6                                            # muzzle velocity: launch angle ~0 (floored 1e-3)
    assert ai.spread_100(100, 4.0, 100, flat, SQUARE) == pytest.approx(2.0, rel=1e-3)
    assert ai.spread_100(25, 4.0, 100, flat, SQUARE) == pytest.approx(4.0, rel=1e-3)
    assert ai.spread_100(100, 4.0, 50, flat, SQUARE) == pytest.approx(4.0, rel=1e-3)
    assert ai.spread_100(100, 4.0, 200, flat, SQUARE) == pytest.approx(2.0, rel=1e-3)      # p capped at 1
    assert ai.spread_100(0, 4.0, 100, flat, SQUARE) == pytest.approx(20.0, rel=1e-3)       # p floored at 0.01
    v = 45.0
    theta = 0.5 * math.asin(100 * 9.81 / v ** 2)
    assert ai.spread_100(20, 3.7, 95, v, SQUARE) == pytest.approx(
        math.sqrt(3.7 / (0.2 * 0.95 ** 2 * (1 - theta / math.pi))), rel=1e-6)
    assert ai.spread_100(20, 0, 0, v, SQUARE) == 0.0


def test_spread_gauss_is_the_equal_variance_side():
    g = {"spread": "gauss", "shrink": "radius"}
    assert ai.spread_100(20, 4.0, 100, 50, g) == pytest.approx(2.0 * 0.8 * math.sqrt(12), rel=1e-6)
    g = {"spread": "gauss", "shrink": "area"}
    assert ai.spread_100(20, 4.0, 100, 50, g) == pytest.approx(2.0 * math.sqrt(0.8) * math.sqrt(12), rel=1e-6)


def test_spread_follows_sim_json_and_rank(passports):
    s0 = ai.final_stats(passports[ARCHERS])
    m = passports[ARCHERS]["missile"]
    assert s0["spread_100"] == pytest.approx(ai.spread_100(m["accuracy"] + m["marksmanship"],
                                                          m["calibration_area_m"], m["calibration_distance_m"],
                                                          m["muzzle_velocity"], SQUARE))
    s9 = ai.final_stats(passports[ARCHERS], {"rank": 9})
    assert s9["spread_100"] == pytest.approx(ai.spread_100(20 + 27, 3.7, 95, 45, SQUARE))
    assert s9["spread_100"] < s0["spread_100"]
    assert ai.final_stats(passports[SPEAR])["spread_100"] == 0.0


def test_the_card_over_the_passport(passports):
    """The card's final numbers win; walk and charge follow the run's change; the come shares; another key fails."""
    p = passports[SPEAR]
    base = ai.final_stats(p)
    assert base["men_come"] == 1.0 and base["hp_come"] == 1.0
    s = ai.final_stats(p, {"run": p["speed"]["run"] * 1.2, "attack": 40, "men_now": 60, "hp_now": 4140})
    assert s["walk"] == pytest.approx(p["speed"]["walk"] * 1.2) and s["charge"] == pytest.approx(
        p["speed"]["charge"] * 1.2)
    assert s["attack"] == 40 and s["men_come"] == 0.5 and s["hp_come"] == pytest.approx(0.5)
    with pytest.raises(KeyError):
        ai.final_stats(p, {"walk": 2.0})
    with pytest.raises(KeyError):
        ai.final_stats(p, {"accuracy": 50})
    a = ai.build({1: [{"key": SPEAR, "card": {"men_now": 60}}, SPEAR], 2: []})
    assert a.numbers[0, ai.INDEX["men_come"]] == 0.5 and a.numbers[1, ai.INDEX["men_come"]] == 1.0


def test_stack():
    b = ai.stack([ai.build({1: OURS, 2: THEIRS}), ai.build({1: OURS, 2: THEIRS}, 2)])
    assert b.numbers.shape == (2, 80, 49) and b.labels.shape == (2, 80, 3) and b.empty.shape == (2, 80)


def test_the_state_group_is_a_fresh_units_before_the_battle(passports):
    """Before the battle (and for a hidden enemy, units.dynamic) a slot's state is a fresh unit's: morale 1 (/ 2), not
    wavering, fresh, its ammunition full (0 without a missile weapon); armed() - a shooter with ammunition left."""
    a = ai.build({1: OURS, 2: THEIRS})
    I = ai.INDEX
    for i in np.nonzero(~a.empty)[0]:
        row = a.numbers[i]
        assert row[I["morale"]] == 0.5 and row[I["wavering"]] == 0 and row[I["fatigue"]] == 0
        assert row[I["ammo_come"]] == (1.0 if row[I["range"]] > 0 else 0.0)
    assert (ai.armed(a.numbers) == (a.numbers[:, I["range"]] > 0)).all()
    empty = a.numbers.copy()
    empty[:, I["ammo_come"]] = 0
    assert not ai.armed(empty).any()                       # out of ammunition: no shooter
