"""Line g43 (25.10.2026): the new units - Huntsmen, Nuln Ironsides, Hochland Long Rifles, the four Gutter Runners, the
Warplock Jezzails - their passports from the game's database, their effects in the catalogue, the v3 army scheme's pools
(v2 and v1 bit for bit) and the network's new token columns (FX_NEW, FX_STATS2, SIGHT). No torch: the mechanics' tests
are test_nn_g43_sim.py (the training container)."""
import json

import numpy as np
import pytest

from tools.nn.armies import generate as G
from tools.nn.armies import pools as P
from tools.nn.model import effects as mfx
from tools.nn.model import observation as ob
from tools.nn.model import passport, sources
from tools.nn.sim import params as sim_params

HUNTSMEN, IRONSIDES, LONG_RIFLES = ("wh2_dlc13_emp_inf_huntsmen_0", "wh3_dlc25_emp_inf_nuln_ironsides",
                                    "wh3_dlc25_emp_inf_hochland_long_rifles")
RUNNERS = ("wh2_main_skv_inf_gutter_runners_0", "wh2_main_skv_inf_gutter_runners_1",
           "wh2_main_skv_inf_gutter_runner_slingers_0", "wh2_main_skv_inf_gutter_runner_slingers_1")
JEZZAILS = "wh2_dlc12_skv_inf_warplock_jezzails_0"
NEW = (HUNTSMEN, IRONSIDES, LONG_RIFLES) + RUNNERS + (JEZZAILS,)
U = passport.load()
SNARE, BLACKPOWDER = "wh2_main_unit_passive_snare_net", "wh3_dlc25_unit_passive_blackpowder_discipline"
BREAKER, POISON = "wh2_dlc09_unit_contact_shieldbreaker", "wh_main_unit_contact_poison"


def test_the_passports_hold_the_database_numbers():
    j = U[JEZZAILS]
    assert j["men"] == 32 and j["shield"]["missile_block_chance"] == 55 and j["radii_ratio"] == 0.6
    assert j["missile"]["range_m"] == 275 and j["missile"]["reload_s"] == 12.0 and j["missile"]["direct"]
    assert j["missile"]["contact_phase"] == BREAKER and j["spotting"] == {"scrub": 160, "tree": 160}
    assert U[LONG_RIFLES]["missile"]["range_m"] == 260 and U[LONG_RIFLES]["spacing"]["chaotic"]
    assert U[IRONSIDES]["missile"]["reload_s"] == 7.0 and BLACKPOWDER in U[IRONSIDES]["abilities"]
    for k in RUNNERS:
        r = U[k]
        assert r["men"] == 120 and r["hiding_scalar"] == 0.8 and "stalk" in r["attributes"], k
        assert r["spotting"]["scrub"] == 80 and SNARE in r["abilities"] and r["speed"]["run"] == 5.4, k
    assert U[RUNNERS[0]]["missile"]["fire_arc_deg"] == 180.0 and "mounted_fire_move" in U[RUNNERS[0]]["attributes"]
    assert U[RUNNERS[1]]["missile"]["contact_phase"] == POISON and U[RUNNERS[3]]["missile"]["contact_phase"] == POISON
    assert U[RUNNERS[2]]["missile"]["category"] == "sling" and "stalk" in U[HUNTSMEN]["attributes"]
    # every passport has the new fields: spotting, the penetration's numbers and a man's resistance
    for k, u in U.items():
        if k in json.loads(passport.PASSPORTS.read_text(encoding="utf-8"))["units"]:
            assert "spotting" in u and u["projectile_resistance"] >= 1, k
            assert u["missile"] is None or "penetration_max" in u["missile"], k


def test_a_bullet_strikes_by_the_penetration_rule():
    # medium_1 (max 1, cap medium): two infantrymen; low (cap small): one; very_low: one; a Giant Rat (resistance 2) stops
    # medium_1 in the first
    assert U[JEZZAILS]["missile"]["penetration"] == U[LONG_RIFLES]["missile"]["penetration"] == "medium_1"
    assert passport.men_struck(U[JEZZAILS]) == passport.men_struck(U[LONG_RIFLES]) == 2
    assert U[IRONSIDES]["missile"]["penetration"] == "low" and passport.men_struck(U[IRONSIDES]) == 1
    assert passport.men_struck(U["wh_main_emp_inf_handgunners"]) == 1 and passport.men_struck(U[HUNTSMEN]) == 1
    assert passport.men_struck(U[JEZZAILS], resistance=U["wh2_dlc16_skv_mon_wolf_rats_0"]["projectile_resistance"]) == 1
    assert passport.men_struck(U["wh_main_emp_inf_swordsmen"]) == 1             # no missile weapon
    # every unit of the old pools strikes one man (their battles do not change)
    old = json.loads(P.POOLS.read_text(encoding="utf-8"))["factions"]
    for f in old.values():
        for spec in f["units"] + f.get("units_v2", []):
            assert passport.men_struck(U[spec["key"]]) == 1, spec["key"]


def test_the_simulators_static_fields():
    pp = sim_params.load()
    j, r, i = pp.static(JEZZAILS), pp.static(RUNNERS[0]), pp.static(IRONSIDES)
    assert j["spot_m"] == 160 and j["hide_k"] == 1.0 and j["pen_max"] == 1 and j["pen_cap"] == 2 and j["shield"] == 0.55
    assert r["spot_m"] == 80 and r["hide_k"] == 0.8 and r["stalk"] and not j["stalk"] and r["fire_move"]
    assert i["reload_db"] == 7.0 and i["pen_cap"] == 1 and i["fxd"] >= 0 and j["fxd"] == -1
    assert pp.static("wh2_dlc16_skv_mon_wolf_rats_0")["pen_res"] == 2 and r["size_k"] == 1
    assert j["sfx"] >= 0 and r["sfx"] == -1 and pp.static(RUNNERS[1])["sfx"] >= 0       # shield breaker, poison


def test_the_effects_catalogue():
    doc = mfx.load()
    e = doc["effects"]
    assert e["stalk"]["modelled"] and e["stalk"]["rules"] == ["stalk"]
    assert not e["snipe"]["modelled"] and not e["ignore_trees"]["modelled"] and not e["hide_forest"]["modelled"]
    net = e[SNARE]
    assert net["modelled"] and net["range_m"] == 35.0 and net["kind"] == "passive"
    assert {(s["stat"], s["value"], tuple(s["on"])) for s in net["stats"]} == {
        ("scalar_speed", 0.9, ("enemies",)), ("scalar_charge_speed", 0.9, ("enemies",))}
    bp = e[BLACKPOWDER]
    assert bp["modelled"] and bp["delay_s"] == 10.0 and sorted(bp["off_when"]) == ["climbing", "moving"]
    assert {(s["stat"], s["value"]) for s in bp["stats"]} == {("stat_accuracy", 40.0), ("stat_reloading", 10.0)}
    br = e[BREAKER]
    assert br["kind"] == "contact" and br["modelled"] and br["duration_s"] == 10.0
    assert [(s["stat"], s["how"], s["value"], s["sim"]) for s in br["stats"]] == [
        ("stat_missile_block_chance", "add", -24.0, "shield")]
    # the order is append-only: the 19 effects before line g43 keep their places, the cavalry's (29.10.2026) after g43's
    assert len(doc["order"]) == ob.FX_MID + (len(ob.FX_NEW) + len(ob.FX_CAV) + len(ob.FX_MON)) // 2
    assert set(doc["order"][ob.FX_MID:ob.FX_MID + ob.FX_G43]) == {
        "ignore_trees", "snipe", "stalk", BREAKER, SNARE, BLACKPOWDER}
    # ... and the artillery's cant_run (29.10.2026: catalogued, no rule - its run speed is its walk's)
    # ... the Skaven monsters' terror and regeneration (29.10.2026)
    assert doc["order"][ob.FX_MID + ob.FX_G43:] == ["causes_fear", "immune_to_psychology", "cant_run", "causes_terror",
                                                    "wh_main_unit_passive_regeneration",
                                                    # the Pusbags (31.10.2026)
                                                    "wh3_dlc29_unit_passive_pestilent_perfection",
                                                    "wh3_dlc29_unit_passive_unshakable_pathogen"]


def test_the_v3_pools_add_the_new_units_and_v2_stays():
    v1, v2, v3 = (P.load(scheme=s) for s in ("v1", "v2", "v3"))
    for f in v3:
        k1, k2, k3 = ({u.key for u in p[f].units} for p in (v1, v2, v3))
        assert k1 < k2 < k3 and not (k2 & set(NEW))
    assert {u.key for p in v3.values() for u in p.units} >= set(NEW)
    seen = set()
    for seed in range(200):
        b = G.battle(seed, armies="v3")
        seen |= {u["key"] for s in b["sides"].values() for u in s["units"]}
        b2 = G.battle(seed, armies="v2")
        assert not {u["key"] for s in b2["sides"].values() for u in s["units"]} & set(NEW)
    assert seen >= set(NEW)


def sight_battle(vis):
    """One battle: own Gutter Runners and Jezzails (side 1) against spearmen with shields and Ironsides; vis [4]."""
    keys = [RUNNERS[0], JEZZAILS, "wh_main_emp_inf_spearmen_1", IRONSIDES]
    _, state = sources.synthetic(batch=1, own=2, enemy=2, seed=3, keys=keys)
    setup = ob.Setup(keys=[keys], side=np.array([[1, 1, 2, 2]]), bounds=np.array([sources.CROSSROADS], np.float32),
                     factions=[("wh2_main_skv_skaven", "wh_main_emp_empire")], attacker=np.ones(1, int))
    return setup, dict(state, vis=np.array([vis]), men=setup.men0.copy())


def test_the_sight_columns():
    setup, state = sight_battle([False, True, True, False])
    obs, _ = ob.observe(state, setup, 1)
    cobs, _ = ob.observe(state, setup, 1, full=True)
    t, c = obs.tokens[0], cobs.tokens[0]
    h, p, s = (ob.INDEX[n] for n in ob.SIGHT)
    own = setup.side[0] == 1
    hid = np.array([False, True, True, False]) == False      # noqa: E712 - the vis given above, negated
    for i in range(4):
        if own[i]:
            assert t[i, h] == float(hid[i]), i                 # own units: hidden from the enemy
        assert c[i, h] == float(hid[i]), i                     # the critic sees every unit's
    keys = setup.keys[0]
    want = passport.sight(keys)
    for i in range(4):
        if own[i] or not hid[i]:
            assert t[i, p] == pytest.approx(want[i, 0]) and t[i, s] == pytest.approx(want[i, 1]), i
    assert want[keys.index(JEZZAILS)].tolist() == pytest.approx([0.25, 1.0])
    assert want[keys.index(RUNNERS[0])].tolist() == pytest.approx([0.0, 0.5])
    assert not set(ob.INDEX[n] for n in ob.SIGHT) & set(ob.OWN_ONLY)


def test_the_new_effect_pairs_and_stats_are_the_tokens_tail():
    c = len(ob.FX_LAST)          # (the cavalry's pairs after them, the monsters' fire stats and pairs: 29.10.2026)
    g = len(ob.FX_NEW) + len(ob.FX_STATS2) + len(ob.SIGHT) + c
    assert ob.NAMES[-g:] == ob.FX_NEW + ob.FX_STATS2 + ob.SIGHT + ob.FX_LAST and ob.TAIL[-6:] == (
        ob.FX_NEW, ob.FX_STATS2, ob.SIGHT, ob.FX_CAV, ob.FX_STATS3, ob.FX_MON)
    assert ob.FX_LAST == ob.FX_CAV + ob.FX_STATS3 + ob.FX_MON
    assert ob.FX_NEW + ob.FX_CAV + ob.FX_MON == mfx.NAMES[2 * ob.FX_MID:] and ob.NAMES[:ob.INDEX["volley_ready"]][-2 * ob.FX_MID:] == mfx.NAMES[
        :2 * ob.FX_MID]
    # the token before line g43: today's first columns; recorded tokens of that width widen with zeros
    old = ob.TOKEN - g
    assert ob.old_columns(old) == list(range(old))
    from tools.nn.observe.tokens import widen
    tok = np.random.default_rng(1).random((2, 3, old)).astype(np.float32)
    out = widen(tok)
    assert out.shape == (2, 3, ob.TOKEN) and np.array_equal(out[..., :old], tok) and not out[..., old:].any()


def test_the_shield_breaker_and_the_net_show_as_numbers():
    order = mfx.keys()
    fx = (1 << order.index(BREAKER)) | (1 << order.index(BLACKPOWDER))
    setup, state = sight_battle([True, True, True, True])
    state = dict(state, fx_on=np.array([[0, 0, fx, fx]]))
    cobs, _ = ob.observe(state, setup, 1, full=True)
    t = cobs.tokens[0]
    col = {n: ob.INDEX[n] for n in ob.FX_STATS2}
    for i in (2, 3):
        assert t[i, col["fxs_stat_missile_block_chance_add"]] == pytest.approx(-0.24), i     # -24 / 100
    # Blackpowder Discipline on its owner (the Ironsides, 3) only: +40 accuracy, +10 reload skill, / 100
    assert t[3, col["fxs_stat_accuracy_add"]] == pytest.approx(0.4)
    assert t[3, col["fxs_stat_reloading_add"]] == pytest.approx(0.1)
    assert t[2, col["fxs_stat_accuracy_add"]] == 0 and t[0, col["fxs_stat_missile_block_chance_add"]] == 0
