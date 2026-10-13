"""The bonus price as the training uses it (tools/nn/abilities_price/train_price.py). numpy only."""
import json

import numpy as np

from tools.nn.abilities_price import ability as abl
from tools.nn.abilities_price import bonuses as bz
from tools.nn.abilities_price import train_price as tp
from tools.nn.abilities_price import validate as va

SYG = "wh_main_character_abilities_stand_your_ground"
FS = "wh_main_character_abilities_foe_seeker"
DO = "wh_main_character_abilities_deadly_onslaught"


def test_the_config_prices_the_three_valid_abilities_with_a_recharge_fit_each():
    pr = tp.load()
    assert pr.valid == {SYG, FS, DO}
    assert not pr.ok("wh_main_character_abilities_rally") and not pr.ok(
        "wh2_main_character_abilities_verminous_valour")
    for k in pr.valid:
        r = pr.spec["recharge"][k]
        assert len(r["beta"]) == len(r["mu"]) == len(r["sd"]) == tp.SITUATION_N == len(va.SITUATION)
    assert set(pr.rates) >= {"melee_damage", "melee_defence", "vigour", "morale", "charge", "speed", "charge_speed"}
    v = pr.curves["vigour"]
    assert len(v["rates"]) == len(v["edges"]) - 1 and all(x >= 0 for x in v["rates"])


def test_the_price_is_the_benefit_less_debuffs_and_the_recharge_cost_mean_over_copies():
    rng = np.random.default_rng(3)
    pr = tp.load()
    trs = [va.random_trace(rng, S=60, U=6, active=25)[0] for _ in range(3)]
    p = pr.price(FS, trs, t=40.0)
    per = [pr.gold(FS, tr) for tr in trs]
    assert np.isclose(p["benefit"], np.mean([b for b, _, _ in per]))
    assert np.isclose(p["price"], p["benefit"] - p["debuffs"] - p["recharge"])
    assert p["recharge"] >= 0 and np.allclose(p["copies"], [b - d - p["recharge"] for b, d, _ in per])


def test_flat_kinds_are_rate_times_counter_and_vigour_takes_the_tier_rate_at_the_press():
    rng = np.random.default_rng(5)
    pr = tp.load()
    a = pr.abilities[DO]
    tr = va.random_trace(rng, S=60, U=6, active=31)[0]
    ben, deb, counts = pr.gold(DO, tr)
    assert deb == 0 and np.isclose(ben, sum(pr.rates[k] * c for k, c in a.counters(tr, pr.bricks).items()))
    # foe seeker's vigour: the counter x the curve's rate of the owner's tier at sample 0
    tr2 = dict(tr)
    tr2["fatigue"] = tr["fatigue"].copy()
    tr2["fatigue"][0, 0] = 20000.0                                    # very tired at the press
    _, _, c2 = pr.gold(FS, tr2)
    k = pr._vigour_rate(tr2)
    assert np.isclose(k, pr.curves["vigour"]["rates"][int(bz.tier(20000.0))])
    ben2, _, _ = pr.gold(FS, tr2)
    flat = sum(pr.rates[kd] * c for kd, c in c2.items() if kd != "vigour")
    assert np.isclose(ben2, flat + k * c2.get("vigour", 0.0))


def test_zero_work_gives_zero_benefit_and_the_situation_reads_sample_zero_only():
    pr = tp.load()
    S1, U = 61, 4
    z = np.zeros((S1, U), dtype=bool)
    tr = {"alive": ~z, "r": z.copy(), "m": z.copy(), "mv": z.copy(), "sprint": z.copy(), "charge": np.zeros((S1, U)),
          "mmax": np.full(U, 60.0), "morale": np.full((S1, U), 60.0), "fatigue": np.zeros((S1, U)),
          "x": np.zeros((S1, U)), "z": np.zeros((S1, U)), "on": np.clip(19 - np.arange(S1), 0, None).astype(float),
          "owner": 0, "dt_s": 1.0}
    tr["on"][0] = 0.0
    p = pr.price(SYG, [tr], t=10.0)
    assert p["work"] == 0 and p["benefit"] == 0                       # nobody fights, morale steady: no useful work
    x = tp.situation(tr, 10.0)
    tr["m"][5:] = True                                                 # later melee: the situation does not see it
    assert np.array_equal(x, tp.situation(tr, 10.0)) and x[1] == 10.0 / 600.0


def test_the_spec_from_a_lab_card_floors_its_bins_and_keeps_only_the_curve_kinds():
    lab = {"rates": {"vigour": {"k": 0.02}, "morale": {"k": 0.03}},
           "curves": {"whole battles": {
               "vigour": {"tier": [{"lo_x": -0.5, "k": -0.1}, {"lo_x": 0.5, "k": 0.01}]},
               "morale": {"morale_at_press": [{"lo_x": 16.0, "k": 1.0}]}}}}
    rates, curves = tp.lab_spec(lab)
    assert rates == {"vigour": 0.02, "morale": 0.03} and set(curves) == {"vigour"}
    assert curves["vigour"]["rates"][:2] == [0.0, 0.01] and curves["vigour"]["rates"][2:] == [0.02] * 4
    _, both = tp.lab_spec(lab, ("vigour", "morale"))
    assert both["morale"]["rates"][1] == 1.0 and both["morale"]["rates"][0] == 0.03
