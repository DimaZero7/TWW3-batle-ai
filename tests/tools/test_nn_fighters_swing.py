"""The swing probe (charge_probe plan 'swing') and its reading (tools/nn/fighters/swingdata.py): small units cut down
before the go, health and places every 0.1 s; the two attack clocks it tells apart. No torch."""
from tools.nn import charge_probe as cp
from tools.nn.fighters import swingdata


def test_the_swing_plan_cuts_units_to_few_men_and_samples_every_tenth_of_a_second():
    config, _, _ = cp.run_config("swing", 1)
    assert config["tick_ms"] == config["men_ms"] == 100
    lanes = config["lanes"]
    assert len(lanes) == 5 and all(ln["target_mode"] == "both" and ln["gap_m"] == 3 for ln in lanes)
    for ln in lanes:
        n = 1 if ln["kind"] == "one" else 5
        for who, key in (("a", ln["a_key"]), ("t", ln["t_key"])):
            men = next(m for k, m, _ in cp.UNITS.values() if k == key)
            assert men - int(men * ln["damage"][who]["share"]) == n                # kill_number_of_men's floor


def test_the_two_clocks_differ_by_p_plus_half_a_second_over_the_interval():
    p, interval = swingdata.p_hit("wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_clanrats_1")
    assert 0.08 <= p <= 0.9 and interval > 3
    per_swing, miss05 = p / interval, p / (p * interval + 0.5)
    assert abs(per_swing / miss05 - (p + 0.5 / interval)) < 1e-9
    hit, hp_man = swingdata.mean_hit("wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_clanrats_1")
    assert 0 < hit < hp_man
