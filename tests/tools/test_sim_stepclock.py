"""The step replay's hidden clocks rebuilt from a recording (tools/nn/sim/stepclock.py): numpy only."""
import json

import numpy as np

from tools.nn import gamedata
from tools.nn.sim import stepclock

CAL = {"dt": 0.5, "fade_s": 30.0, "hist_s": 60.0, "recent_window_s": 4.0, "reset_s": 10.0, "min_share": 0.75,
       "min_runup": 10.0, "decay": 13.0, "rout_pin_s": 7.0}


def battle(T=40):
    """Unit 0 (side 1, a formation) and unit 1 (side 2, the lord) standing 100 m apart; every field set."""
    f = {k: np.zeros((T, 2)) for k in ("x", "z", "a", "mp", "ms")}
    f["x"][:, 0], f["x"][:, 1] = -50.0, 50.0
    f["hp"] = np.ones((T, 2))
    f["men"] = np.tile([100.0, 1.0], (T, 1))
    for k in ("m", "r", "s", "w", "fire", "mv", "f"):
        f[k] = np.zeros((T, 2), bool)
    return gamedata.Battle(run="t", own_ai="attack", enemy_role="defend", result={}, t=np.arange(T, dtype=float), f=f,
                           target=np.full((T, 2), -1), names=("own_a", "enemy_lord"), keys=("k0", "k1"),
                           side=np.array([1, 2]))


INFO = {"hp0": np.array([1000.0, 500.0]), "run": np.array([4.0, 5.0]), "men0": np.array([100.0, 1.0]),
        "reload": np.zeros(2), "lord": np.array([False, True])}


def test_leaving_the_map_is_not_a_loss_and_sets_gone():
    g = battle()
    g.f["r"][10:, 0] = True
    g.f["hp"][5:, 0] = 0.4            # 60 % lost at row 5
    g.f["men"][20:, 0] = 0.0          # off the map at row 20: no men, the health kept
    L = stepclock.losses(g, INFO["hp0"])
    assert L[5, 0] == np.float64(600.0) and L[20:, 0].sum() == 0
    alive, dead, gone = stepclock.status(g)
    assert gone[20:, 0].all() and not gone[19, 0] and not dead[:, 0].any()
    c = stepclock.clocks(g, [19, 25], INFO, CAL)
    assert c["gone"][0, 0] == 0 and c["gone"][1, 0] == 1 and c["gone_s"][1, 0] > 5


def test_casualty_windows_hold_the_last_4_and_60_seconds_newest_first():
    g = battle()
    g.f["hp"][10:, 0] = 0.9           # 100 HP lost in row 10
    g.f["hp"][12:, 0] = 0.8           # 100 more in row 12
    c = stepclock.clocks(g, [12, 13, 15, 16, 30], INFO, CAL)
    h = c["lost_hist"]
    assert h.shape[1] == 120
    assert np.allclose(h[0, :2, 0], 50.0) and np.allclose(h[0, 4:6, 0], 50.0)   # row 12 newest, row 10 two rows back
    assert np.isclose(c["recent"][0, 0], 200.0) and np.isclose(c["recent"][2, 0], 100.0)            # row 15: only row 12 within 4 s
    assert c["recent"][3, 0] == 0.0 and np.isclose(c["extended"][4, 0], 200.0)


def test_rout_rally_and_waver_clocks():
    g = battle()
    g.f["w"][5:8, 0] = True
    g.f["r"][8:14, 0] = True
    g.f["r"][20:, 0] = True
    c = stepclock.clocks(g, [7, 10, 16, 22], INFO, CAL)
    assert c["waver_s"][0, 0] == 2.5 and c["waver_s"][1, 0] == 0
    assert list(c["rout_count"][:, 0]) == [0, 1, 1, 2]
    assert c["rout_s"][1, 0] == 2 and c["rout_s"][3, 0] == 2
    assert c["rally_s"][2, 0] == 2 and c["rally_s"][0, 0] >= 1e6


def test_melee_clock_bridges_short_gaps_and_the_charge_follows_a_run_in():
    g = battle()
    # unit 0 runs at the lord from row 0 at 5 m/s (over 0.75 x its run speed 4), contact at row 18
    for k in range(1, 19):
        g.f["x"][k, 0] = -50.0 + 5.0 * k
    g.f["x"][18:, 0] = 40.0
    g.target[:, 0] = 1
    g.f["m"][18:25, :] = True
    g.f["m"][28:, :] = True          # 3 s out: the same fight
    c = stepclock.clocks(g, [20, 27, 30], INFO, CAL)
    assert c["contact_s"][0, 0] == 3 and c["contact_s"][2, 0] == 7 + 3
    assert c["out_s"][1, 0] == 3 and c["melee_s"][2, 0] == 10
    assert 0.8 < c["charge"][0, 0] < 1.0 and c["ran_in"][0, 0] == 1
    assert c["charge"][0, 1] == 0                     # the lord stood


def test_dealt_goes_to_the_named_attacker_and_lord_fall():
    g = battle()
    g.f["m"][10:, :] = True
    g.target[10:, 0] = 1
    g.f["hp"][12:, 1] = 0.5           # the lord loses 250 HP in melee at row 12
    g.f["hp"][15:, 1] = 0.0           # and dies at row 15
    g.f["men"][15:, 1] = 0.0
    c = stepclock.clocks(g, [12, 20], INFO, CAL)
    assert np.isclose(c["dealt"][0, 0], 250.0) and np.isclose(c["taken"][0, 1], 250.0)
    assert c["shot_taken"][0, 1] == 0
    assert c["lord_dead_s"][0, 1] == -1 and c["lord_dead_s"][1, 1] > 4 and c["dead_s"][1, 1] > 4


def test_casts_and_ability_timers():
    on = np.zeros((40, 1), bool)
    on[5:15, 0] = True
    cast = stepclock.casts(on)
    assert list(np.flatnonzero(cast[:, 0])) == [5]
    o, cd = stepclock.ability_timers(cast, [4, 5, 10, 30], active_s=10.0, recharge_s=20.0)
    assert list(o[:, 0]) == [0.0, 10.0, 5.0, 0.0]
    assert list(cd[:, 0]) == [0.0, 30.0, 25.0, 5.0]


def test_effect_rows_reads_nn_effects(tmp_path):
    g = battle(10)
    lines = [{"event": "nn_effects", "t": 2000, "u": "enemy_lord", "fx": ["a_key"]},
             {"event": "nn_effects", "t": 6000, "u": "enemy_lord", "fx": []}]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    fx = stepclock.effect_rows(tmp_path, g)
    assert list(np.flatnonzero(fx["a_key"][:, 1])) == [2, 3, 4, 5]
