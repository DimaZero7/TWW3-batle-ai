"""Learning by observation (tools/nn/observe): the reading of new orders and the advantage (numpy only)."""
import math

import numpy as np

from tools.nn.observe import advantage as adv
from tools.nn.observe import labels as lab
from tools.nn.observe import record
from tools.nn.sim.orders import ATTACK, HOLD, MOVE

SIDE = np.array([1, 1, 1, 2, 2])


def frames(T=5):
    """5 units (own 0-2, enemy 3-4) standing still, no targets, the order points on the units."""
    N = len(SIDE)
    raw = {k: np.zeros((T, N), bool) for k in ("r", "s", "m", "mv", "f", "fire")}
    raw["x"] = np.tile(np.array([0.0, 0.0, 0.0, 300.0, 300.0]), (T, 1))
    raw["z"] = np.tile(np.array([0.0, 50.0, 100.0, 0.0, 50.0]), (T, 1))
    raw["b"] = np.full((T, N), 90.0)
    raw["men"] = np.full((T, N), 100.0)
    raw["ox"], raw["oz"] = raw["x"].copy(), raw["z"].copy()
    raw["target"] = np.full((T, N), -1)
    raw["vis"] = np.ones((T, N), bool)
    return raw


def read(raw, missile=None, human=True, target_ok=None, widths=None):
    T, N = raw["x"].shape
    ctrl = np.tile(SIDE == 1, (T, 1))
    ok = np.tile(SIDE == 2, (T, 1)) if target_ok is None else target_ok
    return lab.read(raw, SIDE, 1, ctrl, ok, np.zeros(N, bool) if missile is None else missile, human, widths)


def test_a_new_engine_target_is_an_attack_labelled_on_the_decision_before_it():
    raw = frames()
    raw["target"][2:, 0] = 3
    raw["mv"][2:, 0] = raw["f"][2:, 0] = True
    out = read(raw)
    assert out["kind"][1, 0] == ATTACK and out["target"][1, 0] == 3
    assert out["run"][1, 0] and out["run_known"][1, 0]
    assert (out["kind"][:, 0] >= 0).sum() == 1                 # the same target later: nothing new


def test_a_shooter_firing_at_will_and_an_unseen_target_give_no_label():
    raw = frames()
    raw["target"][2:, 0] = 3
    raw["fire"][:, 0] = True
    missile = np.array([True, False, False, False, False])
    assert (read(raw, missile)["kind"] < 0).all()
    raw = frames()
    raw["target"][2:, 0] = 4
    ok = np.tile(SIDE == 2, (5, 1))
    ok[:, 4] = False
    assert (read(raw, target_ok=ok)["kind"] < 0).all()


def test_a_moved_point_is_a_move_running_a_set_with_withdraw_and_a_near_point_a_hold():
    raw = frames()
    raw["ox"][3:, 1], raw["oz"][3:, 1] = 120.0, 60.0
    raw["mv"][3:, 1] = raw["f"][3:, 1] = True
    raw["ox"][2:, 2], raw["oz"][2:, 2] = 5.0, 103.0            # 6 m away, it does not move
    out = read(raw)
    assert out["kind"][2, 1] == MOVE and out["px"][2, 1] == 120.0 and out["pz"][2, 1] == 60.0
    assert out["alt"][2, 1] and out["run"][2, 1]
    assert out["kind"][1, 2] == HOLD
    # the game's halt: the unit stands the first second after, its new point up to STOP_M away
    raw2 = frames()
    raw2["ox"][2:, 2] = 30.0
    raw2["mv"][:2, 2] = True
    assert read(raw2)["kind"][1, 2] == HOLD
    # walking: no set
    raw["f"][:] = False
    out = read(raw)
    assert out["kind"][2, 1] == MOVE and not out["alt"][2, 1]


def test_the_front_point_goes_to_the_formations_centre():
    raw = frames()
    raw["ox"][2:, 1] = 100.0                                   # facing +x (b 90): the front is ahead of the centre
    raw["mv"][2:, 1] = True
    widths = [None, 20.0, None, None, None]
    out = read(raw, widths=widths)
    back = lab.half_depth(100.0, 20.0)
    assert back > 0 and math.isclose(out["px"][1, 1], 100.0 - back)


def test_the_game_ais_melee_unit_moving_its_point_in_melee_is_no_order_a_humans_is():
    raw = frames()
    raw["m"][:, 1] = True
    raw["ox"][2:, 1] = 200.0
    raw["mv"][2:, 1] = True
    assert (read(raw, human=False)["kind"] < 0).all()
    assert read(raw, human=True)["kind"][1, 1] == MOVE


def test_infer_orders_and_the_map_clip():
    raw = {k: v[0] for k, v in frames().items()}
    raw["target"][0] = 3
    raw["ox"][1], raw["oz"][1] = 5000.0, 50.0
    raw["mv"][1] = True
    names = [f"u{i}" for i in range(len(SIDE))]
    out = record.infer_orders(raw, names, SIDE, own=1)
    assert out["u0"] == {"unit": "u0", "kind": "attack", "target": "u3"}
    assert out["u1"]["kind"] == "move" and out["u2"]["kind"] == "hold"
    ox, oz = record.clip_points(np.array([5000.0, np.nan]), np.array([-9000.0, 1.0]), (-768, 768, -800, 736))
    assert ox[0] == 768 and oz[0] == -800 and np.isnan(ox[1])


def test_the_reward_the_advantage_and_the_window():
    gold = np.array([[0.0, 0.0], [0.0, 0.1], [0.05, 0.3], [0.05, 0.3]])
    r1 = adv.rewards(gold, winner=1, side=1)
    assert np.allclose(r1, [0.1, 0.15, 1.0, 0.0])
    r2 = adv.rewards(gold, winner=1, side=2)
    assert np.allclose(r2, [-0.1, -0.15, -1.0, 0.0])
    v = np.array([0.5, 0.2, -0.1, 0.0])
    a, ret = adv.gae(r1, v, gamma=1.0, lam=1.0)
    G = np.array([1.25, 1.15, 1.0, 0.0])
    assert np.allclose(a[:3], G[:3] - v[:3]) and a[3] == 0 and ret[3] == 0
    t = np.array([0.0, 10.0, 20.0, 30.0])
    trade, centred = adv.window(gold, t, 1, seconds=20.0)
    assert np.isclose(trade[0], 0.3 - 0.05) and np.isclose(centred.mean(), 0.0)
