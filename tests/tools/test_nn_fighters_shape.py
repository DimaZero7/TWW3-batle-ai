"""The fighters lab's shape numbers from men's places (tools/nn/fighters/shape.py; numpy, no torch)."""
import numpy as np

from tools.nn.fighters import shape


def block(files, ranks, h, v, front_z, facing=1):
    """Men of a formation: files x ranks, h across, v between ranks, the front rank at front_z, facing +z (facing 1)
    or -z (-1)."""
    x = (np.arange(files) - (files - 1) / 2) * h
    z = front_z - facing * np.arange(ranks) * v
    return np.array([(a, b) for b in z for a in x])


def test_near_counts_two_fronts():
    # two 20-file fronts 2.0 m apart: the first ranks within 3.15 m of each other, the second ranks 3.6 m off
    a = block(20, 3, 1.5, 1.6, 0.0, facing=1)
    e = block(20, 3, 1.5, 1.6, 2.0, facing=-1)
    r = shape.pair_second(a, e, 0.0, 180.0)
    assert r["near_3.15"] == 20
    assert r["contact"]
    assert abs(r["front_gap"] - 2.0) < 1e-6
    assert abs(r["depth_m"] - 3.2 / 0.94) < 0.1                           # p3-p97 of three ranks, over 0.94
    assert 27 < r["width_m"] < 32
    assert r["wrap"] == 0.0


def test_no_contact_far():
    a = block(10, 2, 1.5, 1.6, 0.0)
    e = block(10, 2, 1.5, 1.6, 10.0, facing=-1)
    r = shape.pair_second(a, e, 0.0, 180.0)
    assert not r["contact"] and r["near_3.15"] == 0


def test_wrap_counts_men_round_the_flank():
    e = block(10, 2, 1.5, 1.6, 0.0, facing=-1)            # the enemy faces -z, its front at z 0
    a = block(10, 2, 1.5, 1.6, -2.0, facing=1)            # a in front of it, facing +z
    a = np.concatenate([a, np.array([[9.0, 1.5], [-9.0, 1.5]])])   # two men beside its body
    r = shape.pair_second(a, e, 0.0, 180.0)
    assert r["wrap"] > 0


def test_router_stretch():
    p = block(20, 4, 1.5, 1.6, 0.0)
    later = p + np.array([0.0, 10.0])                    # running +z
    r = shape.router_second(p, later)
    assert 4 < r["along"] < 6                            # p90 - p10 of 4 ranks 1.6 apart
    assert 20 < r["across"] < 25


def test_xy_decimetres():
    assert shape.xy([10, 20, 30, 40]).tolist() == [[1.0, 2.0], [3.0, 4.0]]
