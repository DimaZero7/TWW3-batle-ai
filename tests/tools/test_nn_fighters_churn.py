"""tools/nn/fighters/churndata.py: who is in reach, which ManList slots keep their man (numpy)."""
import numpy as np

from tools.nn.fighters import churndata as C


def test_in_reach_is_within_3_15_m_of_an_enemy_man():
    rec = {"a": np.array([[0.0, 0.0], [0.0, 5.0]]), "b": np.array([[3.0, 0.0], [10.0, 0.0]]),
           "c": np.array([[0.5, 1.0], [0.0, 1.0]])}
    side = {"a": 1, "b": 2, "c": 1}
    got = C.in_reach(rec, side)
    assert got["a"].tolist() == [True, False]
    assert got["b"].tolist() == [True, False]
    assert got["c"].tolist() == [True, False]          # 3.16 m: out


def test_a_slot_is_sure_while_it_never_jumps_and_stays_in_the_list():
    s0 = np.array([[0.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
    s1 = np.array([[1.0, 0.0], [9.0, 0.0]])              # the second died: the last man took his slot (4 m jump)
    s2 = np.array([[2.0, 0.0], [9.5, 0.0]])
    assert C.sure_slots([s0, s1, s2], 5.0).tolist() == [True, True, False]
    assert C.sure_slots([s0, s1, s2], 2.5).tolist() == [True, False, False]
    assert C.sure_slots([s0, None, s2], 5.0).tolist() == [False, False, False]
