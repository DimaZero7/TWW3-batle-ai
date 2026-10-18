"""tools/nn/fighters/card.py: men per HP and HP of a step replay window, whole or from its settle-th second (numpy)."""
import numpy as np

from tools.nn.fighters import card


def _battle():
    # one window from row 0, one unit of 100 men (HP 100 a man): the game loses 1 man and 1 % of its health a second;
    # the copies lose 2 % a second and kill 1 man a second only from the 5th second on
    T, J, C = 12, card.J, 2
    g_hp = 1.0 - 0.01 * np.arange(T)[:, None]
    g_men = 100.0 - np.arange(T)[:, None]
    h = np.arange(J)
    sim_hp = (1.0 - 0.02 * (h + 1))[None, :, None, None].repeat(C, 2)
    sim_men = (100.0 - np.clip(h + 1 - 5, 0, None))[None, :, None, None].repeat(C, 2).astype(float)
    z = np.zeros((1, J, C, 1), bool)
    return {"starts": np.array([0]), "u_men0": np.array([100]), "u_lord": np.array([False]), "u_hp0": np.array([10000.0]),
            "g_men": g_men, "g_hp": g_hp, "g_r": np.zeros((T, 1), bool), "g_s": np.zeros((T, 1), bool),
            "g_m": np.ones((T, 1), bool), "sim_men": sim_men, "sim_hp": sim_hp, "sim_r": z, "sim_s": z, "sim_gone": z,
            "sim_m": np.ones((1, J, C, 1), bool), "keys": np.array(["wh_main_emp_inf_swordsmen"])}


def test_whole_window_and_settled_seconds():
    rec, _ = card.records(_battle())
    assert np.allclose(rec[0, 3:7], [9.0, 9.0, 18.0, 4.0])      # 9 s: game HP in men's worth, men; copies' HP, men
    rec5, _ = card.records(_battle(), settle=5)
    assert np.allclose(rec5[0, 3:7], [4.0, 4.0, 8.0, 4.0])      # seconds 5..9 only
