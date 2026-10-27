"""The executor's local trade (tools/nn/train/local_trade.py, run.py --exec-local-trade): its reward-to-go (plain
arithmetic; the raw trade and its centring: test_nn_local_trade_torch.py)."""
import numpy as np

from tools.nn.train import local_trade


class TestAdvantage:
    def test_a_reward_carried_back_at_gl_without_one_minus_gl(self):
        b = np.zeros((3, 1, 2))
        b[2, 0, 0] = 1.0
        done = np.zeros((3, 1))
        a = local_trade.advantage(b, done, 0.5)
        assert a[:, 0, 0].tolist() == [0.25, 0.5, 1.0] and not a[:, 0, 1].any()

    def test_stops_at_the_battle_end(self):
        b = np.zeros((3, 1, 1))
        b[2, 0, 0] = 1.0
        done = np.array([[0.0], [1.0], [0.0]])          # the battle ended at t = 1: t = 2 is the next battle's
        a = local_trade.advantage(b, done, 0.5)
        assert a[:, 0, 0].tolist() == [0.0, 0.0, 1.0]

