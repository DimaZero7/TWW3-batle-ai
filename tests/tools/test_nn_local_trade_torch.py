"""The executor's local trade with torch (tools/nn/train/local_trade.py, run.py --exec-local-trade): the raw trade of a
decision (gold struck, the routs and the lords by the share of health struck, the own loss) and its centring by class.
torch: the training container, on the CPU."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tools.nn.train import craft, local_trade, reward  # noqa: E402

W = reward.Weights()


def units(**over):
    """One battle, 4 slots: 0 (spears) and 1 (shooters) side 1, 2 (melee) and 3 (the lord) side 2; cost 500 each
    (budget 1000), full health, targets none."""
    f = (lambda v, dt=torch.float32: torch.tensor([v], dtype=dt))
    u = {"side": f([1, 1, 2, 2], torch.long), "men": f([100.0, 80, 100, 1]), "men0": f([100.0, 80, 100, 1]),
         "gone": f([False] * 4, torch.bool), "s": f([False] * 4, torch.bool), "r": f([False] * 4, torch.bool),
         "cost": f([500.0] * 4), "hp_abs": f([1000.0] * 4), "hp0": f([1000.0] * 4), "gold_out": f([0.0] * 4),
         "target": f([-1] * 4, torch.long), "lord": f([False, False, False, True], torch.bool),
         "lost_worst": f([0.0] * 4), "range": f([0.0, 150, 0, 0]), "ammo0": f([0.0, 20, 0, 0]),
         "run": f([4.0, 4, 4, 5]), "large": f([False] * 4, torch.bool), "bonus_v_large": f([15.0, 0, 0, 0])}
    for k, v in over.items():
        u[k] = f(v, u[k].dtype)
    return u


def trade(before, after, mine=(True, True, False, False), weights=W):
    a, b = local_trade.snapshot(before, weights), local_trade.snapshot(after, weights)
    x, eng = local_trade.raw(a, b, weights, torch.tensor([list(mine)]))
    return x[0].tolist(), eng[0].tolist()


class TestRaw:
    def test_health_struck_the_rout_by_share_of_health_and_the_own_loss(self):
        # 0 and 1 strike enemy 2 for 100 gold of health each (400 of its 1000 hp), it routs: its loss beyond health
        # half each; 1 loses 150 gold of health
        after = units(gold_out=[100.0, 100, 0, 0], target=[2, 2, -1, -1], hp_abs=[1000.0, 700, 600, 1000],
                      r=[False, False, True, False])
        x, eng = trade(units(), after)
        # enemy 2: 200 of health, routing 0.5 x the 300 left = 150 beyond health, 75 each
        assert x[0] == pytest.approx((100 + 75) / 1000) and x[1] == pytest.approx((100 + 75 - 150) / 1000)
        assert x[2] == 0 and x[3] == 0 and eng == [True, True, False, False]

    def test_the_target_at_the_start_else_at_the_end_and_friends_not_counted(self):
        # 0 aimed at its friend 1 at the start (no enemy): its strike is health only; 1 had none, at the end enemy 2
        before = units(target=[1, -1, -1, -1])
        after = units(gold_out=[50.0, 50, 0, 0], target=[1, 2, -1, -1], hp_abs=[1000.0, 1000, 800, 1000],
                      gone=[False, False, True, False])
        x, _ = trade(before, after)
        # enemy 2 gone: lost whole (500), its health part 100, so 400 beyond health - all on 1 (0's strike not on it)
        assert x[0] == pytest.approx(50 / 1000) and x[1] == pytest.approx((50 + 400) / 1000)

    def test_lords_the_enemy_fall_on_its_attackers_the_own_on_the_lord(self):
        after = units(gold_out=[30.0, 0, 0, 0], target=[3, -1, -1, -1], hp_abs=[1000.0, 1000, 1000, 940],
                      men=[100.0, 80, 100, 0])
        x, _ = trade(units(), after)
        # the enemy lord dead: 500 lost (30 health struck + 470 beyond on 0) and the lord term 0.3 on 0
        assert x[0] == pytest.approx(500 / 1000 + W.lord)
        # seen from side 2: its lord loses his cost and the lord term
        x2, _ = trade(units(), after, mine=(False, False, True, True))
        assert x2[3] == pytest.approx(-500 / 1000 - W.lord) and x2[2] == 0

    def test_lord_rout_a_routing_lord_a_share_of_a_death(self):
        w = reward.Weights(lord_rout=0.5)
        after = units(gold_out=[10.0, 0, 0, 0], target=[3, -1, -1, -1], hp_abs=[1000.0, 1000, 1000, 980],
                      r=[False, False, False, True])
        x, _ = trade(units(), after, weights=w)
        # 10 health struck off the lord, beyond health 0.5 x (500 - 10) = 245; lord term 0.3 x 0.5
        assert x[0] == pytest.approx((10 + 245) / 1000 + w.lord * 0.5)

    def test_nothing_struck_or_lost_not_engaged(self):
        x, eng = trade(units(), units())
        assert x == [0, 0, 0, 0] and eng == [False] * 4


class TestLocal:
    def test_centred_by_its_class_s_running_mean_and_zero_out_of_the_fight(self):
        loc = local_trade.Local(W)
        st0 = SimpleNamespace(u=units())
        loc.before(st0)
        st1 = SimpleNamespace(u=units(gold_out=[100.0, 0, 0, 0], target=[2, -1, -1, -1],
                                      hp_abs=[1000.0, 1000, 800, 1000]))
        rows, acting = torch.tensor([0]), torch.tensor([[True, True, False, False]])
        p = loc.after(st1, rows, 1, acting)
        # the spears' first engaged decision: its own mean is the base (paid 0); the shooters out of the fight: 0
        assert p[0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
        sp = craft.CLASSES.index("spear")
        assert float(loc.base[sp]) == pytest.approx(0.1)
        loc.before(st1)
        st2 = SimpleNamespace(u=units(gold_out=[400.0, 0, 0, 0], target=[2, -1, -1, -1],
                                      hp_abs=[1000.0, 1000, 200, 1000]))
        p = loc.after(st2, rows, 1, acting)
        # the base: (0.1 x (1 - RATE) + 0.3) / ((1 - RATE) + 1), this decision's raw counted
        k = 1 - local_trade.RATE
        base = (0.1 * k + 0.3) / (k + 1)
        assert float(p[0, 0]) == pytest.approx(0.3 - base) and float(p[0, 1]) == 0
        assert float(loc.base[sp]) == pytest.approx(base)
        s = loc.take_stats()
        assert s["engaged"] == 0.5 and s["raw_mean"] == pytest.approx(0.2) and s["classes"]["spear"]["n"] == 2

    def test_none_pays_the_raw_and_only_the_units_that_decided(self):
        loc = local_trade.Local(W, center="none")
        loc.before(SimpleNamespace(u=units()))
        st = SimpleNamespace(u=units(gold_out=[100.0, 50, 0, 0], target=[2, 2, -1, -1],
                                     hp_abs=[1000.0, 1000, 700, 1000]))
        p = loc.after(st, torch.tensor([0]), 1, torch.tensor([[True, False, False, False]]))
        assert p[0].tolist() == pytest.approx([0.1, 0.0, 0.0, 0.0])

    def test_second_side_row(self):
        # row 1 of B = 1: side 2's units
        loc = local_trade.Local(W, center="none")
        loc.before(SimpleNamespace(u=units()))
        st = SimpleNamespace(u=units(gold_out=[0.0, 0, 80, 0], target=[-1, -1, 0, -1], hp_abs=[840.0, 1000, 1000, 1000]))
        p = loc.after(st, torch.tensor([1]), 1, torch.ones(1, 4, dtype=torch.bool))
        assert p[0].tolist() == pytest.approx([0.0, 0.0, 0.08, 0.0])

