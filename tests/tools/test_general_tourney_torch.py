"""The tourney of the learned group collection (tools/nn/general/tourney.py): one K twice with all the draws common
gives the same battle (trade difference 0); every plan of a battle keeps its K (exactly K_eff groups of ours); the
advantages. torch: the training container, on the CPU."""
import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import executor, tourney                 # noqa: E402
from tools.nn.model import config, policy                      # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
GCFG = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2, groups="learned")


def nets(seed=0):
    torch.manual_seed(seed)
    actor = executor.wrap(policy.Actor(TINY).eval(), GCFG).eval()
    enemy = policy.Actor(TINY).eval()
    return actor, enemy


def cfg(**k):
    return dataclasses.replace(tourney.Config(states=2, ks=(3,), copies=2, limit_s=40.0, armies="v2", max_units=6,
                                              compile=False), **k)


def test_one_k_twice_with_common_draws_is_the_same_battle():
    actor, enemy = nets()
    res = tourney.play(actor, enemy, cfg(ks=(3, 3), copies=1, mode="lanes"), seeds=[11, 12])
    tr = res["trade"].reshape(2, 2)
    assert np.array_equal(tr[:, 0], tr[:, 1]), tr
    rec = res["records"]
    assert len(rec["battle"]) > 0
    for s in range(2):
        a, b = (rec["battle"] == 2 * s), (rec["battle"] == 2 * s + 1)
        assert torch.equal(rec["main"][a], rec["main"][b]) and torch.equal(rec["lp_assign"][a], rec["lp_assign"][b])


def test_every_plan_keeps_its_battles_k():
    actor, enemy = nets(1)
    res = tourney.play(actor, enemy, cfg(ks=(1, 2, 4), copies=2, limit_s=60.0), seeds=[21, 22])
    rec = res["records"]
    G = GCFG.places
    K = torch.as_tensor(res["K"])[rec["battle"]]
    assert torch.equal(rec["k_used"], K)
    assert torch.equal(rec["K"], K)
    who = (rec["own"] & rec["free"]).sum(1)
    assert torch.equal(rec["active"][:, :G].sum(1), torch.minimum(K, who))
    assert bool((rec["deploy"].sum() >= len(res["K"])))            # every battle's deployment plan is there
    assert res["trade"].shape == (2 * 3 * 2,) and np.isfinite(res["trade"]).all()


def test_advantages():
    trade = np.array([1.0, 3.0, 0.0, 0.0, 5.0, 5.0, 2.0, 4.0])
    state = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    K = np.array([2, 2, 3, 3, 2, 2, 3, 3])
    a, ak, mean = tourney.advantages(trade, state, K, (2, 3))
    assert a.tolist() == [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]
    assert mean.tolist() == [[2.0, 0.0], [5.0, 3.0]]
    assert ak.tolist() == [[1.0, -1.0], [1.0, -1.0]]


def test_round_ks_and_seeds():
    c = tourney.Config(ks_random=4, ks_max=10)
    ks = tourney.round_ks(c, 3)
    assert len(set(ks)) == 4 and all(1 <= k <= 10 for k in ks) and ks == tourney.round_ks(c, 3)
    assert tourney.round_seeds(c, 1) != tourney.round_seeds(c, 2)
