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


def test_the_copies_of_a_start_begin_alike_in_one_batch():
    """batch: the battles of one start (every K, every copy) begin with the same state - armies, the randomised
    numbers, places, the battle id (the simulator's draws) - and the start's men and memories; only K differs. Two
    starts differ."""
    from tools.nn.train import rollout, scenes
    actor, enemy = nets()
    c = cfg(ks=(2, 5), copies=2)
    params = rollout.params_with_limit(c.limit_s)
    source = scenes.Generated([31, 32], c.max_units, params, "cpu", sequential=True, armies=c.armies)
    s_of = np.repeat(np.arange(2), 4)
    k_of = np.tile(np.repeat(np.array([2, 5]), 2), 2)
    side_of = np.ones(8, dtype=int)
    env, rt = tourney._battles(actor, enemy, c, source, params, "cpu", s_of, k_of, side_of)
    assert rt.b["K"].tolist() == k_of.tolist()
    u = env.st.u
    for b in range(1, 8):
        same = s_of[b] == s_of[0]
        eq = all(torch.equal(v[b], v[0]) for v in u.values())
        assert eq == same, b
    for b in (1, 2, 3, 5, 6, 7):
        r = 0 if b < 4 else 4
        assert all(torch.equal(v[b], v[r]) for v in u.values())
        assert env.st.keys[b] == env.st.keys[r] and float(env.st.t[b]) == float(env.st.t[r])
        assert int(env.st.attacker[b]) == int(env.st.attacker[r])
        for s in (1, 2):
            mem = env.mem[s]
            for k, v in (vars(mem) if hasattr(mem, "__dict__") else mem._asdict()).items():
                v = torch.as_tensor(v) if not torch.is_tensor(v) and hasattr(v, "shape") else v
                if torch.is_tensor(v) and v.dim() and v.shape[0] == 8:
                    assert torch.equal(v[b], v[r]), (s, k)
    assert bool((u["row"][0] == u["row"][3]).all()) and not bool((u["row"][0] == u["row"][4]).all())
