"""tools/nn/general/runtime.py "hold" (run.py --general-plan-lambda L): the general's plans' advantage chained over the
battle - the ended plans wait on the CPU until their battle ends (or is cut: a restart, narrow), then each battle's
side's plans get signal.plan_gae's advantage in their "ret"; their outcomes (the outcome Q) wait with them. torch: the
training container, on the CPU."""
import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import learn, signal                    # noqa: E402
from tools.nn.general.runtime import COUNTS, Counts, Runtime  # noqa: E402
from tools.nn.train import ppo, rollout                        # noqa: E402

from tests.tools.test_general_runtime_torch import battles, nets  # noqa: E402


def stub(B=2, lam=0.9, gamma=0.99):
    """A Runtime with only what the hold reads: rows [2B], the battles' numbers, n_dec, the counters."""
    rt = Runtime.__new__(Runtime)
    rt.B, rt.dev, rt.gamma, rt.plan_lam = B, torch.device("cpu"), gamma, lam
    rt.hold = {"parts": {}, "left": {}, "idx": {}, "next": 0}
    rt.r = {"n_dec": torch.zeros(2 * B, dtype=torch.long)}
    rt.b = {"bid": torch.arange(B, dtype=torch.long) + 10}
    rt.stats = Counts(rt.dev, 6, 4)
    rt.ready = []
    rt.q_wait = None
    return rt


def rec_of(rows, ret, base, tag):
    n = len(rows)
    return {"ret": torch.tensor(ret, dtype=torch.float32), "base": torch.tensor(base, dtype=torch.float32),
            "new": torch.ones(n, dtype=torch.bool), "tag": torch.tensor(tag, dtype=torch.float32),
            "kind": torch.zeros(n, 3, dtype=torch.long)}


def test_the_plans_wait_for_their_battles_end_and_get_the_chained_advantage():
    rt = stub(B=2, lam=0.9, gamma=0.99)
    # rows: 0, 1 the battles' side 1; 2, 3 their side 2. Plans end interleaved over the battles and sides.
    rt.r["n_dec"][:] = torch.tensor([20, 25, 22, 30])
    rt._hold_add(torch.tensor([0, 1, 2]), rec_of([0, 1, 2], [0.5, 0.2, -0.1], [0.4, 0.1, 0.0], [0, 100, 200]), False)
    rt.r["n_dec"][:] = torch.tensor([10, 15, 12, 30])
    rt._hold_add(torch.tensor([2, 0]), rec_of([2, 0], [0.3, -0.2], [0.35, 0.1], [201, 1]), False)
    assert rt.ready == [] and rt.held() == 5
    # battle 0 (rows 0 and 2) ends: its last plans, then both sides' sequences go
    rt.r["n_dec"][:] = torch.tensor([5, 15, 7, 30])
    rt._hold_add(torch.tensor([0, 2]), rec_of([0, 2], [0.6, -0.4], [0.5, -0.3], [2, 202]), True)
    assert len(rt.ready) == 1 and rt.held() == 1                   # battle 1's side 1 plan still waits
    out = rt.ready[0]
    assert out["tag"].tolist() == [0, 1, 2, 200, 201, 202]          # each sequence in its time order
    g = 0.99
    for tags, delta, n in (([0, 1, 2], [0.1, -0.3, 0.1], [20, 10, 5]), ([200, 201, 202], [-0.1, -0.05, -0.1],
                                                                       [22, 12, 7])):
        disc = np.array([g ** n[0], g ** n[1], 0.0])
        want = signal.plan_gae(np.array(delta), disc, np.array([1.0, 1.0, 0.0]), 0.9)
        sel = [out["tag"].tolist().index(t) for t in tags]
        got = (out["ret"] - out["base"])[sel].numpy()
        assert got == pytest.approx(want, abs=1e-5)
    st = rt.stats
    assert st["gae_seqs"] == 2 and st["gae_plans"] == 6 and st["gae_cut"] == 0


def test_a_cut_battle_releases_its_plans_with_the_critics_tail():
    rt = stub(B=2, lam=1.0, gamma=1.0)
    rt.r["n_dec"][:] = 10
    rt._hold_add(torch.tensor([1, 1]), rec_of([1, 1], [0.2, 0.5], [0.0, 0.1], [0, 1]), False)
    rt._release(rt._keys(torch.tensor([1, 3])).tolist(), cut=True)
    out = rt.ready[0]
    # lambda 1, gamma 1: the first plan's advantage is its delta + the second's; the second's delta holds its tail
    assert (out["ret"] - out["base"]).tolist() == pytest.approx([0.2 + 0.4, 0.4])
    assert rt.held() == 0 and rt.hold["parts"] == {} and rt.stats["gae_cut"] == 1


def test_a_waiting_plans_outcome_waits_with_it():
    rt = stub(B=1)
    rt.r["n_dec"][:] = 10
    r = rec_of([0, 1], [0.1, 0.2], [0.0, 0.0], [0, 1])
    r["qid"] = torch.tensor([7, 8])
    rt._hold_add(torch.tensor([0]), {k: v[:1] for k, v in r.items()}, False)
    z = torch.zeros(2)
    rt.q_wait = {"qid": torch.tensor([7, 9]), "y": torch.tensor([0.3, 0.4]), "span": z + 60, "ended": z.bool(),
                 "bid": torch.tensor([10, 10]), "state": torch.tensor([1, 1]), "row": torch.tensor([0, 0])}
    out = rt._q_take()
    assert out["qid"].tolist() == [9]                               # 7's record still waits: its outcome too
    assert rt.q_wait["qid"].tolist() == [7]
    rt._release(rt._keys(torch.tensor([0])).tolist(), cut=True)
    assert rt._q_take()["qid"].tolist() == [7]


def test_in_the_battles_the_plans_wait_and_a_narrow_releases_them_into_an_update():
    env = battles(B=4, mix={"self": 1.0})
    actor, crit = nets()
    env.step(actor, crit)
    rt = env.general_rt
    rt.ret, rt.gamma, rt.plan_lam = "critic", 0.9994, 0.8
    env.general_rt.r["t_plan"] -= 28.0                                # re-plans inside the chunk
    batch = rollout.collect(env, actor, crit, 4)
    assert batch["general"]["ready"] is None and rt.held() > 0       # the ended plans wait for their battles' end
    n = rt.held()
    keep = torch.tensor([1, 3])
    env.narrow(keep)                                                  # battles 0 and 2 dropped: their plans go, cut
    assert rt.held() < n
    st = rt.take_stats()
    assert st["plan_lambda"] == 0.8 and st["gae_cut_share"] == 1.0 and st["gae_plans_per_battle"] >= 1
    rt.begin_chunk(env)
    ready = rt.end_chunk()["ready"]
    assert ready is not None and len(ready["ret"]) == n - rt.held()
    assert not ready["new"].any()                                     # waited over a chunk: old plans
    batch["general"]["ready"] = ready
    opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
    out = ppo.update(actor, crit, opt, batch, dataclasses.replace(ppo.PPOConfig(), minibatch=8),
                     general=learn.LearnConfig(epochs=1))
    assert out["general_plans"] == len(ready["ret"]) and np.isfinite(out["general_adv_std"])


def test_the_counters_know_the_chains_names():
    for k in ("gae_seqs", "gae_plans", "gae_cut", "gae_d_sum", "gae_d_sq", "gae_a_sum", "gae_a_sq"):
        assert k in COUNTS
