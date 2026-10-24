"""The learned group collection (GeneralConfig.groups "learned", tools/nn/general/assign.py): exactly K groups within
the capacity bounds, the log-probability of the sequential choice equal to the parallel one, the rule's general
unchanged by the new parts, the K given or the head's, the ARI. torch: the training container, on the CPU."""
import dataclasses

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import assign as gassign                 # noqa: E402
from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import groups as ggroups                 # noqa: E402
from tools.nn.general import net as gnet                       # noqa: E402

from tests.tools.test_general_repairs_torch import CFG, G, SECT, army, inputs   # noqa: E402

LEARNED = dataclasses.replace(CFG, groups="learned")


def model(cfg=LEARNED, seed=0):
    torch.manual_seed(seed)
    return gnet.General(cfg, SECT).double().eval()


def random_case(gen, P=64, N=12, G=10):
    logits = torch.randn(P, N, G, generator=gen, dtype=torch.float64) * 3
    who = torch.rand(P, N, generator=gen) < 0.7
    K = torch.randint(1, G + 1, (P,), generator=gen)
    return logits, who, K


def test_capacity_bounds():
    k, m, cap = gassign.capacity(torch.tensor([20, 20, 5, 0, 7]), torch.tensor([3, 4, 8, 3, 1]), 0.5, 1.5)
    assert k.tolist() == [3, 4, 5, 0, 1]
    assert m.tolist() == [3, 2, 1, 0, 3]
    assert cap.tolist() == [10, 8, 2, 0, 11]


def test_the_choice_makes_exactly_k_groups_within_capacity_whatever_the_logits():
    gen = torch.Generator().manual_seed(3)
    for greedy in (False, True):
        for skew in (0.0, 50.0):
            logits, who, K = random_case(gen)
            logits = logits.clone()
            logits[..., 0] += skew                       # every unit wants group 0: the mask still spreads them
            k_eff, m, cap = gassign.capacity(who.sum(1), K, 0.5, 1.5)
            a, lp, ent = gassign.sample(logits, who, k_eff, m, cap, greedy)
            assert bool(((a >= 0) == who).all())
            cnt = torch.nn.functional.one_hot(a.clamp(min=0), G).long() * who[..., None].long()
            cnt = cnt.sum(1)                                                        # [P, G]
            nonempty = (cnt > 0).sum(1)
            assert torch.equal(nonempty, k_eff)
            g = torch.arange(G)
            live = g[None] < k_eff[:, None]
            assert bool(((cnt >= m[:, None]) | ~live).all()) and bool(((cnt <= cap[:, None]) | ~live).all())
            assert bool((lp <= 1e-9).all()) and bool((ent >= -1e-9).all())


def test_the_steps_log_probability_equals_the_parallel_one():
    gen = torch.Generator().manual_seed(5)
    logits, who, K = random_case(gen)
    k_eff, m, cap = gassign.capacity(who.sum(1), K, 0.5, 1.5)
    torch.manual_seed(0)
    a, lp, ent = gassign.sample(logits, who, k_eff, m, cap)
    lp2, ent2, _ = gassign.log_prob(logits, who, k_eff, m, cap, a)
    torch.testing.assert_close(lp2, lp, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ent2, ent, rtol=1e-5, atol=1e-5)


def test_the_rules_general_does_not_read_the_new_parts():
    """groups "rule": the assignment's and the K head's weights take no part (an older checkpoint loads them fresh and
    plays as before); the plan has none of the learned collection's extras."""
    m, inp = model(CFG), inputs()
    torch.manual_seed(1)
    c, d = m(inp, greedy=True)
    assert "lp_assign" not in c
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.startswith(("assign.", "k_head.")):
                p.add_(torch.randn_like(p))
    torch.manual_seed(1)
    c2, d2 = m(inp, greedy=True)
    for k in ("main", "perm", "groups"):
        assert torch.equal(c[k], c2[k]), k
    for k in ("kind", "target", "lp", "value"):
        assert torch.equal(d[k], d2[k]), k


def test_making_a_general_leaves_torchs_generator_as_without_the_new_parts():
    """The new parts are made on a fork of the generator: after a General's making it stands where it stood before
    them (a fresh network's other parts and the draws after it are as without them)."""
    seen = []
    real = gassign.Assign

    def spy(*a, **k):
        seen.append(torch.get_rng_state())
        return real(*a, **k)
    try:
        gassign.Assign = spy
        torch.manual_seed(7)
        gnet.General(CFG, SECT)
    finally:
        gassign.Assign = real
    assert len(seen) == 1 and torch.equal(torch.get_rng_state(), seen[0])


def test_the_learned_general_makes_k_groups_of_ours_and_keeps_the_enemys_rule():
    m, inp = model(), inputs(P=4)
    for K in (1, 2, 3, 5):
        inp["K"] = torch.full((4,), K, dtype=torch.long)
        torch.manual_seed(K)
        c, d = m(inp)
        assert torch.equal(c["active"][:, :G].sum(1), torch.full((4,), min(K, 8)))
        rule = ggroups.collect(inp["num"], inp["lab"], inp["pos"], inp["own"], inp["free"], G)
        assert torch.equal(c["raw"][:, 8:], rule[:, 8:])
        assert torch.equal(c["K"], inp["K"])
        assert bool((c["lp_k"] == 0).all())
        # the update's terms: the same membership's log-probability, with gradients into assign alone
        t = m.assign_terms(inp, c["raw"], c["K"])
        torch.testing.assert_close(t["lp"], c["lp_assign"], rtol=1e-5, atol=1e-5)
        t["lp"].sum().backward()
        grads = {n.split(".")[0] for n, p in m.named_parameters() if p.grad is not None and float(p.grad.abs().sum())}
        assert grads == ({"assign"} if K > 1 else set()), grads           # (K 1: one group, nothing to choose)
        m.zero_grad(set_to_none=True)
        # a re-computation with the plan's membership locked gives its cards
        with torch.no_grad():
            c2 = m.cards(dict(inp, main=c["main"], perm=c["perm"]))
        torch.testing.assert_close(c2["groups"], c["groups"].detach(), rtol=0, atol=1e-12)


def test_without_k_the_head_chooses_it():
    m, inp = model(), inputs(P=3)
    torch.manual_seed(0)
    c, _ = m(inp)
    assert bool(((c["K"] >= 1) & (c["K"] <= 8)).all())
    assert bool((c["lp_k"] < 0).all())
    assert torch.equal(c["active"][:, :G].sum(1), c["K"])


def test_ari():
    a = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
    b = torch.tensor([[2, 2, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    mask = torch.ones_like(a, dtype=torch.bool)
    r = ggroups.ari(a, b, mask, 3)
    assert abs(float(r[0]) - 1.0) < 1e-6                 # the same partition, other labels
    assert 0.0 < float(r[1]) < 1.0
    assert float(r[2]) < 0.1
    mask[0, :] = False
    mask[0, 0] = True
    assert abs(float(ggroups.ari(a, b, mask, 3)[0]) - 1.0) < 1e-6    # one unit: trivial


def test_config_bounds():
    with pytest.raises(ValueError):
        gconfig.GeneralConfig(groups="learned", assign_lo=1.2)
    with pytest.raises(ValueError):
        gconfig.GeneralConfig(groups="learned", assign_hi=0.9)
    assert gconfig.GeneralConfig(groups="learned").assign_hi == 1.5
    _ = army
