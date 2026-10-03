"""tools.nn.model.widen: a network widened k times computes exactly what it did, and its new units learn.

torch is not in the project's .venv: skipped there, run in the training container (as test_nn_model.py).
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.config import ROOT  # noqa: E402
from tools.nn.model import config, critic, heads, policy, sources, widen  # noqa: E402
from tools.nn.model import observation as ob  # noqa: E402
from tools.nn.train import checkpoint  # noqa: E402

CFG = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=2, critic_heads=2)
LORDS = ["wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0", "wh_main_emp_inf_spearmen_0"]


def states(batch=2):
    """Three decisions of a made-up battle with lords (ready abilities) and a blow at the second."""
    setup, state = sources.synthetic(batch=batch, own=3, enemy=3, seed=5, keys=LORDS)
    N = 6
    state.update(men=np.maximum(state["men"], 1.0), ms=np.full((batch, N), 2.0), r=np.zeros((batch, N), bool),
                 s=np.zeros((batch, N), bool))
    for k in range(3):
        for t in ("on", "cd"):
            state[f"ab{k}_{t}"] = np.zeros((batch, N))
    hp = np.full((batch, N), 0.9)
    hit = hp.copy()
    hit[:, 4] = 0.6
    return setup, [dict(state, t=np.full(batch, 30.0 + i), hp=h) for i, h in enumerate((hp, hit, hit))]


def networks(seed=0):
    """A small actor and critic with every part non-trivial: random LayerNorm gains and biases, the
    ability encoder's last layer (zero when fresh) random."""
    torch.manual_seed(seed)
    actor, crit = policy.Actor(CFG), critic.Critic(CFG)
    with torch.no_grad():
        for m in list(actor.modules()) + list(crit.modules()):
            if isinstance(m, torch.nn.LayerNorm):
                m.weight.uniform_(0.5, 1.5)
                m.bias.normal_(std=0.2)
        actor.abilities.net[2].weight.normal_(std=0.1)
        actor.dist.weight.normal_(std=0.5)
    return actor.eval(), crit.eval()


def outputs(actor, crit, setup, sts, dtype):
    """Per decision (memory carried): (logits, memory, value)."""
    out, m, mc, h = [], None, None, None
    for st in sts:
        obs, m = ob.observe(st, setup, 1, m)
        cobs, mc = ob.observe(st, setup, 1, mc, full=True)
        o, c = (widen.double(policy.to_torch(x)) if dtype == torch.float64 else policy.to_torch(x) for x in (obs, cobs))
        with torch.no_grad():
            logits, h = actor(o, h)
            out.append((logits, h, crit(c)))
    return out


def assert_same(small, wide, d, atol):
    for (lo, ho, vo), (lw, hw, vw) in zip(small, wide):
        assert set(lo) == set(lw) and "ability" in lo
        for k in lo:
            assert lo[k].shape == lw[k].shape and torch.allclose(lo[k], lw[k], rtol=0, atol=atol), k
        assert hw.shape[-1] == 2 * d
        for copy in hw.split(d, -1):                                      # the memory: two equal copies
            assert torch.allclose(ho, copy, rtol=0, atol=atol)
        assert torch.allclose(vo, vw, rtol=0, atol=atol)
        a, b = heads.sample(lo, greedy=True, abilities=True), heads.sample(lw, greedy=True, abilities=True)
        assert all(torch.equal(getattr(a, f), getattr(b, f)) for f in ("kind", "point", "target", "run", "ability"))


def test_the_wider_config_is_the_wide_preset():
    assert widen.wider_config(config.SMALL, 2) == config.WIDE
    wide = widen.wider_config(CFG, 2)
    assert (wide.d, wide.heads, wide.pointer, wide.critic_d, wide.critic_heads) == (64, 4, 32, 64, 4)
    assert wide.d // wide.heads == CFG.d // CFG.heads and wide.layers == CFG.layers   # head width, depth kept


def test_a_widened_network_computes_exactly_what_it_did_in_float64():
    setup, sts = states()
    with widen.float64():
        actor, crit = (widen.double(n) for n in networks())
        w = widen.Widener(2, seed=1)
        wide, wide_crit = w.actor(actor), w.critic(crit)
        assert wide.cfg == widen.wider_config(CFG, 2)
        assert_same(outputs(actor, crit, setup, sts, torch.float64), outputs(wide, wide_crit, setup, sts, torch.float64),
                    CFG.d, atol=1e-10)


@pytest.mark.parametrize("k", [2, 3])
def test_a_widened_network_computes_what_it_did_in_float32(k):
    setup, sts = states()
    actor, crit = networks(2)
    w = widen.Widener(k, seed=3)
    wide, wide_crit = w.actor(actor), w.critic(crit)
    small, big = outputs(actor, crit, setup, sts, torch.float32), outputs(wide, wide_crit, setup, sts, torch.float32)
    for (lo, ho, vo), (lw, hw, vw) in zip(small, big):
        for key in lo:
            assert torch.allclose(lo[key], lw[key], rtol=0, atol=1e-5), key
        assert all(torch.allclose(ho, c, atol=1e-5) for c in hw.split(CFG.d, -1)) and torch.allclose(vo, vw, atol=1e-5)


def test_the_new_units_get_a_gradient_and_the_copies_differ():
    setup, sts = states()
    actor, _ = networks()
    wide = widen.Widener(2, seed=0).actor(actor).train()
    d, f = CFG.d, CFG.ff * CFG.d
    # the copies' readers differ (zero-sum noise), the writers are equal copies
    assert not torch.equal(wide.heads.kind.weight[:, :d], wide.heads.kind.weight[:, d:])
    assert torch.equal(wide.blocks[0].f2.weight[:d], wide.blocks[0].f2.weight[d:])
    # new units: outgoing weights zero, incoming random
    assert torch.all(wide.blocks[0].f2.weight[:, f:] == 0) and wide.blocks[0].f1.weight[f:].abs().sum() > 0
    assert torch.all(wide.blocks[0].out.weight[:, d:] == 0) and torch.all(wide.heads.k.weight[CFG.pointer:] == 0)
    obs, _ = ob.observe(sts[0], setup, 1)
    logits, _ = wide(policy.to_torch(obs))
    sum(v.masked_fill(v < heads.NEG / 2, 0).sum() for v in logits.values()).backward()
    for g in (wide.blocks[0].f2.weight.grad[:, f:], wide.blocks[1].out.weight.grad[:, d:],
              wide.heads.k.weight.grad[CFG.pointer:], wide.encoder.unit[2].weight.grad[:, d:]):
        assert g.abs().sum() > 0
    gw = wide.heads.point.weight.grad
    assert torch.allclose(gw[:, :d], gw[:, d:], rtol=1e-5, atol=1e-5)    # the readers' halves see equal inputs ...
    gx = wide.blocks[0].f2.weight.grad
    assert not torch.allclose(gx[:d], gx[d:])                   # ... the copies get different gradients


def test_a_widened_checkpoint_loads_old_ones_still_load_and_training_continues(tmp_path, monkeypatch):
    from tools.nn.train import run
    actor, crit = networks()
    small = checkpoint.save(tmp_path / "small.pt", actor, crit, "small", {"update": 7})
    wide_path = tmp_path / "wide.pt"
    widen.widen_file(small, wide_path, widen.Widener(2))
    data = checkpoint.read(wide_path)
    assert data["preset"] == "smallx2" and data["meta"]["factor"] == 2 and data["meta"]["update"] == 7
    assert checkpoint.load_policy(wide_path).cfg == widen.wider_config(CFG, 2)
    assert checkpoint.load_critic(wide_path).cfg.critic_d == 2 * CFG.critic_d
    assert checkpoint.load_policy(small).cfg == CFG                       # the small one as it was
    # a run from the wide network with a small version as "past" (--pool 1: random.pt only, here the small one)
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    monkeypatch.setattr(checkpoint, "RANDOM", small)
    args = run.parser().parse_args(["--name", "w", "--init", str(wide_path), "--battles", "4", "--steps", "2",
                                    "--updates", "4", "--minutes", "5", "--device", "cpu", "--no-eval",
                                    "--mix", '{"past": 0.5, "nearest": 0.5}', "--anchor", "0.05",
                                    "--pool", "1", "--snapshot-every", "1",
                                    "--bank", "8", "--max-units", "4"])
    trained, summary, out = run.train(args)
    assert summary["updates"] == 4 and trained.cfg == widen.wider_config(CFG, 2)
    pasts = {json.loads(x)["past"] for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()}
    assert pasts == {"small.pt"}                                          # the small past version played
    assert checkpoint.read(out / "latest.pt")["preset"] == "smallx2"      # --preset follows --init's record
    assert checkpoint.load_policy(out / "latest.pt").cfg.d == 2 * CFG.d


SOURCE = ROOT / "build/nn-train/test5/r7_lostworst/m15.pt"


@pytest.mark.skipif(not SOURCE.exists(), reason="no trained checkpoint (build/ is not in Git)")
def test_the_trained_network_widened_acts_the_same_on_simulator_battles():
    actor, crit = checkpoint.load_policy(SOURCE), checkpoint.load_critic(SOURCE).eval()
    w = widen.Widener(2, seed=0)
    wide, wide_crit = w.actor(actor), w.critic(crit)
    obs, cobs, reset = widen.battles(actor, crit, n=4, decisions=16)
    exact, stored = widen.check(actor, crit, wide, wide_crit, 2, 0, obs, cobs, reset)
    assert widen.worst(exact) < 1e-9 and exact["greedy_same"] == 1.0
    for key, v in stored.items():
        if key.startswith("logits."):         # float32: as close as the small network is to its own float64
            assert v <= max(1e-5, 2 * stored["floor"][key[7:]]), key
    assert stored["memory"] < 1e-5 and stored["value"] < 1e-5
