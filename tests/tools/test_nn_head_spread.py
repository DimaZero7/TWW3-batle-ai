"""The per-head spread of v2 (tools/nn/train/head_entropy.py) and the partial reset (head_reset.py) (torch).

torch is not in the project's .venv: skipped there, run in the training container.
"""
import dataclasses
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import critic as mcritic  # noqa: E402
from tools.nn.model import config, policy  # noqa: E402
from tools.nn.model.heads import Action  # noqa: E402
from tools.nn.sim.orders import ATTACK, KEEP, MOVE  # noqa: E402
from tools.nn.train import checkpoint, head_entropy, head_reset, league, ppo, rollout  # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
MIRROR = [("arena", "attack")]


def tiny(seed=0):
    torch.manual_seed(seed)
    return policy.Actor(TINY).eval(), mcritic.Critic(TINY).eval()


def batch_of(actor, crit, steps=3):
    env = rollout.Battles(league.layout(6, 1, {"self": 0.5, "hold_shoot": 0.5}), MIRROR)
    return rollout.collect(env, actor, crit, steps)


def collapse(actor, near=8.0):
    """Make the place and the commitment heads choose one thing (as build/net_audit found them): the sector the
    nearest (the near weight's softplus ~near a sector), one cell, one duration. (Not by scaling the weights: the
    logits' soft cap, chain.CAP, then flattens them all to +30.)"""
    h = actor.heads
    with torch.no_grad():
        h.place_near.bias.fill_(near)
        for m in (h.fine[2], h.commit):
            m.weight.zero_()
            m.bias.zero_()
            m.bias[0] = 25.0


def spread(actor, batch):
    """{head: share of the maximum} of the actor on the batch's rows (head_entropy.measure)."""
    T, R = batch["reward"].shape
    obs = rollout.full_obs(batch["obs"], batch.get("abil_static"))
    with torch.no_grad():
        logits, _ = actor.sequence(obs, batch["h0"], batch["reset"], batch["action"])
    logits = {k: v.reshape(-1, *v.shape[2:]) for k, v in logits.items()}
    act = Action(**{k: getattr(batch["action"], k).reshape(-1, *getattr(batch["action"], k).shape[2:])
                    for k in batch["action"].parts()})
    ctrl = obs["ctrl"].reshape(-1, obs["ctrl"].shape[-1])
    free = obs["free"].reshape(-1, ctrl.shape[-1])
    # every row of a unit that takes orders, whatever kind it chose (the place's logits are under its kind)
    every = Action(torch.full_like(act.kind, MOVE), act.point, act.target, act.run, act.ability, act.commit)
    return {h: float(v[0]) for h, v in head_entropy.measure(logits, every, free, ctrl).items()}, logits


def test_the_spec_is_head_equals_share_of_the_maximum():
    assert head_entropy.parse("sector=0.4, cell=0.5,hold=0.5") == {"sector": 0.4, "cell": 0.5, "hold": 0.5}
    assert head_entropy.parse("") == {}
    for bad in ("place=0.4", "sector", "sector=1.5", "sector=0"):
        with pytest.raises(ValueError):
            head_entropy.parse(bad)
    assert head_reset.parse("sector,cell,hold") == ["sector", "cell", "hold"]
    with pytest.raises(ValueError):
        head_reset.parse("sector,place")


def test_each_floor_goes_up_below_its_target_and_down_above_within_its_bounds():
    w = head_entropy.next_weights({"sector": 0.01, "hold": 0.01, "cell": 0.05},
                                  {"sector": 0.1, "hold": 0.9, "cell": 0.1}, {"sector": 0.4, "hold": 0.5, "cell": 0.5},
                                  0.003, 0.05, 2.0)
    assert w == pytest.approx({"sector": 0.02, "hold": 0.005, "cell": 0.05})
    # a head with no rows this update keeps its weight (a new one: the lowest)
    w = head_entropy.next_weights({"sector": 0.01}, {"sector": None}, {"sector": 0.4, "run": 0.5}, 0.003, 0.05, 2.0)
    assert w == pytest.approx({"sector": 0.01, "run": 0.003})


def test_the_share_is_one_for_alike_choices_and_counts_only_allowed_choices_and_used_rows():
    B, N = 2, 3
    NEG = -1e9
    sector = torch.zeros(B, N, 8)
    sector[..., 4:] = NEG                                   # 4 allowed: uniform among them -> share 1
    sector[1, 0] = torch.tensor([30.0, 0, 0, 0, NEG, NEG, NEG, NEG])      # one sure choice -> ~0
    logits = {"kind": torch.zeros(B, N, 5), "target": torch.zeros(B, N, N), "sector": sector,
              "fine": torch.zeros(B, N, 16), "commit": torch.zeros(B, N, 4), "run": torch.zeros(B, N)}
    kind = torch.tensor([[MOVE, ATTACK, KEEP], [MOVE, MOVE, MOVE]])
    a = Action(kind, torch.zeros(B, N, dtype=torch.long), torch.zeros(B, N, dtype=torch.long),
               torch.zeros(B, N, dtype=torch.bool), None, torch.zeros(B, N, dtype=torch.long))
    ctrl = torch.ones(B, N, dtype=torch.bool)
    m = head_entropy.measure(logits, a, ctrl, ctrl)
    share, nats, rows = m["sector"]
    assert float(rows) == 4                                  # the moves only
    assert float(share) == pytest.approx(3 / 4, abs=1e-3)    # three alike, one sure
    assert float(m["hold"][2]) == 5 and float(m["hold"][0]) == pytest.approx(1.0)   # every new order, not keep
    assert float(m["target"][2]) == 1 and float(m["run"][0]) == pytest.approx(1.0)
    assert float(nats) == pytest.approx(3 / 4 * math.log(4), abs=1e-3)


def test_the_reset_makes_the_collapsed_heads_wide_and_leaves_the_kind_and_the_target_as_they_were():
    actor, crit = tiny()
    collapse(actor)
    batch = batch_of(actor, crit)
    before, lg0 = spread(actor, batch)
    assert before["sector"] < 0.2 and before["cell"] < 0.2 and before["hold"] < 0.2
    params0 = {n: p.detach().clone() for n, p in actor.named_parameters()}
    done = head_reset.reset(actor, ["sector", "cell", "hold"])
    assert {n.split(".")[1] for n in done} == {"place_q", "place_k", "place_near", "fine", "commit"}
    after, lg1 = spread(actor, batch)
    assert after["sector"] > 0.5 and after["cell"] > 0.9 and after["hold"] > 0.9, after
    for k in ("kind", "target", "run"):                     # untouched heads: the same logits
        assert torch.allclose(lg0[k], lg1[k], atol=1e-5), k
        assert after.get(k) == pytest.approx(before.get(k), abs=1e-5)
    changed = {n for n, p in actor.named_parameters() if not torch.equal(params0[n], p)}
    assert changed <= set(done) and "heads.place_q.weight" in changed     # (place_near.weight: 0 before and after)
    # as a new network's: the same init as ChainHeads' (the near weight softplus 1, i.e. the fresh preference)
    fresh, _ = tiny(5)
    assert torch.equal(actor.heads.place_near.weight, fresh.heads.place_near.weight)
    assert torch.allclose(torch.nn.functional.softplus(actor.heads.place_near.bias), torch.ones(1))
    with pytest.raises(ValueError):
        head_reset.reset(policy.Actor(config.preset("small", d=32, layers=2, heads=2, pointer=16)), ["sector"])


def test_the_anchors_get_the_fresh_heads_and_adam_forgets_their_moments():
    actor, crit = tiny()
    ref, _ = tiny()
    opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
    for p in actor.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    done = head_reset.reset(actor, ["hold"], seed=7)
    head_reset.copy(actor, ref, set(done))
    head_reset.forget(opt, actor, done)
    named = dict(actor.named_parameters())
    for n, p in ref.named_parameters():
        if n in done:
            assert torch.equal(p, named[n]) and named[n] not in opt.state
        else:
            assert named[n] in opt.state


def test_a_floor_on_the_sector_widens_it_in_an_update_and_every_head_is_logged():
    actor, crit = tiny()
    collapse(actor, near=3.0)
    batch = batch_of(actor, crit)
    base = dataclasses.replace(ppo.PPOConfig(), epochs=4, minibatch=6, lr=1e-2, target_kl=0.0)
    out = {}
    for name, heads in (("none", ()), ("floor", (("sector", 1.0),))):
        a = policy.Actor(TINY).eval()
        a.load_state_dict(actor.state_dict())
        c = mcritic.Critic(TINY).eval()
        c.load_state_dict(crit.state_dict())
        opt = torch.optim.Adam(list(a.parameters()) + list(c.parameters()), lr=base.lr)
        st = ppo.update(a, c, opt, batch, dataclasses.replace(base, heads=heads))
        for h in ("kind", "sector", "cell", "hold", "run"):
            assert 0.0 <= st[f"head_share_{h}"] <= 1.0 and np.isfinite(st[f"head_ent_{h}"]), h
        out[name] = spread(a, batch)[0]["sector"]
    assert out["floor"] > out["none"] + 0.01, out


def test_run_resets_the_heads_and_keeps_the_floors_weights_in_the_chain(tmp_path, monkeypatch):
    from tools.nn.train import run
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    monkeypatch.setitem(config.PRESETS, "v2", TINY)
    actor, crit = tiny()
    collapse(actor)
    init = checkpoint.save(tmp_path / "init.pt", actor, crit, "v2")
    args = run.parser().parse_args(["--name", "hs", "--init", str(init), "--reward", "v2", "--battles", "4",
                                    "--steps", "2", "--updates", "2", "--minutes", "5", "--device", "cpu", "--no-eval",
                                    "--mix", '{"self": 0.5, "nearest": 0.5}', "--bank", "8", "--max-units", "4",
                                    "--keep-every", "1e-6", "--anchor", "0.05", "--reset-heads", "sector,cell,hold",
                                    "--head-entropy", "sector=0.99,cell=0.01", "--head-entropy-start", "0.01"])
    trained, _, out = run.train(args)
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("head_share_sector" in r and "head_ent_hold" in r for r in rows)
    assert rows[0]["head_entropy_weight"] == {"sector": 0.01, "cell": 0.01}
    assert rows[1]["head_entropy_weight"] == {"sector": 0.0125, "cell": 0.01}      # x 1.25 below, the lowest above
    state = checkpoint.train_state(out / "m0.pt")["state"]
    assert set(state["head_w"]) == {"sector", "cell"}
    # the place's layers are not the collapsed ones any more
    assert not torch.equal(trained.heads.place_q.weight, actor.heads.place_q.weight)
