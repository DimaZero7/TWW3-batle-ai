"""The ability head's own signal (ppo.py ability_own / AbilityInputGrad, ability_branch.py bonus price and lp_all, run.py
--ability-own-signal). torch is not in the project's .venv: skipped there, run in the training container."""
import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")

from tools.nn.abilities_price import train_price  # noqa: E402
from tools.nn.model import config, policy  # noqa: E402
from tools.nn.model import critic as mcritic  # noqa: E402
from tools.nn.model.heads import Action  # noqa: E402
from tools.nn.train import ability_branch as ab  # noqa: E402
from tools.nn.train import checkpoint, league, ppo, rollout  # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
MIRROR = [("arena", "attack")]
ABILITY = ("heads.ability_q", "heads.ability_k", "heads.ability_none")


def tiny(seed=0):
    torch.manual_seed(seed)
    return policy.Actor(TINY).eval(), mcritic.Critic(TINY).eval()


def all_valid(tmp_path):
    """A bonus price spec (config/ability_price.json's numbers) with every pressable ability valid."""
    spec = json.loads(train_price.FILE.read_text(encoding="utf-8"))
    spec["valid"] = sorted(train_price.load().abilities)
    p = tmp_path / "price.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return str(p)


def collected(T=3, **kw):
    actor, crit = tiny()
    env = rollout.Battles(league.layout(4, 1, {"self": 0.5, "hold_shoot": 0.5}), MIRROR)
    env.brancher = ab.Brancher(ab.Config(share=1.0, missed=1.0, copies=2, horizon_s=3.0, batch=4, openloop=True,
                                         own=True, **kw))
    batch = rollout.collect(env, actor, crit, T)
    return actor, crit, env, batch


def grads(model):
    return {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in model.named_parameters()}


def nonzero(g):
    return g is not None and float(g.abs().sum()) > 0


def test_the_main_surrogate_has_no_gradient_on_the_ability_head_under_the_own_signal():
    actor, crit, env, batch = collected()
    assert batch["abil_branch"] is not None and "lp_all" in batch["abil_branch"]
    for own in (False, True):
        cfg = dataclasses.replace(ppo.PPOConfig(), minibatch=10_000, ability=0.0, ability_own=own, entropy=0.0,
                                  target_kl=0.0)
        opt = torch.optim.SGD(list(actor.parameters()) + list(crit.parameters()), lr=0.0)
        out = ppo.update(actor, crit, opt, batch, cfg)
        g = grads(actor)
        ab_grad = any(nonzero(v) for n, v in g.items() if n.startswith(ABILITY))
        assert nonzero(g["heads.kind.weight"])
        if own:
            assert not ab_grad, [n for n, v in g.items() if n.startswith(ABILITY) and nonzero(v)]
            # the old main part (the batch's whole log-probability less the ability's) is the network's: ratio 1
            assert abs(out["kl"]) < 1e-5 and abs(out["ability_kl"]) < 1e-5 and out["clip"] == 0
        else:
            assert ab_grad                       # (the check means something: without the flag it trains the head)


def test_the_body_gets_the_ability_terms_gradient_scaled_and_the_head_its_whole():
    actor, crit, env, batch = collected()
    br = batch["abil_branch"]
    assert bool(br["mask"].any())
    br = {k: br[k].reshape(-1, br[k].shape[-1]) for k in ("adv", "mask", "lp")}
    br["adv"] = torch.where(br["mask"], torch.ones_like(br["adv"]), br["adv"])
    obs = rollout.full_obs(batch["obs"], batch.get("abil_static"))
    act = Action(**{k: getattr(batch["action"], k).reshape(-1, *getattr(batch["action"], k).shape[2:])
                    for k in batch["action"].parts()})
    ctrl = obs["ctrl"].reshape(-1, obs["ctrl"].shape[-1])
    seen, logit = {}, {}
    for s in (1.0, 0.1, 0.0):
        actor.zero_grad(set_to_none=True)
        with ppo.AbilityInputGrad(actor, s):
            logits, _ = actor.sequence(obs, batch["h0"], batch["reset"], batch["action"])
        logits = {k: v.reshape(-1, *v.shape[2:]) for k, v in logits.items()}
        term, _ = ppo.ability_term(logits, act, br, ctrl, 0.2)
        term.backward()
        seen[s] = grads(actor)
        logit[s] = logits["ability"].detach()
    assert torch.equal(logit[1.0], logit[0.1]) and torch.equal(logit[1.0], logit[0.0])   # the forward pass unchanged
    body = [n for n, g in seen[1.0].items() if not n.startswith(ABILITY) and nonzero(g)]
    assert "heads.norm.weight" in body and any(not n.startswith("heads.") for n in body)
    for n in body:
        assert torch.allclose(seen[0.1][n], 0.1 * seen[1.0][n], rtol=1e-4, atol=1e-9), n
        assert not nonzero(seen[0.0][n]), n
    for n, g in seen[1.0].items():
        if n.startswith(ABILITY) and g is not None:
            assert torch.allclose(seen[0.1][n], g) and torch.allclose(seen[0.0][n], g), n
    assert not actor.heads.ability_q._forward_pre_hooks                       # the hooks are gone after


def test_a_valid_ability_is_priced_by_its_bonuses_and_the_rates_are_logged(tmp_path):
    actor, crit, env, batch = collected(bonus=True, bonus_file=all_valid(tmp_path), critic=True)
    b = env.brancher
    assert b.price_kind() == "openloop+bonus"
    last = b.last
    bonus = last["bonus"]
    ok = bonus["ok"]
    assert bool(ok.any())
    # the price the head learns from is the bonus price where valid (budget share), and the advantage follows it
    assert torch.allclose(last["price"][ok], bonus["share"][ok])
    assert torch.allclose(bonus["gold"], bonus["benefit"] - bonus["recharge"], atol=1e-3)
    st = b.stats
    assert "used_rms" in st and "bonus_s" in st
    for k, v in st.items():
        if k.startswith("press_rate"):
            assert 0.0 <= v <= 1.0, (k, v)
    assert "press_rate" in st and any(k.startswith("ready_") for k in st)
    # the critic learns the same kind of price: its buffer holds the used prices
    assert len(b.critic) == len(last["entries"])
    assert torch.allclose(b.critic.y[-len(last["price"]):], last["price"].float())


def test_run_own_signal_logs_a_constant_weight_and_saves_the_price_kind(tmp_path, monkeypatch):
    from tools.nn.train import run
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    monkeypatch.setitem(config.PRESETS, "v2", TINY)
    actor, crit = tiny()
    init = checkpoint.save(tmp_path / "init.pt", actor, crit, "v2")
    args = run.parser().parse_args(["--name", "own", "--init", str(init), "--reward", "v2", "--battles", "4",
                                    "--steps", "2", "--updates", "2", "--minutes", "5", "--device", "cpu", "--no-eval",
                                    "--mix", '{"self": 0.5, "nearest": 0.5}', "--bank", "8", "--max-units", "4",
                                    "--keep-every", "1e-6", "--ability-branch", "1.0", "--ability-branch-missed",
                                    "1.0", "--ability-branch-copies", "1", "--ability-branch-horizon", "2",
                                    "--ability-branch-batch", "4", "--ability-branch-openloop", "--ability-critic",
                                    "--ability-own-signal", "--head-entropy", "ability=0.5"])
    _, _, out = run.train(args)
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["ability_branch"]["weight"] for r in rows] == [1.0, 1.0]
    assert all("ability_kl" in r and "press_rate" in r["ability_branch"] for r in rows)
    ts = checkpoint.train_state(out / "latest.pt")
    assert ts["ability_critic_kind"] == "openloop+bonus"


# --- the ability head's own lr (run.py --ability-lr-mult) and extra epochs (--ability-epochs, ppo.ability_epochs) ---

def own_cfg(**kw):
    return dataclasses.replace(ppo.PPOConfig(), minibatch=10_000, ability=1.0, ability_own=True, ability_body_grad=0.1,
                               target_kl=0.0, **kw)


def copies(actor, crit):
    a, c = policy.Actor(TINY), mcritic.Critic(TINY)
    a.load_state_dict(actor.state_dict())
    c.load_state_dict(crit.state_dict())
    return a.eval(), c.eval()


def priced(batch):
    """The batch with a nonzero advantage on every priced row (the term has a gradient to give)."""
    br = dict(batch["abil_branch"])
    m = br["mask"]
    br["adv"] = torch.where(m, torch.linspace(-1, 1, m.numel()).reshape(m.shape), br["adv"])
    return dict(batch, abil_branch=br)


def test_the_head_group_is_the_ability_head_only_and_a_multiplier_of_1_keeps_one_group():
    from tools.nn.train import run
    actor, crit = tiny()
    names = [n for n, _ in actor.named_parameters() if n.startswith(ppo.ABILITY_HEAD)]
    assert names and all(n.startswith(ABILITY) for n in names)
    assert len(run.optimizer(actor, crit, 1e-4).param_groups) == 1
    opt = run.optimizer(actor, crit, 1e-4, ability_lr=3.0)
    assert len(opt.param_groups) == 3
    head = {id(p) for p in ppo.ability_head_params(actor)}
    for g in opt.param_groups:
        inside = id(g["params"][0]) in head
        assert g["lr"] == pytest.approx(3e-4 if inside else 1e-4)
        assert all((id(p) in head) == inside for p in g["params"])
    # the groups keep the one-group order (resume maps the moments by it)
    flat = [id(p) for g in opt.param_groups for p in g["params"]]
    assert flat == [id(p) for p in list(actor.parameters()) + list(crit.parameters())]


def test_the_multiplier_moves_only_the_heads_weights_faster():
    from tools.nn.train import run
    actor, crit, env, batch = collected()
    batch = priced(batch)
    moved = {}
    for mult in (1.0, 10.0):
        a, c = copies(actor, crit)
        before = {n: p.detach().clone() for n, p in a.named_parameters()}
        torch.manual_seed(1)                      # (the minibatch's row order)
        ppo.update(a, c, run.optimizer(a, c, 1e-3, ability_lr=mult), batch, own_cfg())
        moved[mult] = {n: p.detach() - before[n] for n, p in a.named_parameters()}
    assert any(nonzero(d) for n, d in moved[1.0].items() if n.startswith(ppo.ABILITY_HEAD))
    for n, d in moved[1.0].items():
        if n.startswith(ppo.ABILITY_HEAD):        # Adam's first step is lr x mhat / (sqrt(vhat) + eps): x the multiplier
            # (atol: the float32 rounding of weight + step, ~1e-7 of the weight)
            assert torch.allclose(moved[10.0][n], 10.0 * d, rtol=1e-3, atol=1e-6), n
        else:                                     # one minibatch: the rest took the very same step
            assert torch.equal(moved[10.0][n], d), n


def test_adam_moments_carry_between_one_group_and_the_heads_group(tmp_path):
    from tools.nn.train import run
    actor, crit, env, batch = collected()
    batch = priced(batch)
    for saved_mult, new_mult in ((1.0, 4.0), (4.0, 1.0)):
        a, c = copies(actor, crit)
        opt = run.optimizer(a, c, 1e-3, ability_lr=saved_mult)
        ppo.update(a, c, opt, batch, own_cfg())
        path = checkpoint.save(tmp_path / f"s{saved_mult}.pt", a, c, "v2",
                               train={"optim": opt.state_dict(), "state": {}})
        new = run.optimizer(a, c, 1e-3, ability_lr=new_mult)
        assert run.resume(new, path) == {}
        for p in list(a.parameters()) + list(c.parameters()):
            old_st, new_st = opt.state.get(p, {}), new.state.get(p, {})
            assert set(old_st) == set(new_st) and old_st
            for k in old_st:
                assert torch.equal(old_st[k].cpu(), new_st[k].cpu())
        head = {id(p) for p in ppo.ability_head_params(a)}
        for g in new.param_groups:                  # the learning rates stay the new optimizer's
            assert g["lr"] == pytest.approx(1e-3 * (new_mult if id(g["params"][0]) in head else 1.0))


def test_the_heads_logits_from_the_stored_thought_are_the_forward_passes():
    actor, crit, env, batch = collected()
    obs = rollout.full_obs(batch["obs"], batch.get("abil_static"))
    with ppo.ThoughtTap(actor, True) as tap, torch.no_grad():
        assert tap.on
        logits, _ = actor.sequence(obs, batch["h0"], batch["reset"], batch["action"])
        u = tap.take()
    assert u is not None and not actor.heads.ability_none._forward_hooks          # the hook is gone after
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs.items()}
    with torch.no_grad():
        mine = ppo.head_ability_logits(actor.heads, u, flat, flat["ctrl"])
    assert torch.equal(mine, logits["ability"].reshape(mine.shape))


def test_extra_ability_epochs_move_only_the_head_and_log_their_kl():
    from tools.nn.train import run
    actor, crit, env, batch = collected()
    batch = priced(batch)
    after, out = {}, {}
    for epochs in (1, 3):
        a, c = copies(actor, crit)
        torch.manual_seed(1)
        out[epochs] = ppo.update(a, c, run.optimizer(a, c, 1e-3), batch, own_cfg(ability_epochs=epochs))
        after[epochs] = {n: p.detach().clone() for n, p in list(a.named_parameters()) + list(c.named_parameters())}
    for n, p in after[1].items():
        if not n.startswith(ppo.ABILITY_HEAD):
            assert torch.equal(after[3][n], p), n          # the body, the other heads and the critic: as with one pass
    assert any(not torch.equal(after[3][n], p) for n, p in after[1].items() if n.startswith(ppo.ABILITY_HEAD))
    assert "ability_extra_kl" not in out[1] and "ability_branch_kl" in out[1] and "ability_branch_clip" in out[1]
    o = out[3]
    assert o["ability_extra_steps"] == 2 * o["minibatches"] and o["ability_extra_rows"] > 0
    assert 0.0 <= o["ability_extra_clip"] <= 1.0 and 0.0 <= o["ability_branch_clip"] <= 1.0
    # the extra passes go on from where the main one left the head: further from the rollout's policy
    assert o["ability_extra_kl"] > o["ability_branch_kl"] >= 0.0


def test_run_takes_the_heads_lr_and_epochs(tmp_path, monkeypatch):
    from tools.nn.train import run
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    monkeypatch.setitem(config.PRESETS, "v2", TINY)
    actor, crit = tiny()
    init = checkpoint.save(tmp_path / "init.pt", actor, crit, "v2")
    args = run.parser().parse_args(["--name", "lr", "--init", str(init), "--reward", "v2", "--battles", "4",
                                    "--steps", "2", "--updates", "2", "--minutes", "5", "--device", "cpu", "--no-eval",
                                    "--mix", '{"self": 0.5, "nearest": 0.5}', "--bank", "8", "--max-units", "4",
                                    "--ability-branch", "1.0", "--ability-branch-missed", "1.0",
                                    "--ability-branch-copies", "1", "--ability-branch-horizon", "2",
                                    "--ability-branch-batch", "4", "--ability-branch-openloop", "--ability-own-signal",
                                    "--ability-lr-mult", "3", "--ability-epochs", "2"])
    _, _, out = run.train(args)
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all("ability_branch_kl" in r and "ability_branch_clip" in r for r in rows)
    assert any("ability_extra_kl" in r for r in rows)
    with pytest.raises(SystemExit):
        run.train(run.parser().parse_args(["--name", "x", "--init", str(init), "--ability-epochs", "2",
                                           "--device", "cpu"]))
