"""run.py --general-only 1 (docs/en/training/general.md "Only the general learns"): the executor (the unit network and
its task input) and the critic frozen bit for bit, the general's weights learning; the ability branches off with their
state carried through; the flag's default changes nothing; without a general the run stops. torch: the training
container, on the CPU."""
import argparse
import contextlib
import io
import json

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import executor                          # noqa: E402
from tools.nn.model import config, policy                      # noqa: E402
from tools.nn.model import critic as mcritic                   # noqa: E402
from tools.nn.train import checkpoint, run                     # noqa: E402

from tests.tools.test_general_branch_torch import GCFG, TINY  # noqa: E402

BASE = ["--reward", "v2", "--battles", "4", "--steps", "4", "--minutes", "5", "--device", "cpu", "--no-eval",
        "--mix", '{"self": 0.5, "nearest": 0.5}', "--bank", "8", "--max-units", "4", "--general-card", "16",
        "--general-hidden", "16", "--general-heads", "2", "--general-return", "critic", "--general-replan-s", "2"]
ABILITY = ["--ability-branch", "1.0", "--ability-branch-missed", "1.0", "--ability-branch-copies", "1",
           "--ability-branch-horizon", "2", "--ability-branch-batch", "4", "--ability-branch-openloop",
           "--ability-critic", "--ability-own-signal"]


def is_general(name):
    return name.startswith("general.")


def same(x, y):
    """Nested dicts / lists / tensors equal bit for bit."""
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if torch.is_tensor(x):
        if not (torch.is_tensor(y) and x.shape == y.shape and x.dtype == y.dtype):
            return False
        if x.is_floating_point():                                   # (NaN where NaN was: the same)
            n = x.isnan()
            return torch.equal(n, y.isnan()) and torch.equal(x[~n], y[~n])
        return torch.equal(x, y)
    if isinstance(x, float) and isinstance(y, float) and x != x and y != y:
        return True
    return x == y


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """init (a unit network) -> a: one update with --general (everything learns: Adam's moments for every weight) ->
    b: two updates of --general-only 1 from a's latest.pt. -> the checkpoints, b's log and printed lines."""
    tmp = tmp_path_factory.mktemp("gonly")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(checkpoint, "DIR", tmp)
        mp.setitem(config.PRESETS, "v2", TINY)
        torch.manual_seed(0)
        init = checkpoint.save(tmp / "init.pt", policy.Actor(TINY).eval(), mcritic.Critic(TINY).eval(), "v2")
        _, _, a = run.train(run.parser().parse_args(["--name", "a", "--init", str(init), "--general", "--updates", "1"]
                                                    + BASE + ABILITY))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, _, b = run.train(run.parser().parse_args(["--name", "b", "--init", str(a / "latest.pt"), "--general",
                                                         "--general-only", "1", "--updates", "2"] + BASE + ABILITY))
    return {"init": init, "a": a / "latest.pt", "b": b / "latest.pt", "b_log": b / "log.jsonl",
            "b_out": buf.getvalue()}


def test_freeze_leaves_only_the_generals_weights_trainable():
    torch.manual_seed(0)
    actor, crit = executor.wrap(policy.Actor(TINY), GCFG), mcritic.Critic(TINY)
    total = sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in crit.parameters())
    gen = sum(p.numel() for p in executor.general_params(actor))
    frozen, trainable = executor.freeze_but_general(actor, crit)
    assert (frozen, trainable) == (total - gen, gen) and gen > 0
    assert all(p.requires_grad == is_general(n) for n, p in actor.named_parameters())
    assert not any(p.requires_grad for p in crit.parameters())
    assert not any(p.requires_grad for p in actor.task_in.parameters())      # (the executor's task input: frozen too)


def test_a_general_only_update_keeps_the_executor_and_the_critic_bit_for_bit_and_moves_the_general(runs):
    a, b = checkpoint.read(runs["a"]), checkpoint.read(runs["b"])
    exec_keys = [k for k in a["actor"] if not is_general(k)]
    assert exec_keys and all(torch.equal(a["actor"][k], b["actor"][k]) for k in exec_keys)
    assert all(torch.equal(a["critic"][k], b["critic"][k]) for k in a["critic"])
    assert any(not torch.equal(a["actor"][k], b["actor"][k]) for k in a["actor"] if is_general(k))
    # Adam's moments of the frozen weights go through untouched (a later step without the flag resumes them)
    actor = checkpoint.load_policy(runs["b"])
    names = [n for n, _ in actor.named_parameters()] + [f"critic.{n}" for n, _ in mcritic.Critic(TINY).named_parameters()]
    sa, sb = a["train"]["optim"]["state"], b["train"]["optim"]["state"]
    frozen = [i for i, n in enumerate(names) if not is_general(n) and i in sa]
    assert frozen and all(same(sa[i], sb[i]) for i in frozen)
    # the start line: N frozen, M trainable (general)
    gen = sum(p.numel() for p in executor.general_params(actor))
    total = sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in mcritic.Critic(TINY).parameters())
    line = next(x for x in runs["b_out"].splitlines() if x.startswith("general only:"))
    assert line.startswith(f"general only: {total - gen} weights frozen, {gen} trainable (general)")
    assert "--ability-branch" in line


def test_the_log_has_the_generals_steps_and_no_executors_terms_and_the_ability_state_goes_through(runs):
    rows = [json.loads(x) for x in runs["b_log"].read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all("policy_loss" not in r and "kl" not in r and "ability_branch" not in r for r in rows)
    assert all("value_loss" in r and "ev" in r and r["minibatches"] == 0 for r in rows)
    assert any("pl" in r["general"] and "vl" in r["general"] for r in rows)        # the general's PPO ran
    ta, tb = checkpoint.train_state(runs["a"]), checkpoint.train_state(runs["b"])
    assert "ability_critic" in ta
    assert [k for k in run.ABILITY_STATE if k in ta and not same(ta[k], tb.get(k))] == []
    assert tb["state"]["ability_updates"] == ta["state"]["ability_updates"]
    assert tb["state"]["head_w"] == ta["state"]["head_w"] and tb["state"]["floor_w"] == ta["state"]["floor_w"]


def test_by_default_the_executor_learns_as_before(runs):
    assert run.parser().parse_args([]).general_only == 0
    init, a = checkpoint.read(runs["init"]), checkpoint.read(runs["a"])                 # (a: without the flag)
    assert any(not torch.equal(v, a["actor"][k]) for k, v in init["actor"].items())
    assert any(not torch.equal(v, a["critic"][k]) for k, v in init["critic"].items())


def test_general_only_args_turn_off_the_executors_aids_on_a_copy():
    args = run.parser().parse_args(["--general-only", "1", "--ability-branch", "0.2", "--ability-critic",
                                    "--ability-own-signal", "--ability-epochs", "2", "--observe-dir", "x",
                                    "--critic-warmup", "3"])
    new, off = run.general_only_args(args)
    assert (new.ability_branch, new.ability_critic, new.ability_own_signal, new.ability_epochs, new.observe_dir,
            new.critic_warmup) == (0.0, False, False, 1, None, 0)
    assert len(off) == 3 and args.ability_branch == 0.2 and args.observe_dir == "x"       # (the caller's: as given)
    assert isinstance(new, argparse.Namespace) and new.cli_given == args.cli_given
    assert run.general_only_args(run.parser().parse_args([]))[1] == []


def test_general_only_without_a_general_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    monkeypatch.setitem(config.PRESETS, "v2", TINY)
    init = checkpoint.save(tmp_path / "init.pt", policy.Actor(TINY).eval(), mcritic.Critic(TINY).eval(), "v2")
    with pytest.raises(SystemExit, match="--general-only 1 needs a general"):
        run.train(run.parser().parse_args(["--name", "x", "--init", str(init), "--general-only", "1", "--updates", "1"]
                                          + BASE))
