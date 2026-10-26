"""The thin executor (tools/nn/general/thin.py) and run.py --general-freeze 1 (docs/ru/training/general.md "Тонкий
исполнитель"): a new executor of other sizes, random, under a checkpoint's general (its weights as saved, the
executor's own kind table afresh), the source's critic (ModelConfig.critic_ff keeps its width); in training the general
frozen bit for bit while the executor and the critic learn; the two freezes exclusive. torch: the training container,
on the CPU."""
import contextlib
import dataclasses
import io
import json

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import executor, thin                    # noqa: E402
from tools.nn.model import config, policy                      # noqa: E402
from tools.nn.model import critic as mcritic                   # noqa: E402
from tools.nn.train import checkpoint, run                     # noqa: E402

from tests.tools.test_general_branch_torch import GCFG, TINY  # noqa: E402
from tests.tools.test_general_only_torch import BASE, ABILITY  # noqa: E402

SIZES = {"d": 16, "layers": 2, "heads": 2, "ff": 2, "pointer": 8, "memory": True, "eyes": False}


def own_general(name):
    return name.startswith("general.") and not name.startswith(executor.EXECUTOR_OWN)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """A checkpoint with a general (a unit network wrapped, its general's weights moved off their start)."""
    tmp = tmp_path_factory.mktemp("thin")
    torch.manual_seed(0)
    actor = executor.wrap(policy.Actor(TINY), GCFG)
    with torch.no_grad():
        for n, p in actor.named_parameters():
            if n.startswith("general."):
                p.add_(0.01)
    return checkpoint.save(tmp / "src.pt", actor, mcritic.Critic(TINY), "v2", train={"ability_norm": {"x": 1},
                                                                                     "optim": {}, "state": {"a": 1}})


def test_the_critics_feed_forward_is_the_actors_unless_set():
    assert config.ModelConfig().critic_ff == 0
    a = sum(p.numel() for p in mcritic.Critic(TINY).parameters())
    b = sum(p.numel() for p in mcritic.Critic(dataclasses.replace(TINY, critic_ff=TINY.ff)).parameters())
    c = sum(p.numel() for p in mcritic.Critic(dataclasses.replace(TINY, ff=2)).parameters())
    assert a == b and c < a


def test_the_thin_start_keeps_the_general_and_the_critic_and_its_executor_is_new(source, tmp_path):
    src = checkpoint.read(source)
    out = thin.main(["--general-from", str(source), "--out", str(tmp_path / "start.pt"), "--d", "16", "--layers", "2",
                     "--heads", "2", "--ff", "2", "--pointer", "8", "--eyes", "0"])
    new = checkpoint.read(out)
    cfg = checkpoint.config_of(new)
    assert (cfg.d, cfg.layers, cfg.heads, cfg.ff, cfg.pointer, cfg.eyes) == (16, 2, 2, 2, 8, False)
    assert cfg.critic_d == TINY.critic_d and cfg.critic_ff == TINY.ff and cfg.sectors == TINY.sectors
    assert new["general"] == src["general"]
    gen = [k for k in src["actor"] if k.startswith("general.") and k != "general.exec_kinds"]
    assert gen and all(torch.equal(src["actor"][k], new["actor"][k]) for k in gen)
    assert torch.equal(new["actor"]["general.exec_kinds"], src["actor"]["general.kinds"])
    assert all(torch.equal(src["critic"][k], new["critic"][k]) for k in src["critic"])
    assert new["actor"]["task_in.weight"].shape[0] == 16
    assert new["train"] == {"ability_norm": {"x": 1}}                    # (no Adam, no counters: a fresh run)
    actor = checkpoint.load_policy(out)                                  # (loads as any checkpoint)
    assert thin.executor_weights(actor) < thin.executor_weights(checkpoint.load_policy(source))


def test_freeze_general_leaves_the_executor_and_its_kind_table_trainable():
    torch.manual_seed(0)
    actor = executor.wrap(policy.Actor(TINY), GCFG)
    frozen, trainable = executor.freeze_general(actor)
    assert frozen == sum(p.numel() for n, p in actor.named_parameters() if own_general(n)) > 0
    assert trainable == sum(p.numel() for p in actor.parameters()) - frozen
    assert all(p.requires_grad == (not own_general(n)) for n, p in actor.named_parameters())
    assert actor.general.exec_kinds.requires_grad and actor.task_in.weight.requires_grad


@pytest.fixture(scope="module")
def frozen_run(source, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("freeze")
    start = thin.main(["--general-from", str(source), "--out", str(tmp / "start.pt"), "--d", "16", "--layers", "2",
                       "--heads", "2", "--ff", "2", "--pointer", "8", "--eyes", "0"])
    buf = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(buf):
        mp.setattr(checkpoint, "DIR", tmp)
        mp.setitem(config.PRESETS, "v2", TINY)
        _, _, out = run.train(run.parser().parse_args(["--name", "f", "--init", str(start), "--general",
                                                       "--general-freeze", "1", "--updates", "2",
                                                       "--pool-no-untrained"] + BASE + ABILITY))
    return {"start": start, "end": out / "latest.pt", "log": out / "log.jsonl", "out": buf.getvalue()}


def test_a_frozen_general_stays_bit_for_bit_while_the_executor_and_the_critic_learn(frozen_run):
    a, b = checkpoint.read(frozen_run["start"]), checkpoint.read(frozen_run["end"])
    gen = [k for k in a["actor"] if own_general(k)]
    assert gen and all(torch.equal(a["actor"][k], b["actor"][k]) for k in gen)
    execs = [k for k in a["actor"] if not k.startswith("general.")]
    assert any(not torch.equal(a["actor"][k], b["actor"][k]) for k in execs)
    assert any(not torch.equal(a["critic"][k], b["critic"][k]) for k in a["critic"])
    assert any(x.startswith("general frozen:") for x in frozen_run["out"].splitlines())
    rows = [json.loads(x) for x in frozen_run["log"].read_text(encoding="utf-8").splitlines()]
    assert rows and all("pl" not in (r.get("general") or {}) for r in rows)    # (no plan PPO of the general)
    assert all("policy_loss" in r for r in rows)


def test_the_flags_default_off_and_the_two_freezes_exclude_each_other(source, tmp_path, monkeypatch):
    assert run.parser().parse_args([]).general_freeze == 0
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    with pytest.raises(SystemExit, match="--general-only 1 and --general-freeze 1"):
        run.train(run.parser().parse_args(["--name", "x", "--init", str(source), "--general", "--general-only", "1",
                                           "--general-freeze", "1", "--updates", "1"] + BASE))
    init = checkpoint.save(tmp_path / "plain.pt", policy.Actor(TINY).eval(), mcritic.Critic(TINY).eval(), "v2")
    with pytest.raises(SystemExit, match="--general-freeze 1 needs a general"):
        run.train(run.parser().parse_args(["--name", "y", "--init", str(init), "--general-freeze", "1",
                                           "--updates", "1"] + BASE))
