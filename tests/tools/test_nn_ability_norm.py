"""The ability prices' scale per ability (ability_branch.AbilityNorm, Config.norm, run.py --ability-norm). torch is not
in the project's .venv: skipped there, run in the training container."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config  # noqa: E402
from tools.nn.train import ability_branch as ab  # noqa: E402
from tools.nn.train import checkpoint, rollout  # noqa: E402

from tests.tools.test_nn_ability_branch import TINY, env_of, tiny  # noqa: E402

KEYS = ["abilities_big", "abilities_small", "abilities_rare"]


def synthetic(gen, n_big=300, n_small=300, big=0.02, small=0.0005):
    """One update's rows: ability 0 with prices ~big, ability 1 with prices ~small (budget share)."""
    idx = torch.cat([torch.zeros(n_big, dtype=torch.long), torch.ones(n_small, dtype=torch.long)])
    val = torch.cat([big * torch.randn(n_big, generator=gen), small * torch.randn(n_small, generator=gen)])
    return idx, val


def test_per_ability_scale_equalises_the_abilities_where_one_shared_scale_does_not():
    gen = torch.Generator().manual_seed(0)
    norm = ab.AbilityNorm(KEYS)
    for _ in range(5):
        idx, val = synthetic(gen)
        scale = norm.update(idx, val)
    shared = val.pow(2).mean().sqrt()
    rms_of = lambda x: float(x.pow(2).mean().sqrt())                                         # noqa: E731
    big, small = idx == 0, idx == 1
    # one shared scale: the small ability's signal is ~40 times weaker
    assert rms_of(val[big] / shared) / rms_of(val[small] / shared) > 20
    # its own: both near 1
    for sel in (big, small):
        assert 0.8 < rms_of(val[sel] / scale[sel]) < 1.25
    st = norm.stats()
    assert set(st) == {"norm_rms_big", "norm_rms_small"}
    assert st["norm_rms_big"] / st["norm_rms_small"] == pytest.approx(40, rel=0.25)


def test_a_rare_ability_starts_at_the_shared_scale_and_the_floor_holds():
    gen = torch.Generator().manual_seed(1)
    norm = ab.AbilityNorm(KEYS)
    idx, val = synthetic(gen)
    # one row of the rare ability, a tiny price: not blown up - it is filled up with the update's mean square
    idx = torch.cat([idx, torch.tensor([2])])
    val = torch.cat([val, torch.tensor([1e-6])])
    scale = norm.update(idx, val)
    shared = float(val.pow(2).mean().sqrt())
    assert float(scale[-1]) == pytest.approx(shared * np.sqrt((ab.NORM_PRIOR - 1) / ab.NORM_PRIOR), rel=1e-4)
    # all prices ~0: the floor
    norm2 = ab.AbilityNorm(KEYS)
    s = norm2.update(torch.zeros(50, dtype=torch.long), torch.full((50,), 1e-7))
    assert torch.allclose(s, torch.full_like(s, ab.NORM_FLOOR))
    # an unknown ability (index len(keys)) has a row of its own
    s = norm2.update(torch.tensor([len(KEYS)]), torch.tensor([0.01]))
    assert "norm_rms_unknown" in norm2.stats()


def test_the_state_goes_on_by_key_and_an_empty_one_is_fresh():
    gen = torch.Generator().manual_seed(2)
    a = ab.AbilityNorm(KEYS)
    for _ in range(3):
        a.update(*synthetic(gen))
    saved = a.state()
    assert set(saved) == {"abilities_big", "abilities_small"}
    # another table order: mapped by key
    b = ab.AbilityNorm(list(reversed(KEYS)))
    assert b.load(saved) and not ab.AbilityNorm(KEYS).load(None) and not ab.AbilityNorm(KEYS).load({})
    assert b.sq[2] == pytest.approx(a.sq[0]) and b.n[1] == pytest.approx(a.n[1])
    idx, val = synthetic(gen)
    c = ab.AbilityNorm(KEYS)
    c.load(saved)
    assert torch.equal(a.update(idx, val), c.update(idx, val))
    with pytest.raises(ValueError):
        ab.Config(norm="bogus")


def branched(norm, critic=False):
    actor, crit = tiny()
    env = env_of()
    env.brancher = ab.Brancher(ab.Config(share=1.0, missed=1.0, copies=1, horizon_s=2.0, batch=4, openloop=True,
                                         norm=norm, critic=critic))
    torch.manual_seed(5)
    batch = rollout.collect(env, actor, crit, 3)
    return env.brancher, batch["abil_branch"]


def test_shared_is_the_old_formula_and_per_ability_only_rescales_each_ability():
    b0, br0 = branched("shared")
    assert b0.norm is None and not any(k.startswith("norm_rms_") for k in b0.stats)
    last = b0.last
    m = br0["mask"]
    assert bool(m.any())
    # shared (the default): exactly the formula before the flag
    price, pressed = last["price"], last["pressed"]
    rms = float(price.pow(2).mean().sqrt())
    old = (torch.where(pressed, price, -price) / max(rms, ab.ADV_FLOOR)).clamp(-ab.ADV_CLIP, ab.ADV_CLIP)
    t, r, u = (torch.tensor([e[i] for e in last["entries"]]) for i in (0, 1, 2))
    assert torch.equal(br0["adv"][t, r, u], old)
    # per ability, the same battles: the same cells, each price over its own ability's scale
    b1, br1 = branched("per-ability")
    assert torch.equal(br1["mask"], m) and torch.equal(b1.last["price"], price)
    kind = torch.tensor([e[9] for e in b1.last["entries"]])
    scale = torch.as_tensor(b1.norm.last["rms"], dtype=price.dtype)[kind]
    new = (torch.where(pressed, price, -price) / scale).clamp(-ab.ADV_CLIP, ab.ADV_CLIP)
    assert torch.allclose(br1["adv"][t, r, u], new)
    assert any(k.startswith("norm_rms_") for k in b1.stats)


def test_per_ability_scales_the_critics_prices_too():
    b, br = branched("per-ability", critic=True)
    crit = b.critic
    assert b.stats["critic_on"] == 0 and len(crit) > 0      # (before the warm-up: the replayed ones only)
    assert br["adv"].abs().max() <= ab.ADV_CLIP and any(k.startswith("norm_rms_") for k in b.stats)


def test_run_keeps_the_per_ability_scale_in_the_chain(tmp_path, monkeypatch):
    from tools.nn.train import run
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    monkeypatch.setitem(config.PRESETS, "v2", TINY)
    actor, crit = tiny()
    init = checkpoint.save(tmp_path / "init.pt", actor, crit, "v2")

    def args(name, start, norm="per-ability"):
        return run.parser().parse_args(["--name", name, "--init", str(start), "--reward", "v2", "--battles", "4",
                                        "--steps", "2", "--updates", "2", "--minutes", "5", "--device", "cpu",
                                        "--no-eval", "--mix", '{"self": 0.5, "nearest": 0.5}', "--bank", "8",
                                        "--max-units", "4", "--ability-branch", "1.0", "--ability-branch-missed",
                                        "1.0", "--ability-branch-copies", "1", "--ability-branch-horizon", "2",
                                        "--ability-branch-batch", "4", "--ability-branch-openloop",
                                        "--ability-own-signal", "--ability-norm", norm])
    loaded = []
    load = ab.AbilityNorm.load
    monkeypatch.setattr(ab.AbilityNorm, "load", lambda self, d: loaded.append(d) or load(self, d))
    _, _, out = run.train(args("n1", init))
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(any(k.startswith("norm_rms_") for k in r["ability_branch"]) for r in rows)
    saved = checkpoint.train_state(out / "latest.pt")["ability_norm"]
    assert saved and loaded == [None]                         # the first part: fresh
    run.train(args("n2", out / "latest.pt"))
    assert loaded[-1] == saved                                # the next goes on from it
    _, _, out3 = run.train(args("n3", init, "shared"))
    assert "ability_norm" not in checkpoint.train_state(out3 / "latest.pt")
