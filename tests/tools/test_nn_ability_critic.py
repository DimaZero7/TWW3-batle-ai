"""The ability critic (tools/nn/train/ability_critic.py): the branches' price learned and given at every moment.

torch is not in the project's .venv: skipped there, run in the training container.
"""
import json

import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import abilities as mab  # noqa: E402
from tools.nn.model import config  # noqa: E402
from tools.nn.model import observation as ob  # noqa: E402
from tools.nn.train import ability_branch as ab  # noqa: E402
from tools.nn.train import ability_critic as ac  # noqa: E402
from tools.nn.train import checkpoint, rollout  # noqa: E402

from tests.tools.test_nn_ability_branch import TINY, env_of, tiny  # noqa: E402


PATTERNS = torch.rand(5, ac.WIDTH, generator=torch.Generator().manual_seed(99))


def synthetic(n, seed, ratio=1.5, signal=True):
    """Moments as the critic sees them: 40 columns vary (the unit's state, the neighbourhood), the rest is one of 5
    abilities' fixed patterns (passports). The price: a buff (abilities 0, 1) pays in melee and costs out of it, plus a
    count; noise `ratio` x the signal's std (S25: a standard error ~0.6 of the prices' spread). signal False: noise
    only. -> (X, y, the noiseless price)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, ac.WIDTH, generator=g)
    kind = torch.randint(5, (n,), generator=g)
    X[:, 40:] = PATTERNS[kind, 40:]
    melee, buff = (X[:, 0] > 0.5).float(), (kind < 2).float()
    sig = 0.004 * melee * buff - 0.002 * (1 - melee) * buff + 0.002 * X[:, 2]
    noise = ratio * float(sig.std()) * torch.randn(n, generator=g)
    return X, (sig + noise if signal else noise), sig


def feed(crit, X, y, per=105):
    """The pace of training: `per` replayed moments, then a fit, as each update does. -> the first ready update."""
    first = None
    for u in range(0, len(y), per):
        crit.add(X[u:u + per], y[u:u + per])
        crit.fit()
        if first is None and crit.ready():
            first = u // per + 1
    return first


def test_the_critic_learns_a_noisy_price_and_is_used_only_after_its_warm_up():
    crit = ac.AbilityCritic(warmup=2000)
    X, y, _ = synthetic(6300, 1)
    first = feed(crit, X, y)
    assert first is not None and first * 105 >= 2000              # not before the warm-up
    st = crit.last
    assert crit.ready() and st["val"] < st["base"] and st["ev"] > 0.1 and 0 < st["shrink"] <= 1
    Xt, _, sig = synthetic(2000, 2)
    q = crit.predict(Xt)
    # against the noiseless price on new moments: most of the signal's variance explained
    assert 1 - (q - sig).var() / sig.var() > 0.5


def test_a_critic_of_pure_noise_is_never_used_and_shrinks_to_the_mean():
    crit = ac.AbilityCritic(warmup=2000)
    X, y, _ = synthetic(6300, 3, signal=False)
    assert feed(crit, X, y) is None
    assert crit.last["shrink"] < 0.5


def test_the_buffer_keeps_the_last_samples_and_holds_out_a_tenth():
    crit = ac.AbilityCritic(buffer=1000)
    X, y, _ = synthetic(1500, 3)
    crit.add(X, y)
    assert len(crit) == 1000 and crit.seen == 1500 and int(crit.ids[0]) == 500
    assert torch.equal(crit.y, y[500:])
    assert int(((crit.ids % ac.VAL_EVERY) == 0).sum()) == 100


def test_the_state_goes_on_in_a_new_critic_and_another_width_starts_afresh():
    crit = ac.AbilityCritic()
    X, y, _ = synthetic(800, 4)
    crit.add(X, y)
    crit.fit()
    st = checkpoint._to_cpu(crit.state())
    again = ac.AbilityCritic()
    assert again.load(st) and len(again) == 800 and again.seen == 800
    assert torch.allclose(again.predict(X[:50]), crit.predict(X[:50]), atol=1e-3 * float(crit.model.unit))
    assert not ac.AbilityCritic().load(dict(st, width=ac.WIDTH + 1))
    assert not ac.AbilityCritic().load(None)


def test_the_buffer_keeps_the_copies_halves_and_tells_how_much_of_the_price_is_real():
    g = torch.Generator().manual_seed(5)
    sig = torch.randn(3000, generator=g)
    halves = sig[:, None] + torch.randn(3000, 2, generator=g) * 2 ** 0.5    # each half: noise var 2, the mean's 1
    crit = ac.AbilityCritic()
    X, _, _ = synthetic(3000, 6)
    crit.add(X, halves.mean(1), halves)
    assert abs(crit.reliability() - 0.5) < 0.05                    # signal var 1, the price's noise var 1
    st = checkpoint._to_cpu(crit.state())
    again = ac.AbilityCritic()
    assert again.load(st) and torch.equal(again.halves, crit.halves)
    # an older state (no halves): loads, the reliability unknown until new prices come with them
    old = {k: v for k, v in st.items() if k != "halves"}
    again = ac.AbilityCritic()
    assert again.load(old) and again.halves.isnan().all() and again.reliability() is None
    again.add(X[:60], halves[:60].mean(1))                        # without halves: NaN, still unknown
    assert again.reliability() is None and len(again.halves) == len(again.y)


def test_features_read_the_unit_its_slot_and_the_neighbourhood():
    N = 5
    tokens = torch.zeros(1, N, ob.TOKEN)
    I = ob.INDEX
    tokens[0, :, I["men"]] = 1.0
    tokens[0, :, I["visible"]] = 1.0
    own = torch.tensor([[True, True, True, False, False]])
    pos = torch.tensor([[[0, 0], [20, 0], [60, 0], [30, 0], [200, 0]]], dtype=torch.float) / ob.POS
    tokens[0, 1, I["state_routing"]] = 1.0            # a friend routing 20 m away
    tokens[0, 2, I["melee"]] = 1.0                    # a friend in melee 60 m away
    tokens[0, 3, I["melee"]] = 1.0                    # an enemy in melee 30 m away
    abil = torch.zeros(1, N, 3, mab.SIZE)
    abil[0, 0, 1, 0] = 7.0                            # slot 1's first column
    obs = {"tokens": tokens, "ctx": torch.zeros(1, ob.CONTEXT), "own": own, "pos": pos, "abil": abil}
    x = ac.features(obs, torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    assert x.shape == (1, ac.WIDTH)
    assert float(x[0, ac.OWN]) == 7.0 and float(x[0, ob.INDEX["men"]]) == 1.0
    loc = x[0, -ac.LOCAL:]
    assert float(loc[0]) == pytest.approx(30 / ac.FAR) and float(loc[1]) == 0.0
    near35 = (loc[2:8] * 4).tolist()                  # own, own melee, own wavering, own routing, enemy, enemy melee
    near80 = (loc[8:14] * 4).tolist()
    assert near35 == [1, 0, 0, 1, 1, 1]
    assert near80 == [2, 1, 0, 1, 1, 1]


def test_a_ready_critic_prices_every_moment_and_is_a_control_variate_at_the_replayed():
    actor, crit = tiny()
    env = env_of()
    env.brancher = ab.Brancher(ab.Config(share=1.0, missed=0.5, copies=1, horizon_s=2.0, batch=4, critic=True))
    critic = env.brancher.critic
    critic.last = {"samples": 0, "train": 0.0}                       # an untrained model, but "ready"
    critic.model.shrink.fill_(1.0)
    critic.fit = lambda: critic.last
    critic.ready = lambda: True
    batch = rollout.collect(env, actor, crit, 3)
    st = env.brancher.stats
    br = batch["abil_branch"]
    n_rep = st["replayed_presses"] + st["replayed_missed"]
    assert st["moments"] == st["presses"] + st["missed_moments"] > n_rep > 0
    assert len(critic) == n_rep                                       # the replayed prices went into its buffer
    assert int(br["mask"].sum()) == st["moments"] and st["covered"] == 1.0 and st["critic_on"] == 1
    # the logged choice: a press's slot, a ready ability left at the others
    assert int((batch["action"].ability[br["mask"]] >= 0).sum()) == st["presses"]
    assert br["adv"].abs().max() <= ab.ADV_CLIP and float(br["adv"][br["mask"]].abs().sum()) > 0
    for k in ("q_press", "q_std", "critic_s", "critic_samples", "price_sd", "cv_sd", "cv_used"):
        assert k in st
    assert st["cv_sd"] <= st["price_sd"] or st["cv_used"] == 0


def test_before_it_is_ready_only_the_replayed_moments_count():
    actor, crit = tiny()
    env = env_of()
    env.brancher = ab.Brancher(ab.Config(share=1.0, missed=0.0, copies=1, horizon_s=2.0, batch=4, critic=True))
    batch = rollout.collect(env, actor, crit, 3)
    st = env.brancher.stats
    assert st["critic_on"] == 0 and int(batch["abil_branch"]["mask"].sum()) == st["replayed_presses"]
    assert st["covered"] == pytest.approx(round(st["replayed_presses"] / st["moments"], 4))


def test_run_logs_the_critic_and_carries_its_buffer_in_the_chain(tmp_path, monkeypatch):
    from tools.nn.train import run
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    monkeypatch.setitem(config.PRESETS, "v2", TINY)
    actor, crit = tiny()
    init = checkpoint.save(tmp_path / "init.pt", actor, crit, "v2")        # an old checkpoint: no critic in it

    def args(name, start):
        return run.parser().parse_args(["--name", name, "--init", str(start), "--reward", "v2", "--battles", "4",
                                        "--steps", "2", "--updates", "2", "--minutes", "5", "--device", "cpu",
                                        "--no-eval", "--mix", '{"self": 0.5, "nearest": 0.5}', "--bank", "8",
                                        "--max-units", "4", "--ability-branch", "1.0", "--ability-branch-missed", "1.0",
                                        "--ability-branch-copies", "1", "--ability-branch-horizon", "2",
                                        "--ability-branch-batch", "4", "--ability-critic"])
    _, _, out = run.train(args("abc1", init))
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("critic_samples" in r["ability_branch"] and "moments" in r["ability_branch"] for r in rows)
    saved = checkpoint.train_state(out / "latest.pt")["ability_critic"]
    n = len(saved["y"])
    assert n > 0 and saved["X"].dtype == torch.float16
    _, _, out2 = run.train(args("abc2", out / "latest.pt"))
    first = json.loads((out2 / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["ability_branch"]["critic_samples"] > n
    with pytest.raises(SystemExit):
        run.train(run.parser().parse_args(["--name", "abc3", "--ability-critic", "--device", "cpu"]))
