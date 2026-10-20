"""tools/nn/general/executor.py: the executor = today's unit network + the task input at zero weights (S28 skeleton).
With the task input at zero - or none - every output is the old network's to the bit. torch: the training container."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import executor                          # noqa: E402
from tools.nn.model import config, policy                      # noqa: E402
from tools.nn.train import checkpoint, league, rollout         # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
GCFG = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2)
MIRROR = [("arena", "attack")]


def observation(seed=0):
    """A real decision's input of the learner rows (rollout's, with the v2 inputs) and the unit network."""
    torch.manual_seed(seed)
    old = policy.Actor(TINY).eval()
    env = rollout.Battles(league.layout(2, 1, {"self": 1.0}), MIRROR)
    env.step(old)
    a, frame, _ = env.cur
    obs_r, *_ = env._act(old, a, frame, env.rows_learn, None, True)
    return old, obs_r


def task_of(obs, seed=1):
    g = torch.Generator().manual_seed(seed)
    R, N = obs["own"].shape
    return dict(obs, task=torch.randn(R, N, GCFG.task_dim, generator=g), task_on=obs["own"] & obs["ctrl"])


def same(a, b):
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_with_the_task_input_at_zero_the_executor_is_the_old_network_to_the_bit():
    old, obs = observation()
    new = executor.wrap(old, GCFG).eval()
    assert isinstance(new, executor.GeneralActor) and float(new.task_in.weight.detach().abs().sum()) == 0
    with torch.no_grad():
        la, aa, ha = old.act(obs, None, greedy=True)
        lb, ab, hb = new.act(task_of(obs), None, greedy=True)       # a random task: no effect at zero weights
        lc, ac, hc = new.act(obs, None, greedy=True)                # no task at all
    same(la, lb)
    same(la, lc)
    assert torch.equal(ha, hb) and torch.equal(ha, hc)
    for f in aa.parts():
        assert torch.equal(getattr(aa, f), getattr(ab, f)), f


def test_the_task_input_once_trained_changes_the_decision():
    old, obs = observation()
    new = executor.wrap(old, GCFG).eval()
    with torch.no_grad():
        new.task_in.weight.normal_(0, 1.0)
        la, *_ = old.act(obs, None, greedy=True)
        lb, *_ = new.act(task_of(obs), None, greedy=True)
        none = dict(task_of(obs), task_on=torch.zeros_like(obs["own"]))
        lc, *_ = new.act(none, None, greedy=True)                   # nobody has a task: the old network again
    assert not torch.allclose(la["kind"], lb["kind"])
    same(la, lc)


def test_a_checkpoint_keeps_the_general_and_a_unit_network_loads_into_an_executor(tmp_path):
    old, obs = observation()
    new = executor.wrap(old, GCFG).eval()
    with torch.no_grad():
        for p in new.general.parameters():
            p.add_(0.01)
    path = checkpoint.save(tmp_path / "g.pt", new, None, "v2")
    data = checkpoint.read(path)
    assert checkpoint.general_of(data) == GCFG
    back = checkpoint.load_policy(path)
    assert isinstance(back, executor.GeneralActor) and back.gcfg == GCFG
    for (n, p), (m, q) in zip(new.state_dict().items(), back.state_dict().items()):
        assert n == m and torch.equal(p, q), n
    # an old checkpoint loads as the unit network it is
    p2 = checkpoint.save(tmp_path / "u.pt", old, None, "v2")
    assert not isinstance(checkpoint.load_policy(p2), executor.GeneralActor)
    # the general's parameters come last (run.resume keeps the order of the old ones)
    names = [n for n, _ in new.named_parameters()]
    first = min(i for i, n in enumerate(names) if n.startswith(executor.FRESH))
    assert all(n.startswith(executor.FRESH) for n in names[first:])
    assert names[:first] == [n for n, _ in old.named_parameters()]
