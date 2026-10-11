"""The bank of armies built in a worker process (tools/nn/train/bankproc.py): the same bank as built in the training
process, for the same random numbers (torch: run in the training container)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.train import bankproc, drills, league, rollout  # noqa: E402


def same(a, b, path="source"):
    """Every tensor, list, number and generator of two sources equal (generators: the same next draws)."""
    assert type(a) is type(b), path
    if torch.is_tensor(a):
        assert a.dtype == b.dtype and a.device == b.device and torch.equal(a, b), path
    elif isinstance(a, torch.Generator):
        assert torch.equal(a.get_state(), b.get_state()), path
    elif isinstance(a, dict):
        assert list(a) == list(b), path
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif type(a).__module__.startswith("tools.") and hasattr(a, "__dict__"):
        assert sorted(vars(a)) == sorted(vars(b)), path
        for k in vars(a):
            same(getattr(a, k), getattr(b, k), f"{path}.{k}")
    else:
        assert a == b, path


def spec(seed, kite=True, drill=False):
    rng = np.random.default_rng(seed)
    lay = league.layout(8, 1, {"self": 0.5, "past": 0.25, "nearest": 0.25})
    s = bankproc.Spec(rng.integers(0, 10_000, 6), int(rng.integers(1 << 30)), 19, rollout.params_with_limit(None), lay,
                      (1.0, 3))           # (kite battles need room for 19 units a side; the others small)
    if drill:
        code = league.CODE[drills.opponent("kiting")]
        s.layout = league.Layout(np.zeros(8, dtype=int), np.array([1, 2] * 4), np.array([code] * 4 + [league.CODE["self"]] * 4))
        s.drills, s.per_drill, s.max_units = True, 4, 6
    if kite:
        s.kite_share, s.kite_seeds, s.kite_seed = 0.5, rng.integers(0, 10_000, 3), int(rng.integers(1 << 30))
    return s


def picks(src, n=3):
    return [src.pick().tolist() for _ in range(n)]


def test_the_workers_bank_is_the_one_built_here():
    w = bankproc.Ahead()
    try:
        specs = [spec(1), spec(2, kite=False)]
        pending = [w.submit(s) for s in specs]                 # two in a row: taken in order
        for s, p in zip(specs, pending):
            here = bankproc.build(s, "cpu")
            there = w.take(p, "cpu")
            same(here, there)
            assert picks(here) == picks(there)
    finally:
        w.close()
    assert not w.dir.exists()


def test_a_source_with_the_drills_crosses_by_its_seeds():
    s = spec(3, drill=True)
    here = bankproc.build(s, "cpu")
    packed = bankproc._walk(bankproc.build(s, "cpu"), bankproc._pack_leaf, {})
    assert not any(isinstance(v, torch.Generator) for v in vars(packed).values())
    there = bankproc.to_device(packed, "cpu")
    same(here, there)
    assert picks(here) == picks(there)


def test_a_used_generator_is_refused():
    src = bankproc.build(spec(4, kite=False), "cpu")
    src.pick()
    with pytest.raises(RuntimeError, match="used before"):
        bankproc._walk(src, bankproc._pack_leaf, {})


def test_a_failed_build_raises_in_the_training_process():
    w = bankproc.Ahead()
    try:
        bad = spec(5, kite=False)
        bad.max_units = -1
        p = w.submit(bad)
        with pytest.raises(RuntimeError, match="the worker failed"):
            w.take(p, "cpu")
        good = spec(6, kite=False)
        same(bankproc.build(good, "cpu"), w.take(w.submit(good), "cpu"))     # it goes on
    finally:
        w.close()
