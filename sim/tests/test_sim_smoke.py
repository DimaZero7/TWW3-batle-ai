"""The simulator's smoke test: every module of sim imports; generated battles (sim.armies, the training's scheme v3)
run N steps against the scripted opponents (ai_game for side 1, nearest for side 2) without an error, every number
stays finite, men fall, and the same battles run twice end in the same state (the simulator is deterministic).
The battle needs torch (the container); the import check runs on the host too (modules that need torch skipped)."""
import importlib
import pkgutil

import pytest

import sim

SEEDS = (11, 12)
STEPS = 100                        # 50 s at the simulator's 0.5 s step (the first contacts: men fall from ~40 s)


def test_every_module_imports():
    failed = {}
    for m in pkgutil.walk_packages(sim.__path__, "sim."):
        if m.name.startswith("sim.tests") or m.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(m.name)
        except ModuleNotFoundError as e:
            if e.name != "torch":              # the host's .venv has no torch: those modules run in the container
                failed[m.name] = repr(e)
        except Exception as e:                 # noqa: BLE001 (any import error is the finding)
            failed[m.name] = repr(e)
    assert not failed, failed


def run(P):
    torch = pytest.importorskip("torch")
    from sim import battle, scenario, scripts
    from sim import orders as O
    from sim.armies import generate
    armies = [scenario.from_arena(generate.battle(s, armies="v3"), "attack" if i % 2 == 0 else "defend")
              for i, s in enumerate(SEEDS)]
    st = scenario.build(armies, P)
    men0 = float(st.u["men"].sum())
    for _ in range(STEPS):
        orders = O.merge(scripts.ai_game(st), scripts.nearest(st), st.u["side"] == 2)
        O.check(orders, st.N)
        battle.step(st, orders, P, P.dt)
    for k, v in st.u.items():
        if torch.is_floating_point(v):
            assert bool(torch.isfinite(v).all()), k
    return st, men0


def test_generated_battles_run_and_repeat():
    torch = pytest.importorskip("torch")
    from sim.params import load
    P = load()
    a, men0 = run(P)
    b, _ = run(P)
    assert float(a.t.min()) > 0 and float(a.u["men"].sum()) < men0           # time ran, the armies fought
    for k in a.u:
        assert torch.equal(a.u[k], b.u[k]), k
    assert torch.equal(a.t, b.t) and torch.equal(a.done, b.done) and torch.equal(a.winner, b.winner)
