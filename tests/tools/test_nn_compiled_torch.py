"""tools/nn/sim/compiled.py: on CUDA the compiled simulator step and observation compute as the eager ones - the code
the companion runs in the game (docs/en/training/simulator.md "Compiled and eager step"). CUDA only (skipped elsewhere: the
container without a GPU, the host); the first run compiles the step (~2-3 min, then the container's inductor cache)."""
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only: torch.compile of the simulator")

from tools.nn.model import observation as ob                   # noqa: E402
from tools.nn.sim import battle                                 # noqa: E402
from tools.nn.sim.compiled import compiled                      # noqa: E402
from tools.nn.train import opponents, rollout, scenes           # noqa: E402


def _div(a, b):
    return a / b


def test_a_compiled_division_is_exact():
    # Triton's own fp32 division (div.full) gave 4068 / 4068 = 0.99999994: 2672 of the integers 1..20000
    v = torch.arange(1, 20001, device="cuda", dtype=torch.float32)
    assert bool((compiled(_div)(v, v) == 1).all())
    a, b = torch.rand(4096, device="cuda") * 1000 + 1, torch.rand(4096, device="cuda") * 1000 + 1
    assert torch.equal(compiled(_div)(a, b), a / b)


def test_the_compiled_step_and_observation_are_the_eager_ones():
    """The same battles stepped and observed eagerly and compiled (the scripts' orders, both sides): every token column
    and the context within 1e-5 and the memory's damage clock the same over the steps; the whole units' health exactly
    1 (the compiled division took 6e-8 of it: the damage clock started at the first decision of every battle)."""
    params = rollout.params_with_limit(600.0)
    src = scenes.Generated(range(500, 504), 8, params, "cuda", seed=1)
    st_e, setup, _ = rollout.open_rows(src)
    st_c = st_e.clone()
    step_c, look_c = battle.stepper("cuda", True), rollout.observer("cuda", True)
    mem_e = {s: ob.start(st_e.observation(), setup, s) for s in (1, 2)}
    mem_c = {s: ob.start(st_c.observation(), setup, s) for s in (1, 2)}
    for i in range(16):
        for s in (1, 2):
            o_e, mem_e[s] = ob.observe(st_e.observation(), setup, s, mem_e[s])
            o_c, mem_c[s] = look_c(st_c.observation(), setup, s, mem_c[s])
            assert float((o_c.tokens - o_e.tokens).abs().max()) < 1e-5, (i, s)
            assert float((o_c.ctx - o_e.ctx).abs().max()) < 1e-5, (i, s)
            assert torch.equal(mem_c[s].hit_t, mem_e[s].hit_t), (i, s)
        battle.step(st_e, opponents.nearest(st_e), params, params.dt)
        step_c(st_c, opponents.nearest(st_c), params, params.dt)
        whole = (st_e.u["side"] > 0) & (st_e.u["hp_abs"] == st_e.u["hp0"])
        assert bool(whole.any())
        assert bool((st_c.u["hp"][whole] == 1).all()) and bool((st_e.u["hp"][whole] == 1).all())
