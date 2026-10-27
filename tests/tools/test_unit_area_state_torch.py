"""Whose the area is (tools/nn/unit/drill_go.whose_area) and its small reward signal in the loop (train.py
--area-reward). Needs torch: run in the container."""
import pytest

torch = pytest.importorskip("torch")

from tests.tools.test_unit_view_net_torch import battle  # noqa: E402
from tools.nn.sim import battle as Bt  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim import replay  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import drill_go as DG  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402

P = load()
SPEAR, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0"


def rect(st, x, z, side=64.0):
    f = lambda v: torch.full((st.B, st.N), float(v))       # noqa: E731
    return {"x": f(x), "z": f(z), "b": f(0.0), "front": f(side), "depth": f(side)}


def test_ours_theirs_empty():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 300, 180)])
    H = st.N // 2
    assert int(DG.whose_area(st, rect(st, 0, 0))[0]) == DG.OURS             # we in it, no enemy
    assert int(DG.whose_area(st, rect(st, 0, 150))[0]) == DG.EMPTY          # nobody in it
    assert int(DG.whose_area(st, rect(st, 0, 300))[0]) == DG.THEIRS         # the enemy holds it
    st.u["z"][0, H] = 20.0                                                    # the enemy in our area with us, no fight
    assert int(DG.whose_area(st, rect(st, 0, 0))[0]) == DG.THEIRS
    st.u["r"][0, H] = True                                                    # ... routing: it holds nothing
    assert int(DG.whose_area(st, rect(st, 0, 0))[0]) == DG.OURS


def test_a_fight_with_the_enemy_in_the_area_makes_it_ours_wherever_our_centre_is():
    st = battle([(SPEAR, 0, -20, 0)], [(CLANRAT, 0, 20, 180)])
    H = st.N // 2
    o = replay.hold(st)
    o.kind[0, H], o.target[0, H] = O.ATTACK, 0
    for _ in range(60):
        Bt.step(st, o, P)
        if bool(st.u["m"][0, 0]) and bool((st.u["hit_sec"].view(1, st.N, st.N)[0, H, 0] > 0)):
            break
    assert bool(st.u["m"][0, 0])
    area = rect(st, 0, float(st.u["z"][0, H]), side=20.0)                    # only the enemy's centre in the area
    assert int(DG.whose_area(st, area)[0]) == DG.OURS


def test_the_signal_is_added_per_second_and_the_enemys_only_after_arrival():
    cfg = {"battles": 4, "seconds": 40, "cell_m": 8.0, "bounds": None}
    sp = DG.specs(4, torch.Generator().manual_seed(11), fixed=[("held", "fast"), ("early", "fast")])
    seen = {}

    def watch(st, hp0, was_done, area, step, setup=None):
        wa = DG.whose_area(st, area)
        arr = seen["ender"].arrived
        wa = torch.where((wa == DG.THEIRS) & ~arr, torch.zeros_like(wa), wa)
        seen.setdefault("sum", torch.zeros(st.B))
        seen["sum"] += torch.where(was_done, torch.zeros(st.B), wa.float()) * P.dt
    out = {}
    for w in (0.0, 0.01):
        torch.manual_seed(0)
        setup = T.go_setup(cfg, P, "cpu", sp=sp)
        seen.clear()
        seen["ender"] = setup["ender"]
        b, _ = T.rollout(Nt.UnitNet(), None, dict(cfg, area_reward=w), P, "cpu", T.Duty("cpu"), setup, watch=watch)
        out[w] = (b["reward"].sum(0), seen["sum"].clone())
    diff = out[0.01][0] - out[0.0][0]
    assert torch.allclose(diff, 0.01 * out[0.01][1], atol=1e-5)
    assert torch.equal(out[0.0][1], out[0.01][1])


def test_the_default_is_on_for_u2_only():
    assert T.AREA_REWARD == 0.001
