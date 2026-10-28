"""The order changes' cost in the unit loop (tools/nn/unit/train.py order_changes, --order-change-cost): a change costs,
the same order does not, the leash's replacement is not the network's change. Needs torch: run in the container."""
import pytest

torch = pytest.importorskip("torch")

from tests.tools.test_unit_view_net_torch import battle  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim import replay  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import decode as Dc  # noqa: E402
from tools.nn.unit import drill_hold as DH  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402

P = load()
SPEAR, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0"
BI, II = torch.tensor([0]), torch.tensor([0])


def order(kind=O.MOVE, x=0.0, z=0.0, target=-1, face=-1.0):
    o = O.hold(1, 2)
    o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.target[0, 0], o.face[0, 0] = kind, x, z, target, face
    return o


def ch(a, b):
    return bool(T.order_changes(a, b, BI, II)[0])


def test_what_is_a_change():
    base = order(O.MOVE, 0, 0, face=10.0)
    assert not ch(order(O.MOVE, 0, 0, face=10.0), base)                 # the same order
    assert not ch(order(O.MOVE, 6, 0, face=20.0), base)                 # a point 6 m off, a facing 10 deg off
    assert ch(order(O.MOVE, 12, 0, face=10.0), base)                    # a point 12 m off
    assert ch(order(O.MOVE, 0, 0, face=40.0), base)                     # a facing 30 deg off
    assert ch(order(O.MOVE, 0, 0), base)                                # the facing taken away
    assert ch(order(O.HOLD), base) and ch(order(O.ATTACK, target=1), base)
    assert ch(order(O.ATTACK, target=1), order(O.ATTACK, target=0)) and not ch(order(O.ATTACK, target=1),
                                                                                order(O.ATTACK, target=1))
    assert not ch(order(O.KEEP), base)


def test_the_leash_replacement_is_not_the_networks_change():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 140, 180)])
    H = st.N // 2
    own = torch.zeros((1, st.N), dtype=torch.bool)
    own[0, 0] = True
    f = lambda v: torch.full((1, st.N), float(v))       # noqa: E731
    area = {"x": f(0), "z": f(0), "b": f(0), "front": f(60), "depth": f(60)}
    st.u["vz"][0, 0] = 3.0
    st.u["z"][0, 0] = 28.0                                               # about to leave, the target far out
    intended = replay.hold(st)
    intended.kind[0, 0], intended.target[0, 0] = O.ATTACK, H
    leash = Dc.Leash(1, st.N)
    out = leash.apply(st, intended, area, own)
    assert bool(leash.on[0, 0]) and int(out.kind[0, 0]) == O.MOVE          # the decoder replaced it
    assert not bool(T.order_changes(intended, intended, BI, II)[0])        # the network's own order: unchanged


def test_the_cost_lands_on_the_reward():
    cfg = {"battles": 4, "seconds": 20, "cell_m": 8.0, "bounds": None}
    sp = DH.specs(4, torch.Generator().manual_seed(2))
    out = {}
    for c in (0.0, 0.01):
        torch.manual_seed(0)
        b, s = T.rollout(Nt.UnitNet(), None, dict(cfg, order_change=c), P, "cpu", T.Duty("cpu"),
                         T.hold_setup(cfg, P, "cpu", sp=sp))
        out[c] = (b["reward"].sum(), s["order_changes"])
    assert out[0.01][1] == out[0.0][1] > 0
    assert float(out[0.0][0] - out[0.01][0]) == pytest.approx(0.01 * out[0.0][1], rel=1e-4)
