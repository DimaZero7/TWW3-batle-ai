"""Branches on the approach (tools/nn/unit/branch.py): the copy, the moment, the two scripted branches, the behaviour at
the contact, the bonus and its decay; the rollout's bonus at the contact's decision. Needs torch: run in the container."""
import pytest

torch = pytest.importorskip("torch")

from tests.tools.test_unit_view_net_torch import battle  # noqa: E402
from tools.nn.sim import battle as Bt  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim import replay  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import branch as Br  # noqa: E402
from tools.nn.unit import drill_hold as DH  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
SPEAR, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0"


def square(st, side=200.0):
    f = lambda v: torch.full((st.B, st.N), float(v))       # noqa: E731
    return {"x": f(0), "z": f(0), "b": f(0), "front": f(side), "depth": f(side)}


def test_take_and_cat_give_the_same_battles():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 60, 180)])
    two = Br.cat([st, Br.take(st, torch.tensor([0]))])
    assert two.B == 2
    for k, v in st.u.items():
        assert torch.equal(two.u[k][0], v[0]) and torch.equal(two.u[k][1], v[0]), k


def test_the_moment_is_the_enemys_charge_distance_plus_lead():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 200, 180)])
    H = st.N // 2
    reach = float(st.u["charge_dist"][0, H] + st.u["run"][0, H] * Br.LEAD_S)
    gap0 = float(Br.edge_gap(st, SP)[0])
    for d, want in ((reach + 2, False), (reach - 2, True)):
        st.u["z"][0, H] = 200.0 - (gap0 - d)
        assert bool(Br.moment(st, SP)[0]) == want


def test_behaviour_stand_attack_other():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 60, 180)])
    H = st.N // 2
    Bt.step(st, replay.hold(st), P)
    assert int(Br.behaviour(st, Br.bracing_deg(P))[0]) == Br.STAND
    st.u["b"][0, 0] = 180.0                                  # facing away
    assert int(Br.behaviour(st, Br.bracing_deg(P))[0]) == Br.OTHER
    o = replay.hold(st)
    o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
    Bt.step(st, o, P)
    assert int(Br.behaviour(st, Br.bracing_deg(P))[0]) == Br.ATTACK


def test_the_branches_play_from_the_copy_and_leave_it_alone():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 50, 180)])
    before = {k: v.clone() for k, v in st.u.items()}
    ts, ta = Br.play(st, square(st), P, Bt.step, horizon_s=20.0, cap_s=40.0, rows=3)
    assert ts.shape == (1,) and ta.shape == (1,) and torch.isfinite(ts).all() and torch.isfinite(ta).all()
    assert float(ts[0]) != float(ta[0])
    for k, v in before.items():
        assert torch.equal(st.u[k], v), k


def test_the_bonus_pays_the_better_branch_and_decays():
    beh = torch.tensor([Br.STAND, Br.ATTACK, Br.OTHER, Br.STAND])
    ts = torch.tensor([0.2, 0.2, 0.2, -0.1])
    ta = torch.tensor([0.0, 0.0, 0.0, 0.1])
    bon, best = Br.bonus(beh, ts, ta, 2.0)
    assert bon.tolist() == pytest.approx([0.4, 0.0, 0.0, 0.0]) and best.tolist() == [0, 0, 0, 1]
    assert [Br.weight_at(1.0, k, 4) for k in (0, 2, 4, 9)] == [1.0, 0.5, 0.0, 0.0]
    assert Br.weight_at(1.0, 50, 0) == 1.0


def test_the_rollout_adds_the_bonus_at_the_contact():
    torch.manual_seed(0)
    cfg = {"battles": 6, "seconds": 60, "cell_m": 8.0, "bounds": None}
    sp = DH.specs(6, torch.Generator().manual_seed(2))
    for s in sp:
        s["gap"] = 80.0
    net = Nt.UnitNet()
    duty = T.Duty("cpu")
    br = {"share": 1.0, "weight": 1.0, "lead_s": Br.LEAD_S, "horizon_s": 10.0}
    torch.manual_seed(1)
    b1, s1 = T.rollout(net, None, cfg, P, "cpu", duty, T.hold_setup(cfg, P, "cpu", sp=sp), branch=br)
    torch.manual_seed(1)
    b0, s0 = T.rollout(net, None, cfg, P, "cpu", duty, T.hold_setup(cfg, P, "cpu", sp=sp),
                       branch=dict(br, weight=0.0))
    info = s1["_branch"]
    assert bool(info["branched"].any()) and "branch" in s1
    diff = b1["reward"] - b0["reward"]
    bon, _ = Br.bonus(info["behaviour"], info["t_stand"], info["t_attack"], 1.0)
    for b in range(6):
        d = int(info["contact_d"][b])
        want = float(bon[b]) if bool(info["branched"][b]) and d >= 0 else 0.0
        assert float(diff[:, b].sum()) == pytest.approx(want, abs=1e-5)
        if want:
            assert float(diff[d, b]) == pytest.approx(want, abs=1e-5)
