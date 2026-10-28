"""Drill U3 'several enemies' (tools/nn/unit/drill_multi.py) and the four-variant branches (branch.KIT_MULTI). Needs
torch: run in the container."""
import math

import pytest

torch = pytest.importorskip("torch")

from tests.tools.test_unit_view_net_torch import battle  # noqa: E402
from tools.nn.sim import battle as Bt  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim import replay  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import branch as Br  # noqa: E402
from tools.nn.unit import drill_multi as DM  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
SPEAR, CLANRAT, SLAVE = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0", \
    "wh2_main_skv_inf_skavenslave_spearmen_0"


def rel(st, b, j):
    u = st.u
    return (math.degrees(math.atan2(float(u["x"][b, j] - u["x"][b, 0]), float(u["z"][b, j] - u["z"][b, 0])))
            - float(u["b"][b, 0]) + 180) % 360 - 180


def test_two_enemies_from_the_layouts_sides_and_with_the_timing_wanted():
    sp = DM.specs(60, torch.Generator().manual_seed(7), hard_share=0.0)
    st, own, area = DM.build(sp, P, "cpu")
    H = st.N // 2
    pp = __import__("tools.nn.unit.drill_hold", fromlist=["_passports"])._passports()
    for b, s in enumerate(sp):
        ra, rb = DM.LAYOUTS[s["layout"]]
        assert abs((rel(st, b, H) - ra + 180) % 360 - 180) <= DM.JITTER + 1e-3
        assert min(abs((rel(st, b, H + 1) - r + 180) % 360 - 180) for r in (rb, -rb)) <= DM.JITTER + 1e-3
        ta = s["gap"] / pp[s["foe"]]["speed"]["run"]
        tb = s["gap_b"] / pp[s["foe_b"]]["speed"]["run"]
        if s["timing"] == "together":
            assert tb == pytest.approx(ta, abs=1e-6)
        else:
            assert DM.LAG_S[0] - 1e-6 <= tb - ta <= DM.LAG_S[1] + 1e-6
    assert {s["layout"] for s in sp} == set(DM.LAYOUTS) and {s["timing"] for s in sp} == set(DM.TIMINGS)
    assert (st.u["side"][:, H:H + 2] == 2).all() and len(DM.exam_specs()) == 360


def test_the_behaviour_names_whom_it_attacks_or_faces():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 60, 180), (SLAVE, 60, 0, 270)])
    H = st.N // 2
    Bt.step(st, replay.hold(st), P)
    k = Br.KIT_MULTI
    assert k.names() == ["attack_A", "attack_B", "stand_A", "stand_B", "attack_A_then_B"]
    assert int(k.behaviour(st, Br.bracing_deg(P))[0]) == 2          # facing A (ahead)
    st.u["b"][0, 0] = 90.0                                           # now facing B (east)
    assert int(k.behaviour(st, Br.bracing_deg(P))[0]) == 3
    st.u["b"][0, 0] = 225.0                                          # facing neither
    assert int(k.behaviour(st, Br.bracing_deg(P))[0]) == Br.OTHER
    o = replay.hold(st)
    o.kind[0, 0], o.target[0, 0] = O.ATTACK, H + 1
    Bt.step(st, o, P)
    assert int(k.behaviour(st, Br.bracing_deg(P))[0]) == 1


def test_four_variants_play_from_the_copy():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 50, 180), (SLAVE, 70, 0, 270)])
    before = {k: v.clone() for k, v in st.u.items()}
    f = lambda v: torch.full((1, st.N), float(v))       # noqa: E731
    area = {"x": f(0), "z": f(0), "b": f(0), "front": f(200), "depth": f(200)}
    rep = {}
    t = Br.KIT_MULTI.play(st, area, P, Bt.step, horizon_s=15.0, cap_s=30.0, rows=2, report=rep)
    assert t.shape == (1, 5) and torch.isfinite(t).all() and len(set(t[0].tolist())) > 1
    assert rep["braced"].shape == (1, 5)
    for k, v in before.items():
        assert torch.equal(st.u[k], v), k


def test_the_bonus_is_the_margin_over_the_second_best():
    trades = torch.tensor([[0.1, 0.3, 0.25, 0.0], [0.2, 0.2, 0.2, 0.2], [0.5, 0.1, 0.0, 0.0]])
    beh = torch.tensor([1, 0, 2])
    bon, best = Br.bonus_of(beh, trades, 2.0)
    assert best.tolist()[0] == 1 and best.tolist()[2] == 0
    assert bon.tolist() == pytest.approx([2 * 0.05, 0.0, 0.0])


def test_the_end_waits_for_both_enemies():
    sp = DM.specs(2, torch.Generator().manual_seed(8))
    st, own, area = DM.build(sp, P, "cpu")
    H = st.N // 2
    end = DM.Ender(st.B, "cpu")
    st.u["men"][0, H] = 0.0                                           # battle 0: A dead, B still there
    st.u["men"][1, H], st.u["men"][1, H + 1] = 0.0, 0.0              # battle 1: both dead
    end.after_step(st, SP)
    assert st.done.tolist() == [False, True] and int(end.why[1]) == 2


def test_a_short_rollout_of_the_drill_with_branches():
    torch.manual_seed(0)
    cfg = {"battles": 4, "seconds": 40, "cell_m": 8.0, "bounds": None}
    sp = DM.specs(4, torch.Generator().manual_seed(9))
    for s in sp:
        s["gap"], s["gap_b"] = 80.0, 90.0
    br = {"share": 1.0, "weight": 1.0, "lead_s": Br.LEAD_S, "horizon_s": 10.0}
    batch, stats = T.rollout(Nt.UnitNet(), None, cfg, P, "cpu", T.Duty("cpu"), T.multi_setup(cfg, P, "cpu", sp=sp),
                             branch=br)
    assert stats["bans_broken"] == 0 and stats["nan"] == 0
    info = stats["_branch"]
    assert info["trades"].shape == (4, 5) and info["names"] == Br.KIT_MULTI.names()
    assert bool(info["branched"].any()) and set(stats["branch"]["best"]) == set(Br.KIT_MULTI.names())


def test_hard_skirmishes_a_weak_one_first_a_dangerous_one_later_not_opposite():
    sp = DM.specs(400, torch.Generator().manual_seed(12))
    hard = [s for s in sp if s["hard"]]
    assert 0.4 < len(hard) / len(sp) < 0.6
    for s in hard:
        assert DM.danger(s["foe_b"]) >= DM.HARD_RATIO * DM.danger(s["foe"])
        assert s["layout"] in DM.HARD_LAYOUTS and s["timing"] == "staggered" and s["lag"] >= DM.LAG_S[0]
    assert any(s["theirs"] == s["ours"] for s in hard)                 # mirrors (Empire against Empire ...)
    st, own, area = DM.build(hard[:8], P, "cpu")                       # the simulator takes them
    for b, s in enumerate(hard[:8]):
        assert st.keys[b][st.N // 2] == s["foe"] and st.keys[b][st.N // 2 + 1] == s["foe_b"]
    ex = DM.exam_specs()
    assert sum(s["hard"] for s in ex) == 120


def test_the_switch_variant_and_its_tracking():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 40, 180), (SLAVE, 120, 0, 270)])
    H = st.N // 2
    k = Br.KIT_MULTI
    beh = torch.tensor([0])                                            # attacked A at the first contact
    o = replay.hold(st)
    o.kind[0, 0], o.target[0, 0] = O.ATTACK, H + 1
    Bt.step(st, o, P)
    assert int(k.track(st, beh, SP)[0]) == 0                           # B still far: not the switch
    st.u["x"][0, H + 1] = 40.0                                         # B within its moment
    assert bool(k.switch_moment(st, SP)[0])
    assert int(k.track(st, beh, SP)[0]) == 4
    assert int(k.track(st, torch.tensor([2]), SP)[0]) == 2             # (a stand stays a stand)
    f = lambda v: torch.full((1, st.N), float(v))       # noqa: E731
    st2 = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 45, 180), (SLAVE, 110, 0, 270)])
    area = {"x": f(0), "z": f(0), "b": f(0), "front": f(200), "depth": f(200)}
    t = k.play(st2, area, P, Bt.step, horizon_s=30.0, cap_s=60.0, rows=1)
    assert float(t[0, 4]) != float(t[0, 0])                            # the switch played out differently


def test_the_exam_kit_adds_attack_a_then_face_b():
    k = Br.KIT_MULTI_EXAM
    assert k.names()[-2:] == ["attack_A_then_B", "attack_A_then_face_B"]
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 40, 180), (SLAVE, 40, 0, 270)])
    H = st.N // 2
    o = replay.hold(st)
    o.face[0, 0] = 90.0                                                # a hold facing B (east)
    Bt.step(st, o, P)
    assert int(k.track(st, torch.tensor([0]), SP)[0]) == 5
    st2 = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 45, 180), (SLAVE, 110, 0, 270)])
    f = lambda v: torch.full((1, st2.N), float(v))       # noqa: E731
    area = {"x": f(0), "z": f(0), "b": f(0), "front": f(200), "depth": f(200)}
    t = k.play(st2, area, P, Bt.step, horizon_s=30.0, cap_s=60.0, rows=1)
    assert t.shape == (1, 6) and float(t[0, 5]) != float(t[0, 4])


def test_mix_trains_the_three_drills_in_turn(tmp_path):
    log, same = T.main(["--drill", "mix", "--battles", "3", "--seconds", "15", "--updates", "5", "--minibatch", "64",
                        "--out", str(tmp_path)])
    assert [r["drill"] for r in log] == ["hold", "go", "multi", "hold2", "shoot1"] and same
    assert all(r["bans_broken"] == 0 and r["nan"] == 0 for r in log)
