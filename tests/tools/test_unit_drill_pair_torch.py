"""Drill U4 'friends beside' (tools/nn/unit/drill_pair.py): two network units with areas side by side / in echelon,
each sees the other; the end waits for both; the loop runs two viewers a battle with the side's reward shared. Needs
torch: run in the container."""
import pytest

torch = pytest.importorskip("torch")

from tests.tools.test_unit_view_net_torch import card  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import decode as Dc  # noqa: E402
from tools.nn.unit import drill_hold as DH  # noqa: E402
from tools.nn.unit import drill_pair as DP  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402
from tools.nn.unit import view as V  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]


def test_two_units_each_in_its_own_area_side_by_side_or_in_echelon():
    sp = DP.specs(40, torch.Generator().manual_seed(3))
    st, own, area = DP.build(sp, P, "cpu")
    assert own[:, :2].all() and not own[:, 2:].any()
    H = st.N // 2
    for b, s in enumerate(sp):
        for i in (0, 1):
            ai = {k: v[b, i] for k, v in area.items() if k != "on"}
            assert float(Dc.signed_dist(st.u["x"][b, i], st.u["z"][b, i], ai)) > 0       # inside its own
            assert DH.min_cells() * DH.CELL <= float(ai["front"]) <= DP.MAX_PAIR_CELLS * DH.CELL
        a0 = {k: v[b, 0] for k, v in area.items() if k != "on"}
        a1 = {k: v[b, 1] for k, v in area.items() if k != "on"}
        s1, f1 = Z.local(a1["x"], a1["z"], a0["x"], a0["z"], s["b"])
        gap = float(s1) - float(a0["front"] + a1["front"]) / 2                      # the areas' gap across the front
        assert gap == pytest.approx(s["gap_cells"] * DH.CELL, abs=1e-3)
        assert float(f1) == pytest.approx(s["shift"] * DH.CELL, abs=1e-3)
        assert float(st.u["b"][b, 0]) == float(st.u["b"][b, 1])
        n_foes = int((st.u["side"][b] == 2).sum())
        assert n_foes == len(DP.ENEMY_SETS[s["enemies"]]) and int((st.u["side"][b, H:] == 2).sum()) == n_foes
    assert {s["layout"] for s in sp} == set(DP.LAYOUTS) and len(DP.exam_specs()) == 240


def test_each_unit_sees_the_other_in_its_rows_and_on_its_map():
    sp = DP.specs(4, torch.Generator().manual_seed(5))
    st, own, area = DP.build(sp, P, "cpu")
    now = V.raw(st, SP)
    for i in (0, 1):
        who = torch.zeros_like(own)
        who[:, i] = True
        v = V.build(now, V.Memory(st.B, st.N).hist, card(st), who, area={k: x for k, x in area.items()})
        other = 1 - i
        assert all(other in v["slot"][b].tolist() for b in range(st.B))
        rows = (v["slot"] == other)
        assert (v["whose"][rows] == V.OWN).all()
        assert (v["map"][:, Z.L["own"]].flatten(1).sum(1) > 0).all()


def test_the_end_waits_for_both_of_ours():
    sp = DP.specs(2, torch.Generator().manual_seed(6))
    st, own, area = DP.build(sp, P, "cpu")
    end = DP.Ender(st.B, "cpu")
    st.u["r"][0, 0] = True                                        # battle 0: one of ours routs
    st.u["r"][1, 0], st.u["men"][1, 1] = True, 0.0                # battle 1: both out
    end.after_step(st, SP)
    assert st.done.tolist() == [False, True] and int(end.why[1]) == 1


def test_the_loop_runs_two_viewers_a_battle_with_the_reward_shared():
    torch.manual_seed(0)
    cfg = {"battles": 3, "seconds": 20, "cell_m": 8.0, "bounds": None}
    sp = DP.specs(3, torch.Generator().manual_seed(7))
    batch, stats = T.rollout(Nt.UnitNet(), Nt.Critic(), cfg, P, "cpu", T.Duty("cpu"), T.pair_setup(cfg, P, "cpu", sp=sp))
    assert stats["bans_broken"] == 0 and stats["nan"] == 0
    assert batch["act"]["where"].shape[1] == 6 and batch["bi"].tolist() == [0, 0, 1, 1, 2, 2]
    rw = batch["reward"]
    assert torch.equal(rw[:, 0::2], rw[:, 1::2])                     # the side's reward, the same for both units


def test_the_exam_counts_jostling_when_our_formations_touch():
    from tools.nn.unit import exam_pair as EP
    sp = DP.specs(2, torch.Generator().manual_seed(8))
    st, own, area = DP.build(sp, P, "cpu")
    st.u["x"][1, 1], st.u["z"][1, 1] = st.u["x"][1, 0] + 5.0, st.u["z"][1, 0]      # battle 1: ours overlap
    w = EP.Watch(st.B, st.N, P, "cpu")
    w(st, st.u["hp_abs"][:, 0].clone(), torch.zeros(st.B, dtype=torch.bool), area, 1)
    assert float(w.touch_s[0]) == 0.0 and float(w.touch_s[1]) == P.dt and float(w.both_s[0]) == P.dt
