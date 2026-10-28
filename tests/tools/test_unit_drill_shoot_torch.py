"""Drill U5 'shooters - fire at one target' (tools/nn/unit/drill_shoot.py): the shooters' roster, the position and the
fire sector, the targets' mask (in the sector and in range, wherever), the order's sector numbers, the shooter's leash
(no walking to a target beyond range), old checkpoints padded. Needs torch: run in the container."""
import math

import pytest

torch = pytest.importorskip("torch")

from tests.tools.test_unit_view_net_torch import battle, card  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim import replay  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import decode as Dc  # noqa: E402
from tools.nn.unit import drill_shoot as DS  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402
from tools.nn.unit import view as V  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
ARCHER, CLANRAT = "wh2_dlc13_emp_inf_archers_0", "wh2_main_skv_inf_clanrat_spearmen_0"


def test_the_shooters_roster_leaves_the_hybrids_out():
    S = DS.shooters()
    keys = S["emp"] + S["skv"]
    for k in ("wh_dlc04_emp_inf_free_company_militia_0", "wh3_dlc25_emp_inf_nuln_ironsides", "wh2_dlc13_emp_inf_huntsmen_0"):
        assert k not in keys
    assert ARCHER in keys and "wh2_dlc12_skv_inf_warplock_jezzails_0" in keys and S["emp"] and S["skv"]


def test_the_position_the_sector_and_the_enemys_start_and_goal():
    sp = DS.specs(40, torch.Generator().manual_seed(2))
    st, own, area, goal = DS.build(sp, P, "cpu")
    H = st.N // 2
    for b, s in enumerate(sp):
        a0 = {k: v[b, 0] for k, v in area.items() if k in ("x", "z", "b", "front", "depth")}
        assert float(Dc.signed_dist(st.u["x"][b, 0], st.u["z"][b, 0], a0)) > 0
        dx, dz = float(st.u["x"][b, H]), float(st.u["z"][b, H])
        off = abs((math.degrees(math.atan2(dx, dz)) - s["sb"] + 180) % 360 - 180)
        assert off <= s["sw"] / 2 and DS.START_M[0] <= math.hypot(dx, dz) <= DS.START_M[1]
        gd = math.hypot(float(goal["x"][b, 0]), float(goal["z"][b, 0]))
        assert (gd < 40) if s["mode"] == "at" else (DS.PAST_M[0] - 1e-3 <= gd <= DS.PAST_M[1] + 1e-3)
    assert {s["mode"] for s in sp} == set(DS.MODES) and len(DS.exam_specs()) == 240


def _sector(st, sb, sw, side=100.0):
    f = lambda v: torch.full((st.B, st.N), float(v))       # noqa: E731
    return {"x": f(0), "z": f(0), "b": f(0), "front": f(side), "depth": f(side), "sb": f(sb), "sw": f(sw)}


def test_the_targets_are_in_the_sector_and_in_range_wherever():
    st = battle([(ARCHER, 0, 0, 0)], [(CLANRAT, 0, 120, 180)])               # 120 m ahead: beyond the 100 m area
    rng = float(st.u["range"][0, 0])
    who = torch.zeros((1, st.N), dtype=torch.bool)
    who[0, 0] = True
    v = V.build(V.raw(st, SP), V.Memory(1, st.N).hist, card(st), who, area=_sector(st, 0.0, 90.0))
    assert rng >= 120 and bool(v["target_ok"][0, 1])                           # in the sector and range: yes
    v = V.build(V.raw(st, SP), V.Memory(1, st.N).hist, card(st), who, area=_sector(st, 180.0, 90.0))
    assert not bool(v["target_ok"][0, 1])                                       # out of the sector
    st.u["z"][0, st.N // 2] = rng + 20.0
    v = V.build(V.raw(st, SP), V.Memory(1, st.N).hist, card(st), who, area=_sector(st, 0.0, 90.0))
    assert not bool(v["target_ok"][0, 1])                                       # out of range
    ar = _sector(st, 0.0, 0.0)                                                   # no sector: the area's rule
    st.u["z"][0, st.N // 2] = 120.0
    v = V.build(V.raw(st, SP), V.Memory(1, st.N).hist, card(st), who, area=ar)
    assert not bool(v["target_ok"][0, 1])


def test_the_order_carries_the_sector():
    st = battle([(ARCHER, 0, 0, 30)], [(CLANRAT, 0, 300, 180)])
    now = V.raw(st, SP)
    ar = {k: v[:, 0] for k, v in _sector(st, 120.0, 90.0).items()}
    o = dict(zip(Nt.ORDER, Nt.order_of(now[:, 0], ar, torch.tensor([Nt.PACE_ANY]), torch.tensor([True]))[0].tolist()))
    assert o["has_sector"] == 1.0 and o["sector_cos"] == pytest.approx(0.0, abs=1e-5)
    assert o["sector_sin"] == pytest.approx(1.0) and o["sector_width"] == pytest.approx(0.5)
    ar0 = {k: v[:, 0] for k, v in _sector(st, 0.0, 0.0).items()}
    o0 = dict(zip(Nt.ORDER, Nt.order_of(now[:, 0], ar0, torch.tensor([0]), torch.tensor([True]))[0].tolist()))
    assert o0["has_sector"] == 0.0 and o0["sector_cos"] == 0.0


def test_a_shooter_does_not_walk_to_a_target_beyond_range():
    st = battle([(ARCHER, 0, 0, 0)], [(CLANRAT, 0, 300, 180)])
    H = st.N // 2
    own = torch.zeros((1, st.N), dtype=torch.bool)
    own[0, 0] = True
    leash = Dc.Leash(1, st.N)
    o = replay.hold(st)
    o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
    out = leash.apply(st, o, _sector(st, 0.0, 90.0), own)
    assert int(out.kind[0, 0]) == O.HOLD and float(out.face[0, 0]) == pytest.approx(0.0, abs=1e-3)
    assert bool(leash.on[0, 0])
    st.u["z"][0, H] = 100.0                                                    # within range now: it shoots
    out = leash.apply(st, o, _sector(st, 0.0, 90.0), own)
    assert int(out.kind[0, 0]) == O.ATTACK and not bool(leash.on[0, 0])


def test_an_old_checkpoint_is_padded_with_zero_columns(tmp_path):
    net, critic = Nt.UnitNet(), Nt.Critic()
    opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()))
    nsd, csd = net.state_dict(), critic.state_dict()
    k = "trunk.core.net.0.weight"
    cut = len(Nt.ORDER) - Nt.ORDER_BEFORE_SECTOR
    for sd in (nsd, csd):
        w = sd[k]
        sd[k] = torch.cat([w[:, :w.shape[1] - 1 - cut], w[:, -1:]], 1)      # as before the sector's fields
    torch.save({"kind": T.KIND, "net": nsd, "critic": csd, "opt": opt.state_dict(), "cfg": {}, "updates": 3},
               tmp_path / "old.pt")
    n2, c2, ck = T.restore(tmp_path / "old.pt")
    assert ck["padded"]
    w2 = n2.trunk.core.net[0].weight
    assert float(w2[:, -1 - cut:-1].abs().sum()) == 0.0 and torch.equal(w2[:, -1], net.trunk.core.net[0].weight[:, -1])


def test_a_short_rollout_of_the_drill():
    torch.manual_seed(0)
    cfg = {"battles": 4, "seconds": 30, "cell_m": 8.0, "bounds": None}
    sp = DS.specs(4, torch.Generator().manual_seed(4))
    batch, stats = T.rollout(Nt.UnitNet(), Nt.Critic(), cfg, P, "cpu", T.Duty("cpu"), T.shoot_setup(cfg, P, "cpu", sp=sp))
    assert stats["bans_broken"] == 0 and stats["nan"] == 0
