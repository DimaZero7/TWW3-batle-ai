"""The unit's view (tools/nn/unit/view.py: zone, rows, memory) and the unit network (tools/nn/unit/net.py: weights by
module, the hard bans, an enemy outside the zone changes nothing). Needs torch: run in the container."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import passport  # noqa: E402
from tools.nn.sim import scenario  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import view as V  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
SPEAR, SLAVE, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0", \
    "wh2_main_skv_inf_clanrat_spearmen_0"
ARCHER = "wh2_dlc13_emp_inf_archers_0"


def battle(own, enemies):
    """own / enemies: [(key, x, z, b)]; the first own unit is the viewer."""
    def units(rows):
        return [{"key": k, "x": x, "z": z, "b": b} for k, x, z, b in rows]
    return scenario.build([{"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(own)},
                                                     2: {"faction": "wh2_main_skv_skaven", "units": units(enemies)}}}], P)


def card(st):
    return torch.from_numpy(np.stack([passport.table(row) for row in st.keys]))


def view_of(st, hist=None, who=None, area=None, **kw):
    now = V.raw(st, SP)
    if hist is None:
        hist = V.Memory(st.B, st.N).hist
    if who is None:
        who = torch.zeros((st.B, st.N), dtype=torch.bool)
        who[:, 0] = True
    return V.build(now, hist, card(st), who, area=area, **kw)


def turn_battle(st, dx, dz, deg):
    """The whole battle moved by (dx, dz) and turned by deg about the origin."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, z = st.u["x"].clone(), st.u["z"].clone()
    st.u["x"] = x * c + z * s + dx
    st.u["z"] = -x * s + z * c + dz
    vx, vz = st.u["vx"].clone(), st.u["vz"].clone()
    st.u["vx"], st.u["vz"] = vx * c + vz * s, -vx * s + vz * c
    st.u["b"] = (st.u["b"] + deg) % 360
    return st


def three():
    """Me (spearmen at the origin facing north), a clanrat unit 60 m ahead facing me, an own archer unit beside me,
    skavenslaves 500 m off (outside the zone)."""
    return battle([(SPEAR, 0, 0, 0), (ARCHER, 45, 0, 0)], [(CLANRAT, 0, 60, 180), (SLAVE, 500, 300, 270)])


class TestZoneAndRows:
    def test_the_zone_holds_who_can_reach_me_and_rows_are_raw_and_relative(self):
        st = three()
        H = st.N // 2
        v = view_of(st)
        slots = v["slot"][0].tolist()
        assert slots[:3] == [0, H, 1] and all(s == -1 for s in slots[3:])          # me, the enemy, my own; no far one
        assert v["whose"][0, :3].tolist() == [V.SELF, V.ENEMY, V.OWN]
        e = dict(zip(V.ROW, v["rows"][0, 1].tolist()))
        assert e["right"] == pytest.approx(0, abs=1e-5) and e["forward"] == pytest.approx(0.6, abs=1e-5)
        assert e["front_cos"] == pytest.approx(-1, abs=1e-5) and e["enemy"] == 1.0
        assert e["front"] == pytest.approx(32.0 / 50, abs=1e-3) and e["men"] == pytest.approx(1.0)
        me = dict(zip(V.ROW, v["rows"][0, 0].tolist()))
        assert me["self"] == 1.0 and me["right"] == 0.0 and me["front_cos"] == pytest.approx(1.0)
        assert float(v["rows"][0, 3:].abs().sum()) == 0 and not v["mask"][0, 3:].any()
        assert float(v["map"][0, Z.L["enemy"]].sum()) > 0 and float(v["map"][0, Z.L["own"]].sum()) > 0

    def test_moving_and_turning_the_whole_battle_changes_nothing(self):
        a = view_of(three())
        b = view_of(turn_battle(three(), 130.0, -70.0, 63.0))
        for k in ("rows", "mem", "map"):
            assert torch.allclose(a[k], b[k], atol=2e-4), k
        assert torch.equal(a["slot"], b["slot"])

    def test_an_unseen_enemy_is_not_in_the_view(self):
        st = three()
        H = st.N // 2
        st.u["vis"][0, H] = False
        v = view_of(st)
        assert H not in v["slot"][0].tolist()
        assert float(v["map"][0, Z.L["enemy"]].abs().sum()) == 0

    def test_no_time_of_the_battle_anywhere(self):
        for names in (V.RAW, V.ROW, Nt.ORDER):
            assert not any(n in ("t", "time", "time_left", "limit", "clock") for n in names)

    def test_the_zone_reaches_by_range_and_run_and_by_any_part_of_the_formation(self):
        # run 3 + 3 m/s x 15 s = 90 m: spearmen vs clanrats edge to edge 85 m in, 95 m out
        st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 0, 180)])
        H = st.N // 2
        reach = (float(st.u["run"][0, 0]) + float(st.u["run"][0, H])) * V.REACH_S
        now = V.raw(st, SP)
        half = float(now[0, 0, V.I["depth"]]) / 2 + float(now[0, H, V.I["depth"]]) / 2
        for d, inside in ((reach - 5, True), (reach + 5, False)):
            st.u["z"][0, H] = half + d
            assert (H in view_of(st)["slot"][0].tolist()) == inside
        # side by side, the centres far, the edges near: in
        # (both fronts along x: 14.8 + 16 m half fronts, the centres reach + 25 m apart, the edges reach - 5.8)
        st.u["z"][0, H], st.u["x"][0, H], st.u["b"][0, H] = 0.0, reach + 25.0, 0.0
        assert H in view_of(st)["slot"][0].tolist()
        st.u["x"][0, H] = reach + 36.0
        assert H not in view_of(st)["slot"][0].tolist()

    def test_at_most_k_rows_the_nearest_enemies_first(self):
        enemies = [(SLAVE, 40.0 * (k % 5) - 80, 40 + 18.0 * (k // 5), 180) for k in range(18)]
        own = [(SPEAR, 0, 0, 0), (ARCHER, -40, -20, 0)]
        st = battle(own, enemies)
        v = view_of(st)
        assert int(v["mask"][0].sum()) == V.K
        whose = v["whose"][0].tolist()
        assert whose[0] == V.SELF and whose[1:] == sorted(whose[1:], key=lambda w: (w != V.ENEMY))
        gaps = V.zone(V.raw(st, SP), torch.tensor([0]), torch.tensor([0]))[1][0]
        picked = [s for s, w in zip(v["slot"][0].tolist(), whose) if w == V.ENEMY]
        assert max(float(gaps[s]) for s in picked) <= min(float(gaps[s]) for s in range(st.N // 2, st.N)
                                                          if s not in picked and float(st.u["side"][0, s]) > 0)


class TestMemory:
    def test_snapshots_are_put_into_my_frame_of_now(self):
        st = three()
        H = st.N // 2
        mem = V.Memory(st.B, st.N)
        for k in range(V.T):
            st.u["z"][0, H] = 60.0 - 2.0 * k                     # the enemy walks at me, 2 m a second
            mem.push(V.raw(st, SP))
        st.u["b"][0, 0] = 90.0                                   # now I face east
        v = view_of(st, mem.hist)
        m = v["mem"][0, 1]                                       # the enemy's snapshots, newest first
        assert v["mem_mask"][0, 1].all()
        fwd = m[:, V.ROW.index("forward")]
        right = m[:, V.ROW.index("right")]
        assert fwd.abs().max() < 1e-5                            # facing east, the enemy north of me: on my left
        assert (-right).tolist() == pytest.approx([(60.0 - 2.0 * (V.T - 1 - t)) / 100 for t in range(V.T)], abs=1e-5)
        assert v["mem"][0, 0].shape == (V.T, len(V.ROW))         # my own row has its memory too
        assert v["mem_mask"][0, 0].all()

    def test_an_empty_or_unseen_snapshot_is_empty(self):
        st = three()
        H = st.N // 2
        mem = V.Memory(st.B, st.N)
        st.u["vis"][0, H] = False
        mem.push(V.raw(st, SP))
        st.u["vis"][0, H] = True
        v = view_of(st, mem.hist)
        assert not v["mem_mask"][0, 1].any() and float(v["mem"][0, 1].abs().sum()) == 0


def net_inputs(st, area=None, pace=Nt.PACE_ANY, everyone=False, who=None):
    v = view_of(st, area=area, everyone=everyone, who=who)
    R = v["rows"].shape[0]
    return v, Nt.order_vec(R, pace=pace, has_area=area is not None), torch.full((R,), 8.0)


class TestNet:
    def test_weights_by_module(self):
        net = Nt.UnitNet()
        w = Nt.report(net)
        assert 250_000 <= w["total"] <= 360_000
        assert 25_000 <= w["map"] <= 35_000 and 15_000 <= w["rows"] <= 22_000 and 25_000 <= w["memory"] <= 40_000
        assert 60_000 <= w["among"] <= 80_000 and 60_000 <= w["me"] <= 80_000 and 40_000 <= w["core"] <= 50_000
        assert w["face"] == 128 * 64 + 64 and w["pace"] == 258
        assert sum(v for k, v in w.items() if k != "total") == w["total"]
        assert 250_000 <= Nt.weights(Nt.Critic()) <= 360_000

    def test_the_bans_are_never_broken(self):
        torch.manual_seed(0)
        st = battle([(SPEAR, 0, 0, 0), (SPEAR, 40, 0, 0)], [(CLANRAT, 0, 50, 180), (CLANRAT, 70, 70, 200)])
        H = st.N // 2
        who = torch.zeros((1, st.N), dtype=torch.bool)
        who[0, :2] = True
        area = {"x": torch.full((1, st.N), 10.0), "z": torch.full((1, st.N), 10.0), "b": torch.full((1, st.N), 30.0),
                "front": torch.full((1, st.N), 60.0), "depth": torch.full((1, st.N), 50.0)}
        net = Nt.UnitNet()
        for pace in (Nt.PACE_SLOW, Nt.PACE_FAST, Nt.PACE_ANY):
            v, o, cm = net_inputs(st, area, pace, who=who)
            lg = net(v, o, cm)
            for k in range(300):
                a = Nt.sample(lg)
                ok = v["cells_ok"].flatten(1).gather(1, a["where"][:, None])[:, 0]
                assert ok.all()
                t_ok = torch.cat([v["target_ok"], torch.ones((2, 1), dtype=torch.bool)], 1)
                assert t_ok.gather(1, a["target"][:, None]).all()
                if pace == Nt.PACE_SLOW:
                    assert (a["pace"] == Nt.WALK).all()
                if pace == Nt.PACE_FAST:
                    assert (a["pace"] == Nt.RUN).all()
            probs = torch.softmax(lg["where"].detach(), 1)
            assert float(probs[~v["cells_ok"].flatten(1)].sum()) == 0.0
            lp, ent = Nt.log_prob(lg, a)
            assert torch.isfinite(lp).all() and torch.isfinite(ent).all()
        # the enemy at (70, 70) is outside the area: never a target; the cells: centres inside the turned rectangle
        assert not v["target_ok"][0, (v["slot"][0] == H + 1).nonzero()[0, 0]]
        assert v["cells_ok"][0].sum() > 0 and not v["cells_ok"][0].all()

    def test_an_enemy_outside_the_zone_changes_nothing_bit_for_bit(self):
        torch.manual_seed(1)
        net = Nt.UnitNet().eval()
        crit = Nt.Critic().eval()
        H = None
        outs = []
        for far in ((500.0, 300.0, 270.0), (-400.0, 450.0, 10.0), (300.0, -500.0, 135.0)):
            st = three()
            H = st.N // 2
            st.u["x"][0, H + 1], st.u["z"][0, H + 1], st.u["b"][0, H + 1] = far
            st.u["men"][0, H + 1] = 100.0 + far[2]
            v, o, cm = net_inputs(st)
            with torch.no_grad():
                outs.append(net(v, o, cm))
        for k in Nt.HEADS:
            assert torch.equal(outs[0][k], outs[1][k]) and torch.equal(outs[0][k], outs[2][k]), k
        st = three()
        st.u["x"][0, H] = 10.0                                    # the near enemy moves: the decision changes
        v, o, cm = net_inputs(st)
        with torch.no_grad():
            assert not torch.equal(net(v, o, cm)["face"], outs[0]["face"])
            # the critic sees the whole skirmish: the far enemy is in its view
            vc, oc, cc = net_inputs(st, everyone=True)
            assert H + 1 in vc["slot"][0].tolist() and crit(vc, oc, cc).shape == (1,)
