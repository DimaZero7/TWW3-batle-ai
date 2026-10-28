"""A new width of the line (Orders.width; tools/nn/sim/reform.py; config/nn/sim.json formation.reform): the limits, the
re-form's time standing / at a run / in melee / on a move against the game's (the probe meleewidth,
build/charge-probe/runs/20261028-162747; data/roster), the melee exit's window, the front rank kept, what is ignored.
Needs torch: run in the container."""
import pytest

torch = pytest.importorskip("torch")

from tests.tools.test_sim_face_melee import enemy_attacks, fight, run  # noqa: E402
from tools.nn.sim import battle, geometry, reform, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load, reform_share  # noqa: E402

P = load()
WIN = P.battle["melee_breakoff_secs"]
SP = P.sim["formation"]["spacing_m"]
RF = P.sim["formation"]["reform"]
SPEAR, CLANRAT, GENERAL = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrats_1", "wh_main_emp_cha_general_0"
H_, V_ = 1.48, 1.6          # the spearmen's template (config/nn/units.json spacing)


def alone(width=30.0, key=SPEAR):
    """Our unit (slot 0) at the origin facing +z, the order's width 30 m as the probe's; clanrats 600 m away."""
    desc = {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire",
                                         "units": [{"key": key, "x": 0.0, "z": 0.0, "b": 0.0, "width": width}]},
                                     2: {"faction": "wh2_main_skv_skaven",
                                         "units": [{"key": CLANRAT, "x": 0.0, "z": 600.0, "b": 180.0}]}}}
    return scenario.build([desc], P)


def order(st, width, run_=False, kind=O.HOLD, x=None, z=None, target=-1):
    o = replay.hold(st)
    o.kind[0, 0], o.width[0, 0], o.run[0, 0], o.target[0, 0] = kind, width, run_, target
    o.x[0, 0] = float(st.u["x"][0, 0]) if x is None else x
    o.z[0, 0] = float(st.u["z"][0, 0]) if z is None else z
    return o


def keep(st):
    o = replay.hold(st)
    o.kind[0, 0] = O.KEEP
    return o


def ready_after(st, first, then=keep, limit_s=60.0):
    """Seconds from the order until the re-form is over (wnew back to -1); the widths a step."""
    ws = []
    for k in range(int(limit_s / P.dt)):
        battle.step(st, first if k == 0 else then(st), P)
        ws.append(float(st.u["width"][0, 0]))
        if float(st.u["wnew"][0, 0]) < 0:
            return (k + 1) * P.dt, ws
    return None, ws


def files(w, h=H_):
    return int(w / h + 1e-4)


class TestLimits:
    def test_the_narrowest_and_the_widest(self):
        st = alone()
        lo, hi = reform.limits(st.u, RF, SP)
        assert float(lo[0, 0]) == pytest.approx(5 * H_)            # 120 men x 0.04 = 4.8 -> 5 files (mechanic.md 3)
        assert float(hi[0, 0]) == pytest.approx(120 * H_)          # one rank
        assert float(reform.snap(st.u, torch.full_like(lo, 3.0), RF, SP)[0, 0]) == pytest.approx(5 * H_)
        assert float(reform.snap(st.u, torch.full_like(lo, 15.0), RF, SP)[0, 0]) == pytest.approx(10 * H_)

    def test_the_share_by_class(self):
        assert reform_share("inf_mel", RF) == 0.04 and reform_share("cav_shk", RF) == 0.10
        assert reform_share("mon_lrg", RF) == 0.10 and reform_share("art_fld", RF) == 0.0
        assert reform_share("inf_mel", dict(RF, on=0)) == 0.0


class TestTime:
    def test_standing_at_a_walk_widening_takes_4_s_more(self):
        st = alone()
        T, ws = ready_after(st, order(st, 60.0))
        want = 4.0 + (files(60) - files(30)) * H_ / 2 / 1.5
        assert T == pytest.approx(want, abs=P.dt)
        assert 13.0 <= T <= 16.0                                     # the roster's 30 -> 60 m: 13-16 s in the game
        assert float(st.u["width"][0, 0]) == pytest.approx(files(60) * H_)
        assert ws[len(ws) // 2] == pytest.approx((ws[0] + ws[-1]) / 2, rel=0.1)    # evenly over T

    def test_standing_at_a_run_narrowing_and_widening(self):
        st = alone()
        T, _ = ready_after(st, order(st, 10.0, run_=True))
        assert T == pytest.approx((120 // files(10) + (120 % files(10) > 0) - 1) * V_ / 3.0, abs=P.dt)
        assert T == pytest.approx(11.3, rel=0.15)                    # the probe: 30 -> 10 m at a run 11.3 s
        st = alone()
        T, _ = ready_after(st, order(st, 80.0, run_=True))
        assert T == pytest.approx(7.8, rel=0.15)                     # 30 -> 80 m at a run 7.8 s (no extra 4 s)

    def test_on_a_move_the_narrow_line_is_ready_at_the_arrival_the_wide_one_on_the_way(self):
        st = alone()
        T, _ = ready_after(st, order(st, 10.0, run_=True, kind=O.MOVE, z=-80.0), limit_s=80.0)
        assert T == pytest.approx(80.0 / 3.0, rel=0.1)               # the way's time (the probe: ready at 28.8 s)
        st = alone()
        T, _ = ready_after(st, order(st, 80.0, run_=True, kind=O.MOVE, z=-80.0), limit_s=80.0)
        assert T < 80.0 / 3.0 / 2                                    # widened on the way (the probe: 10.8 s)


class TestMelee:
    def test_in_melee_it_re_forms_at_a_walk_and_strikes_nobody_for_24_s(self):
        st, H = fight()
        o = enemy_attacks(st, order(st, 15.0))
        rows = run(st, o, 13.0)
        assert float(st.u["wnew"][0, 0]) < 0                          # ready within 13 s (the probe 11.1 s)
        assert float(st.u["width"][0, 0]) == pytest.approx(files(15) * H_)
        assert bool(st.u["m"][0, 0])                                  # still in contact
        rows += [(t + 13.0, d, lost) for t, d, lost in run(st, enemy_attacks(st, keep(st)), WIN + 6 - 13.0)]
        assert all(d == 0.0 for t, d, _ in rows if t < WIN - 1e-6)
        assert sum(d for t, d, _ in rows if t > WIN + 1e-6) > 0      # blows back after the window

    def test_the_time_in_melee_matches_the_probe(self):
        st, H = fight()
        T, _ = ready_after(st, enemy_attacks(st, order(st, 15.0)), then=lambda s: enemy_attacks(s, keep(s)))
        assert T == pytest.approx(11.1, rel=0.15)                     # spearmen 30 -> 15 m in melee: 11.1 s
        st, H = fight()
        T, _ = ready_after(st, enemy_attacks(st, order(st, 60.0)), then=lambda s: enemy_attacks(s, keep(s)))
        assert T == pytest.approx(8.2, rel=0.25)                      # 30 -> 60 m: 8.2 s (no extra 4 s in melee)

    def test_the_same_width_opens_nothing(self):
        st, H = fight()
        o = enemy_attacks(st, order(st, float(st.u["width"][0, 0])))
        rows = run(st, o, 6.0)
        assert sum(d for _, d, _ in rows) > 0


class TestRules:
    def test_a_held_unit_keeps_its_front_rank(self):
        st = alone()
        _, d0 = geometry.dims(st.u, SP)
        front0 = float(st.u["z"][0, 0] + d0[0, 0] / 2)
        ready_after(st, order(st, 10.0, run_=True), then=lambda s: order(s, 10.0, run_=True))
        _, d1 = geometry.dims(st.u, SP)
        assert float(d1[0, 0]) > float(d0[0, 0]) + 10
        assert float(st.u["z"][0, 0] + d1[0, 0] / 2) == pytest.approx(front0, abs=0.3)

    def test_the_order_re_given_each_step_does_not_restart(self):
        st = alone()
        T, _ = ready_after(st, order(st, 60.0), then=lambda s: order(s, 60.0))
        assert T == pytest.approx(4.0 + (files(60) - files(30)) * H_ / 2 / 1.5, abs=P.dt)

    def test_no_width_changes_nothing_attack_and_a_lone_man_ignore_it(self):
        st = alone()
        w0 = float(st.u["width"][0, 0])
        for _ in range(10):
            battle.step(st, order(st, -1.0), P)
        assert float(st.u["width"][0, 0]) == w0 and float(st.u["wnew"][0, 0]) < 0
        st = alone()
        battle.step(st, order(st, 60.0, kind=O.ATTACK, target=st.N // 2), P)
        assert float(st.u["wnew"][0, 0]) < 0
        st = alone(key=GENERAL)
        w0 = float(st.u["width"][0, 0])
        battle.step(st, order(st, 60.0), P)
        assert float(st.u["wnew"][0, 0]) < 0 and float(st.u["width"][0, 0]) == w0

    def test_a_rout_ends_the_re_form(self):
        st = alone()
        battle.step(st, order(st, 60.0), P)
        st.u["r"][0, 0] = True
        battle.step(st, keep(st), P)
        assert float(st.u["wnew"][0, 0]) < 0

    def test_the_orders_check_a_width(self):
        o = O.hold(1, 2)
        o.width[0, 0] = float("nan")
        with pytest.raises(ValueError):
            O.check(o, 2)
