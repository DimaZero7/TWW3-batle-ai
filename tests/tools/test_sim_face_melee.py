"""A facing order given in melee (contact.face_breakoff, turn.face_melee_s; build/diag_g49/turn_melee/mechanic.md): the
melee exit's window of 24 s on the enemies touched at the order, 10 s of striking nobody, an attack on that enemy ends
it, another facing restarts it, a hold opens nothing; the turn in melee takes the measured time by the angle. Needs
torch: run in the container."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
R = P.battle
WIN, IMM = R["melee_breakoff_secs"], R["melee_breakoff_total_immunity_secs"]
SPEAR, CLANRAT, SLAVE = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0", \
    "wh2_main_skv_inf_skavenslave_spearmen_0"


def army(side1, side2):
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3]} for r in rows]
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(side1)},
                                     2: {"faction": "wh2_main_skv_skaven", "units": units(side2)}}}


def fight(second=False):
    """Our spearmen (slot 0, facing east) in contact with clanrats (slot H, attacking them); second: skavenslaves far
    away (slot H + 1) to bring in later. 12 s of fighting first."""
    rows2 = [(CLANRAT, 50, 0, 270)] + ([(SLAVE, 0, 400, 180)] if second else [])
    st = scenario.build([army([(SPEAR, -50, 0, 90)], rows2)], P)
    H = st.N // 2
    front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
    touch = min(0.0, P.sim["contact"]["reach_m"])
    st.u["x"][0, 0] = -(depth[0, 0] / 2 + touch / 2)
    st.u["x"][0, H] = depth[0, H] / 2 + touch / 2
    for _ in range(24):
        battle.step(st, enemy_attacks(st), P)
    assert bool(st.u["m"][0, 0])
    return st, H


def enemy_attacks(st, mine=None):
    H = st.N // 2
    o = replay.hold(st) if mine is None else mine
    for j in range(H, st.N):
        if float(st.u["men0"][0, j]) > 0:
            o.kind[0, j], o.target[0, j] = O.ATTACK, 0
    return o


def mine(st, kind=O.HOLD, face=-1.0, target=-1):
    o = replay.hold(st)
    o.kind[0, 0], o.face[0, 0], o.target[0, 0] = kind, face, target
    o.kind[0, 1:st.N // 2] = O.KEEP
    return enemy_attacks(st, o)


def keep(st):
    o = replay.hold(st)
    o.kind[0, 0] = O.KEEP
    return enemy_attacks(st, o)


def run(st, first, seconds, then=keep, j=None):
    """[(t since the order, HP our unit took off slot j that step, HP it lost)]."""
    j = st.N // 2 if j is None else j
    out = []
    for k in range(int(round(seconds / P.dt))):
        hj, h0 = float(st.u["hp_abs"][0, j]), float(st.u["hp_abs"][0, 0])
        battle.step(st, first if k == 0 else then(st), P)
        out.append(((k + 1) * P.dt, hj - float(st.u["hp_abs"][0, j]), h0 - float(st.u["hp_abs"][0, 0])))
    return out


class TestWindow:
    def test_a_facing_in_melee_strikes_nobody_for_24_s_and_the_enemy_strikes_on(self):
        st, H = fight()
        rows = run(st, mine(st, face=0.0), WIN + 6)
        assert all(d == 0.0 for t, d, _ in rows if t < WIN - 1e-6)
        assert sum(d for t, d, _ in rows if t > WIN + 1e-6) > 0          # back to blows after the window
        assert sum(lost for t, _, lost in rows if t < WIN) > 0           # it is struck all the while
        ctl, _ = fight()
        assert sum(d for t, d, _ in run(ctl, keep(ctl), 10.0)) > 0       # (the control keeps fighting)

    def test_a_hold_without_a_facing_opens_nothing(self):
        st, H = fight()
        rows = run(st, mine(st), 6.0)
        assert sum(d for _, d, _ in rows) > 0

    def test_an_attack_on_that_enemy_ends_the_window(self):
        st, H = fight()
        run(st, mine(st, face=0.0), 5.0)
        rows = run(st, mine(st, kind=O.ATTACK, target=H), 2.0)
        assert sum(d for _, d, _ in rows) > 0

    def test_another_facing_restarts_it(self):
        st, H = fight()
        run(st, mine(st, face=0.0), 15.0)
        rows = run(st, mine(st, face=180.0), WIN + 4)
        assert all(d == 0.0 for t, d, _ in rows if t < WIN - 1e-6)
        assert sum(d for t, d, _ in rows if t > WIN + 1e-6) > 0

    def test_the_first_10_s_nobody_then_only_the_enemies_touched_at_the_order_are_spared(self):
        st, H = fight(second=True)
        B_ = H + 1
        first = mine(st, face=0.0)
        battle.step(st, first, P)
        # the second enemy (not touched at the order) comes into contact at once, on our other side
        front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
        touch = min(0.0, P.sim["contact"]["reach_m"])
        st.u["x"][0, B_] = float(st.u["x"][0, 0]) - (depth[0, 0] / 2 + depth[0, B_] / 2 + touch)
        st.u["z"][0, B_], st.u["b"][0, B_] = float(st.u["z"][0, 0]), 90.0
        on_b = run(st, keep(st), WIN - 2.0, j=B_)
        assert all(d == 0.0 for t, d, _ in on_b if t < IMM - P.dt - 1e-6)          # 10 s: nobody
        assert sum(d for t, d, _ in on_b if IMM + 1 < t) > 0                      # then the new one, yes
        assert float(st.u["fbk_s"][0, 0 * st.N + H]) > 0                           # the first one: still spared


class TestTurnInMelee:
    @pytest.mark.parametrize("turn", [90.0, 180.0])
    def test_the_facing_waits_the_measured_delay_then_turns_at_the_standing_rate(self, turn):
        t_cal = P.sim["turn"]
        delay, rate = t_cal["face_melee_delay_s"], t_cal["formation_deg_s"]
        assert delay == 8.5
        st, H = fight()
        b0 = float(st.u["b"][0, 0])
        goal = float(O.snap_face(torch.tensor([(b0 + turn) % 360]))[0])
        series = []
        o = mine(st, face=(b0 + turn) % 360)
        for k in range(int(round((delay + 6) / P.dt))):
            battle.step(st, o if k == 0 else keep(st), P)
            series.append(((k + 1) * P.dt, float(st.u["b"][0, 0])))
        assert all(b == pytest.approx(b0) for t, b in series if t < delay - 1e-6)
        start = next(t for t, b in series if b != pytest.approx(b0))
        assert start == pytest.approx(delay, abs=P.dt)
        off = abs((goal - b0 + 180) % 360 - 180)
        done = next(t for t, b in series if b == pytest.approx(goal))
        assert done == pytest.approx(start + (off / (rate * P.dt) - 1) * P.dt, abs=P.dt + 1e-6)
        # the game: 90 deg ~ 9.6 s, 180 deg ~ 10.8 s (+-2-3 s)
        assert abs(done - {90.0: 9.6, 180.0: 10.8}[turn]) <= 3.0

    def test_without_a_facing_it_drifts_as_before(self):
        st, H = fight()
        b0 = float(st.u["b"][0, 0])
        run(st, keep(st), 5.0)
        assert abs((float(st.u["b"][0, 0]) - b0 + 180) % 360 - 180) <= P.sim["contact"]["melee_turn_deg_s"] * 5 + 1e-4
