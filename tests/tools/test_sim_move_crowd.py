"""contact.move_crowd (config/nn/sim.json): a formation on the move through its own standing formation is slowed by their
men (the shares of morale.rout_crowd), and goes straight through (R-work notes/p4/way.md: no unit in the search for the way)."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SPEAR = "wh_main_emp_inf_spearmen_0"


def units(rows):
    return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3]} for r in rows]


def run_through(params, friend=True, walk=False):
    """A unit runs along x from -100; a friend (or none) stands across its way at x 0. -> (the slowest and the free speed,
    the most off the line, the friend's shift)."""
    own = [(SPEAR, -100.0, 0.0, 90.0)] + ([(SPEAR, 0.0, 0.0, 90.0)] if friend else []) + [(SPEAR, -300.0, 700.0, 90.0)]
    a = {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(own)},
                                  2: {"faction": "wh2_main_skv_skaven", "units": units([(SPEAR, 300.0, 700.0, 270.0)])}}}
    st = scenario.build([a], params)
    u = st.u
    f0 = float(u["x"][0, 1])
    vs, off = [], 0.0
    for _ in range(int(round(50.0 / params.dt))):
        o = replay.hold(st)
        o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.MOVE, 100.0, 0.0, not walk
        battle.step(st, o, params)
        x = float(u["x"][0, 0])
        if -60 < x < 60:
            vs.append(math.hypot(float(u["vx"][0, 0]), float(u["vz"][0, 0])))
            off = max(off, abs(float(u["z"][0, 0])))
    return min(vs), max(vs), off, abs(float(u["x"][0, 1]) - f0)


def test_it_goes_straight_through_at_the_shared_speed():
    share = P.sim["morale"]["rout_crowd"]["own"][1]
    slow, free, off, _ = run_through(P)
    assert slow == pytest.approx(free * share, rel=0.08)
    assert off < 3.0


def test_with_no_friend_and_with_the_rule_off_it_keeps_its_speed():
    slow, free, _, _ = run_through(P, friend=False)
    assert slow == pytest.approx(free, rel=0.02)
    slow, free, _, _ = run_through(P.with_cal("contact", move_crowd=0))
    assert slow == pytest.approx(free, rel=0.02)
