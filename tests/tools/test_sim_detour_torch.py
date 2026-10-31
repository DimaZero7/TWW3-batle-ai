"""The way round 'cannot stand' ground in the battle (tools/nn/sim/detour.py steer / plan, battle.py; build/diag_map/
pathing.md; config/nn/sim.json terrain.passable.detour): a unit ordered past a closed block walks round it corner to
corner and reaches its point, its centre never on a closed node and off the block by about half its front + pad_m; a
wider front takes a wider and longer way; a straight line clear of the growth is the old step bit for bit; the switch off
is the old stand at the edge; a battle laid turned on the map walks round as well; an attack walks round to its target; a
gap narrower than the front is passed at the narrowest growth. Needs torch (the container)."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, detour, geometry, ground, scenario, terrain  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import Params, load  # noqa: E402

P = load()
SPEAR, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrats_1"
N = 301                                     # nodes, 2 m apart: x, z in -300..300


def node(v):
    return int(np.floor((v + 300) / 2))


def block(x0=-60.0, x1=60.0, z0=40.0, z1=60.0):
    b = np.zeros((N, N), np.uint8)
    b[node(z0):node(z1), node(x0):node(x1)] = 1
    return b


def a_map(blocked):
    mid = terrain.register(np.zeros((N, N), np.float32), -300.0, -300.0, 2.0)
    ground.register_layers(mid, None, None, None, blocked)
    return mid


def army(x=30.0, z=0.0, width=None, enemy=(250.0, -250.0)):
    u = {"key": SPEAR, "x": x, "z": z, "b": 0}
    if width:
        u["width"] = width
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "ai": False, "units": [u]},
                                     2: {"faction": "wh2_main_skv_skaven", "ai": False,
                                         "units": [{"key": CLANRAT, "x": enemy[0], "z": enemy[1], "b": 180}]}}}


def orders(st, kind=O.MOVE, x=30.0, z=120.0, target=-1):
    o = O.hold(st.B, st.N)
    o.kind[:, 0], o.x[:, 0], o.z[:, 0], o.run[:, 0], o.target[:, 0] = kind, x, z, True, target
    return o


def keep(st):
    o = O.hold(st.B, st.N)
    o.kind[:] = O.KEEP
    return o


def params(on=True):
    sim = copy.deepcopy(P.sim)
    pas = copy.deepcopy(sim["terrain"]["passable"])
    pas["detour"]["on"] = on
    sim["terrain"] = {**sim["terrain"], "passable": pas}
    return Params(P.units, P.rules, sim, P.abilities, P.effects, P.spells)


@pytest.fixture(autouse=True)
def _clean():
    for d in (ground._LAYERS, ground._DEV, detour._F, detour._TAB, detour._DEV, detour._STACK):
        d.clear()
    yield
    for d in (ground._LAYERS, ground._DEV, detour._F, detour._TAB, detour._DEV, detour._STACK):
        d.clear()


def walk(st, Pq, first, steps, record=None):
    battle.step(st, first, Pq)
    out = []
    for _ in range(steps):
        battle.step(st, keep(st), Pq)
        if record:
            out.append(record(st))
    return out


def clear_at(mid, x, z):
    F = detour._map(mid, detour.cfg_of(P))[1][0]
    return float(detour.at(F, -300.0, -300.0, 2.0, x, z))


def test_a_unit_walks_round_a_closed_block_to_its_point():
    Pq = params()
    st = scenario.build([army()], Pq)
    mid = a_map(block())
    terrain.set_map(st, mid)
    xs = walk(st, Pq, orders(st), 240, lambda s: (float(s.u["x"][0, 0]), float(s.u["z"][0, 0])))
    x, z = xs[-1]
    assert abs(x - 30.0) < 3 and abs(z - 120.0) < 3                     # at its point
    assert float(st.u["pa_c"][0, 0]) >= 1 and float(st.u["blk_s"][0, 0]) == 0
    front = float(geometry.dims(st.u, float(P.sim["formation"]["spacing_m"]))[0][0, 0])
    clr = min(clear_at(mid, a, b) for a, b in xs)
    assert clr > front / 2 + 1.5 - 2.5                                    # the centre off the block by ~W/2 + pad
    assert max(a for a, _ in xs) > 60.0                                   # round the nearer (east) end
    way = sum(np.hypot(a2 - a1, b2 - b1) for (a1, b1), (a2, b2) in zip(xs[:-1], xs[1:]))
    assert 120 < way < 190                                                # longer than the straight 120 m, no wander


def _clearance_and_way(width):
    Pq = params()
    st = scenario.build([army(width=width)], Pq)
    mid = a_map(block())
    terrain.set_map(st, mid)
    xs = walk(st, Pq, orders(st), 260, lambda s: (float(s.u["x"][0, 0]), float(s.u["z"][0, 0])))
    way = sum(np.hypot(a2 - a1, b2 - b1) for (a1, b1), (a2, b2) in zip(xs[:-1], xs[1:]))
    return min(clear_at(mid, a, b) for a, b in xs), way, xs[-1]


def test_a_wider_front_takes_a_wider_and_longer_way():
    c1, w1, e1 = _clearance_and_way(16.0)
    c2, w2, e2 = _clearance_and_way(48.0)
    assert c2 > c1 + 8 and w2 > w1
    for x, z in (e1, e2):
        assert abs(x - 30.0) < 3 and abs(z - 120.0) < 3


def _state_after(on, blocked, place=None, steps=120):
    Pq = params(on)
    st = scenario.build([army(), army(-40.0, -30.0)], Pq)
    terrain.set_map(st, a_map(blocked), place=place)
    walk(st, Pq, orders(st, x=20.0, z=-90.0), steps)                  # south, away from the block
    return {k: v.clone() for k, v in st.u.items() if torch.is_tensor(v) and k not in detour.STATE + ("terr",)}, st


def test_a_clear_line_is_the_old_step_bit_for_bit():
    off, _ = _state_after(False, block())
    on, st = _state_after(True, block())
    assert float(st.u["pa_c"].sum()) == 0
    for k in off:
        assert torch.equal(off[k], on[k]), k


def test_the_switch_off_stands_at_the_edge():
    Pq = params(False)
    st = scenario.build([army()], Pq)
    terrain.set_map(st, a_map(block()))
    zs = walk(st, Pq, orders(st), 200, lambda s: float(s.u["z"][0, 0]))
    assert max(zs) < 40.0 and float(st.u["blk_s"][0, 0]) > 10


def test_a_battle_laid_turned_on_the_map_walks_round_too():
    # the battle's frame turned (a = -90 deg) and moved (px 50): its point (x, z) stands on the map at (-z + 50, x)
    # (terrain.set_map) - from the map's (50, 30) to (-70, 30) across a block at the map's x -10..10, z -30..90
    Pq = params()
    st = scenario.build([army()], Pq)
    blk = np.zeros((N, N), np.uint8)
    blk[node(-30.0):node(90.0), node(-10.0):node(10.0)] = 1
    mid = a_map(blk)
    terrain.set_map(st, mid, place=(torch.tensor([50.0]), torch.tensor([0.0]), torch.tensor([-90.0])))
    xs = walk(st, Pq, orders(st), 260, lambda s: terrain.to_map(s.u, s.u["x"][:, :1], s.u["z"][:, :1]))
    mx, mz = (float(v[0, 0]) for v in xs[-1])
    assert abs(mx - (-70.0)) < 3 and abs(mz - 30.0) < 3
    assert float(st.u["pa_c"][0, 0]) >= 1 and float(st.u["blk_s"][0, 0]) == 0
    assert min(clear_at(mid, float(a[0, 0]), float(b[0, 0])) for a, b in xs) > 10


def test_an_attack_walks_round_to_its_target():
    Pq = params()
    st = scenario.build([army(enemy=(30.0, 120.0))], Pq)
    terrain.set_map(st, a_map(block()))
    H = st.N // 2
    got = walk(st, Pq, orders(st, O.ATTACK, target=H), 240, lambda s: bool(s.u["m"][0, 0]))
    assert any(got) and float(st.u["blk_s"][0, 0]) == 0 and float(st.u["pa_c"][0, 0]) >= 1


def test_a_gap_narrower_than_the_front_is_passed_at_the_narrowest_growth():
    b = np.zeros((N, N), np.uint8)
    b[node(40.0):node(60.0), :] = 1
    b[node(40.0):node(60.0), node(-50.0):node(-38.0)] = 0                # a 12 m gap at x -50..-38
    Pq = params()
    st = scenario.build([army(width=40.0)], Pq)
    terrain.set_map(st, a_map(b))
    xs = walk(st, Pq, orders(st), 300, lambda s: (float(s.u["x"][0, 0]), float(s.u["z"][0, 0])))
    x, z = xs[-1]
    assert abs(x - 30.0) < 3 and abs(z - 120.0) < 3
    assert float(st.u["pa_nf"][0, 0]) == 0


def test_the_devices_graph_is_the_hosts():
    mid = a_map(block())
    cfg = detour.cfg_of(P)
    F, x0, z0, step = detour._map(mid, cfg)[1]
    r = 12.0
    c = detour.corners(F, x0, z0, step, r, cfg["contour_m"], cfg["eps_m"], cfg["push_m"], cfg["tol_m"])
    D1, NX1 = detour.graph(F, x0, z0, step, c, r, cfg["tol_m"], cfg["sample_m"])
    D2, NX2 = detour._graph_t(F, x0, z0, step, c, r, cfg["tol_m"], cfg["sample_m"], "cpu")
    fin = np.isfinite(D1)
    assert (fin == np.isfinite(D2)).mean() > 0.99
    both = fin & np.isfinite(D2)
    assert np.allclose(D1[both], D2[both], rtol=1e-4, atol=1e-3)


def test_ways_are_planned_on_the_battles_clock_every_plan_every_s():
    Pq = params()
    st = scenario.build([army()], Pq)
    terrain.set_map(st, a_map(block()))
    battle.step(st, keep(st), Pq)                                        # t 0 -> 0.5
    battle.step(st, orders(st), Pq)                                      # at t 0.5: not a planning step (1 s)
    assert float(st.u["pa_c"][0, 0]) == 0 and not bool(st.u["pa_ok"][0, 0])
    battle.step(st, keep(st), Pq)                                        # at t 1.0: planned
    assert float(st.u["pa_c"][0, 0]) == 1 and int(st.u["pa_n"][0, 0]) >= 1
    n = float(st.u["pa_c"][0, 0])
    for _ in range(10):                                                  # the goal stays: no new way
        battle.step(st, keep(st), Pq)
    assert float(st.u["pa_c"][0, 0]) == n


def test_a_line_surely_clear_by_its_ends_is_not_checked():
    # far from the block, a short move: (F(start) + F(goal) - length) / 2 > r - no plan, no record
    Pq = params()
    st = scenario.build([army(30.0, -150.0)], Pq)
    terrain.set_map(st, a_map(block()))
    walk(st, Pq, orders(st, x=40.0, z=-170.0), 20)
    assert not bool(st.u["pa_ok"][0, 0]) and float(st.u["pa_c"][0, 0]) == 0
