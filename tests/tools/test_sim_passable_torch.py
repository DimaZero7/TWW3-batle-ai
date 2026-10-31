"""'Cannot stand' (tools/nn/sim/ground.py blocked, battle.py; build/diag_map/scan.md; config/nn/sim.json
terrain.passable): the layer read from a map folder's passable.npz on the heights' grid, a move order whose point is on a
closed node refused whole (the order in force goes on), a unit's centre not stepping onto a closed node (it stops at the
edge), a unit on one walking off, the switch off and an empty layer giving the old numbers bit for bit. Needs torch
(the container)."""
import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, ground, scenario, terrain  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import Params, load  # noqa: E402

P = load()
SPEAR, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrats_1"
N = 301                                     # nodes, 2 m apart: x, z in -300..300


def node(v):
    return int(np.floor((v + 300) / 2))


def wall(z0=40.0, z1=60.0):
    """blocked [N, N]: the band z0 <= z < z1 across the whole map closed."""
    b = np.zeros((N, N), np.uint8)
    b[node(z0):node(z1), :] = 1
    return b


def a_map(blocked=None):
    mid = terrain.register(np.zeros((N, N), np.float32), -300.0, -300.0, 2.0)
    ground.register_layers(mid, None, None, None, blocked)
    return mid


def lone(x=0.0, z=0.0):
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": [{"key": SPEAR, "x": x, "z": z, "b": 0}]},
                                     2: {"faction": "wh2_main_skv_skaven",
                                         "units": [{"key": CLANRAT, "x": 250.0, "z": -250.0, "b": 0}]}}}


def move(st, x, z, kind=O.MOVE):
    o = O.hold(st.B, st.N)
    o.kind[:, 0], o.x[:, 0], o.z[:, 0], o.run[:, 0] = kind, x, z, True
    return o


def keep(st):
    o = O.hold(st.B, st.N)
    o.kind[:] = O.KEEP
    return o


@pytest.fixture(autouse=True)
def _clean():
    ground._LAYERS.clear()
    ground._DEV.clear()
    yield
    ground._LAYERS.clear()
    ground._DEV.clear()


def test_the_layer_reads_from_the_map_folder_on_the_heights_grid(tmp_path):
    H = np.zeros((11, 11), np.float32)
    b = np.zeros((11, 11), np.uint8)
    b[3, 4] = 1
    np.savez_compressed(tmp_path / "heights.npz", height=H, ground=np.zeros(H.shape, np.uint8), x0=np.float64(-5),
                        z0=np.float64(-5), step=np.float64(1.0))
    (tmp_path / "meta.json").write_text(json.dumps({"ground_names": ["grass"]}), encoding="utf-8")
    np.savez_compressed(tmp_path / "passable.npz", blocked=b, x0=np.float64(-5), z0=np.float64(-5), step=np.float64(1.0))
    mid = terrain.load_npz(tmp_path / "heights.npz")
    assert ground.has_passable(mid)
    st = scenario.build([lone()], P)
    terrain.set_map(st, mid)
    # node (ix 4, iz 3) = x -1..0, z -2..-1
    got = ground.blocked(st.u, torch.tensor([[-0.5, 0.5, -0.5]]), torch.tensor([[-1.5, -1.5, -0.5]]))
    assert got.tolist() == [[True, False, False]]
    assert "blk_s" in st.u and "blk_o" in st.u
    # another grid: refused
    np.savez_compressed(tmp_path / "passable.npz", blocked=b, x0=np.float64(-4), z0=np.float64(-5), step=np.float64(1.0))
    with pytest.raises(AssertionError):
        terrain.load_npz(tmp_path / "heights.npz")


def test_a_move_order_to_a_closed_point_is_refused_whole():
    st = scenario.build([lone()], P)
    terrain.set_map(st, a_map(wall()))
    x0, z0 = float(st.u["x"][0, 0]), float(st.u["z"][0, 0])
    for _ in range(20):
        battle.step(st, move(st, 0.0, 50.0), P)            # the point inside the band
    assert int(st.u["order_kind"][0, 0]) == O.HOLD                 # the order in force (none: hold) goes on
    assert float(st.u["x"][0, 0]) == pytest.approx(x0) and float(st.u["z"][0, 0]) == pytest.approx(z0)
    assert float(st.u["oz"][0, 0]) == pytest.approx(z0)            # its ordered position stays
    assert float(st.u["blk_o"][0, 0]) == 20 and float(st.u["blk_s"][0, 0]) == 0
    # a withdraw too; a moving unit keeps its old order
    battle.step(st, move(st, 0.0, -30.0), P)
    for _ in range(4):
        battle.step(st, move(st, 0.0, 45.0, O.WITHDRAW), P)
    assert int(st.u["order_kind"][0, 0]) == O.MOVE and float(st.u["oz"][0, 0]) == pytest.approx(-30.0)
    assert float(st.u["z"][0, 0]) < z0                              # on its way to -30


def test_a_unit_stops_at_the_edge_of_a_closed_band():
    st = scenario.build([lone()], P)
    terrain.set_map(st, a_map(wall()))
    battle.step(st, move(st, 0.0, 120.0), P)                        # a free point beyond the band
    zs = []
    for _ in range(120):
        battle.step(st, keep(st), P)
        zs.append(float(st.u["z"][0, 0]))
    assert int(st.u["order_kind"][0, 0]) == O.MOVE                  # the order was taken
    assert max(zs) < 40.0 and max(zs) > 36.0                        # it walked up to the band and stays short of it
    assert zs[-1] == zs[-2] and float(st.u["blk_s"][0, 0]) > 10
    assert not bool(st.u["mv"][0, 0])                               # it stands


def test_a_unit_on_a_closed_node_walks_off():
    st = scenario.build([lone(0.0, 50.0)], P)                       # inside the band
    terrain.set_map(st, a_map(wall()))
    battle.step(st, move(st, 0.0, 0.0), P)
    for _ in range(60):
        battle.step(st, keep(st), P)
    assert float(st.u["z"][0, 0]) < 40.0


def _params(on):
    sim = copy.deepcopy(P.sim)
    sim["terrain"] = {**sim["terrain"], "passable": {**sim["terrain"]["passable"], "on": on}}
    return Params(P.units, P.rules, sim, P.abilities, P.effects, P.spells)


def _run(blocked, on=True, layer=True):
    Pq = _params(on)
    st = scenario.build([lone(), lone(20.0, -20.0)], Pq)
    mid = terrain.register(np.zeros((N, N), np.float32), -300.0, -300.0, 2.0)
    if layer:
        ground.register_layers(mid, None, None, None, blocked)
    terrain.set_map(st, mid)
    battle.step(st, move(st, 0.0, 120.0), Pq)
    for _ in range(100):
        battle.step(st, keep(st), Pq)
    return {k: v.clone() for k, v in st.u.items() if torch.is_tensor(v) and k not in ("blk_s", "blk_o", "terr")}


def test_the_switch_off_and_an_empty_layer_are_the_old_numbers_bit_for_bit():
    old = _run(None, layer=False)                                    # no layer at all
    empty = _run(np.zeros((N, N), np.uint8))                         # a layer with nothing closed
    off = _run(wall(), on=False)                                     # the band, the rules off
    on = _run(wall())
    for k in old:
        assert torch.equal(old[k], empty[k]), k
        assert torch.equal(old[k], off[k]), k
    assert not torch.equal(old["z"], on["z"])                        # (the band stops it when on)
