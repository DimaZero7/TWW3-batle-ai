"""The unit drills on a real map (tools/nn/unit/realmap.py; terrain.set_map place / to_map): a skirmish keeps its own
coordinates and reads the map's layers through its frame, the places keep the skirmish on the map and its units on clear
ground, the exam's places are fixed, the terrain readings. Needs torch (the container)."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import geometry, scenario, terrain  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import realmap  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402
from tools.nn.unit import view as V  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

P = load()
SPEAR, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0"


def write_map(folder, ax=0.1, az=0.0, clear=None, blocked=None):
    """A 1 m height map h = ax x + az z over x, z -512..512 (heights.npz, as map_grid's), clear [rows, cols] or all;
    passable.npz: blocked [rows, cols] or not clear; meta.json."""
    g = np.arange(-512.0, 513.0, 1.0)
    Zg, Xg = np.meshgrid(g, g, indexing="ij")
    H = (ax * Xg + az * Zg + 200).astype(np.float32)
    C = np.ones(H.shape, np.uint8) if clear is None else clear
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "heights.npz", height=H, clear=C, ground=np.zeros(H.shape, np.uint8),
                        x0=np.float64(-512), z0=np.float64(-512), step=np.float64(1.0))
    np.savez_compressed(folder / "passable.npz", blocked=(1 - (C > 0)).astype(np.uint8) if blocked is None else blocked,
                        x0=np.float64(-512), z0=np.float64(-512), step=np.float64(1.0))
    (folder / "meta.json").write_text('{"ground_names": ["grass"]}', encoding="utf-8")
    return folder


def duel():
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": [{"key": SPEAR, "x": 0, "z": 0, "b": 0}]},
                                     2: {"faction": "wh2_main_skv_skaven",
                                         "units": [{"key": CLANRAT, "x": 0, "z": 50, "b": 180}]}}}


def test_a_place_turns_and_shifts_the_map_under_the_battle():
    mid = terrain.register(np.arange(-20.0, 21.0)[None, :].repeat(41, 0).astype(np.float32), -20.0, -20.0, 1.0)  # h = x
    st = scenario.build([duel()] * 2, P)
    terrain.set_map(st, mid)
    x = torch.tensor([[3.0], [3.0]])
    z = torch.tensor([[0.0], [0.0]])
    assert terrain.height(st.u, x, z)[:, 0].tolist() == pytest.approx([3.0, 3.0])        # no place: as they are
    terrain.set_map(st, mid, battles=[1], place=(torch.tensor([0.0, 5.0]), torch.tensor([0.0, 0.0]),
                                                  torch.tensor([0.0, 90.0])))
    # battle 1: turned 90 deg (its +x is the map's -z ... its +z the map's +x) and shifted 5 m along x
    xm, zm = terrain.to_map(st.u, x, z)
    assert float(xm[1, 0]) == pytest.approx(5.0, abs=1e-5) and float(zm[1, 0]) == pytest.approx(-3.0, abs=1e-5)
    h = terrain.height(st.u, torch.tensor([[0.0], [0.0]]), torch.tensor([[4.0], [4.0]]))
    assert h[:, 0].tolist() == pytest.approx([0.0, 9.0], abs=1e-4)      # battle 1: its (0, 4) is the map's (9, 0)


def test_the_places_keep_the_skirmish_on_the_map_and_its_units_on_clear_ground(tmp_path):
    clear = np.ones((1025, 1025), np.uint8)
    clear[:, :512] = 0                                       # x < 0: not clear
    realmap.activate("m", root=write_map(tmp_path / "m", clear=clear).parent)
    try:
        cfg = {"battles": 16, "seconds": 10, "cell_m": 8.0, "bounds": None}
        s = T.hold_setup(cfg, P, "cpu", torch.Generator().manual_seed(3))
        st = s["st"]
        x0 = st.u["x"].clone()
        assert torch.equal(st.u["x"], x0) and "tpx" in st.u                 # the battle keeps its coordinates
        ex = realmap.extents(s)
        for b, (cx, cz, R, start) in enumerate(ex):
            a = math.radians(float(st.u["tpr"][b, 0]))
            px, pz = float(st.u["tpx"][b, 0]), float(st.u["tpz"][b, 0])
            wx = cx * math.cos(a) + cz * math.sin(a) + px
            wz = -cx * math.sin(a) + cz * math.cos(a) + pz
            assert abs(wx) + R <= 512 + 1e-3 and abs(wz) + R <= 512 + 1e-3
            for (x, z) in start:
                assert x * math.cos(a) + z * math.sin(a) + px >= -0.5            # on clear ground (x >= 0)
        # the exam's places: the same for every build
        realmap.activate("m", exam=True, root=tmp_path)
        s1 = T.hold_setup(cfg, P, "cpu", sp=s["specs"])
        s2 = T.hold_setup(cfg, P, "cpu", sp=s["specs"])
        assert torch.equal(s1["st"].u["tpx"], s2["st"].u["tpx"]) and torch.equal(s1["st"].u["tpr"], s2["st"].u["tpr"])
    finally:
        realmap.deactivate()


def test_the_zone_map_reads_the_ground_through_the_place(tmp_path):
    realmap.activate("m", exam=True, root=write_map(tmp_path / "m", ax=0.2).parent)
    try:
        cfg = {"battles": 4, "seconds": 10, "cell_m": 8.0, "bounds": None}
        s = T.hold_setup(cfg, P, "cpu", torch.Generator().manual_seed(1))
        st = s["st"]
        raw = V.raw(st, P.sim["formation"]["spacing_m"])
        me = raw[:, 0]
        # the height layer: the map's own ground at each cell (h = 0.2 x on the map) less the unit's
        cm = torch.full((st.B,), 8.0)
        hl, fwd, right = Z.ground_layers(me[:, V.I["x"]], me[:, V.I["z"]], me[:, V.I["b"]], cm, me[:, V.I["terr"]],
                                         me[:, V.I["gh"]], place=[me[:, V.I[k]] for k in ("tpx", "tpz", "tpr")])
        # the slope across the map's x seen from the unit: |grad| 0.2 split between forward and right
        g = torch.sqrt(fwd ** 2 + right ** 2)
        assert float(g.mean()) == pytest.approx(0.2, abs=0.01)
        assert float(me[0, V.I["gh"]]) == pytest.approx(float(terrain.height(st.u, st.u["x"][:, :1], st.u["z"][:, :1])[0, 0]))
    finally:
        realmap.deactivate()


def test_the_meter_reads_the_height_over_the_enemy_in_melee(tmp_path):
    realmap.activate("m", exam=True, root=write_map(tmp_path / "m", ax=0.0, az=0.0).parent)
    try:
        st = scenario.build([duel()], P)
        mid = realmap._ACTIVE["mid"]
        # our unit on a 2 m step above the enemy: a map h = 2 for z < 25 (our side), 0 beyond
        g = np.zeros((101, 101), np.float32)
        g[:75, :] = 2.0
        m2 = terrain.register(g, -50.0, -50.0, 1.0)
        terrain.set_map(st, m2, place=(torch.zeros(1), torch.zeros(1), torch.zeros(1)))
        st.u["m"][0, 0] = True
        st.u["m"][0, st.N // 2] = True
        meter = realmap.Meter(1, "cpu")
        bi, ii = torch.tensor([0]), torch.tensor([0])
        meter.step(st, bi, ii, P, torch.zeros(1, dtype=torch.bool))
        r = meter.result()
        assert r["dh_melee_m"] == pytest.approx(2.0) and r["melee_s"] == pytest.approx(P.dt)
        assert mid >= 0
    finally:
        realmap.deactivate()


def test_the_maps_by_name_list_or_all_and_their_files(tmp_path):
    write_map(tmp_path / "a")
    write_map(tmp_path / "b")
    assert realmap.names("all", tmp_path) == ["a", "b"]
    assert realmap.names("b, a", tmp_path) == ["b", "a"]
    (tmp_path / "b" / "passable.npz").unlink()
    with pytest.raises(FileNotFoundError, match="passable.npz"):
        realmap.names("all", tmp_path)                       # a map without its 'cannot stand' layer: refused
    with pytest.raises(FileNotFoundError):
        realmap.activate("b", root=tmp_path)


def test_the_places_keep_every_unit_box_off_the_closed_nodes(tmp_path):
    blocked = np.zeros((1025, 1025), np.uint8)
    blocked[:, 300:700] = 1                                  # x -212..188 closed: a 400 m band
    realmap.activate("m", root=write_map(tmp_path / "m", blocked=blocked).parent)
    try:
        cfg = {"battles": 12, "seconds": 10, "cell_m": 8.0, "bounds": None}
        s = T.hold_setup(cfg, P, "cpu", torch.Generator().manual_seed(5))
        st = s["st"]
        assert bool(s["map_place"]["free"].all())
        for b, (cx, cz, R, pts) in enumerate(realmap.extents(s)):
            px, pz, deg = (float(st.u[k][b, 0]) for k in ("tpx", "tpz", "tpr"))
            wx, wz = realmap.to_world(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), px, pz, deg)
            assert bool(realmap.free_at(wx, wz).all())
            assert len(pts) >= 5 * int((st.u["side"][b] > 0).sum())      # centres and box corners
    finally:
        realmap.deactivate()


def test_two_maps_share_the_skirmishes_in_turn_in_the_exam(tmp_path):
    write_map(tmp_path / "a")
    write_map(tmp_path / "b", ax=0.0)
    mids = [realmap.activate("a,b", exam=True, root=tmp_path)]
    try:
        ms = realmap.maps()
        assert [m["name"] for m in ms] == ["a", "b"]
        assert realmap.pick(5, None).tolist() == [0, 1, 0, 1, 0]
        cfg = {"battles": 6, "seconds": 10, "cell_m": 8.0, "bounds": None}
        s = T.hold_setup(cfg, P, "cpu", torch.Generator().manual_seed(2))
        terr = s["st"].u["terr"][:, 0].tolist()
        assert terr == [ms[b % 2]["mid"] for b in range(6)] and mids[0] == ms[0]["mid"]
        # training: a random map a skirmish, both used
        realmap.activate("a,b", root=tmp_path)
        w = realmap.pick(400, torch.Generator().manual_seed(0))
        assert 150 < int((w == 0).sum()) < 250
        # (re-activating does not register the maps again)
        assert [m["mid"] for m in realmap.maps()] == [m["mid"] for m in ms]
        # one map: no draw (the places' random stream as with a single map before)
        realmap.activate("a", root=tmp_path)
        g = torch.Generator().manual_seed(9)
        assert realmap.pick(7, g).tolist() == [0] * 7 and torch.rand((), generator=g) == torch.rand(
            (), generator=torch.Generator().manual_seed(9))
    finally:
        realmap.deactivate()
