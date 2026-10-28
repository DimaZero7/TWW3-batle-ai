"""The ground's height (tools/nn/sim/terrain.py; build/diag_g49/terrain/mechanic.md; config/nn/sim.json terrain): the
height maps (bilinear, per battle, flat by default), the rules one by one - speed on a slope, melee and missile damage
by height, the morale's hill, the line of fire over friends and over the ground - and a flat ground giving the old
numbers. Needs torch (the container)."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, melee, missile, scenario, terrain  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.train import opponents  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
SPEAR, CLANRAT, SWORD = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0", "wh_main_emp_inf_swordsmen"
XBOW, HANDGUN = "wh_main_emp_inf_crossbowmen", "wh_main_emp_inf_handgunners"
GX = np.arange(-300.0, 301.0, 2.0)


def plane(ax=0.0, az=0.0, c=0.0):
    """A height map h = ax x + az z + c over +-300 m (2 m nodes, as the game's)."""
    Z, X = np.meshgrid(GX, GX, indexing="ij")
    return terrain.register((ax * X + az * Z + c).astype(np.float32), -300.0, -300.0, 2.0)


def army(own, foes, attacker=1):
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3]} for r in rows]
    return {"attacker": attacker, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(own)},
                                            2: {"faction": "wh2_main_skv_skaven", "units": units(foes)}}}


class TestMaps:
    def test_bilinear_held_at_the_edge_and_flat_by_default(self):
        mid = plane(0.5, 0.2, 3.0)
        st = scenario.build([army([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 100, 180)])] * 3, P)
        assert not terrain.on(st.u) and float(terrain.height(st.u, st.u["x"], st.u["z"]).abs().sum()) == 0
        terrain.set_map(st, mid, battles=[0, 2])
        x = torch.tensor([[1.3, -7.7], [1.3, -7.7], [1.3, 1000.0]])
        z = torch.tensor([[2.9, 11.1], [2.9, 11.1], [2.9, 0.0]])
        h = terrain.height(st.u, x, z)
        assert h[0].tolist() == pytest.approx([0.5 * 1.3 + 0.2 * 2.9 + 3, 0.5 * -7.7 + 0.2 * 11.1 + 3], abs=1e-4)
        assert h[1].tolist() == [0.0, 0.0]                                   # battle 1: flat
        assert float(h[2, 1]) == pytest.approx(0.5 * 300 + 3, abs=1e-4)      # beyond the map: its edge
        dh = terrain.pair_dh(st.u)
        H = st.N // 2
        assert float(dh[0, 0, H]) == pytest.approx(0.2 * 100, abs=1e-3) and float(dh[1, 0, H]) == 0

    def test_a_cloned_or_narrowed_batch_keeps_its_maps(self):
        mid = plane(0.0, 1.0)
        st = scenario.build([army([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 100, 180)])] * 2, P)
        terrain.set_map(st, mid, battles=[1])
        keep = torch.tensor([1])
        u = {k: v[keep] for k, v in st.u.items()}
        assert float(terrain.ground(u)[0, st.N // 2]) == pytest.approx(100.0, abs=1e-3)


def test_speed_on_a_slope():
    f = terrain.speed_factor(torch.tensor([0.0, 0.25, 1.0, -0.1, -0.8]), 1.5)
    assert f.tolist() == pytest.approx([1.0, 0.8, 0.5, 1.1, 1.5])
    # spearmen walking up a 0.25 slope (+z) / down it: x0.8 / x1.25 of their walk once at speed
    out = {}
    for name, az in (("up", 0.25), ("down", -0.25), ("flat", 0.0)):
        st = scenario.build([army([(SPEAR, 0, -100, 0)], [(CLANRAT, 500, 500, 180)])], P)
        terrain.set_map(st, plane(0.0, az))
        o = O.hold(st.B, st.N)
        o.kind[0, 0], o.x[0, 0], o.z[0, 0] = O.MOVE, 0.0, 100.0
        for _ in range(20):
            battle.step(st, o, P)
        out[name] = math.hypot(float(st.u["vx"][0, 0]), float(st.u["vz"][0, 0]))
    walk = float(st.u["walk"][0, 0])
    assert out["flat"] == pytest.approx(walk, rel=1e-3)
    assert out["up"] == pytest.approx(walk / 1.25, rel=1e-2) and out["down"] == pytest.approx(walk * 1.25, rel=1e-2)


def _melee_pair(az):
    st = scenario.build([army([(SWORD, 0, -10, 0)], [(CLANRAT, 0, 10, 180)])], P)
    front, depth = geometry.dims(st.u, SP)
    H = st.N // 2
    st.u["z"][0, 0], st.u["z"][0, H] = -float(depth[0, 0]) / 2, float(depth[0, H]) / 2
    if az is not None:
        terrain.set_map(st, plane(0.0, az))
    pw = geometry.pairwise(st.u, SP)
    if terrain.on(st.u):
        pw["dh"] = terrain.pair_dh(st.u)
    return st, pw, H


def test_melee_damage_by_the_height_between_the_fighting_men():
    gap = P.sim["terrain"]["pair_gap_m"]
    for az, own, theirs in ((1.0, 0.7, 1.3), (-1.0, 1.3, 0.7), (0.1, 1 - 0.3 * 0.1 * gap, 1 + 0.3 * 0.1 * gap)):
        st, pw, H = _melee_pair(az)
        f = terrain.melee_factor(st.u, pw, P)
        # (the swordsmen at -z strike the clanrats uphill when az > 0: x (1 - 0.3 dh))
        assert float(f[0, 0, H]) == pytest.approx(own, abs=1e-4) and float(f[0, H, 0]) == pytest.approx(theirs, abs=1e-4)
    # the blow's damage in the strikes: in proportion (no overkill near the clanrats' health here)
    st, pw, H = _melee_pair(None)
    contact = pw["enemy"] & (pw["gap"] <= 1.0)
    st.u["contact_s"] = torch.full_like(st.u["men"], 300.0)
    flat = melee.strikes(st.u, pw, contact, P, torch.zeros_like(st.u["men"]))[1]
    st2, pw2, _ = _melee_pair(0.1)
    st2.u["contact_s"] = torch.full_like(st2.u["men"], 300.0)
    up = melee.strikes(st2.u, pw2, contact, P, torch.zeros_like(st2.u["men"]))[1]
    assert float(up[0, 0, H]) < float(flat[0, 0, H]) and float(up[0, H, 0]) > float(flat[0, H, 0])


def test_missile_damage_by_height():
    f = terrain.missile_factor(torch.tensor([-40.0, -80.0, 20.0, 0.0]), P)
    assert f.tolist() == pytest.approx([1.3, 1.3, 0.85, 1.0])
    # the launch angle: a target below is shot flatter (more accurately), one above steeper
    v, d = torch.tensor(45.0), torch.tensor(100.0)
    flat = 0.5 * math.asin(100 * 9.81 / 45 ** 2)
    assert float(missile.acc_theta(torch.tensor(flat), v, d, None)) == pytest.approx(flat)
    assert float(missile.acc_theta(torch.tensor(flat), v, d, torch.tensor(0.0))) == pytest.approx(flat, abs=1e-5)
    assert float(missile.acc_theta(torch.tensor(flat), v, d, torch.tensor(-30.0))) < flat < float(
        missile.acc_theta(torch.tensor(flat), v, d, torch.tensor(30.0)))


def test_the_hill_morale():
    # spearmen at z = 0 on a slope rising to -z: 0.15 x 100 m = 15 m above clanrats 100 m off; 300 m off: none near
    for dz, hill in ((100.0, True), (300.0, False)):
        st = scenario.build([army([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, dz, 180)])], P)
        terrain.set_map(st, plane(0.0, -0.15))
        pw = geometry.pairwise(st.u, SP)
        alive = st.u["men"] > 0
        assert bool(terrain.hill(st.u, pw, alive, P)[0, 0]) == hill
        assert not bool(terrain.hill(st.u, pw, alive, P)[0, st.N // 2])


def test_the_ground_closes_the_line_of_fire():
    # a ridge 15 m high halfway between handgunners and clanrats 100 m off
    Z, X = np.meshgrid(GX, GX, indexing="ij")
    ridge = terrain.register((15.0 * np.exp(-((Z - 50.0) / 6.0) ** 2)).astype(np.float32), -300.0, -300.0, 2.0)
    for key in (HANDGUN, XBOW):
        st = scenario.build([army([(key, 0, 0, 0)], [(CLANRAT, 0, 100, 180)])], P)
        pw = geometry.pairwise(st.u, SP)
        tgt = torch.full((1, st.N), -1, dtype=torch.long)
        tgt[0, 0] = st.N // 2
        terrain.set_map(st, plane(0.0, 0.0))
        assert not bool(terrain.ground_closed(st.u, pw, tgt, P)[0, 0])
        terrain.set_map(st, ridge)
        assert bool(terrain.ground_closed(st.u, pw, tgt, P)[0, 0])


def test_a_shooter_on_a_hill_shoots_over_its_infantry():
    # handgunners 10 m behind own spearmen 70 m wide, clanrats 90 m beyond them: flat - the spearmen close the line; the
    # handgunners on a rise 10 m above the rest - clear
    Z, X = np.meshgrid(GX, GX, indexing="ij")
    rise = terrain.register(np.where(Z < -33.0, 10.0, 0.0).astype(np.float32), -300.0, -300.0, 2.0)
    desc = army([(HANDGUN, 0, -40, 0), (SPEAR, 0, -30, 0)], [(CLANRAT, 0, 60, 180)])
    desc["sides"][1]["units"][1]["width"] = 70.0                       # (a screen as wide as the shooters)
    st = scenario.build([desc], P)
    H = st.N // 2
    tgt = torch.full((1, st.N), -1, dtype=torch.long)
    tgt[0, 0] = H
    pw = geometry.pairwise(st.u, SP)
    assert bool(missile.friend_rays(st.u, pw, tgt, P)[0, 0])
    terrain.set_map(st, rise)
    pw["dh"] = terrain.pair_dh(st.u)
    assert not bool(missile.friend_rays(st.u, pw, tgt, P)[0, 0])
    assert not bool(terrain.ground_closed(st.u, pw, tgt, P)[0, 0])


def test_a_level_map_gives_the_flat_numbers():
    desc = army([(SPEAR, 0, -60, 0), (XBOW, 40, -90, 0)], [(CLANRAT, 0, 60, 180), (CLANRAT, 40, 60, 180)])
    out = []
    for level in (False, True):
        st = scenario.build([desc] * 2, P)
        if level:
            terrain.set_map(st, plane(0.0, 0.0, 50.0))                  # level ground 50 m up: only differences count
        for _ in range(120):
            battle.step(st, opponents.nearest(st), P)
        out.append(st)
    for k in ("hp_abs", "men", "x", "z", "morale"):
        assert torch.allclose(out[0].u[k], out[1].u[k], rtol=1e-3, atol=1e-2), k
