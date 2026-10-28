"""The map's ground types and trunks (tools/nn/sim/ground.py; build/diag_g49/terrain/ground.md, trees.md, stealth.md;
config/nn/sim.json terrain.ground / terrain.trees, stealth): the layers on the height grid, the multipliers by each man's
cell, the trunks stopping projectiles past their first ~42 m and slowing the men, the forest hiding hide_forest units,
and a map without layers giving the old numbers bit for bit. Needs torch (the container)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, ground, scenario, stealth, terrain  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
SPEAR, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrats_1"
ARCHER = "wh2_dlc13_emp_inf_archers_0"
N = 301                                     # nodes, 2 m apart: x, z in -300..300
NAMES = ["forest", "grass", "shallow_water"]


def flat_map(ground_code=None, trunks=None):
    """A flat map (+-300 m, 2 m nodes) with its ground codes [N, N] (NAMES' indices; None: grass) and trunks."""
    mid = terrain.register(np.zeros((N, N), np.float32), -300.0, -300.0, 2.0)
    g = np.full((N, N), 1, np.uint8) if ground_code is None else ground_code
    ground.register_layers(mid, g, NAMES, trunks)
    return mid


def army(own, foes):
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3]} for r in rows]
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(own)},
                                     2: {"faction": "wh2_main_skv_skaven", "units": units(foes)}}}


def node(x):
    return int(np.floor((x + 300) / 2))


def test_trunk_grids_sum_the_literal_trunks():
    t = {"x": np.array([0.2, 0.7, 10.0], np.float32), "z": np.array([0.1, 0.4, 10.0], np.float32),
         "y": np.array([0.0, 1.0, 0.0], np.float32), "r": np.array([0.4, 0.3, 1.0], np.float32),
         "h": np.array([10.0, 14.0, 5.0], np.float32)}
    lam, top, base, s0, s1, s2 = ground.trunk_grids(t, (5, 5), -1.0, -1.0, 1.0)
    # the first two in the node at (0, 0) (index 1, 1): 2 (0.4 + 0.3) per m2, spread over the 3 x 3 m box round it
    # (ground.LAM_BOX_M); its top 15 and base 0 over the same box
    assert lam[1, 1] == pytest.approx(1.4 / 9) and lam[0, 0] == pytest.approx(1.4 / 9) and lam[3, 3] == 0
    assert top[0, 2] == pytest.approx(15.0) and base[2, 0] == pytest.approx(0.0) and top[4, 4] < -1e8
    assert lam.sum() == pytest.approx(1.4)                     # the third beyond the grid
    k = ground.MOVE_BOX_M
    assert s0[1, 1] == pytest.approx(2 / k ** 2) and s1[1, 1] == pytest.approx(0.7 / k ** 2)
    assert s2[1, 1] == pytest.approx((0.16 + 0.09) / k ** 2)


def test_classes_of_the_captured_names():
    c = ground.classes_of(np.array([[0, 1, 2]], np.uint8), NAMES)
    assert c.tolist() == [[1, 0, 2]]
    c = ground.classes_of(np.array([[0, 1, 2, 3]], np.uint8), ["deep_water", "mud", "rock", "scrub"])
    assert c.tolist() == [[3, 0, 4, 4]]


def test_a_map_without_layers_steps_bit_for_bit_as_before():
    def go(with_layers):
        st = scenario.build([army([(SPEAR, 0, -40, 0), (ARCHER, 30, -60, 0)], [(CLANRAT, 0, 40, 180)])], P)
        mid = terrain.register(np.zeros((N, N), np.float32), -300.0, -300.0, 2.0)
        if with_layers:
            ground.register_layers(mid, None, None, None)           # empty layers: no ground, no trees
        terrain.set_map(st, mid)
        o = O.hold(st.B, st.N)
        o.kind[0, 0], o.target[0, 0] = O.ATTACK, st.N // 2
        o.kind[0, 1], o.target[0, 1] = O.ATTACK, st.N // 2
        for _ in range(60):
            battle.step(st, o, P)
        return {k: v.clone() for k, v in st.u.items() if torch.is_tensor(v)}
    ground._LAYERS.clear()
    ground._DEV.clear()
    a = go(False)
    b = go(True)
    ground._LAYERS.clear()
    ground._DEV.clear()
    for k in ("x", "z", "men", "hp", "mp"):
        assert torch.equal(a[k], b[k]), k


def test_the_forest_covers_its_men_from_projectiles():
    # ground.missile_cover: each man on a forest cell is hit at forest_cover (0.75, the probe cover) of his chance
    share = torch.zeros((1, 3, ground.NCLASS))
    share[0, 0, 1], share[0, 1, 1] = 1.0, 0.5
    u = {"x": torch.zeros((1, 3))}
    c = ground.missile_cover(u, share, P)
    assert P.sim["terrain"]["ground"]["forest_cover"] == 0.75
    assert c[0].tolist() == pytest.approx([0.75, 0.875, 1.0])
    off = P.with_cal("terrain", ground={k: v for k, v in P.sim["terrain"]["ground"].items() if k != "forest_cover"})
    assert ground.missile_cover(u, share, off)[0].tolist() == [1.0, 1.0, 1.0]

    # in a battle: archers on clanrats in the woods take 0.75 of the HP they take on grass (no trees here; nobody hides)
    Q = P.with_cal("stealth", on=0)
    def lost(code_fill):
        code = np.full((N, N), code_fill, np.uint8)
        mid = flat_map(code)
        st = scenario.build([army([(ARCHER, 0, -60, 0)], [(CLANRAT, 0, 30, 180)])], Q)
        terrain.set_map(st, mid)
        st.u["hid_d"][:] = False
        o = O.hold(st.B, st.N)
        hp0 = float(st.u["hp_abs"][0, st.N // 2])
        for _ in range(40):
            battle.step(st, o, Q)
        ground._LAYERS.clear()
        ground._DEV.clear()
        return hp0 - float(st.u["hp_abs"][0, st.N // 2])
    grass, wood = lost(1), lost(0)
    assert grass > 0 and wood / grass == pytest.approx(0.75, rel=0.08)


def test_large_units_in_forest_and_small_in_shallow_water():
    code = np.full((N, N), 1, np.uint8)
    code[:, : node(0)] = 0                                     # x < 0: forest
    code[:, node(0):] = 2                                      # x >= 0: shallow water
    mid = flat_map(code)
    st = scenario.build([army([(SPEAR, -100, 0, 0), (SPEAR, 100, 0, 0)], [(CLANRAT, 0, 250, 180)])], P)
    terrain.set_map(st, mid)
    pw = geometry.pairwise(st.u, SP)
    sh = ground.shares(st.u, pw["front"], pw["depth"], P)
    assert float(sh[0, 0, 1]) == 1.0 and float(sh[0, 1, 2]) == 1.0
    sp, at = ground.factors(st.u, sh, P)
    assert float(sp[0, 0]) == 1.0 and float(sp[0, 1]) == pytest.approx(0.8) and float(at[0, 1]) == pytest.approx(0.8)
    st.u["large"][0, 0] = True                                 # a large unit in the forest: x0.8
    sp, at = ground.factors(st.u, sh, P)
    assert float(sp[0, 0]) == pytest.approx(0.8) and float(at[0, 0]) == pytest.approx(0.8)
    st.u["strider"][0, 0] = True                               # a strider: no penalty
    sp, at = ground.factors(st.u, sh, P)
    assert float(sp[0, 0]) == 1.0 and float(at[0, 0]) == 1.0
    # half the men in the water: half the penalty
    st.u["x"][0, 1] = 0.0
    sh = ground.shares(st.u, pw["front"], pw["depth"], P)
    sp, _ = ground.factors(st.u, sh, P)
    assert float(sh[0, 1, 2]) == pytest.approx(0.5) and float(sp[0, 1]) == pytest.approx(0.9)
    ground._LAYERS.clear()
    ground._DEV.clear()


def wall(x0, x1, every=1.0, r=0.4, h=14.0):
    """Trunks filling the band x0..x1 across z -300..300, every `every` m."""
    xs, zs = np.meshgrid(np.arange(x0, x1, every), np.arange(-300, 300, every))
    n = xs.size
    return {"x": xs.ravel().astype(np.float32), "z": zs.ravel().astype(np.float32), "y": np.zeros(n, np.float32),
            "r": np.full(n, r, np.float32), "h": np.full(n, h, np.float32)}


def keep_of(trunks, gap=100.0):
    mid = flat_map(trunks=trunks)
    st = scenario.build([army([(ARCHER, 0, 0, 90)], [(CLANRAT, gap, 0, 270)])], P)
    terrain.set_map(st, mid)
    pw = geometry.pairwise(st.u, SP)
    tgt = torch.full_like(st.u["side"], -1, dtype=torch.int64)
    tgt[0, 0] = st.N // 2
    k = ground.trunk_keep(st.u, pw, tgt, None, P)
    ground._LAYERS.clear()
    ground._DEV.clear()
    return float(k[0, 0]), st


def test_a_trunk_stops_a_projectile_only_past_its_first_42_m():
    # the archers' arrow: 0.93 s x 45 m/s ~ 42 m clear of trees (the database's vegetation_ignore_time)
    k_none, st = keep_of(None)
    assert k_none == 1.0
    assert float(st.u["veg_m"][0, 0]) == pytest.approx(0.93 * 45, abs=0.1)
    k_near, _ = keep_of(wall(10, 25))                          # a strip within the first 42 m: nothing stopped
    k_far, _ = keep_of(wall(60, 75))                           # the same strip further: stops most
    assert k_near == pytest.approx(1.0) and k_far < 0.05
    # a thin strip: exp(-lam x its width), lam = 2 r per m2 (a trunk every 5 m: 2 x 0.4 / 25)
    k_thin, _ = keep_of(wall(60, 70, every=5.0))
    assert k_thin == pytest.approx(np.exp(-0.8 / 25 * 10), rel=0.25)
    # over the trunks' tops it flies free
    k_low, _ = keep_of(wall(60, 75, h=0.5))
    assert k_low > 0.95


def test_trunks_slow_the_men_and_a_strider_passes():
    trunks = wall(-300, 300, every=5.0)
    mid = flat_map(trunks=trunks)
    st = scenario.build([army([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 250, 180)])], P)
    terrain.set_map(st, mid)
    pw = geometry.pairwise(st.u, SP)
    f = ground.trunk_speed(st.u, pw["front"], pw["depth"], P)
    R = float(st.u["radius"][0, 0])
    loss = 2 * P.sim["terrain"]["trees"]["detour"] * (0.4 + R) ** 2 / 25
    assert float(f[0, 0]) == pytest.approx(1 / (1 + loss), rel=0.05)
    st.u["strider"][0, 0] = True
    assert float(ground.trunk_speed(st.u, pw["front"], pw["depth"], P)[0, 0]) == 1.0
    ground._LAYERS.clear()
    ground._DEV.clear()


def test_a_hide_forest_unit_hides_in_the_woods_and_is_seen_within_60_m():
    code = np.full((N, N), 1, np.uint8)
    code[:, : node(0)] = 0                                     # x < 0: forest
    mid = flat_map(code)
    assert stealth.on(P)

    def placed(x, gap):
        st = scenario.build([army([(SPEAR, x, 0, 90)], [(CLANRAT, x + 150, 0, 270)])], P)
        terrain.set_map(st, mid)
        pw = geometry.pairwise(st.u, SP)
        g = float(pw["gap"][0, st.N // 2, 0])
        st.u["x"][0, 0] = st.u["x"][0, 0] + (g - gap)          # the gap edge to edge
        return st
    st = placed(-100, 200)
    battle.step(st, O.hold(st.B, st.N), P)
    assert bool(st.u["hid"][0, 0])                             # deployed in the woods: hidden from the first step
    assert not bool(st.u["hid"][0, st.N // 2])                 # the clanrats (hide_forest) on grass: never
    st = placed(-100, 200)
    st.u["hid_d"][:] = False                                   # (not a deployment: it walked in)
    for _ in range(9):
        battle.step(st, O.hold(st.B, st.N), P)
    assert not bool(st.u["hid"][0, 0])                         # 4.5 s in the woods
    battle.step(st, O.hold(st.B, st.N), P)
    assert bool(st.u["hid"][0, 0]) and not bool(st.u["vis"][0, 0])      # 5 s: hidden
    # in the woods it is seen within the clanrats' spot_dist_tree 60 m x its hiding_scalar 1.0
    for gap, seen in ((64.0, False), (56.0, True)):
        st = placed(-100, gap)
        st.u["hid"][0, 0], st.u["hid_s"][0, 0], st.u["hid_t"][0, 0] = True, 5.0, 10.0
        battle.step(st, O.hold(st.B, st.N), P)
        assert bool(st.u["vis"][0, 0]) == seen, gap
    # on open ground the same unit never hides
    st = placed(60, 200)
    st.u["hid_d"][:] = False
    for _ in range(12):
        battle.step(st, O.hold(st.B, st.N), P)
    assert not bool(st.u["hid"][0, 0])
    ground._LAYERS.clear()
    ground._DEV.clear()
