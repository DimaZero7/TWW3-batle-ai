"""What projectiles strike, by the game's memory (build/diag_mem/missile; config/nn/sim.json artillery.gun_share, terrain.trees
los / fan): a battery's guns are bodies of their own (missile.battery), trunks on the low line make arcing fire lob
(ground.trunk_lob), the trunk rays fan over the target's span. Needs torch (the container)."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import artillery, battle, geometry, ground, missile, scenario, terrain  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
WARP = "wh2_main_skv_art_warp_lightning_cannon"
ARCHER, CLANRAT, HGUN = "wh2_dlc13_emp_inf_archers_0", "wh2_main_skv_inf_clanrats_0", "wh_main_emp_inf_handgunners"


def with_(**kv):
    p = copy.deepcopy(P)
    for k, v in kv.items():
        sec, key = k.split("__")
        d = p.sim[sec] if sec != "trees" else p.sim["terrain"]["trees"]
        if v is None:
            d.pop(key, None)
        else:
            d[key] = v
    return p


def army(own, foes):
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3]} for r in rows]
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(own)},
                                     2: {"faction": "wh2_main_skv_skaven", "units": units(foes)}}}


def test_the_guns_bodies_in_the_static_row():
    s = P.static(WARP)
    assert (s["eng_r"], s["eng_q"], s["eng_h"]) == (3.5, 0.5, 4.5)
    assert (P.static(ARCHER)["eng_r"], P.static(ARCHER)["eng_q"]) == (0.0, 1.0)


def test_a_battery_is_hit_by_its_guns_bodies_and_an_infantry_target_as_before():
    st = scenario.build([army([(ARCHER, 0, 0, 0), (ARCHER, 300, 0, 0)], [(WARP, 0, 125, 180), (CLANRAT, 300, 125, 180)])], P)
    pw = geometry.pairwise(st.u, SP)
    H = st.N // 2
    p = missile.hit_chance(st.u, pw, P)
    q, sh = missile.battery(st.u, pw, P, p)
    assert float(q[0, 0, H]) > 3 * float(p[0, 0, H])           # the guns: large bodies (the crew's 10 m template alone)
    assert 0.5 < float(sh[0, 0, H]) < 1.0                       # most hits on the guns (the game: ~70 %)
    assert torch.equal(q[0, :, H + 1], p[0, :, H + 1]) and float(sh[0, 1, H + 1]) == 0.0   # the clanrats: untouched
    q0, sh0 = missile.battery(st.u, pw, with_(artillery__gun_share="count"), p)
    assert sh0 is None and torch.equal(q0, p)
    # a lost gun leaves its stretch: two guns left, half the guns' chance
    st.u["eng"][0, H] = 2
    q2, _ = missile.battery(st.u, pw, P, p)
    assert float(q2[0, 0, H]) < float(q[0, 0, H])


def test_archers_break_the_guns_with_bodies_not_with_the_count():
    def guns_lost(params):
        st = scenario.build([army([(ARCHER, 0, 0, 0)], [(WARP, 0, 125, 180)])], params)
        H = st.N // 2
        w0 = float(st.u["eng_w"][0, H])
        for _ in range(int(round(30 / params.dt))):
            o = O.hold(st.B, st.N)
            o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
            battle.step(st, o, params, params.dt)
        return w0 - float(st.u["eng_w"][0, H])
    lost_b, lost_c = guns_lost(P), guns_lost(with_(artillery__gun_share="count"))
    assert lost_b > 500 and lost_c < 100                        # the game: 1552 HP of the guns in ~240 shots


N = 301


def flat_map(trunks=None):
    mid = terrain.register(np.zeros((N, N), np.float32), -300.0, -300.0, 2.0)
    ground.register_layers(mid, np.full((N, N), 1, np.uint8), ["forest", "grass", "shallow_water"], trunks)
    return mid


def wall(x0, x1, z0=-300, z1=300, every=1.0, r=0.4, h=14.0):
    xs, zs = np.meshgrid(np.arange(x0, x1, every), np.arange(z0, z1, every))
    n = xs.size
    return {"x": xs.ravel().astype(np.float32), "z": zs.ravel().astype(np.float32), "y": np.zeros(n, np.float32),
            "r": np.full(n, r, np.float32), "h": np.full(n, h, np.float32)}


def setup(trunks, shooter=ARCHER, gap=100.0):
    mid = flat_map(trunks)
    st = scenario.build([army([(shooter, 0, 0, 90)], [(CLANRAT, gap, 0, 270)])], P)
    terrain.set_map(st, mid)
    pw = geometry.pairwise(st.u, SP)
    tgt = torch.full_like(st.u["side"], -1, dtype=torch.int64)
    tgt[0, 0] = st.N // 2
    return st, pw, tgt


def teardown():
    ground._LAYERS.clear()
    ground._DEV.clear()


def test_trunks_on_the_low_line_make_arcing_fire_lob_and_the_lob_flies_over():
    try:
        st, pw, tgt = setup(wall(60, 75, every=3.0))
        q = ground.trunk_lob(st.u, pw, tgt, None, P)
        assert float(q[0, 0]) > 0.5                              # a wood on the line: the unit mostly lobs
        assert ground.trunk_lob(st.u, pw, tgt, None, with_(trees__los=None)) is None
        keep = float(ground.trunk_keep(st.u, pw, tgt, None, P)[0, 0])
        low = float(ground.trunk_keep(st.u, pw, tgt, None, with_(trees__los=None))[0, 0])
        assert keep > low + 0.3                                 # the lob flies over the 14 m trunks at 60-75 m
        # a unit lobbing already (friends, hills) is not counted twice
        lob = torch.zeros_like(tgt, dtype=torch.bool)
        lob[0, 0] = True
        assert float(ground.trunk_lob(st.u, pw, tgt, lob, P)[0, 0]) == 0.0
        # the line check counts from the muzzle (a strip within the first 42 m closes it; the arrows ignore it)
        teardown()
        st, pw, tgt = setup(wall(10, 25, every=3.0))
        assert float(ground.trunk_lob(st.u, pw, tgt, None, P)[0, 0]) > 0.5
        teardown()
        st, pw, tgt = setup(None)
        assert float(ground.trunk_keep(st.u, pw, tgt, None, P)[0, 0]) == 1.0
        # direct fire never lobs past trees: handgunners keep their low line
        teardown()
        st, pw, tgt = setup(wall(60, 75, every=3.0), shooter=HGUN)
        assert float(ground.trunk_lob(st.u, pw, tgt, None, P)[0, 0]) == 0.0
    finally:
        teardown()


def test_the_rays_fan_over_the_targets_span():
    try:
        # trunks only in front of the target's left half (z > 0 at 70-80 m): the fan's rays split, the nearest
        # man's do not
        st, pw, tgt = setup(wall(70, 80, z0=2, z1=40, every=2.0))
        fan = ground.trunk_keep(st.u, pw, tgt, None, P)
        near = ground.trunk_keep(st.u, pw, tgt, None, with_(trees__fan=None))
        assert float(fan[0, 0]) != pytest.approx(float(near[0, 0]), abs=1e-3)
    finally:
        teardown()
