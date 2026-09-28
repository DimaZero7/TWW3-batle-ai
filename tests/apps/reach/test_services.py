"""apps.reach: who reaches whom (middle of the shooters to the nearest rank of the target), the window."""
import json
import math
import random

import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture(scope="module")
def reach():
    lua = new_runtime()
    m = load(lua, "apps.reach.services")
    encode = lua.eval("function(t) return require('apps.core.json').encode(t) end")
    m.t = lambda x: lua.table_from(x, recursive=True)
    m.win = lambda own, enemy, bearing=0, params=None: json.loads(encode(
        m.window(m.t(own), m.t(enemy), bearing, m.t(params or {}))))
    return m


def block(uid, x, z, bearing, front, depth, range_m=0):
    return {"id": uid, "x": x, "z": z, "bearing": bearing, "front_m": front, "depth_m": depth, "range_m": range_m}


def test_distance_to_a_block(reach):
    b = reach.t(block("b", 0, 0, 0, 20, 10))   # faces +Z: spans x -10..10, z -10..0
    assert reach.to_block(reach.t({"x": 0, "z": 5}), b) == pytest.approx(5)
    assert reach.to_block(reach.t({"x": 13, "z": -14}), b) == pytest.approx(5)
    assert reach.to_block(reach.t({"x": 3, "z": -4}), b) == 0


def test_first_arrow_from_the_middle_whatever_the_depth(reach):
    # Measured: first arrow at 125-126 m from the middle of the block to the target's nearest rank.
    for depth in (6, 18, 47):
        shooter = block("a", 0, 0, 0, 20, depth, 130)
        near = block("t", 0, 125 - depth / 2, 180, 30, 9)
        far = block("t", 0, 127 - depth / 2, 180, 30, 9)
        assert reach.reaches(reach.t(shooter), reach.t(near))[0]
        assert not reach.reaches(reach.t(shooter), reach.t(far))[0]


def armies(gap, tilt=0.0, enemy_range=130, enemy_archers=True, archer_depth=18.0):
    """Ours like the test army: wall 6 x 30 m (9.3 deep), 4 + 4 archer squares 20 x 18 behind;
    theirs like the game's defender: spearmen in a line, archers 20 m behind it."""
    own = [block(f"w{i}", -73.5 + 29.4 * i, 0, 0, 28.4, 9.3) for i in range(6)]
    own += [block(f"a{i}", -40.6 + 27.1 * i, -12.3, 0, 20.2, archer_depth, 130) for i in range(4)]
    own += [block(f"b{i}", -40.6 + 27.1 * i, -39.4, 0, 20.2, archer_depth, 130) for i in range(4)]
    enemy = [block(f"s{i}", -80 + 20 * i, gap, 180 + tilt, 18.4, 8.0) for i in range(8)]
    if enemy_archers:
        enemy += [block(f"e{i}", -55 + 22 * i, gap + 8 + 20, 180 + tilt, 20.2, 18, enemy_range) for i in range(6)]
    return own, enemy


def test_window_of_the_test_army(reach):
    own, enemy = armies(200)
    r = reach.win(own, enemy)
    assert r["reason"] == "window" and r["shooters"] == 8
    # Our first row reaches their line from a gap of 125 - 21.3 = 103.7 m; their archers
    # (middle 8 + 20 + 9 = 37 m behind their front) reach our wall from 128 - 37 = 91 m.
    assert 200 - r["window"]["from"] == pytest.approx(103.7, abs=0.05)
    assert 200 - r["safe_to"] == pytest.approx(91.0, abs=0.05)
    assert r["reached"] == 4                          # the second row does not reach yet
    assert 200 - r["advance_m"] == pytest.approx(98.7, abs=0.05)   # 5 m into the window
    assert not r["under_fire_now"]


def test_without_their_shooters_we_go_5_m_into_our_reach(reach):
    own, enemy = armies(200, enemy_archers=False)
    r = reach.win(own, enemy)
    assert r["reason"] == "no_enemy_shooters" and r.get("safe_to") is None
    assert 200 - r["advance_m"] == pytest.approx(103.7 - 5, abs=0.05) and r["reached"] == 4


def test_no_window_when_they_outrange_us(reach):
    own, enemy = armies(200, enemy_range=170)
    r = reach.win(own, enemy)
    assert r["reason"] == "no_window" and r["reached"] == 0
    assert 200 - r["advance_m"] == pytest.approx(168 - 37, abs=0.05)


def test_already_under_fire(reach):
    own, enemy = armies(85)
    r = reach.win(own, enemy)
    # Too close, under their fire: the stop is behind us, back in the window.
    assert r["under_fire_now"] and 85 - r["advance_m"] == pytest.approx(98.7, abs=0.05)


def test_the_stop_stays_put_once_we_stand_in_the_window(reach):
    far = reach.win(*armies(200))
    for gap in (200 - far["advance_m"], 100.5, 96.0):
        r = reach.win(*armies(gap))
        assert gap - r["advance_m"] == pytest.approx(200 - far["advance_m"], abs=0.05)


def test_nobody_shoots(reach):
    own, enemy = armies(200, enemy_archers=False)
    r = reach.win([u for u in own if not u["range_m"]], enemy)
    assert r["reason"] == "no_shooters" and r.get("advance_m") is None


@pytest.mark.parametrize("seed", range(40))
def test_any_armies_the_stop_keeps_their_shooters_out_and_ours_in(reach, seed):
    """Random sizes, gaps and a turned enemy line: at the chosen stop none of theirs reaches us
    and exactly the reported number of ours reaches one of theirs."""
    rnd = random.Random(seed)
    tilt = rnd.uniform(-8, 8)
    own = [block(f"w{i}", rnd.uniform(-90, 90), rnd.uniform(-3, 3), 0, rnd.uniform(8, 40), rnd.uniform(6, 20))
           for i in range(rnd.randint(1, 8))]
    own += [block(f"a{i}", rnd.uniform(-60, 60), -rnd.uniform(12, 60), 0, 20, rnd.choice([8, 12, 18, 27]),
                  rnd.choice([90, 130, 150])) for i in range(rnd.randint(0, 8))]
    gap = rnd.uniform(150, 300)
    b = 180 + tilt
    fx, fz = math.sin(math.radians(b)), math.cos(math.radians(b))
    enemy = []
    for i in range(rnd.randint(1, 10)):
        along, back = rnd.uniform(-90, 90), rnd.choice([0, rnd.uniform(15, 40)])
        shooter = back > 0 and rnd.random() < 0.8
        # Their block: "along" across their line, "back" behind their front (they face -Z).
        x = -math.cos(math.radians(b)) * along - fx * back
        z = gap + math.sin(math.radians(b)) * along - fz * back
        enemy.append(block(f"e{i}", x, z, b, rnd.uniform(10, 40), rnd.uniform(6, 25),
                           rnd.choice([90, 130, 160]) if shooter else 0))
    r = reach.win(own, enemy)
    if r.get("advance_m") is None:          # nobody shoots
        assert r["reason"] == "no_shooters" and not any(u["range_m"] for u in own + enemy)
        return
    a = r["advance_m"]
    moved = [dict(u, z=u["z"] + a) for u in own]
    p = reach.params(None)
    for s in enemy:
        if s["range_m"]:
            for u in moved:
                d = reach.to_block(reach.middle(reach.t(s)), reach.t(u))
                assert d > s["range_m"] - p.fire_short_m - 1e-6 or r["under_fire_now"], (s["id"], u["id"], d)
    count = 0
    for s in moved:
        if s["range_m"]:
            m = reach.middle(reach.t(s))
            if any(reach.to_block(m, reach.t(t)) <= s["range_m"] - p.fire_short_m - p.own_margin_m + 1e-6
                   for t in enemy):
                count += 1
    assert count == r["reached"], (r, count)
