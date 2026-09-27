"""apps.battlefield: the frame between the two main groups (level 1 and 2)."""
import json
import math
from pathlib import Path

import pytest

from tests.lua_runtime import load, new_runtime

CASES = sorted((Path(__file__).parents[2] / "cases" / "vision").glob("*.json"))


@pytest.fixture(scope="module")
def lua():
    runtime = new_runtime()
    runtime.globals().bf = load(runtime, "apps.battlefield.services")
    return runtime


def run(lua, own, enemy, params=None):
    encode = lua.eval("function(t) return require('apps.core.json').encode(t) end")
    data = {"own": {"points": own}, "enemy": {"points": enemy}}
    return json.loads(encode(lua.globals().bf.frame(lua.table_from(data, recursive=True), lua.table_from(params or {}))))


def rect(x0, x1, z0, z1, step=2):
    """Points filling an axis-aligned rectangle."""
    pts = []
    x = x0
    while x <= x1 + 1e-9:
        z = z0
        while z <= z1 + 1e-9:
            pts += [x, z]
            z += step
        x += step
    return pts


def test_armies_facing_north(lua):
    # Our army 40 m wide around z = -100..-90, theirs 60 m wide at z = 100..110.
    f = run(lua, rect(-20, 20, -100, -90), rect(-30, 30, 100, 110))
    assert f["status"] == "ok" and f["bearing"] == pytest.approx(0)
    assert f["centres_m"] == pytest.approx(200)
    assert f["own"]["front_m"] == pytest.approx(5) and f["own"]["back_m"] == pytest.approx(-5)
    assert f["enemy"]["front_m"] == pytest.approx(195) and f["gap_m"] == pytest.approx(190)
    assert f["enemy"]["width_m"] == pytest.approx(60) and f["own"]["width_m"] == pytest.approx(40)
    # Symmetric about the axis: 30 m (their wider half) + 40 m margin.
    assert f["half_width_m"] == pytest.approx(70) and f["margin_m"] == 40


def test_diagonal_armies_keep_the_same_numbers(lua):
    straight = run(lua, rect(-20, 20, -100, -90), rect(-30, 30, 100, 110))
    # Rotate everything by 45 degrees: the frame turns, the numbers stay.
    c, s = math.cos(math.radians(45)), math.sin(math.radians(45))

    def turn(pts):
        out = []
        for i in range(0, len(pts), 2):
            x, z = pts[i], pts[i + 1]
            out += [x * c + z * s, -x * s + z * c]
        return out
    turned = run(lua, turn(rect(-20, 20, -100, -90)), turn(rect(-30, 30, 100, 110)))
    assert turned["bearing"] == pytest.approx(45)
    for key in ("gap_m", "centres_m", "half_width_m"):
        assert turned[key] == pytest.approx(straight[key])


def test_contact(lua):
    f = run(lua, rect(-20, 20, -10, 5), rect(-20, 20, 0, 15))
    assert f["status"] == "contact" and f["gap_m"] < 0


def test_frame_round_trip_and_corners(lua):
    f = run(lua, rect(-20, 20, -100, -90), rect(-30, 30, 100, 110), {"margin_m": 50})
    bf = lua.globals().bf
    field = lua.table_from({"origin": f["origin"], "bearing": f["bearing"]}, recursive=True)
    p = bf.to_world(field, 120, -35)
    back = bf.to_frame(field, p)
    assert back.along == pytest.approx(120) and back.across == pytest.approx(-35)
    xs = sorted(c["x"] for c in f["corners"])
    zs = sorted(c["z"] for c in f["corners"])
    assert xs[0] == pytest.approx(-80) and xs[-1] == pytest.approx(80)   # 30 + 50 each side
    assert zs[0] == pytest.approx(-100) and zs[-1] == pytest.approx(110)  # our back to their back


def test_bad_input(lua):
    with pytest.raises(Exception, match="Main group points required"):
        run(lua, [], [0, 0])
    with pytest.raises(Exception, match="same centre"):
        run(lua, [0, 0], [0, 0])


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_real_game_ai_layouts(path, lua):
    """Level 2: our army 300 m south of each real game AI layout."""
    case = json.loads(path.read_text(encoding="utf-8"))
    enemy = [v / 10 for u in case["units"] for v in u["points_dm"]]
    ex = enemy[0::2]
    ez = enemy[1::2]
    cx, cz = sum(ex) / len(ex), sum(ez) / len(ez)
    own = rect(cx - 30, cx + 30, cz - 310, cz - 290)
    f = run(lua, own, enemy)
    # 300 m between centres minus our half depth (10 m) and their front half depth (up to ~50 m).
    assert f["status"] == "ok" and 230 < f["gap_m"] < 290
    assert f["enemy"]["width_m"] == pytest.approx(max(ex) - min(ex), rel=0.3)
    assert f["half_width_m"] >= 40 + f["enemy"]["width_m"] / 2 - 1
