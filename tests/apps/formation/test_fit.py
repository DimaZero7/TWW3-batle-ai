"""apps.formation.fit: the formation on the map — another width, then moved, then turned.

The map is a function "does this unit fit here" (in battle: the stand mask);
here it is a rock given by hand.
"""
import json

import pytest

from tests.apps.formation.test_services import unit
from tests.lua_runtime import load, new_runtime


@pytest.fixture
def fit():
    lua = new_runtime()
    f = load(lua, "apps.formation.services")
    encode = lua.eval("function(t) return require('apps.core.json').encode(t) end")
    corners = lua.eval("""
        function(p)
            local b = math.rad(p.bearing)
            local fx, fz, rx, rz = math.sin(b), math.cos(b), math.cos(b), -math.sin(b)
            local out = {}
            for _, c in ipairs({{-p.front_m / 2, 0}, {p.front_m / 2, 0}, {p.front_m / 2, p.depth_m}, {-p.front_m / 2, p.depth_m}}) do
                out[#out + 1] = {p.x + rx * c[1] - fx * c[2], p.z + rz * c[1] - fz * c[2]}
            end
            return out
        end
    """)

    def run(rock, units, bearing=0):
        """rock = (x0, x1, z0, z1): nobody may stand on it (any corner or the middle inside)."""
        def fits(p):
            pts = [(c[1], c[2]) for c in corners(p).values()]
            pts.append((sum(x for x, _ in pts) / 4, sum(z for _, z in pts) / 4))
            return not any(rock[0] <= x <= rock[1] and rock[2] <= z <= rock[3] for x, z in pts)
        data = lua.table_from({"anchor": {"x": 0, "z": 0}, "bearing": bearing, "units": units}, recursive=True)
        data.fits = fits
        return json.loads(encode(f.fit("line_and_blocks", data, None)))
    return run


def army():
    return [unit("w1", "wall"), unit("w2", "wall"), unit("a1", "arc"), unit("a2", "arc"), unit("lord", "lord")]


def test_nothing_in_the_way_as_planned(fit):
    r = fit((500, 600, 500, 600), army())
    assert r["fit"]["ok"] and r["fit"]["how"] == "as_planned"


def test_a_rock_under_the_line_moves_the_whole_formation_facing_the_same_way(fit):
    # A rock right under the middle of the wall, deep enough that no width avoids it.
    r = fit((-5, 5, -60, 1), army())
    assert r["fit"]["ok"] and r["fit"]["how"] == "shift"
    assert r["bearing"] == 0
    s = r["fit"]["shift"]
    assert abs(s["side_m"]) > 0 or s["back_m"] > 0
    # Every unit clear of the rock, all facing the same way.
    for p in r["placements"]:
        assert p["bearing"] == 0
        assert not (-5 <= p["x"] <= 5 and -60 <= p["z"] <= 1) or p["role"] == "lord"


def test_sideways_is_tried_before_back(fit):
    # One width only (no other width to try); a narrow rock under its middle:
    # a step sideways clears it, stepping back would not.
    w = {"id": "w1", "role": "wall", "range_m": 0, "shapes": [{"ordered_m": 20, "front_m": 18, "depth_m": 15}]}
    r = fit((-5, 5, -200, 1), [w])
    assert r["fit"]["how"] == "shift"
    assert r["fit"]["shift"]["back_m"] == 0 and abs(r["fit"]["shift"]["side_m"]) >= 12


def test_no_place_keeps_the_plan_and_says_so(fit):
    r = fit((-1000, 1000, -1000, 1000), army())
    assert not r["fit"]["ok"] and r["fit"]["how"] == "none"
    assert len(r["placements"]) == 5
