"""tools.nn.map_grid: the capture's grid.csv into a height array, the deployment zones of a battle XML."""
import math

import numpy as np
import pytest

from tools.nn import map_grid


def grid_csv(nx=3, nz=2, x0=-2.0, z0=4.0, step=2.0, drop=None):
    lines = ["ix,iz,x,z,height,clear,ground,inside_radar"]
    for iz in range(nz):
        for ix in range(nx):
            if (ix, iz) == drop:
                continue
            lines.append(f'{ix},{iz},{x0 + ix * step},{z0 + iz * step},{100 + ix + 10 * iz},{int(ix != 1)},'
                         f'"{"grass" if iz == 0 else "mud"}",1')
    return "\n".join(lines) + "\n"


def test_grid_is_laid_out_row_z_column_x():
    H, C, G, names, x0, z0, step = map_grid.parse_grid(grid_csv())
    assert H.shape == (2, 3) and H.dtype == np.float32
    assert (x0, z0, step) == (-2.0, 4.0, 2.0)
    assert H[1, 2] == 112 and H[0, 0] == 100
    assert C.tolist() == [[1, 0, 1], [1, 0, 1]]
    assert names == ["grass", "mud"] and G[1].tolist() == [1, 1, 1]


def test_a_missing_cell_is_an_error():
    with pytest.raises(AssertionError):
        map_grid.parse_grid(grid_csv(drop=(1, 1)))


def test_zones_corners_from_centre_size_and_facing():
    xml = """<battle><alliance id="0"><army><deployment_area><centre x="10" y="-20"/><width metres="200"/>
    <height metres="100"/><orientation radians="%s"/></deployment_area></army></alliance>
    <battle_map_definition><name>terrain\\battles\\x\\</name><catchment_area>c1</catchment_area></battle_map_definition>
    </battle>""" % (math.pi / 2)
    (z,) = map_grid.zones(xml)
    assert (z["x"], z["z"], z["width"], z["depth"], z["bearing_deg"]) == (10.0, -20.0, 200.0, 100.0, 90.0)
    xs = sorted(c[0] for c in z["corners"])
    zs = sorted(c[1] for c in z["corners"])
    assert xs[0] == pytest.approx(-40) and xs[-1] == pytest.approx(60)      # facing +x: the depth along x
    assert zs[0] == pytest.approx(-120) and zs[-1] == pytest.approx(80)
    assert map_grid.map_of(xml) == ("terrain\\battles\\x\\", "c1")


def test_tree_bodies_by_the_game_s_rule():
    # a tree is a cylinder: radius min((CollRadius or 0.85) x scale, 2.25), height CollHeight x scale; no metadata row,
    # no body (build/diag_g49/terrain/trees.md 3)
    meta = [{"Filename": "A.rigid_model_v2", "CollHeight": "10", "CollRadius": "0.4"},
            {"Filename": "Big.rigid_model_v2", "CollHeight": "20", "CollRadius": "2.0"},
            {"Filename": "zero.rigid_model_v2", "CollHeight": "5", "CollRadius": "0"}]
    pts = [{"model": "a.RIGID_MODEL_V2", "x": "1", "y": "200", "z": "2", "scale": "1.5"},
           {"model": "Big.rigid_model_v2", "x": "3", "y": "201", "z": "4", "scale": "1.5"},
           {"model": "zero.rigid_model_v2", "x": "5", "y": "202", "z": "6", "scale": "0.5"},
           {"model": "unknown.rigid_model_v2", "x": "7", "y": "203", "z": "8", "scale": "1"}]
    b = map_grid.tree_bodies(pts, meta)
    assert b["x"].tolist() == [1, 3, 5] and b["z"].tolist() == [2, 4, 6] and b["y"].tolist() == [200, 201, 202]
    assert b["r"].tolist() == pytest.approx([0.6, 2.25, 0.425])
    assert b["h"].tolist() == pytest.approx([15.0, 30.0, 2.5])



def reach_csv():
    # 3 x 2 nodes; clear 0 at ix 1 (as grid_csv); reach: (1, 0) a ring node (not clear, reachable),
    # (2, 1) clear but unreachable, (0, 1) outside the read (-1); deep water at (1, 1)
    lines = ["ix,iz,x,z,height,clear,ground,inside_radar,reach_side_1,reach_side_2"]
    reach = {(0, 0): 1, (1, 0): 1, (2, 0): 1, (0, 1): -1, (1, 1): 0, (2, 1): 0}
    ground = {0: "shallow_water", 1: "deep_water", 2: "grass"}
    for iz in range(2):
        for ix in range(3):
            g = "grass" if iz == 0 else ground[ix]
            r = reach[(ix, iz)]
            lines.append(f'{ix},{iz},{-2.0 + ix * 2},{4.0 + iz * 2},100,{int(ix != 1)},"{g}",1,{r},{r}')
    return "\n".join(lines) + "\n"


def test_blocked_is_the_reach_where_read_and_not_clear_elsewhere():
    text = reach_csv()
    H, C, G, names, x0, z0, step = map_grid.parse_grid(text)
    reach = map_grid.parse_reach(text, H.shape)
    assert reach[0].tolist() == [[1, 1, 1], [-1, 0, 0]]
    blocked, info = map_grid.passable_layers(C, G, names, reach)
    assert blocked.dtype == np.uint8
    assert blocked.tolist() == [[0, 0, 0], [0, 1, 1]]      # (1, 0) reachable -> free; (0, 1) not read, clear -> free
    assert (info["not_clear_reach"], info["clear_not_reach"], info["reach_read"]) == (1, 1, 5)
    assert info["blocked_by_ground"]["deep_water"] == [1, 1] and info["blocked_by_ground"]["grass"] == [1, 4]


def test_without_reach_blocked_is_not_clear():
    H, C, G, names, *_ = map_grid.parse_grid(grid_csv())
    assert map_grid.parse_reach(grid_csv(), H.shape) is None
    blocked, info = map_grid.passable_layers(C, G, names)
    assert blocked.tolist() == [[0, 1, 0], [0, 1, 0]] and info["source"].startswith("bm:is_area_clear")


def test_passable_goes_next_to_heights_and_into_meta(tmp_path):
    text = reach_csv()
    H, C, G, names, x0, z0, step = map_grid.parse_grid(text)
    blocked, info = map_grid.passable_layers(C, G, names, map_grid.parse_reach(text, H.shape))
    (tmp_path / "meta.json").write_text('{"name": "m"}', encoding="utf-8")
    meta = map_grid.write_layers(tmp_path, blocked, x0, z0, step, info)
    p = np.load(tmp_path / "passable.npz")
    assert (p["blocked"] == blocked).all() and float(p["x0"]) == x0 and float(p["step"]) == step
    assert meta["name"] == "m" and meta["passable"]["blocked"] == 2
    assert meta["passable"]["blocked_by_ground"]["deep_water"] == "1 of 1"
