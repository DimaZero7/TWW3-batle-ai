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
