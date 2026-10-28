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
