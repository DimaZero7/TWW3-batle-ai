"""apps.orders.facing: the engine holds a unit's facing in 64 sectors, at the sector's middle."""
import pytest

from tests.lua_runtime import load, new_runtime

SECTOR = 360 / 64


@pytest.fixture(scope="module")
def facing():
    return load(new_runtime(), "apps.orders.facing")


# Bearing sent -> what the engine showed (measured, 28.09.2026: the sweep and earlier battles).
MEASURED = [(0, 2.81), (5, 2.81), (6, 8.44), (16, 14.06), (17, 19.69), (25.0, 25.3), (84, 81.56), (85, 87.19),
            (88.59, 87.19), (89, 87.19), (90, 92.81), (95, 92.81), (96, 98.44), (125.18, 126.56), (180, 182.81),
            (340, 340.31), (344, 345.94), (348.48, 345.94), (351.58, 351.56), (357.6, 357.19), (359, 357.19)]


@pytest.mark.parametrize("sent,seen", MEASURED)
def test_engine_model_matches_every_measurement(facing, sent, seen):
    assert facing.engine(sent) == pytest.approx(seen, abs=0.35)


@pytest.mark.parametrize("bearing", [0, 1, 90, 125.18, 180, 270, 348.48, 359.9, 45.0])
def test_snapped_facing_is_held_and_close(facing, bearing):
    snapped = facing.snap(bearing)
    assert facing.engine(facing.command(snapped)) == pytest.approx(snapped, abs=1e-9)
    assert abs((snapped - bearing + 180) % 360 - 180) <= SECTOR / 2 + 1e-9
