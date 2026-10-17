"""tools/nn/general/map_batch.py: the general's map layers of many battles on the fly (torch)."""
import copy
import math
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn import scenario as nn_scenario                       # noqa: E402
from tools.nn.general import geometry as geo                      # noqa: E402
from tools.nn.general import map_batch as mb                      # noqa: E402
from tools.nn.general import map_input as mi                      # noqa: E402
from tools.nn.sim import scenario as sim_scenario                 # noqa: E402
from tools.nn.train import drills                                 # noqa: E402

F64 = torch.float64


def numpy_layers(zones, side):
    """map_input's exact layers of the arena's field with these Rect zones."""
    spec = mi.MapSpec(name="t", field_box=mi.arena_field(), centre=(0.0, 0.0),
                      zones={s: geo.rect_corners(*zones[s]) for s in (1, 2)})
    return mi.unpack(mi.build(spec), side)


def drill_battles(per_drill=6):
    out = []
    for name, d in drills.load().items():
        out += drills.battles(d, range(per_drill))
    return out


@pytest.fixture(scope="module")
def battles():
    return drill_battles()


def test_the_arena_matches_the_file_layers():
    zones = mb.zones_of(sim_scenario.from_arena("arena"))
    assert zones == mi.arena_rects(nn_scenario.load_arena())
    got = mb.layers(mb.rects_tensor([zones, zones], dtype=F64), [1, 2]).numpy()
    for k, side in enumerate((1, 2)):
        want = mi.unpack(mi.build(mi.arena_spec()), side)
        assert np.abs(got[k, :2] - want[:2]).max() <= 0.5 / 255 + 1e-9
        assert np.array_equal(got[k, 2:], want[2:])


def test_drill_zones_hold_their_units_and_match_the_exact_geometry(battles):
    """Turned drill zones: torch against the polygon geometry of map_input (both exact)."""
    for desc, ours in battles[::3]:
        zones = mb.zones_of(desc)
        for s in (1, 2):
            pts = np.array([(u["x"], u["z"]) for u in desc["sides"][s]["units"]])
            assert mb._holds(zones[s], pts)
        got = mb.layers(mb.rects_tensor([zones], dtype=F64), [ours]).numpy()[0]
        want = numpy_layers(zones, ours)
        assert np.array_equal(got[2:], want[2:]), desc.get("frame")


def test_a_half_turned_battle_seen_by_side_2_gives_side_1s_layers(battles):
    """Turn the battle half round the map's centre and swap the sides (drills.half_turn, drills.mirror): the
    network playing side 2 of it gets exactly the layers side 1 got of the first."""
    descs, turned = [], []
    for desc, _ in battles:
        descs.append(desc)
        turned.append(drills.mirror(drills.half_turn(copy.deepcopy(desc))))
    a = mb.of_battles(descs, [1] * len(descs), dtype=F64)
    b = mb.of_battles(turned, [2] * len(turned), dtype=F64)
    assert torch.equal(a, b)
    assert a[:, 2].sum() > 0 and a[:, 3].sum() > 0


def test_a_side_standing_in_an_arena_square_gets_it(battles):
    """Embedded drills deploy as a normal battle (arena squares); a side with a unit the drill put outside
    its square gets the rectangle around its units instead."""
    squares = list(mi.arena_rects(nn_scenario.load_arena()).values())
    got = 0
    for desc, _ in battles:
        z = mb.zones_of(desc)
        for s in (1, 2):
            pts = np.array([(u["x"], u["z"]) for u in desc["sides"][s]["units"]])
            fits = [q for q in squares if mb._holds(q, pts)]
            assert (z[s] == fits[0]) if fits else (z[s] not in squares)
            got += bool(fits)
    assert got > 0


def test_speed():
    """1024 battles of random turned zones: layers on the CPU, float32; zones_of per battle."""
    rng = np.random.default_rng(0)
    B = 1024
    zones = []
    for _ in range(B):
        a = rng.uniform(0, 2 * math.pi)
        ux, uz = math.cos(a), math.sin(a)
        cx, cz = rng.uniform(-500, 500, 2)
        zones.append({1: (cx - 285 * ux, cz - 285 * uz, ux, uz, 150.0, 150.0),
                      2: (cx + 285 * ux, cz + 285 * uz, -ux, -uz, 150.0, 150.0)})
    rects = mb.rects_tensor(zones)
    sides = torch.tensor(rng.integers(1, 3, B))
    mb.layers(rects[:8], sides[:8])
    t0 = time.perf_counter()
    out = mb.layers(rects, sides)
    dt = time.perf_counter() - t0
    print(f"layers of {B} battles on the CPU: {dt:.3f} s")
    assert out.shape == (B, 6, 128, 128)
    assert dt < 10.0
    descs = [d for d, _ in drill_battles(2)]
    t0 = time.perf_counter()
    for d in descs:
        mb.zones_of(d)
    per = (time.perf_counter() - t0) / len(descs)
    print(f"zones_of: {per * 1e3:.2f} ms a battle")
    assert per < 0.01
    if torch.cuda.is_available():
        r = rects.cuda()
        s = sides.cuda()
        mb.layers(r, s)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mb.layers(r, s)
        torch.cuda.synchronize()
        print(f"layers of {B} battles on the GPU: {time.perf_counter() - t0:.3f} s")
