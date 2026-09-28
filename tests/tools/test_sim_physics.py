"""The simulator's physics near obstacles (task 27, docs/ru/architecture/tasks/sim-obstacles.md).

In the game no unit walks or stands on an obstacle: the engine walks round it and a
block squeezes past its edge. The simulation must do the same, for every manoeuvre
of the approach and for the queues round an obstacle, on every army with a map.
"""
import math

import pytest

from tools.sim import formation as sim
from tools.sim import logistics as queues
from tools.sim import walker as walking
from tools.sim.mapgrid import MapGrid

ARMIES = ["rock_march_wide", "gap_march_wide", "spear_march_wide", "skaven_march_wide", "edge_march_wide",
          "rock_attack", "window_game"]
# A block may brush the obstacle's edge (its soldiers squeeze aside), never be driven through it.
MAX_SQUEEZED = 0.3


@pytest.fixture(scope="module")
def runs():
    out = {}
    for name in ARMIES:
        army, _ = sim.load_army(name)
        planner = sim.Planner()
        sides, by_id = sim.simulate(army, planner)
        out[name] = (army, planner, sides, by_id, MapGrid(army["map"]))
    return out


def on_blocked(grid, x, z):
    return not grid.reader(x, z)[0]


def samples(move):
    """Every unit's tracked (t_s, x, z, bearing, front_m, depth_m) of a manoeuvre (front-rank centre)."""
    return move.get("tracks") or {}


@pytest.mark.parametrize("name", ARMIES)
def test_every_manoeuvre_of_the_approach_is_walked_in_time(runs, name):
    _, _, sides, _, _ = runs[name]
    for m in (sides.get("approach") or {}).get("moves", []):
        tracks = samples(m)
        assert set(tracks) == {p["id"] for p in m["after"]}, "every unit walks the manoeuvre"
        assert m["walk_s"] == pytest.approx(max(tr[-1][0] for tr in tracks.values()), abs=walking.DT_S)


@pytest.mark.parametrize("name", ARMIES)
def test_no_unit_walks_or_stands_on_an_obstacle_in_the_approach(runs, name):
    _, _, sides, by_id, grid = runs[name]
    squeezer = walking.Squeezer(grid)
    bad, worst = [], 0.0
    for n, m in enumerate((sides.get("approach") or {}).get("moves", [])):
        for uid, tr in samples(m).items():
            men = by_id[uid]["men"]
            for t, x, z, b, front, depth in tr:
                rect = {"x": x, "z": z, "bearing": b, "front_m": front, "depth_m": depth}
                cx, cz = x - math.sin(math.radians(b)) * depth / 2, z - math.cos(math.radians(b)) * depth / 2
                if on_blocked(grid, cx, cz):
                    bad.append(f"move {n} {uid} t={t}: centre on an obstacle")
                pts, moved = squeezer.squeeze(sim.soldier_points(rect, men))
                if any(on_blocked(grid, pts[i], pts[i + 1]) for i in range(0, len(pts), 2)):
                    bad.append(f"move {n} {uid} t={t}: soldiers on an obstacle after squeezing")
                worst = max(worst, moved / men)
    assert not bad, "\n".join(bad[:10])
    assert worst <= MAX_SQUEEZED, f"a block driven through an obstacle: {worst:.0%} of its men squeezed"


@pytest.mark.parametrize("name", ["rock_march_wide", "gap_march_wide", "spear_march_wide", "skaven_march_wide"])
def test_no_soldier_on_an_obstacle_in_the_queues(runs, name):
    _, planner, sides, by_id, grid = runs[name]
    lg = queues.Logistics(planner.lua)
    for m in [m for m in sides["approach"]["moves"] if m["detour"]]:
        for queue in (True, False):
            summary, _, _ = queues.walk(lg, m["field"], grid, m["before"], m["after"], by_id, queue)
            assert summary["on_blocked_max"] == 0
            assert summary["squeezed_max_share"] <= MAX_SQUEEZED


# ---- the walker on a made-up map ---------------------------------------------------------------

class Grid:
    """A 3 m grid 300 x 300 m around (0, 0); blocked(x, z) -> True where nobody can stand."""

    def __init__(self, blocked):
        import numpy as np
        self.step, self.min_x, self.min_z = 3.0, -150.0, -150.0
        n = 100
        self.clear = np.ones((n, n), dtype=bool)
        self.inside = np.ones((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                x, z = self.min_x + (j + 0.5) * 3, self.min_z + (i + 0.5) * 3
                self.clear[i, j] = not blocked(x, z)

    def reader(self, x, z):
        i, j = int((z - self.min_z) // self.step), int((x - self.min_x) // self.step)
        if not (0 <= i < self.clear.shape[0] and 0 <= j < self.clear.shape[1]):
            return False, False
        c = bool(self.clear[i, j])
        return c, c


def rock(x, z):
    return -20 <= x <= 20 and 10 <= z <= 40


def walk_one(grid, start, goal, bearing=0.0, final_bearing=None, front=20.0, depth=6.0):
    w = walking.Walker(grid)
    w.add("u", start[0], start[1], bearing, front, depth)
    w.order("u", goal[0], goal[1], final_bearing)
    track = w.run(max_s=600)["u"]
    return w, track


def test_open_ground_is_walked_straight_at_the_walking_pace():
    w, track = walk_one(Grid(lambda x, z: False), (0, 0), (0, 60))
    t, x, z, b = track[-1][:4]
    assert (x, z) == pytest.approx((0, 60), abs=0.5)
    assert t == pytest.approx(60 / walking.SPEED_MPS + walking.SETTLE_S, abs=1)   # walks, then settles
    assert all(abs(p[1]) < 0.01 for p in track)


def test_a_rock_in_the_way_is_walked_round_not_through():
    grid = Grid(rock)
    w, track = walk_one(grid, (0, 0), (0, 60))
    assert track[-1][1:3] == pytest.approx((0, 60), abs=0.5)
    assert not any(rock(p[1], p[2]) for p in track)
    assert track[-1][0] > 60 / walking.SPEED_MPS + walking.SETTLE_S + 5   # round the rock takes longer


def test_a_walking_block_faces_where_it_goes_and_turns_to_its_order_at_the_end():
    w, track = walk_one(Grid(lambda x, z: False), (0, 0), (60, 0), bearing=0.0, final_bearing=0.0)
    middle = track[len(track) // 2]
    assert abs(((middle[3] - 90 + 540) % 360) - 180) < 5          # walking east faces east
    assert abs(((track[-1][3] + 540) % 360) - 180) < 1            # and turns north at the place


def test_an_order_onto_the_rock_stops_at_its_edge():
    grid = Grid(rock)
    w, track = walk_one(grid, (0, 0), (0, 20))
    x, z = track[-1][1:3]
    assert not rock(x, z) and z < 10 and w.state("u")["arrived"]


def test_soldiers_on_the_rock_squeeze_to_the_nearest_free_ground():
    grid = Grid(rock)
    pts, moved = walking.Squeezer(grid).squeeze([0, 11, 0, 5, 19, 38])
    assert moved == 2 and not any(rock(pts[i], pts[i + 1]) for i in range(0, len(pts), 2))
    assert pts[2:4] == [0, 5]                                      # a free soldier stays
    assert math.hypot(pts[0] - 0, pts[1] - 11) < 4                 # the nearest free ground, not far away
