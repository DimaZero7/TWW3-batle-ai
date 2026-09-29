"""Task 26: the alignment chased the game's AI round (docs/ru/architecture/tasks/align-chase.md).

The game's AI in defence turns to face our army by itself while its centre stays;
aligning with where it LOOKS moved the army hundreds of metres aside and again and
again. window_game's enemy turns as the game's did (tools/sim/enemy.NativeReaction).
"""
import math

import pytest

from tools.sim import formation as sim


@pytest.fixture(scope="module")
def runs():
    out = {}
    army, _ = sim.load_army("window_game")
    out["front"], _ = sim.simulate(dict(army, align={"line": "front"}))
    # The old line without the map: led onto the rocks, apps.formation.fit searches for
    # minutes (backlog #20, struck 29.09.2026); on open ground the chase is the same as in the game.
    out["enemy_facing"], _ = sim.simulate(dict(army, align={"line": "enemy_facing"}, map=None))
    return out


def centre(placements):
    body = [p for p in placements if p["role"] != "lord"]
    return sum(p["x"] for p in body) / len(body), sum(p["z"] for p in body) / len(body)


def test_the_enemy_turns_to_our_army_as_the_game_s_ai(runs):
    timeline = runs["front"]["enemy"]["timeline"]
    assert len(timeline) >= 2                                       # it turned
    c0, c1 = centre(timeline[0]["placements"]), centre(timeline[-1]["placements"])
    assert math.hypot(c1[0] - c0[0], c1[1] - c0[1]) < 5             # its centre stayed


def total_s(sides):
    """From deployment to the stop: the alignment at deployment, then the approach."""
    return (sides.get("align_walk") or {}).get("walk_s", 0) + sides["approach"]["final"]["t_s"]


def window_reached(sides):
    final = sides["approach"]["final"]
    return final["decision"] == "hold" and (final.get("window") or {}).get("reached", 0) >= 3


def test_with_their_facing_the_army_is_led_aside_and_realigns_again_and_again(runs):
    old = runs["enemy_facing"]
    assert old["alignment"]["check"]["offset_m"] < -200             # the game: -344 m at deployment
    aligns = [r for r in old["approach"]["log"] if r["decision"] == "align"]
    assert len(aligns) >= 2                                         # the game: 5-6 a battle
    assert total_s(old) > 650                                       # the game: 720-750 s


def test_turning_in_place_the_army_never_goes_aside_and_reaches_the_window(runs):
    new, old = runs["front"], runs["enemy_facing"]
    for m in new["approach"]["moves"]:
        if m["decision"] == "align":
            (bx, bz), (ax, az) = centre(m["before"]), centre(m["after"])
            assert math.hypot(ax - bx, az - bz) < 5                 # only a turn
    assert sum(r["decision"] == "align" for r in new["approach"]["log"]) <= 2
    assert window_reached(new) and new["fire"]["lost"]["enemy"]["men"] > 0
    assert abs(new["alignment"]["check"]["offset_m"]) < 1e-6
    assert total_s(new) < total_s(runs["enemy_facing"]) - 200
