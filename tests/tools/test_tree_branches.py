"""Every branch of the tree switched off = its baseline (apps.tree), in the simulation.

The baseline is what the chain did before the branch existed; a branch must
never be needed for the trunk to work (5.1 of docs/ru/architecture/ai-design.md).
"""
import copy

import pytest

from tools.sim import formation as sim


@pytest.fixture(scope="module")
def planner():
    return sim.Planner()


def run(planner, name, switches=None, **changes):
    army, _ = sim.load_army(name)
    army = copy.deepcopy(army)
    army.update(changes)
    if switches:
        army["tree"] = switches
    sides, _ = sim.simulate(army, planner)
    return sides


def test_approach_off_the_army_stands_where_deployed(planner):
    on, off = run(planner, "window_open"), run(planner, "window_open", {"approach": False})
    assert on.get("approach") and not off.get("approach")
    assert "fire" in off   # the next steps still run, from where it stands


def test_map_fit_off_is_the_formation_without_the_map(planner):
    off = run(planner, "test_halves", {"map_fit": False})
    army, _ = sim.load_army("test_halves")
    no_map = run(planner, "test_halves", map=None)
    assert off.get("start_fit") is None
    assert [(p["id"], round(p["x"], 2), round(p["z"], 2)) for p in off["own"]["placements"]] == \
        [(p["id"], round(p["x"], 2), round(p["z"], 2)) for p in no_map["own"]["placements"]]


def test_formation_window_off_is_the_thickest_wall(planner):
    # The enemy standing square to us (no turn after a crooked deployment): the window rule decides.
    army, _ = sim.load_army("window_game_few")
    square = {k: v for k, v in army["enemy"].items() if k != "reaction"}
    on = run(planner, "window_game_few", enemy=square)
    off = run(planner, "window_game_few", {"formation_window": False}, enemy=square)
    assert on["own"]["choice"]["wall_width"] == 30 and on["own"]["window_check"]["ok"]
    assert off["own"]["choice"]["wall_width"] == 15 and "window_check" not in off["own"]


def test_align_off_no_alignment(planner):
    on, off = run(planner, "first_attack_crooked", approach=True), \
        run(planner, "first_attack_crooked", {"align": False}, approach=True)
    assert on["alignment"]["check"]["needed"]
    assert off["alignment"] is None and "own_before" not in off
    assert all(r["decision"] != "align" and r["align"] is None for r in off["approach"]["log"])


def test_safe_detour_off_goes_past_the_rock_as_before(planner):
    on, off = run(planner, "rock_attack"), run(planner, "rock_attack", {"safe_detour": False})
    assert [r["decision"] for r in on["approach"]["log"]] == ["approach", "approach", "blocked"]
    # Before the branch: the place past the rock, beyond the stop line (under their fire).
    log = off["approach"]["log"]
    assert [r["decision"] for r in log] == ["approach", "approach", "hold"]
    assert log[1]["path"]["advance_m"] > 50 and log[1]["path"]["beyond_stop_line"]


def test_under_fire_stop_off_one_action_at_a_time_even_under_fire(planner):
    # Our army deployed already inside their archers' reach.
    close = {"own": dict(sim.load_army("window_open")[0]["own"], anchor=[175, -60])}
    on = run(planner, "window_open", **close)
    off = run(planner, "window_open", {"under_fire_stop": False}, **close)
    assert on["approach"]["log"][0]["window"]["under_fire_now"]
    assert on["approach"]["log"][0]["reason"] == "under_fire"
    assert all(r["reason"] != "under_fire" for r in off["approach"]["log"])


def test_logistics_switch_from_the_old_field():
    assert sim.tree_switches({"logistics": True}) == {"logistics": True}
    assert sim.tree_switches({"tree": {"logistics": False}, "logistics": True}) == {"logistics": False}
    assert sim.tree_switches({}) == {}
