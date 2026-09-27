"""Start-of-battle cases (level 2: modules by contract).

Every tests/cases/start/<name>.json says: for these armies (units from the real
data/roster) the strategy module chooses this strategy (or 'none' because of
these conditions), and the formation module places it with these properties.
A new case is a new file; no test code changes.
"""
import json
from pathlib import Path

import pytest

from tools import roster
from tools.sim.formation import Planner, roster_units

CASES = sorted((Path(__file__).parents[2] / "cases" / "start").glob("*.json"))


@pytest.fixture(scope="module")
def planner():
    return Planner()


@pytest.fixture(scope="module")
def entries():
    return roster.load_roster()


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_case(path, planner, entries):
    case = json.loads(path.read_text(encoding="utf-8"))
    expect = case["expect"]
    plan = planner.start(case["own"]["role"], (0, -150), 0, roster_units(case["own"], entries),
                         roster_units(case["enemy"], entries))
    decision = plan["decision"]
    assert decision["strategy"] == expect["strategy"], decision["candidates"]
    if "failed" in expect:
        chosen = next(c for c in decision["candidates"] if c["key"] == "wall_and_arc")
        assert set(expect["failed"]) <= set(chosen["failed"]), chosen["failed"]
    if "formation" in expect:
        formation = plan["formation"]
        assert formation["status"] == expect["formation"]["status"]
        assert len(formation["overlaps"]) == expect["formation"]["overlaps"]
        # Every own unit got a role and, unless 'other', a place.
        placed = {p["id"] for p in formation["placements"]}
        assert placed | set(formation["unplaced"]) == set(decision["roles"])


def test_there_are_cases():
    assert len(CASES) >= 5
