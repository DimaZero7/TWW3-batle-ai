"""The rules that let the code grow (docs/ru/architecture/ai-design.md, 5.1), checked.

* levels: modules depend only downwards; on one level only the level's trunk
  (plan, tactics) calls its nodes; every app has a level;
* pure code (services, contract) never touches the engine;
* no unit, race or lord keys in logic: only in data modules;
* module size, with the exceptions and why;
* the tree: nodes, parents, baselines;
* time budget of a decision.
"""
import re
import time
from pathlib import Path

import pytest

from tests.lua_runtime import load, new_runtime
from tools import architecture

SRC = Path(__file__).resolve().parents[2] / "src"
APPS = SRC / "apps"

LEVELS, TRUNKS = architecture.LEVELS, architecture.TRUNKS
MAX_LINES = 1000  # user, 28.09.2026 (was 500)
# Longer modules, and why (to be split when they are next changed).
TOO_LONG = {}
REQUIRE = re.compile(r"require\('apps\.([a-z_]+)\.")
UNIT_KEY = re.compile(r"['\"]wh[0-9]?_[a-z0-9_]+['\"]")
ENGINE = re.compile(r"\bbm:|get_context_value|script_unit|battle_manager|:unique_ui_id\(")


def lua_files():
    return sorted(APPS.rglob("*.lua"))


def app_of(path):
    return path.relative_to(APPS).parts[0]


def test_every_app_has_a_level():
    apps = {p.name for p in APPS.iterdir() if p.is_dir()}
    assert not apps - set(LEVELS), f"give these apps a level: {sorted(apps - set(LEVELS))}"


def test_dependencies_go_only_down():
    bad = []
    for path in lua_files():
        app = app_of(path)
        for dep in set(REQUIRE.findall(path.read_text(encoding="utf-8"))) - {app}:
            if LEVELS[dep] > LEVELS[app]:
                bad.append(f"{path.relative_to(APPS)} -> {dep} (up: {LEVELS[app]} -> {LEVELS[dep]})")
            elif LEVELS[dep] == LEVELS[app] and LEVELS[app] > 0 and app not in TRUNKS:
                bad.append(f"{path.relative_to(APPS)} -> {dep} (same level {LEVELS[app]}, not through its trunk)")
    assert not bad, "\n".join(bad)


def test_pure_code_never_touches_the_engine():
    bad = [str(p.relative_to(APPS)) for p in lua_files()
           if p.stem in ("services", "contract", "data") and ENGINE.search(p.read_text(encoding="utf-8"))]
    assert not bad, bad


def test_no_unit_keys_in_logic():
    bad = []
    for p in lua_files():
        if p.stem == "data":
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("--", 1)[0]
            if UNIT_KEY.search(code):
                bad.append(f"{p.relative_to(APPS)}:{n}: {line.strip()}")
    assert not bad, "keys belong in data modules:\n" + "\n".join(bad)


def test_modules_are_not_too_long():
    long = {str(p.relative_to(APPS)).replace("\\", "/"): len(p.read_text(encoding="utf-8").splitlines())
            for p in lua_files()}
    over = {k: v for k, v in long.items() if v > MAX_LINES and k not in TOO_LONG}
    assert not over, f"split these modules (or add them to TOO_LONG with a reason): {over}"
    stale = [k for k in TOO_LONG if long.get(k, 0) <= MAX_LINES]
    assert not stale, f"no longer too long, remove from TOO_LONG: {stale}"


@pytest.fixture(scope="module")
def lua():
    return new_runtime()


def test_the_tree_is_well_formed(lua):
    tree = load(lua, "apps.tree.services")
    nodes = [dict(n) for n in tree.NODES.values()]
    names = [n["name"] for n in nodes]
    assert len(names) == len(set(names))
    kinds = {n["name"]: n["kind"] for n in nodes}
    phases = [n for n in nodes if n["kind"] == "phase"]
    # The trunk is one chain of phases; a branch hangs on a phase and says what happens without it.
    assert sum(1 for n in phases if not n.get("parent")) == 1
    for n in nodes:
        if n.get("parent"):
            assert kinds.get(n["parent"]) == "phase", n["name"]
        if n["kind"] == "branch":
            assert n.get("baseline"), f"branch {n['name']} has no baseline"
        assert n["kind"] in ("phase", "branch")
    children = [n["parent"] for n in phases if n.get("parent")]
    assert len(children) == len(set(children)), "a phase has two next phases"


def test_decisions_keep_their_time_budget():
    """apps.plan (phase 1, with the window check) and apps.tactics (one decision of
    phase 2) on the test army; generous limits, the game's Lua is a bit slower."""
    from tools.sim import formation as sim
    army, _ = sim.load_army("window_open")
    planner = sim.Planner()
    t = time.perf_counter()
    sides, _ = sim.simulate(army, planner)
    whole = time.perf_counter() - t
    decisions = len(sides["approach"]["log"]) + 2
    # Everything the simulation did, per decision (plans, windows, pictures).
    assert whole / decisions < 0.25, f"{whole / decisions * 1000:.0f} ms per decision"
