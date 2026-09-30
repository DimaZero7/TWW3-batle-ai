# Tests without the game

[← Back](README.md) · [Documentation](../README.md) · [Русский](../../ru/testing/tests.md)

Tests run **the same Lua 5.1** as WH3 through `lupa` and check the code without starting the game.

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python -m pytest tests/apps/units -v
```

They check services, adapters on fake objects, the wiring of apps in the entry
points, the tools outside the game and the documentation rules. They do not
model WH3 battle mechanics: engine behaviour is only verified by running the
game ([running](../launch/run.md)).

## Test levels

Rule of 27.09.2026: small modules are tested on their own; above them only that
they are called correctly. The failing level shows where the fault is.

| Level | What | Where |
|---|---|---|
| 1. Functions | A small function does its job | `tests/apps/<app>/test_*.py` |
| 2. Module by measurements | On data from the game a module gives what the game gave: engine facings, the radar frame, the kit's sensors | `tests/apps/orders/test_facing.py`, `tests/apps/map/test_services.py`, `tests/apps/units/` |
| 3. Wiring | An entry on the fake game calls the modules in order and writes the right events | `tests/entries/test_entries.py` |
| 4. Engine checks | Game facts the code relies on | In-game runs with a verdict; on game updates and new engine facts |

Besides the levels:

| Check | What it catches | Where |
|---|---|---|
| Architecture | every app has a level (all are the base today), dependencies go only down, pure code never touches the engine, no unit keys in logic, no module over 1000 lines | `tests/architecture/` |
| Documentation | language mirror, "← Back", live links and pictures, fresh indexes, a page for every module | `tests/docs/` ([rules](../architecture/documentation.md)) |
| Build | PFH5 format, the bundler, every target builds and compiles, True Sight in every pack | `tests/tools/test_build.py` |

**Tests run by themselves:**
- **before every commit** (~5 s) — hook `.githooks/pre-commit`. Enable once:
  `git config core.hooksPath .githooks`; skip only with a reason, `--no-verify`;
- **on every push** — all tests in GitHub Actions (`.github/workflows/tests.yml`).

## Layout

As in photo-fixing, another project of the author, tests mirror the code:

```text
tests/
├── lua_runtime.py          Lua runtime: require('apps.x.y') loads src/apps/x/y.lua
├── apps/<app>/             service and adapter tests
├── entries/                entry smoke tests on a fake bm
│   └── fake_battle.lua     minimal battle manager fake
├── tools/                  pack format, bundler, every build target, roster, readouts, reports
├── architecture/           code rules
└── docs/                   documentation rules
```

## Writing a test

```python
import pytest
from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().intel = load(runtime, "apps.intel.adapter")
    runtime.execute("""
        position_reads = 0
        function enemy(visible)
            local u = {visible = visible}
            function u:is_visible_to_alliance() return self.visible end
            function u:position()
                position_reads = position_reads + 1
                return {get_x=function() return 10 end, get_y=function() return 0 end, get_z=function() return 20 end}
            end
            return u
        end
    """)
    return runtime


def test_hidden_enemy_position_is_never_read(lua):
    lua.execute("""
        intel.observe(enemy(false), {}, 'enemy-2', 0, {})
        assert(position_reads == 0)
    """)
```

Rules:

- **Write assertions in Lua** inside `lua.execute`, so the test sees the
  same types, `nil` and `0/0` as the game.
- **Test an adapter with a fake object**: a table with the needed methods.
  For "hidden data is never read" checks the fake counts calls or raises on
  a forbidden read (as above and in `tests/apps/units/test_range_adapter.py`).
- **Test an entry** with `tests/entries/fake_battle.lua`: callbacks queue up
  and the test drives them with `bm:pump()` and `bm:tick()`.
- A new engine method used by an entry → add it to `fake_battle.lua`.

## Coverage

| Area | File |
|---|---|
| State sensor (ported from the kit) | `tests/apps/units/test_state_adapter.py` |
| Range sensor (ported from the kit) | `tests/apps/units/test_range_adapter.py` |
| `core`: reads, JSON, errors, guards | `tests/apps/core/test_core.py` |
| Health signature for the stall rule | `tests/apps/battle/test_adapter.py` |
| Radar frame and grid | `tests/apps/map/test_services.py` |
| End of a leg: `arrived`, `stopped`, `stuck`, `timeout` | `tests/apps/navigation/test_services.py` |
| Visibility memory | `tests/apps/intel/test_intel.py` |
| The engine's 64 facings | `tests/apps/orders/test_facing.py` |
| CA's planner: rallied units and units idle under fire | `tests/apps/orders/test_planner_adapter.py` |
| Whole entries on the fake game | `tests/entries/test_entries.py` |
| PFH5, bundler, all build targets | `tests/tools/test_build.py` |
| Roster, readout catalogue | `tests/tools/test_roster.py`, `tests/tools/test_readouts.py` |
| Reports: obstacles on a map, the game AI's groups | `tests/tools/test_obstacles.py`, `tests/tools/test_enemy_layout.py` |
