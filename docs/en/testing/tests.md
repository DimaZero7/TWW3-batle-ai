# Tests without the game

[← Back](README.md) · [Documentation](../README.md) · [Русский](../../ru/testing/tests.md)

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python -m pytest tests/apps/units -v
```

Tests run **the same Lua 5.1** as WH3 through `lupa`. They check contracts,
services, adapters on fake objects and the wiring of apps in the entry
points. They do not model WH3 battle mechanics: engine behaviour is only
verified by running the game ([running](../launch/run.md)).

## Layout

As in photo-fixing, tests mirror the code:

```text
tests/
├── lua_runtime.py          Lua runtime: require('apps.x.y') loads src/apps/x/y.lua
├── apps/<app>/             service, contract and adapter tests
├── entries/                entry smoke tests on a fake bm
│   └── fake_battle.lua     minimal battle manager fake
└── tools/                  pack format, bundler, every build target
```

## Writing a test

```python
import pytest
from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().orders = load(runtime, "apps.orders.contract")
    return runtime


class TestValidate:
    def test_hidden_target_is_rejected(self, lua):
        lua.execute("""
            local own = {spears = {alive = true, position = {x=0, z=0}, kind = 'spearmen'}}
            local enemies = {hidden = {visibility = 'not_visible'}}
            local ok, err = pcall(orders.validate,
                {{unit_id='spears', action='attack', target_id='hidden', mode='melee'}}, own, enemies, 14)
            assert(not ok and err:find('target not visible', 1, true))
        """)
```

Rules:

- **Write assertions in Lua** inside `lua.execute`, so the test sees the
  same types, `nil` and `0/0` as the game.
- **Test an adapter with a fake object**: a table with the needed methods.
  For "hidden data is never read" checks the fake counts calls or raises on
  a forbidden read (see `tests/apps/units/test_range_adapter.py`).
- **Test an entry** with `tests/entries/fake_battle.lua`: callbacks queue up
  and the test drives them with `bm:pump()` and `bm:tick()`.
- A new engine method used by an entry → add it to `fake_battle.lua`.

## Coverage

| Area | File |
|---|---|
| State sensor (ported from the kit) | `tests/apps/units/test_state_adapter.py` |
| Range sensor (ported from the kit) | `tests/apps/units/test_range_adapter.py` |
| `core`: reads, JSON, errors | `tests/apps/core/test_core.py` |
| Radar frame and grid | `tests/apps/map/test_services.py` |
| Visibility memory | `tests/apps/intel/test_intel.py` |
| AI policies and contract | `tests/apps/ai/test_policies.py` |
| Command validation | `tests/apps/orders/test_contract.py` |
| Deployment v2 | `tests/apps/deployment/test_deployment.py` |
| Policy sandbox | `tests/apps/sandbox/test_sandbox.py` |
| Whole duel and map capture | `tests/entries/test_entries.py` |
| PFH5, bundler, all build targets | `tests/tools/test_build.py` |

## Test levels

Rule of 27.09.2026: small modules are tested on their own; above them only
that they are called correctly. The failing level shows where the fault is.

| Level | What | Where |
|---|---|---|
| 1. Functions | A small function does its job | `tests/apps/<app>/test_services.py` |
| 2. Module contract | On given armies a module decides as expected (strategy chosen or rejected by the right conditions; formation without overlaps) | `tests/cases/start/*.json` + `tests/apps/strategy/test_cases.py`; a new case is a new file |
| 3. Chain wiring | Modules called in order and passing their data; modules replaced by spies | `tests/apps/plan/`, entries on the fake battle |
| 4. Engine checks | Game facts the AI relies on | In-game runs with verdicts, on game updates and new engine facts |

Level 2 checks properties, not exact numbers, so tuning does not break tests.
