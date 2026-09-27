# Project structure and layers

[Documentation](../README.md) · [Русский](../../ru/architecture/overview.md)

The project follows the photo-fixing layout: code is split into **apps** by
domain, logic lives in **services**, and entry points only wire them
together. The main difference: the code runs inside the game (Lua 5.1 in
WH3), so engine calls are kept in a separate layer — **adapters**.

## Folders

```text
src/
├── apps/              apps: core, battle, map, navigation, units, intel,
│                      orders, deployment, ai, sandbox, telemetry
└── entries/           entry points: duel, arena, map_capture
scenarios/             battle XML scenarios
tools/                 Python/PowerShell outside the game
├── build.py           pack build: python -m tools.build <target>
├── pack/              PFH5 format and the Lua module bundler
├── launcher/          install the pack, start WH3, wait, clean up
├── telemetry/         read JSONL while the game writes it
└── analysis/          height maps, slopes, passages from a saved grid
tests/                 pytest + lupa: the same Lua 5.1 without the game
config/                default.json in Git, local.json per machine
data/                  compact verified map and unit data
docs/ru, docs/en       documentation in two languages
research/              research archive: evidence, probes, legacy scripts
```

## App layers

Each app is a folder `src/apps/<name>/` with files by role:

| File | photo-fixing analogue | Rule |
|---|---|---|
| `*adapter.lua` | `models.py` | **The only place** that calls `bm`, unit objects or `common` (CCO). Returns plain tables |
| `services.lua` | `services.py` | Pure logic over plain tables. No engine, no files, no timers |
| `contract.lua` | `schemas.py` | Data shapes and their validation: commands, deployment plan, profiles |
| `policies/` | — | Only in `ai`: interchangeable strategies |

Not every app has every layer: `units` has adapters and a contract, `intel`
has an adapter and a memory service.

**Entry points** (`src/entries/`) are the `router.py` analogue: they subscribe
to battle phases, run the tick, call adapters and services and write
telemetry. They contain no decision logic.

## Dependency rules

```text
entries     →  any app
sandbox     →  core, ai.contract, orders.contract
deployment  →  core, units.contract, orders.adapter (only in its adapter)
orders      →  core, units.contract
navigation  →  core, map.adapter (vector construction)
ai, battle, intel, map, telemetry, units  →  core (and their own contract)
core        →  nothing
```

The bundler fails on a circular `require`; `tests/tools/test_build.py`
builds every target.

- `core` imports nothing from apps. In photo-fixing `common` knew about
  `auth` and `routes`; that is not allowed here.
- Services do not import adapters. When a service needs the engine, it gets a
  callback (see `deployment.services.verify(measure, pair_distance)`).
- Modules use plain `require('apps.map.services')`. The builder puts every
  module into one script with its own loader ([build](../launch/build.md)).

## Battle data flow

```text
XML scenario  →  WH3 loads the battle and our script from the pack
  →  entry: battle kind check (battle), read sides
  →  Deployment phase: placement (deployment)
  →  Deployed phase: take control (orders), tick every tick_ms:
        units / intel / map  →  observation
        ai                   →  decision
        orders.contract      →  command validation
        orders.adapter       →  engine orders
        telemetry            →  events.jsonl
  →  result, callbacks removed, rematch or end
  →  the launcher copies results to build/<target>/runs/<time>/
```

## Adding an app

1. Create `src/apps/<name>/` and split the code: engine calls in
   `adapter.lua`, logic in `services.lua`, data shapes in `contract.lua`.
2. Write tests in `tests/apps/<name>/`. Services are tested directly,
   adapters with fake engine objects.
3. Document the app in `docs/en/apps/<name>.md` and `docs/ru/apps/<name>.md`,
   add a row to the [list of apps](apps.md) and the indexes.
