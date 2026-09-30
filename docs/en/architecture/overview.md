# Project structure and layers

[← Back](README.md) · [Documentation](../README.md) · [Русский](../../ru/architecture/overview.md)

Code is split into **apps** by domain, logic lives in **services**, and entry
points only wire them together. This is the layout of photo-fixing, another
project of the author (a Python web app); its files are given below for
comparison. The main difference: the code runs inside the game (Lua 5.1 in
WH3), so engine calls are kept in a separate layer — **adapters**.

## Folders

```text
src/
├── apps/              apps: core, battle, map, navigation, units, intel,
│                      observation, orders, telemetry
└── entries/           entry points: ai_vs_ai, archer_range, enemy_layout, manual_record,
                       map_capture, move_probe, nn_arena, roster_capture, unit_readout
scenarios/             battle XML scenarios
tools/                 Python/PowerShell outside the game
├── build.py           pack build: python -m tools.build <target>
├── pack/              PFH5 format and the Lua module bundler
├── launcher/          install the pack, start WH3, wait, clean up
├── telemetry/         read JSONL while the game writes it
├── analysis/          height maps, slopes, passages, run reports
├── nn/                rules from the game's database, the arena's scenario, arena records as arrays
└── docs/              indexes of the documentation hubs
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
| `services.lua` and other pure modules (`orders/facing.lua`, `core/*`) | `services.py` | Pure logic over plain tables. No engine, no files, no timers |

Not every app has both layers: `units` has only adapters, `intel` has an
adapter and a memory service.

**Entry points** (`src/entries/`) are the `router.py` analogue: they subscribe
to battle phases, run the tick, call adapters and services and write
telemetry. They contain no decision logic.

## Dependency rules

```text
entries                         →  any app
observation                     →  units, intel (their adapters)
navigation                      →  core, map.adapter (vector construction)
intel, map, telemetry, units    →  core
orders                          →  only its own facing
battle                          →  nothing
core                            →  nothing
```

- Apps know nothing of the entries; `core` imports nothing from apps. In
  photo-fixing `common` knew about `auth` and `routes`; that is not allowed here.
- Services do not import adapters. When a service needs the engine, it gets a
  callback: `map.services.build_frame(query)` receives the radar query from
  `map.adapter`, and the tests pass a plain function.
- Modules use plain `require('apps.map.services')`. The builder puts every
  module into one script with its own loader ([build](../launch/build.md));
  a circular `require` fails the build.
- Every app has a level in `tools/architecture.py`. All are the base (0) today;
  a future level above may use the base, never the other way round.
- `tests/architecture/` checks the levels, and also that pure code never touches
  the engine, unit keys (`wh_…`) are not written into logic, and no module is
  longer than 1000 lines.

## Battle data flow

```text
XML scenario  →  WH3 loads the battle and our script from the pack
  →  entry: battle kind check, read sides and roles (battle)
  →  Deployment phase: a probe places its units (orders.teleport) or leaves deployment to the game
  →  Deployed phase: speed (battle.speed_guard), deadline (battle.deadline), tick every tick_ms:
        units / intel / observation / map       →  readings
        orders.adapter, orders.planner_adapter  →  orders from the probe's plan or the game's AI
        telemetry                               →  events.jsonl
  →  result (outcome, stall, timeout, deadline), callbacks removed
  →  the launcher copies results to build/<target>/runs/<time>/
```

## Project rules

| Rule | Where |
|---|---|
| Every battle runs with the True Sight mod (Workshop 3628832922) | [running a battle](../launch/run.md#required-mod-true-sight) |
| Battles against the game's AI run only at Normal difficulty | [fair difficulty](../launch/run.md#fair-difficulty) |
| A side's AI gets only **its side's view**: own units in full; for enemies visibility, the position while visible, the last known position. The full view is only ground truth and research data | [observation](../apps/observation.md) |
| Hidden units must not leak: checked by the `--view side` report | [visibility](../game/units/visibility.md) |
| A battle always ends by itself: a real-time deadline, a stall of 10 game minutes without damage, the launcher limit | [entry points](../apps/entries.md) |
| The speed stays at the requested one (×20) even after the outcome is decided | [battle](../apps/battle.md) |
| Orders only through verified engine calls | [orders](../apps/orders.md) |
| What a run collects is chosen by a profile from the readout catalogue | [readout catalogue](../game/readouts.md) |

## Adding an app

1. Create `src/apps/<name>/` and split the code: engine calls in
   `adapter.lua`, logic in `services.lua`.
2. Write tests in `tests/apps/<name>/`. Services are tested directly,
   adapters with fake engine objects.
3. Document the app in `docs/en/apps/<name>.md` and `docs/ru/apps/<name>.md`,
   add a row to the [list of apps](apps.md) and update the indexes
   (`python -m tools.docs.index_doc`).
