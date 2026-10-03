# Documentation

[← Back](../../README.md) · [Project](../../README.md) · **English** | [Русский](../ru/README.md)

This is the project's base: the battle mechanics of Total War: WARHAMMER III that we
measured, our tools to work with the game, and the battle network trained from scratch in our
simulator and checked in the game. The project's earlier algorithmic AI was removed (the
project owner's decision); it is in the Git history.

## Data for training the network

- **[Data for training the network](training/README.md)** — the arena, recorded battles of
  the game's AI, rules from the game's database, measured facts; how to collect and read them.
- [Network model](training/network.md) · [Inputs and model](training/model.md) ·
  [Training](training/training.md) · [Battle simulator](training/simulator.md) ·
  [Random armies](training/armies.md) · [Unit passports](training/units.md) ·
  [Measurements](training/measurements.md) · [In-game check](launch/gate.md)

## Game mechanics

- [What is verified in WH3](game/README.md) — map, units, map catalogue, how we measured it.
- [Morale](game/units/morale.md) · [Melee](game/units/melee.md) · [Pace and fatigue](game/units/pace.md) —
  battle rules checked against recorded battles.
- [The game's database](game/database.md) · [Battle difficulty](game/difficulty.md) · [The game's own battle AI](game/game-ai.md) —
  rule tables, difficulty bonuses, how the game's own AI fights.
- [MP Crossroads (flat)](game/maps/crossroads-flat.md) — the arena's empty flat map.
- [Readout catalogue](game/readouts.md) — what can be collected about units and the battle, what we already collect.
- [Visual atlas](game/atlas.md) — the key research images.

## Infrastructure

### Code in the game (`src`)

| App | Purpose |
|---|---|
| [core](apps/core.md) | Value checks, JSON, error codes, time |
| [battle](apps/battle.md) | Sides, armies, units, attacker/defender roles, speed, deadline and stall |
| [map](apps/map.md) | Radar frame, grid, heights, ground, objects |
| [navigation](apps/navigation.md) | Point reachability for a specific unit, the end of a leg |
| [units](apps/units.md) | Own unit state, missile range, the unit card, soldier positions |
| [intel](apps/intel.md) | Enemy visibility and last known position |
| [observation](apps/observation.md) | One side's view — without hidden data |
| [orders](apps/orders.md) | Verified order calls, handing units to the game's AI, engine facings |
| [telemetry](apps/telemetry.md) | JSONL events, state files, post-battle samples |
| [Entries](apps/entries.md) | Probes and battle recorders: `nn_arena`, `ai_vs_ai`, `unit_readout`, `move_probe`, `archer_range`, `enemy_layout`, `roster_capture`, `manual_record`, `map_capture` |

### Project and rules

- [Project structure and layers](architecture/overview.md) — folders, layers, dependency rules, the tick flow, project rules.
- [List of apps](architecture/apps.md) — apps, entries and tools, where they came from.
- [Errors and unknown values](architecture/error_handling.md) — `known/unknown`, error codes, guarded callbacks.
- [How to keep the docs](architecture/documentation.md) — language mirror, hubs, navigation, generated indexes.

### Environment, build, launch, tests

- [Setup and configuration](environment/setup.md) — Python, `.venv`, `config/local.json`, game path.
- [Building a pack](launch/build.md) — build targets, options, module bundler, PFH5 format.
- [Running a battle and results](launch/run.md) — launcher, fair difficulty, result folders, True Sight mod.
- [Tests without the game](testing/tests.md) — pytest + lupa, fake `bm`, how to write tests.

## Research

- [Research archive](research/README.md) — map-data and launch reports, evidence and legacy scripts.

## All sections

[Game knowledge](game/README.md) · [Training data](training/README.md) ·
[Code modules](apps/README.md) · [Architecture](architecture/README.md) ·
[Environment](environment/README.md) · [Build and launch](launch/README.md) ·
[Testing](testing/README.md) · [Research](research/README.md)
