# List of apps

[← Back](README.md) · [Documentation](../README.md) · [Project structure](overview.md) · [Русский](../../ru/architecture/apps.md)

The base code: apps in the game, entry points and tools outside the game. Most of it
was moved from the `lua-knowledge-kit` research set: module behaviour is preserved, the
12 original sensor tests were ported with their checks unchanged and pass. The original
sources are kept in [research/scripts/legacy-lua](../../../research/scripts/legacy-lua/).

**Reset of 30.09.2026** (the project owner's decision): our algorithmic AI — the strategy, tactics
and formation modules, the AI tree, the policies of the former `duel` and `arena` entries, the policy sandbox,
planned deployment, the battle simulator and the viewer — was removed with its docs and
tests. The old work is in the Git history (the last commit before the reset is `242c3b1`).

## In-game apps (`src/apps`)

| App | Files | What it does | Moved from |
|---|---|---|---|
| [core](../apps/core.md) | `value`, `json`, `errors`, `clock` | Safe value reads, telemetry JSON, error codes, guarded callbacks | Helpers duplicated in the kit's old scripts (`state`, `range`, `visibility`, `post_battle_telemetry`, `capture`) |
| [battle](../apps/battle.md) | `adapter`, `services` | Sides, armies, units, unsupported battle kinds, attacker/defender roles, speed, deadline and stall | `runtime/battle_role.lua` |
| [map](../apps/map.md) | `adapter`, `services` | Radar frame, grid, height, ground, cell clearance, buildings and CCO structures | `map/reader.lua`, `map/objects.lua` |
| [navigation](../apps/navigation.md) | `adapter`, `diagnostics_adapter`, `services` | Cell reachability for a unit, `can_reach_position` diagnostics, the end of a leg (`arrived`/`stopped`/`stuck`/`timeout`) | `map/reachability.lua`, `runtime/reach_observation.lua` |
| [units](../apps/units.md) | `state_adapter`, `range_adapter`, `card_adapter`, `formation_adapter` | 68 own-unit state fields, missile range, unit card and profile, movement and soldiers | `units/state.lua`, `units/range.lua` |
| [intel](../apps/intel.md) | `adapter`, `services` | Enemy visibility, last known position memory | `visibility/reader.lua` |
| [observation](../apps/observation.md) | `adapter`, `services` | One side's view: own units in full, only permitted enemy data; leak self-check | new |
| [orders](../apps/orders.md) | `adapter`, `planner_adapter`, `facing` | Order calls verified in battle, handing units to CA's planner, the engine's 64 facings | [verified commands](../game/units/commands.md) |
| [telemetry](../apps/telemetry.md) | `adapter`, `sampler_adapter` | JSONL events, state files, post-battle samples of both sides | `runtime/post_battle_telemetry.lua` |

## Entry points (`src/entries`)

Every entry measures the game or records a battle; none makes decisions of its own.

| Entry | Build target | Scenario | What it does |
|---|---|---|---|
| [ai_vs_ai](../apps/entries.md#ai_vs_ai) | `ai-vs-ai` | `ai_vs_ai.xml` | The AI army under the game's general battle AI, the player's army under CA's planner; our code makes no battle decisions: it repeats the planner's task, sends rallied and idle units back into battle and records the battle |
| [unit_readout](../apps/entries.md#unit_readout) | `unit-readout` | `unit_readout.xml` | Live check of every documented unit readout and visibility |
| [move_probe](../apps/entries.md#move_probe) | `move-probe` | `move_probe.xml` | One unit through a plan of legs: formation and plain orders, soldier positions |
| [roster_capture](../apps/entries.md#roster_capture) | `roster-capture` | `roster_capture.xml` | Unit cards and formation for the roster |
| [archer_range](../apps/entries.md#archer_range) | `archer-range` | `archer_range.xml` | When archers start shooting by the depth of their block, arrow damage by distance |
| [enemy_layout](../apps/entries.md#enemy_layout) | `enemy-layout` | `enemy_layout.xml` | How the game AI deploys and stands with different armies |
| [manual_record](../apps/entries.md#manual) | `manual` | `manual_hamlet.xml` | Player's manual battle with recording |
| [nn_arena](../apps/entries.md#nn_arena) | `nn-arena` | `nn_arena.xml` | Records the game's AI battles in the arena: every unit of both sides once a second — data for training |
| [map_capture](../apps/entries.md#map_capture) | `map-capture` | `map_capture.xml` | Map grid, objects and reachability to CSV/JSONL |

## Tools outside the game (`tools`)

| Tool | What it does |
|---|---|
| `tools/build.py`, `tools/pack/` | Module bundler, PFH5 writer, manifest ([build](../launch/build.md)) |
| `tools/launcher/launch.ps1` | Install, launch at fair difficulty, wait, clean up ([running](../launch/run.md)) |
| `tools/telemetry/read_jsonl.ps1` | Read JSONL while the game writes it |
| `tools/config.py`, `tools/lua_runtime.py` | Paths and machine settings; Lua 5.1 in Python for the tests |
| `tools/roster.py`, `tools/readouts.py` | The unit roster from runs; the readout catalogue and collection profiles |
| `tools/archer_range.py`, `tools/enemy_layout.py` | Scenarios and settings of the `archer-range` and `enemy-layout` probes |
| `tools/analysis/` | Height map, slopes, passages, passability, obstacles; reports of `move-probe`, `unit-readout`, `archer-range`, `enemy-layout` runs |
| `tools/nn/` | Data for training: rules from the game's database (`gamedb.py`), the arena's scenario (`scenario.py`), arena records as arrays (`gamedata.py`), running in a PyTorch container (`dock.sh`) — [more](../training/README.md) |
| `tools/docs/` | Indexes of the documentation hubs (`python -m tools.docs.index_doc`) |
| `tools/architecture.py` | The level of every app (all are the base today), for the architecture and docs tests |

The launcher and the build grew out of the old `map-capture/launch.ps1`, `finish.ps1`
and `build.py` (paths fixed, same packing logic); they are kept in
[research/scripts/legacy-launch](../../../research/scripts/legacy-launch/).

## Not moved into working code

- Deployment `deployment.lua` (v1) and `deployment_v2.lua` — archive only, in
  [research/scripts/legacy-lua](../../../research/scripts/legacy-lua/).
- One-off research scripts (`map-analysis`, `unit-state`, `unit-range`,
  `unit-actions`) — in [research/scripts](../../../research/scripts/). They
  refer to the old layout and exist to reproduce archived experiments.
- Tournament, agents, task queues — already excluded in the kit.
