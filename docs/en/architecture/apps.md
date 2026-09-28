# List of apps

[Documentation](../README.md) · [Project structure](overview.md) · [Русский](../../ru/architecture/apps.md)

The code was moved from the `lua-knowledge-kit` research set. Module
behaviour is preserved: the 12 original sensor tests were ported with their
checks unchanged and pass. The original sources are kept in
[research/scripts/legacy-lua](../../../research/scripts/legacy-lua/).

## In-game apps (`src/apps`)

| App | Files | What it does | Moved from |
|---|---|---|---|
| [core](../apps/core.md) | `value`, `json`, `errors`, `clock` | Safe value reads, telemetry JSON, error codes, guarded callbacks | Helpers duplicated in `state`, `range`, `visibility`, `post_battle_telemetry`, `harness`, `arena`, `capture` |
| [battle](../apps/battle.md) | `adapter`, `services` | Sides, armies, units, unsupported battle kinds, attacker/defender roles | `runtime/battle_role.lua`, start of `harness.lua` |
| [map](../apps/map.md) | `adapter`, `services` | Radar frame, grid, height, ground, cell clearance, buildings and CCO structures | `map/reader.lua`, `map/objects.lua` |
| [navigation](../apps/navigation.md) | `adapter`, `diagnostics_adapter`, `services` | Cell reachability for a unit, `can_reach_position` diagnostics | `map/reachability.lua`, `runtime/reach_observation.lua` |
| [assessment](../apps/assessment.md) | `services` | Force assessment from both armies: features for strategies | new |
| [strategy](../apps/strategy.md) | `services`, `contract` | Strategy catalogue, conditions, roles, choice with its reasons | new |
| [formation](../apps/formation.md) | `services` | Formation layouts by role: `line_and_blocks` from roster data | new |
| [plan](../apps/plan.md) | `services` | Start-of-battle chain: assessment → strategy → formation | new |
| [vision](../apps/vision.md) | `services` | Groups of visible enemies; the main army is the strongest group | new |
| [battlefield](../apps/battlefield.md) | `services` | Battlefield between the main groups: axis, front lines, 40 m margin | new |
| [alignment](../apps/alignment.md) | `services` | Aligning the army opposite the enemy's main group (no approach) | new |
| [mask](../apps/mask.md) | `services`, `adapter` | "Can we stand here?" mask over the battlefield, 3 m cells | new |
| [approach](../apps/approach.md) | `services` | Approach in 50 m steps to the stop line; one action at a time, alignment first | new |
| [reach](../apps/reach.md) | `services` | Who reaches whom with missiles (from the block's middle); the window to stand in | new |
| [missile](../apps/missile.md) | `services` | Missile damage from the measurement; a step of a missile exchange | new |
| [logistics](../apps/logistics.md) | `services` | Queues past an obstacle: who goes round, which side, in what order and width; can be switched off | new |
| [units](../apps/units.md) | `state_adapter`, `range_adapter`, `card_adapter`, `formation_adapter`, `contract` | 68 own-unit state fields, missile range, unit card and profile, movement and soldiers, formation width bounds | `units/state.lua`, `units/range.lua` |
| [intel](../apps/intel.md) | `adapter`, `services` | Enemy visibility, last known position memory | `visibility/reader.lua` |
| [orders](../apps/orders.md) | `contract`, `adapter`, `planner_adapter` | Command shape and validation, order calls verified in battle | `validate` from `policy_host.lua`, calls from `harness`/`arena` and `commands.md` |
| [deployment](../apps/deployment.md) | `contract`, `services`, `adapter` | Placement contract v2, "collect → validate → apply → verify" transaction | `runtime/deployment_v2.lua` |
| [ai](../apps/ai.md) | `contract`, `services`, `policies/` | Army profiles, decision contract, duel policies | `profile` from `policy_host.lua`, `policy.lua`, `delayed_melee.lua` |
| [observation](../apps/observation.md) | `adapter`, `services` | One side's view: own units in full, only permitted enemy data; leak self-check | new |
| [sandbox](../apps/sandbox.md) | `services` | Loads third-party policies into a restricted environment with an instruction budget | `policy_host.lua` |
| [telemetry](../apps/telemetry.md) | `adapter`, `sampler_adapter` | JSONL events, state files, post-battle samples of both sides | `emit` from harness, `runtime/post_battle_telemetry.lua` |

## Entry points (`src/entries`)

| Entry | Build target | Scenario | What it does |
|---|---|---|---|
| [duel](../apps/entries.md#duel) | `duel` | `ranged_melee.xml` | One unit per side, forced melee, a series with automatic rematches |
| [arena](../apps/entries.md#arena) | `arena` | `triple_melee.xml` | Three pairs at once on one map |
| [ai_vs_ai](../apps/entries.md#ai_vs_ai) | `ai-vs-ai` | `ai_vs_ai.xml` | Both armies under the game's AI, our code only observes |
| [unit_readout](../apps/entries.md#unit_readout) | `unit-readout` | `unit_readout.xml` | Live check of every documented unit readout |
| [move_probe](../apps/entries.md#move_probe) | `move-probe` | `move_probe.xml` | One unit through a plan of legs: formation and plain orders, soldier positions |
| [roster_capture](../apps/entries.md#roster_capture) | `roster-capture` | `roster_capture.xml` | Unit cards and formation for the roster |
| [formation_probe](../apps/entries.md#formation_probe) | `formation-probe` | `formation_probe.xml` | Army formation in battle from `apps.formation`, archer turns |
| [enemy_layout](../apps/entries.md#enemy_layout) | `enemy-layout` | `enemy_layout.xml` | How the game AI stands in defence with different armies |
| [manual_record](../apps/entries.md#manual) | `manual` | `manual_hamlet.xml` | Player's manual battle with recording |
| [map_capture](../apps/entries.md#map_capture) | `map-capture` | `map_capture.xml` | Map grid, objects and reachability to CSV/JSONL |

## Tools outside the game (`tools`)

| Tool | What it does | Moved from |
|---|---|---|
| `tools/build.py`, `tools/pack/` | Module bundler, PFH5 writer, manifest | `game-launch/build.py`, `map-capture/build.py` (paths fixed, same packing logic) |
| `tools/launcher/launch.ps1` | Install, launch, wait, clean up | Logic of `map-capture/launch.ps1` + `finish.ps1`; the old scripts began with `throw` |
| `tools/telemetry/read_jsonl.ps1` | Read JSONL while the game writes it | `telemetry/telemetry.ps1` |
| `tools/analysis/` | Height map, slopes, passages | `map-capture/heightmap.py`, `slopes.py`, `passages.py` |

## Not moved into working code

- `deployment.lua` (v1) — archive only; `deployment-placement-v2` is current.
- One-off research scripts (`map-analysis`, `unit-state`, `unit-range`,
  `unit-actions`) — in [research/scripts](../../../research/scripts/). They
  refer to the old layout and exist to reproduce archived experiments.
- Tournament, agents, task queues — already excluded in the kit.
