# telemetry

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/telemetry.md)

Recording what happened in a battle. Code: `src/apps/telemetry/`.

## adapter — files

| Function | What it does |
|---|---|
| `sink(path, stamp?)` | Returns `emit(event, fields)`: one JSONL row with `event`, `wall_iso` and fields from `stamp(row)` |
| `buffered_sink(path, stamp?)` | The same rows kept in memory; `flush()` writes them with one file open. For frequent rows: flush once a tick and at the end of the battle |
| `read_file`, `write_file`, `append` | Small state files in the game folder |
| `next_sequence(path)` | On-disk counter: unique series ids without the game's random numbers |

`sink` reopens the file for every row, so a game crash never loses buffered
lines.

```lua
local telemetry = require('apps.telemetry.adapter')
local emit = telemetry.sink('tww3_bai_events.jsonl', function(row)
    row.build, row.model_ms = config.build, bm:time_elapsed_ms()
end)
emit('result', {status = 'completed', winner = 1})
```

## sampler_adapter — post-battle samples

A sample of both sides for analysis after the battle: positions,
visibility for each side, CCO health and morale, ammo and its use, target,
ordered position, distances between attackers and damaged units.

> **Never an input for a side's AI.** The sample also reads hidden enemies.

| Function | What it does |
|---|---|
| `new(sides, cco)` | Registry `unit → roster id`, falls back to `unique_ui_id` |
| `sample(state, time_ms)` | One telemetry frame |
| `entity_hit(state, event, time_ms)`, `drain_hits(state)` | Hits from a game event |

<a id="observer_adapter"></a>

## observer_adapter — everything about a human's battle

Extra recording of a battle a human plays ([nn_arena](entries.md#nn_arena), mode `human`): everything
the game gives about every unit of both sides beyond the `nn_sample` row. Full view, never an AI's input.

| Function | What it does |
|---|---|
| `new(opts)` | `units` (name, unit, side), `alliances`, `cco`, `emit`, `now_ms`, `soldiers_every` |
| `decorate(row)` | Adds to a unit's row: `ob`, `ow` — bearing and width of the order in force; `idle` — the unit has no order; `v` — visible to the other side; `uma`, `td`, `dir` — under missile fire, taking damage, damage inflicted recently (CCO) |
| `changes()` | On change: `nn_effects` (a unit's active effects) and `nn_ability_ready` (an ability's readiness; a use is true -> false), both sides |
| `soldiers()` | Every `soldiers_every`-th call: `nn_soldiers`, every soldier's place, dm |

## Outside the game

- [tools/telemetry/read_jsonl.ps1](../../../tools/telemetry/read_jsonl.ps1) —
  reads the log while it is written (used by the launcher).
- Run results — [running a battle](../launch/run.md#results).
