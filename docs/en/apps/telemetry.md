# telemetry

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/telemetry.md)

Recording what happened in a battle. Code: `src/apps/telemetry/`.

## adapter — files

| Function | What it does |
|---|---|
| `sink(path, stamp?)` | Returns `emit(event, fields)`: one JSONL row with `event`, `wall_iso` and fields from `stamp(row)` |
| `read_file`, `write_file`, `append` | Small state files in the game folder |
| `next_sequence(path)` | On-disk counter: unique series ids without the game's random numbers |

The file is reopened for every row, so a game crash never loses buffered
lines.

```lua
local telemetry = require('apps.telemetry.adapter')
local emit = telemetry.sink('tww3_bai_events.jsonl', function(row)
    row.build, row.model_ms = config.build, bm:time_elapsed_ms()
end)
emit('decision', {side = 1, action = 'attack', reason = 'initial_charge'})
```

## sampler_adapter — post-battle samples

A sample of both sides for analysis after the battle: positions,
visibility for each side, CCO health and morale, ammo and its use, target,
ordered position, distances between attackers and damaged units.

> **Never a policy input.** The sample also reads hidden enemies.

| Function | What it does |
|---|---|
| `new(sides, cco)` | Registry `unit → roster id`, falls back to `unique_ui_id` |
| `sample(state, time_ms)` | One telemetry frame |
| `entity_hit(state, event, time_ms)`, `drain_hits(state)` | Hits from a game event |

## Outside the game

- [tools/telemetry/read_jsonl.ps1](../../../tools/telemetry/read_jsonl.ps1) —
  reads the log while it is written (used by the launcher).
- Run results — [running a battle](../launch/run.md#results).
