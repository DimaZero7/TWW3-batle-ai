# Building a pack

[← Back](README.md) · [Documentation](../README.md) · [Running](run.md) · [Русский](../../ru/launch/build.md)

Building never touches the game; it only writes `build/<target>/`.

```bash
.venv/Scripts/python -m tools.build nn-arena --own-ai attack --timeout 900
.venv/Scripts/python -m tools.build ai-vs-ai --speed 3
.venv/Scripts/python -m tools.build map-capture --step 3 --features
```

## Targets

| Target | Entry | Scenario | Pack |
|---|---|---|---|
| `ai-vs-ai` | `entries.ai_vs_ai` | `scenarios/ai_vs_ai.xml` | `tww3_bai_ai_vs_ai.pack` |
| `unit-readout` | `entries.unit_readout` | `scenarios/unit_readout.xml` | `tww3_bai_unit_readout.pack` |
| `move-probe` | `entries.move_probe` | `scenarios/move_probe.xml` | `tww3_bai_move_probe.pack` |
| `manual` | `entries.manual_record` | `scenarios/manual_hamlet.xml` | `tww3_bai_manual.pack` |
| `archer-range` | `entries.archer_range` | `scenarios/archer_range.xml` (`tools/archer_range.py`) | `tww3_bai_archer_range.pack` |
| `enemy-layout` | `entries.enemy_layout` | `scenarios/enemy_layout.xml` (from `config/armies/defender_layouts.json`) | `tww3_bai_enemy_layout.pack` |
| `roster-capture` | `entries.roster_capture` | `scenarios/roster_capture.xml` (from `config/roster/capture.json`) | `tww3_bai_roster_capture.pack` |
| `nn-arena` | `entries.nn_arena` | `scenarios/nn_arena.xml` (`tools/nn/scenario.py`) | `tww3_bai_nn_arena.pack` |
| `map-capture` | `entries.map_capture` | `scenarios/map_capture.xml` | `tww3_bai_map_capture.pack` |

What every entry does: [entry points](../apps/entries.md).

## Options

| Option | Targets | Meaning |
|---|---|---|
| `--speed 1/3/10/20` | all but map-capture and manual | Battle speed (default 20) |
| `--timeout 30..1800` | ai-vs-ai, nn-arena | Model-time limit per battle, s (600); the battle's deadline is computed from it. The arena battles of 30.09.2026 were recorded with 900 |
| `--tick-ms` | all but map-capture | Tick period, ms (1000) |
| `--step 1/2/3/5` | map-capture | Grid cell size, m |
| `--features` | map-capture | After deployment also read objects and cell reachability |
| `--window MIN_X MAX_X MIN_Z MAX_Z` | map-capture | Capture only this part of the map |
| `--own-ai attack\|defend` | nn-arena | Our side under CA's planner attacks (the game's AI defends) or defends (the game's AI attacks) |
| `--layout`, `--enemy-mode native\|defend` | enemy-layout | Enemy layout; the game AI as is or told to defend (default) |
| `--range-mode fire_at_will\|attack\|damage` | archer-range | When archers start shooting by the depth of their block; `damage` — damage to a fearless target at 70–120 m |
| `--damage-rotate N` | archer-range `damage` | Shift the distances by N lanes (the same distance on other ground) |
| `--plan` | move-probe | Plan from `config/move-plans/<name>.json` (default `hamlet`) |
| `--capture` | roster-capture | Another unit list instead of `config/roster/capture.json` |
| `--scenario` | all | Another file from `scenarios/` instead of the target's scenario |
| `--deadline` | all but map-capture | Real-time limit per battle, s (default: from the scripted length and speed) |
| `--stall-minutes` | all but map-capture | End the battle when nobody takes damage for this many game minutes (10). In move-probe, roster-capture, archer-range and enemy-layout at least the scenario length + 2 min |

## What `build/<target>/` contains

- `<pack>.pack` — the archive for the game's `data` folder;
- `<script>.lua` — the bundled script, handy to read while debugging;
- `manifest.json` — `build` (short SHA-256 of all inputs), configuration,
  module list, pack SHA-256, scenario path, `syntax_checked`.

`build` is written into every telemetry row, so a log always names the
exact code that played.

## How the bundler works

`tools/pack/bundle.py` starts at the entry and follows every
`require('...')`. Each module becomes a factory function that receives the
bundle's own loader in place of the global `require`:

```lua
__modules["apps.core.value"] = function(require)
    -- source of src/apps/core/value.lua
end
...
if bm then
    __require("entries.nn_arena").main(bm, {own_ai = "attack", ...}, {common = common, battle_vector = battle_vector})
end
```

Modules are therefore plain Lua and load the same way in tests. A circular
`require` fails the build.

## Pack contents

```text
script\battle\mod\<script>.lua          our script; the game loads it in every battle
script\battle\<folder>\scenario.lua     load_script_libraries()
script\battle\<folder>\<scenario>.xml   battle XML
```

The format is uncompressed PFH5 (`tools/pack/pfh5.py`, after RPFM). Entries
are sorted, so equal inputs give equal bytes. The required mod
`true_sight.pack` from `config/mod-dependencies.json` always goes into the pack header.

Because a script in `script\battle\mod\` loads in every battle, **do not add
a test pack to your campaign mod list**. The launcher installs it only for
the duration of a run.
