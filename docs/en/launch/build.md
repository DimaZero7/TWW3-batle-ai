# Building a pack

[Documentation](../README.md) · [Running](run.md) · [Русский](../../ru/launch/build.md)

Building never touches the game; it only writes `build/<target>/`.

```bash
.venv/Scripts/python -m tools.build duel --runs 3 --speed 20 --timeout 300
.venv/Scripts/python -m tools.build arena
.venv/Scripts/python -m tools.build map-capture --step 3 --features
```

## Targets

| Target | Entry | Scenario | Pack |
|---|---|---|---|
| `duel` | `entries.duel` | `scenarios/ranged_melee.xml` | `tww3_bai_duel.pack` |
| `arena` | `entries.arena` | `scenarios/triple_melee.xml` | `tww3_bai_arena.pack` |
| `ai-vs-ai` | `entries.ai_vs_ai` | `scenarios/ai_vs_ai.xml` | `tww3_bai_ai_vs_ai.pack` |
| `map-capture` | `entries.map_capture` | `scenarios/map_capture.xml` | `tww3_bai_map_capture.pack` |

## Options

| Option | Targets | Meaning |
|---|---|---|
| `--runs 1..10` | duel | Battles in a row in one game process (automatic rematch) |
| `--speed 1/3/10/20` | duel, arena, ai-vs-ai | Battle speed |
| `--timeout 30..1800` | duel, arena, ai-vs-ai | Model-time limit per battle, seconds |
| `--tick-ms` | duel, arena | Decision period, ms (default 1000) |
| `--step 1/2/3/5` | map-capture | Grid cell size, m |
| `--features` | map-capture | After deployment also read objects and cell reachability |

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
    __require("entries.duel").main(bm, {runs = 3, ...}, {common = common, battle_vector = battle_vector})
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
