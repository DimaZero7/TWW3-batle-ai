# Running a battle and results

[Documentation](../README.md) · [Build](build.md) · [Русский](../../ru/launch/run.md)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target duel
```

Options: `-Target duel|arena|map-capture` (required), `-TimeoutSeconds`
(default 1200, including game load), `-KeepGameOpen` (leave the game running
after completion).

> **Status:** the launcher is assembled from the verified logic of the
> research scripts but has not yet started the game in this project. The
> first run is a test of the stand itself.

## What the launcher does

1. Reads `config/default.json` and `config/local.json`, finds the game.
2. Checks that WH3 is not running and no files of a previous run remain.
3. Checks dependencies (SHA-256), then copies them and our pack into `data`
   and verifies the installed pack hash.
4. Writes its **own** mod list `tww3_bai_<target>_mods.txt`. The user's
   `used_mods.txt` is never touched.
5. Starts `Warhammer3.exe game_startup_mode battle <scenario>; <mods>;` and
   records the PID.
6. Reads the event log while the game writes it and prints key lines.
7. Waits for completion: `result` (duel, per battle), `arena_complete`,
   `probe_done`. Stops on `error` / `probe_error` / `skipped`, process exit,
   a new crash report or timeout.
8. Copies results and stops **only its own** process (same PID, path and
   start time).
9. Removes its pack, its mod list and its dependency copies — only when
   unchanged.

## Results

`build/<target>/runs/<YYYYMMDD-hhmmss>/`:

| File | Contents |
|---|---|
| `manifest.json` | The build that played |
| `launch.json` | PID, time, arguments, log offset |
| `events.jsonl` | New events of this run |
| `status.json` | `completed` / `lua_error` / `timeout` / `process_exited` / `crash_report` and cleanup outcome |
| `tww3_bai_map_capture_*.{csv,xml,jsonl}` | For `map-capture`: grid, battle XML, events |

The duel/arena log in the game folder (`tww3_bai_events.jsonl`) is shared
and appended; the launcher copies only the new part. `tww3_bai_pending.txt`
and `tww3_bai_sequence.txt` hold the duel series state and are kept.

## Required mods

By default the game runs without third-party mods. To enable **True Sight:
Improved Line of Sight** (Workshop 3628832922) use the example in
`config/local.example.json`. The launcher:

- takes `true_sight.pack` from the Workshop folder and checks its SHA-256;
- copies it into `data` if absent; never overwrites a different file;
- lists it before our pack;
- removes only its own copy after the run.

The pinned mod version: [config/mod-dependencies.json](../../../config/mod-dependencies.json).
Decision history (Russian): [dependency research](../../ru/research/launch/dependencies.md).

## Timing

From the research measurements: about 83 s to the first `ready`, about 11 s
to reload after a rematch, 7–8 s per battle at ×20. That is why the duel
plays a series in one process ([measurements, Russian](../../ru/research/launch/loading-times.md)).

## Limits

- The duel rematch looks for the confirmation text of the **Russian** game
  UI. Another language needs its own text in `src/entries/duel.lua`
  (`REMATCH_PROMPT`).
- Graphics presets before a run (minimum quality, Ultra unit size in the
  research) are not ported yet: the script for them was missing from the kit.
