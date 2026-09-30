# Running a battle and results

[← Back](README.md) · [Documentation](../README.md) · [Build](build.md) · [Русский](../../ru/launch/run.md)

The launcher installs a built pack in the game, starts one battle, waits for the
outcome, copies the results and cleans up only its own files.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target nn-arena
```

Options:
`-Target ai-vs-ai|unit-readout|move-probe|manual|roster-capture|archer-range|enemy-layout|nn-arena|map-capture`
(required; the target must be [built](build.md) first),
`-TimeoutSeconds` (default 240 s for loading + the battle's deadline `deadline_s` from
`manifest.json`; 1200 without one), `-KeepGameOpen` (leave the game running after completion).

> **Status:** the launcher was verified in game on 2026-09-27 with the
> `ai-vs-ai` target: load, events, result copy and cleanup of its own files worked.
> Since 30.09.2026 it has recorded `nn-arena` battles at Normal difficulty.

## What the launcher does

1. Reads `config/default.json` and `config/local.json`, finds the game.
2. Checks that WH3 is not running and no files of a previous run remain.
3. Checks dependencies (SHA-256), then copies them and our pack into `data`
   and verifies the installed pack hash.
4. Writes its **own** mod list `tww3_bai_<target>_mods.txt`. The user's
   `used_mods.txt` is never touched.
5. Sets Normal difficulty for the battle ([below](#fair-difficulty)).
6. Starts `Warhammer3.exe game_startup_mode battle <scenario>; <mods>;` and
   records the PID.
7. Reads the event log while the game writes it and prints key lines.
8. Waits for completion: `result` (`probe_done` for `map-capture`). Stops on
   `error` / `probe_error` / `skipped`, process exit, a new crash report or timeout.
9. Copies results and stops **only its own** process (same PID, path and
   start time).
10. Removes its pack, its mod list and its dependency copies — only when
    unchanged; puts the game's preferences file back.

## Results

`build/<target>/runs/<YYYYMMDD-hhmmss>/`:

| File | Contents |
|---|---|
| `manifest.json` | The build that played |
| `launch.json` | PID, time, arguments, log offset, battle difficulty |
| `events.jsonl` | New events of this run |
| `status.json` | `completed` / `lua_error` / `timeout` / `process_exited` / `crash_report`, cleanup and preferences outcome |
| `tww3_bai_map_capture_*.{csv,xml,jsonl}` | For `map-capture`: grid, battle XML, events |

The event log in the game folder (`tww3_bai_events.jsonl`) is shared by every target
but `map-capture` and is only appended; the launcher copies the new part.

## Fair difficulty

**Every battle against the game's AI runs at Normal difficulty** (the user's decision,
30.09.2026): on Very Hard the AI gets bonuses (morale, reload speed; in `_kv_morale_tables` and
`_kv_rules_tables`), and a future AI is meant to play people, who get no such bonuses. For the run the launcher sets
`battle_difficulty 1` (0 easy, 1 normal, 3 very hard) in
`%APPDATA%\The Creative Assembly\Warhammer3\scripts\preferences.script.txt` and afterwards puts
the user's file back byte for byte; a copy sits in `build/preferences.script.backup.txt` while
the battle runs and is restored even after an interrupted run (by the next one). The difficulty
is in `launch.json` (`battle_difficulty`, `user_battle_difficulty`), the result in `status.json`
(`preferences_restored`). A battle's fairness is checked only by `battle_difficulty` = 1 in
`launch.json`: the starting morale is the same at both difficulties
([battle difficulty](../game/difficulty.md)). The arena's analysis takes only such battles
(`tools/nn/gamedata.runs`).

## Required mod: True Sight

**Every battle of the project runs with [True Sight: Improved Line of Sight](https://steamcommunity.com/sharedfiles/filedetails/?id=3628832922)**
(Workshop 3628832922, by GunPawDa). There is no vanilla mode.

- The version is pinned in [config/mod-dependencies.json](../../../config/mod-dependencies.json):
  `true_sight.pack`, 1065 bytes, SHA-256 `790c54d3…ac511`, profile `true-sight-v1`.
- Every build writes `true_sight.pack` into our pack header and into
  `manifest.json` (`mod_profile`, `dependencies`).
- The launcher refuses a build without the mod, and stops when the mod is not
  downloaded or its hash changed.
- Before a run the launcher copies `true_sight.pack` from the Workshop folder
  into `data` if absent, never overwrites a different file, lists it before
  our pack and removes only its own copy afterwards.

Subscribe to the mod in Steam Workshop once. If the author releases an
update, the hash no longer matches and runs stop: review the new version and
record it in `mod-dependencies.json`.
Decision history (Russian): [dependency research](../../ru/research/launch/dependencies.md).

## Timing

From the research measurements: about 83 s to the first `ready`, about 11 s
to reload after a rematch, 7–8 s per battle at ×20
([measurements, Russian](../../ru/research/launch/loading-times.md)). An `nn-arena` battle
takes about 2 minutes with loading (loading ~90 s, the battle 25–45 s real time at ×20).
Every run is one battle now; a series of battles in one process would save the loading.

**What reading the game from Lua costs** (27.09.2026, the probes were removed in
the reset of 30.09.2026):

- Reading all the armies' soldiers (about 2000 calls) together with working out
  groups and the battlefield took 41–47 ms; in another battle one read of all soldiers took about
  50 ms a tick.
- At ×20 a tick of 1 s of game time is 50 ms of real time. A script that reads
  every soldier every tick fills the whole tick, and the battle runs in jerks.
  Do such reads less often (for example once in 5 s, when they are not noticed).
- Querying the map grid in battle: [surface sampling](../game/map/sampling.md#querying-during-battle-the-hamlet-on-the-moorlands-route).

## Limits

- Graphics presets before a run (minimum quality, Ultra unit size in the
  research) are not ported yet: the script for them was missing from the kit.
