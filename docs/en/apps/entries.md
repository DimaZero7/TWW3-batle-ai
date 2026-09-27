# Entry points

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/entries.md)

An entry wires apps into a specific battle: it subscribes to phases, runs
the tick, calls adapters and services and writes telemetry. It contains no
decision logic. Code: `src/entries/`. Each entry is a module with
`main(bm, config, globals)`, called by the bundled script.

<a id="duel"></a>

## duel — duel with automatic rematch

Build: `python -m tools.build duel --runs 3`. Scenario `ranged_melee.xml`:
one Kossars unit per side, script names `bai_ranged_a` / `bai_ranged_b`. With
other names the duel accepts a manual "one foot lord per side" battle.

1. `loaded` → battle kind check → read sides → `ready`.
2. `Deployment` phase: end deployment automatically after 1 s.
3. `Deployed` phase: re-read `tww3_bai_policy.lua` (if present), set speed,
   take control, `prepare_melee`.
4. Tick: battle result → timeout → control confirmation → policy decision
   per side → order → `decision` and `snapshot`.
5. `result`: `completed` (winner from the engine), `timeout` (a forced draw,
   never a "natural" one), `incomplete`.
6. With battles left in the series: write `tww3_bai_pending.txt`, end the
   battle, press "rematch" through the UI API and confirm.

Events: `loaded`, `ready`, `initial_unit`, `deployment`, `start`,
`control_requested`, `control_acquired`, `decision`, `snapshot`,
`final_unit`, `result`, `restart_*`, `error`, `skipped`.

<a id="arena"></a>

## arena — three pairs on one map

Build: `python -m tools.build arena`. Scenario `triple_melee.xml`: pairs
`bai_arena_<pair>_<side>` — Kossars 120, Tzar Guard 100, Snow Leopard 1.
Control is taken before deployment; afterwards units return to their XML
positions. A pair ends at the first rout or death. A unit of another pair
within 120 m raises `isolation_warning`. Army morale is shared by all pairs;
every result says so.

<a id="ai_vs_ai"></a>

## ai_vs_ai — two game AIs

Build: `python -m tools.build ai-vs-ai --speed 3`. Scenario `ai_vs_ai.xml`:
4 Kislev units per side (2 Kossars, Tzar Guard, Winged Lancers). Our code
gives no unit orders.

- The army the engine treats as AI stays with the **general battle AI**.
- The army the engine treats as the player's (`army:is_player_controlled()`)
  goes to the game's AI planner ([planner_adapter](orders.md)) with "attack the
  enemy force", re-issued every 15 s.
- Who controls what: the `ai_assigned` event (`general_battle_ai` / `script_ai_planner`).
- Every 5 ticks: per-unit `snapshot` and `progress` (men and standing units
  per side); at the end `final_unit` and `result`.

First run on 2026-09-27 (before True Sight was required): ~30 s load, both
armies engaged on their own; at 237 s side 1 had 258 men and 4 units, side 2
had 147 men and one unit (the game was closed before `result`).

<a id="unit_readout"></a>

## unit_readout — every unit readout and visibility

Build: `python -m tools.build unit-readout --speed 20`. Scenario
`unit_readout.xml` on **The Moorlands Route** (`catchment_03`): Empire vs
Empire; each side has a general, spearmen, archers, a spearmen ambush in the
northern forest and stalk Huntsmen on open grass.

Stages (game seconds): idle 0 → march 8 → ranged 30 → cease fire 75 → melee 85
→ side 1 scouts at 80/40/15 m (140/148/156) → side 2 scouts (164/172/180) →
halt 188 → done 196. Every second:

| Event | What it is |
|---|---|
| `side_view` ×2 | [Side view](observation.md): own units in full, only permitted enemy data, own shooters' range |
| `full_view` | Full summary (ground truth, sees hidden units) |
| `enemy_gate` (every 5 s) | State sensor from the enemy's side: visibility only |
| `unit_profile` (once) | Type, men, range, attributes (incl. `stalk`), behaviours, abilities, rank |

Rows are buffered and written with one file open per tick (~1,100 rows per
battle instead of ~16,000 in the first version, which slowed the game and the
launcher down).

Reports:

```bash
.venv/Scripts/python -m tools.analysis.unit_readout build/unit-readout/runs/<time>
.venv/Scripts/python -m tools.analysis.unit_readout build/unit-readout/runs/<time> --view side --side 1
```

Result on 2026-09-27 (`20260927-142521`, True Sight, ×20, 196 game s ≈ 10 s real):
full summary **168 / 168**, side 1 view **14 / 14**, side 2 **14 / 14**.
Hiding measurements: [visibility](../game/units/visibility.md).

<a id="map_capture"></a>

## map_capture — map collection

Build: `python -m tools.build map-capture --step 3 [--features]`.
Scenario `map_capture.xml`, files prefixed `tww3_bai_map_capture_`.

1. 3 s after load: pause, `ready.xml`, radar frame (`frame`, `corner`),
   grid (`grid_begin`).
2. Batches of 4096 cells → `grid.csv`:
   `ix,iz,x,z,height,clear,ground,inside_radar`.
3. With `--features`: take control of both units, finish deployment at ×20,
   pause, then buildings (`building`), the CCO list (`structure_context`) and
   cell reachability for both sides (`reach_side_1`, `reach_side_2` in the CSV).
4. `probe_done` or `probe_error`.

Process the grid without the game with [tools/analysis](../../../tools/analysis/):
`heightmap.py`, `slopes.py`, `passages.py`. Method: [map](../game/map/README.md).

## Common

- **A battle always ends by itself.** Three safety nets: a real-time
  `deadline_s` (`battle.deadline`, result `deadline`); a stall — nobody takes
  damage for `stall_ms` of game time, 10 minutes by default
  (`battle.services.new_stall_detector`, result `stalled`); the launcher limit
  = 240 s for loading + every battle's deadline. Build options: `--deadline`,
  `--stall-minutes`.

- **Speed.** Once the outcome is decided and units flee, the engine drops the
  speed to ×1 by itself. `battle.speed_guard` restores the requested speed
  every 0.5 s until the `Complete` phase and writes `speed_restored` (verified
  on 2026-09-27 at ×20: 1 → 20 right after `result`). A pause (speed 0) is left alone.

- Reload guard: a global flag (`tww3_bai_duel`, `tww3_bai_arena`,
  `tww3_bai_map_capture`).
- Every callback goes through `errors.guard` ([errors](../architecture/error_handling.md)).
- Checked without the game in `tests/entries/test_entries.py` on a fake `bm`.
