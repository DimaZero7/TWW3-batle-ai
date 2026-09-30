# Entry points

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/entries.md)

An entry wires apps into a specific battle: it subscribes to phases, runs
the tick, calls adapters and services and writes telemetry. It contains no
decision logic: every entry either measures the game or records a battle.
Code: `src/entries/`. Each entry is a module with `main(bm, config, globals)`,
called by the bundled script.

<a id="ai_vs_ai"></a>

## ai_vs_ai — two game AIs

Build: `python -m tools.build ai-vs-ai --speed 3`. Scenario `ai_vs_ai.xml`:
4 Kislev units per side (2 Kossars, Tzar Guard, Winged Lancers). Our code
makes no battle decisions and gives units no orders directly.

- The army the engine treats as AI stays with the **general battle AI**.
- The army the engine treats as the player's (`army:is_player_controlled()`)
  goes to the game's AI planner ([planner_adapter](orders.md)) with "attack the
  enemy force", re-issued every 15 s. Units that rally after a rout or stand idle
  under fire are brought back as in [nn_arena](#nn_arena) (`rejoined`, `idle_kick`).
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

<a id="move_probe"></a>

## move_probe — movement of one unit

Build: `python -m tools.build move-probe --plan hamlet`. Scenario
`move_probe.xml` on **The Moorlands Route** (`catchment_03`): unit
`probe_spears` (Empire spearmen) and a far general `far_general` held by the
script.

The plan `config/move-plans/<name>.json` is a list of legs:

- `shape` — reform in place at another width;
- `traverse` — one plain order to a far point.

A leg: teleport to the start, 2 s to settle, one `goto_location_angle_width`,
then every second `move_sample` (unit movement and every soldier's position
in decimetres) until `leg_end` with reason `arrived`, `stopped`, `stuck` or
`timeout` (`apps.navigation.services`).

Analysis: `tools/analysis/move_probe.py`; results — [hamlet](../../../research/analysis/hamlet/README.md) (in Russian).

<a id="roster_capture"></a>

## roster_capture — roster capture

Build: `python -m tools.build roster-capture`; the scenario `roster_capture.xml`
is generated from `config/roster/capture.json`. In deployment every unit
writes `unit_card` (card, profile); then every unit with more than one man
reforms in place at its own slot for each width in the list, all at once —
`shape_result` with soldier positions and timing. Result: `tools/roster.py
update`, see [unit roster](../game/units/roster.md).

<a id="archer_range"></a>

## archer_range — when archers shoot and how much damage

Build: `python -m tools.build archer-range [--range-mode fire_at_will|attack|damage]`;
the scenario `archer_range.xml` is written by `tools/archer_range.py`. The Moorlands Route,
four lanes 200 m apart: in each an Empire archer unit and in front of it a spearmen unit as
the target; the script holds them all.

- `fire_at_will` — archers of different widths (60, 30, 15, 8 m — 8 to 46 m deep) stand with
  fire at will, the target steps closer;
- `attack` — the target stands, the archers are ordered to attack it (they walk into range);
- `damage` — archers 20 m wide, a fearless target at its own distance (70–120 m); the archers
  shoot until out of arrows. `--damage-rotate N` shifts the distances by N lanes (the same
  distance on other ground).

Every tick `range_sample` per lane: arrows, the engine's firing flag, `unit_in_range`, the
archers' front and rear ranks and the target's nearest rank (from the soldiers), men and
health. Analysis: `tools/analysis/archer_range.py`; results — [missile range](../game/units/missile-range.md)
and [missile damage](../game/units/missile-damage.md).

<a id="enemy_layout"></a>

## enemy_layout — how the game AI deploys and stands

Build: `python -m tools.build enemy-layout --layout <name> [--enemy-mode native|defend]`;
scenario from `config/armies/defender_layouts.json` (`tools/enemy_layout.py`). Our
army stands, held by the script; the enemy is the game AI, which deploys its army itself
when deployment ends. In `defend` mode (default) it gets
`script_ai_planner:defend_position` after deployment (where it stands, 80 m radius);
`native` — the AI as the battle sets it.

For 90 game seconds: `enemy_sample` every tick (every unit's movement and whether our side
sees it), `enemy_snapshot` with every soldier — in deployment, after it and at the end,
`own_snapshot` at the end. This is the full view, for research. Analysis:
`tools/analysis/enemy_layout.py`; results — [how it stands](../../../research/analysis/enemy-layout/README.md) (in Russian).

<a id="manual"></a>

## manual_record — manual battle with recording

Build: `python -m tools.build manual --deadline 3600 --stall-minutes 30`.
Scenario `manual_hamlet.xml`: six spearmen units near the hamlet under the
player's command. The script never commands them, it only records:
`own_sample` every second (movement and soldiers of all units), `order_seen`
when the ordered point changes, `order_end` (`arrived`, `stopped`, `stuck`,
`timeout`). The player ends the battle; otherwise after an hour or 30 game
minutes without damage.

<a id="nn_arena"></a>

## nn_arena — recording battles to train a network

Build: `python -m tools.build nn-arena --own-ai attack|defend --timeout 900`.
`--timeout 900` is a battle limit of 900 s of game time, as in the records of 30.09.2026;
without it the build gives 600 s.
Scenario `nn_arena.xml` is written by `tools/nn/scenario.py` from `config/nn/arena.json`: the
empty flat map MP Crossroads (flat), each side a lord, 4 spearmen and 2 archers of the Empire,
the armies' fronts 350 m apart. The script makes no battle decisions: side 2 is always led by
the game's general battle AI, side 1 by CA's planner ([planner_adapter](orders.md)). The script
repeats the planner's task, sends rallied and idle units back into battle (below) and records
the battle. The side that wins on timeout defends:

- `attack` — the planner attacks (the order repeated every 15 s), the game's AI defends;
- `defend` — the planner defends where the army stands, the game's AI attacks.

Every second `nn_sample` records every unit of both sides (the full view):

| Field | What it is |
|---|---|
| `n`, `side` | The unit's script name (`own_lord`, `enemy_spear_1`, …) and side |
| `x`, `z`, `b` | Place, m, and facing, ° |
| `men`, `hp` | Men alive, share of health |
| `mp`, `ms` | Morale from CCO: `MoralePercent` and `MoraleState` |
| `r`, `s`, `w` | Routing, shattered, wavering |
| `m`, `mv`, `f` | In melee, moving, running |
| `a`, `fire` | Arrows left, firing now |
| `t` | Current target (a script name) |
| `fat`, `k` | Fatigue, kills |
| `ox`, `oz` | Ordered position |
| `lf`, `rf`, `bf` | Threat to the left flank, right flank, rear |

The end: the last snapshot `nn_final` and `result` — the outcome, men and health of each side,
how many `rejoined` and `idle_kick` there were. A battle takes about 2 minutes at ×20 including
loading. Only battles at Normal difficulty count in analysis ([fair difficulty](../launch/run.md#fair-difficulty)).
The records are read by `tools/nn/gamedata.py` — [data for training](../training/README.md).

CA's planner has two weaknesses the script mends (30.09.2026):

- a unit that routed and rallied is no longer led by the planner — it stood to the end; now it
  goes back into the planner with the last order (`rejoined`);
- a unit stands idle under arrows (no target, not moving, not in melee, losing health): 15-75%
  of the time our units were shot at, 0-3% for the game's AI. After 6 s idle it goes back into
  the planner; if it again stands 6 s under fire, but no sooner than 15 s after the first kick,
  it gets a planner of its own that attacks the nearest enemy (`idle_kick`, stages 1 and 2).

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
  = 240 s for loading + the battle's deadline. Build options: `--deadline`,
  `--stall-minutes`.

- **Speed.** Once the outcome is decided and units flee, the engine drops the
  speed to ×1 by itself. `battle.speed_guard` restores the requested speed
  every 0.5 s until the `Complete` phase and writes `speed_restored` (verified
  on 2026-09-27 at ×20: 1 → 20 right after `result`). A pause (speed 0) is left alone.

- Reload guard: a global flag `tww3_bai_<entry>` (for example
  `tww3_bai_nn_arena`, `tww3_bai_map_capture`).
- Every callback goes through `errors.guard` ([errors](../architecture/error_handling.md)).
- Checked without the game in `tests/entries/test_entries.py` on a fake `bm`.
