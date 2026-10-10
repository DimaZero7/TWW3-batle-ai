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

<a id="lord_swarm"></a>

## lord_swarm — a lord surrounded by infantry

Build: `python -m tools.build lord-swarm [--swarm infantry|lords|all] [--repeats N]`; the scenario
`lord_swarm.xml` is written by `tools/nn/lord_swarm.py`. MP Crossroads (flat), two lanes 600 m
apart: the Empire General attacked by Skaven clanrat spearmen and Stormvermin halberds, the
Skaven Warlord by Empire spearmen and halberdiers. Both sides are held by script, fearless. A
trial puts 1-4 attacking units 20 m around each lord (front, back, left, right) and orders them to
attack him; `lords` sends the other lord in too (one lane at a time). A lane's trial ends 40 s after
the lord's first contact; then every unit is healed and the next trial begins.

Every 0.2 s `swarm_sample` per lane: the lord's and each attacker's health (CCO `HealthValue`),
men, kills, melee flag, place; every 1 s `swarm_men`: the attackers' soldiers within 1.5-6 m of
the lord (CCO `ManList` positions). Analysis: `python -m tools.nn.lord_swarm` (`--sim`: the
simulator on the same trials); results — [a lord surrounded](../game/units/lord-swarm.md).
The `damage` plan (`--swarm damage --repeats 4`): one unit in front of each lord in turn - Empire swordsmen
or greatswords on the Warlord, clanrats or Stormvermin on the General; `--blows` - the blows: the lord's
health per drop (one infantry blow, 0.2 s) and the attackers' health and men per lord blow (drops within
0.6 s); `--blows --sim` - the same rates in the simulator.

<a id="charge_probe"></a>

## charge_probe - the melee probe (one mechanic, game and simulator alike)

Build: `python -m tools.build charge-probe --probe-plan charge|hit|move --probe-battle N`; every battle of a plan in
turn: `python -m tools.nn.charge_probe run --plan charge|hit|move [--battles 1,2]` (a build and `launch.ps1` each).
MP Crossroads (flat), 2-5 lanes 240 m apart. In a lane an attacker and a target `gap_m` apart (front to
front); everyone is held by script and fearless. The attacker's order: attack at a run (`attack_run`), attack
at a walk (`attack_walk`), a move order at a run to a point 5 m past the target's front (`move_run`), a
recharge (`recharge`: 10 s into contact a run 40 m back, then attack again), a melee exit (`withdraw`: 10 s
into contact a run 150 m back, the order never changed again) or a script (`script`: at given seconds a facing
order in place - `goto_location_angle_width` with a new bearing - or a move to a point; the turning tests). The
attacker may use its own ability `a_ability_after_s` after contact (`a_ability`: Foe-Seeker). `men_all_s`: the
attacker's soldier places every `men_ms` from the start, wherever the enemy is. The target stands and at its
first second of contact is ordered to attack the attacker (`stand`), stands and is never ordered (`hold`:
braced spears), attacks too (`both`) or stands facing away (`rear`). A lane may have a lord behind the target
who at the moment of contact uses an ability (Stand Your Ground) or not (the control). Unused lords stand far.

Plans: `charge` - 8 battles (swordsmen on clanrats under the four orders; clanrats on braced spearmen, at a
walk, from behind and on swordsmen; the General and the Warlord charging; spearmen on skavenslave spearmen
from 150 and from 20 m - the charge speed); `hit` - 2 battles (swordsmen, greatswords, spearmen with shields
on unarmoured skavenslaves, flagellants on clanrats; clanrat spearmen on Empire spearmen with the General
behind them: in the first battle he uses Stand Your Ground at contact, in the second not); `move` - 2 battles
(`build/movelords`): turning (spearmen facing 90 deg and back, then 180 deg; spearmen at a run to a point 150 m
behind and 120 m to the side; the General and the Warlord facing 180 and 90 deg, then at a run to a point behind;
0.25 s samples, soldier places) and the melee exit (swordsmen and spearmen leave 10 s into contact, chased by
clanrats ordered to attack them or left standing; the General fights clanrats and uses Foe-Seeker 50 s in).
`syg2` - 2 battles (clanrat spearmen on Empire spearmen with the General casting Stand Your Ground once the centres
are `lord.at_m` apart, the same pair without him; flagellants, swordsmen on clanrats; clanrats on held flagellants);
`pair` - 1 battle (clanrat spearmen attack Empire spearmen from 80 m for 240 s; the spearmen hold, answer, or
`push`: a move order at a walk to a point `push_m` 60 m ahead, through the attacker); `fatleave` - 1 battle (both
attacking from 3 m without a run-up, also 60 m wide on 15 m wide - `a_w` / `t_w`; three units leaving held clanrats).
`skirmish` - 2 battles of 10 lanes (rows of 5, 200 m apart, the rows 600 m apart; battle 2 swaps the rows):
skirmish mode. A shooter stands (the lane's target, `target_mode = 'skirmish'`: in place, fire at will, at the
start the mode on or off by `t_skirmish`, event `probe_skirmish` - was the mode on before and is it now; at the
go a ranged attack at a walk on the attacker, as the bridge aims a held shooter), infantry attacks it at a run
from 60 m, 90 s: slave slingers and Night Runners (chased by swordsmen), the Empire's archers, crossbowmen and
handgunners (chased by clanrats), each with the mode and without. The shooter's row has its ammo, fire flag,
damage dealt and `sk`; the soldiers' places every 1 s within 200 m. Analysis: `python -m tools.nn.charge_probe
skirmish [folders]` - from what distance the shooter starts to move away, where and how fast, does it shoot on
the move, how many shots it got off, was it caught.
`skirmish2` - 2 battles of 12 lanes: what sets skirmish mode off. Slave slingers, archers and crossbowmen with
the mode (standing as in `skirmish`); lane kinds: `pass` - enemy infantry (12 m front) runs past the shooter,
centre 30 m beside it (edge ~9 m), to a point 100 m behind its front, never attacking it (the attacker's mode
`pass`); `neighbour` - the infantry attacks the shooter's neighbour (a unit of its side, 20 m front, centre 30 m
beside it; mode `attack_t2`, the neighbour's width `t2_width`); `control` - the infantry attacks the shooter;
60 s each; `long` - fast infantry (clanrats 4.2 m/s, flagellants 3.6 after the slingers) chases the shooter for
120 s in a long lane of its own (~1080 m): the speed-up and its cause (fatigue, the run flag). The short lanes on
2 rows of 5 places 150 m apart, the long ones in 3 columns of their own; battle 2 rotates the lanes. The
`skirmish` table splits them by shooter and kind; caught = the shooter's own melee flag; for `long` the speed,
the run share and fatigue per 30 s.
Missile probe: `reform_men` (the target re-forms by a 5 m move with its width at that many men), `friend` (a friend
placed `friend_fwd` / `friend_lat` from the shooter, sampled as `f`); plans `thin`, `pistol`, `moving2`, `lof`.
Morale probe: plans `strong2` (the strong-enemy scale, 6 enemies, 120-30 m) and `rally2` (the rally clock).

Every 0.5 s `probe_sample` per lane: men, health (CCO `HealthValue`), melee / moving / running flags, place,
bearing, kills, fatigue, status keys (CCO `StatusList`: `braced`, `melee`...) of the attacker, the target and
the lord; `probe_contact` (the first and second contact), `probe_phase` (recharge), `probe_ability`; every
1 s while the units are within 60 m and up to 30 s after contact, every soldier's place of both (`probe_men`,
CCO `ManList`). A lane ends `fight_s` after its first contact. Analysis: `python -m tools.nn.charge_probe
report [--sim]` - the speed on the way in (the last 30 and 10 m), the target's and the attacker's health
lost in 0-1, 0-2, 0-5, 5-15, 15-30 s after contact and per second from 15 s, men within 1.5 / 2.5 / 3.5 m of
an enemy; `--sim` - the same analysis of the simulator on the same lanes (same places and orders, 8 copies);
the "0-..." windows start at the sample before contact (a blow in the contact sample counts). `turns [--sim]` -
turning (seconds to within 10 deg of the new bearing, the centre's speed by seconds, whether the men kept their
ranks); `exits [--sim]` - the melee exit (how long the leaver stays in melee, its kills and the enemy's HP lost
in 0-24 and 26-50 s after the order) and the lord's fatigue with the ability. The simulator has no facing order:
the twin sets the bearing at once.
Results: [melee](../game/mechanics/melee.md#in-game-check-the-melee-probe).

<a id="lord_fall"></a>

## lord_fall — an army whose lord is killed or routs

The question: does an army lose morale when its lord is **killed**, and how much, against when he
**routs**. All 75 recorded lord falls of the fair battles were routs (the lord shattered with 2-50 %
health), none a death; the game's database has `general_died_recently` -16, `general_dead` -10 and
`general_fled_recently` -16. The result (below): a death costs the army -16, then -10, a rout on the field
only his aura (leaving the map: -16 for ~120 s, from recorded battles); the simulator now does the same (`morale.lord_fall`, [a lord's fall](../training/simulator.md#lords)).

Build: `python -m tools.build lord-fall --faction emp|skv|vmp --treatment kill|rout|none`; the battle
file comes from `tools/nn/lord_fall.py`. MP Crossroads (flat). One battle = one army (side 1) and one
thing done to its lord; the opponent (Skaven against the Empire, the Empire against Skaven and the
Vampire Counts) is fearless, so its routs never lift our army's morale. The army: the lord 20 m behind
his line, two infantry units in melee with two of the enemy's (`fight`, ~63 m from the lord: inside
his aura), two standing idle 130-166 m to the side (`idle`: out of the aura, 115+ m from any enemy -
the shock alone, without the melee's drift). Units from the game's database: Empire spearmen (120,
leadership 60), clanrat spearmen (160, 45), skeleton spearmen `wh_main_vmp_inf_skeleton_warriors_1`
and the vampire lord `wh_main_vmp_cha_vampire_lord_0` (160, 35; not in the simulator's data).

20 s after the first contact (at the latest 60 s after the start) the lord is: `kill` - killed
(`unit:reduce_hitpoints_unary(1)`, then `uc:kill()` if he still stands 1 s later, event
`lord_fall_fallback`), `rout` - routed (`uc:morale_behavior_rout()`), `none` - left alone (the
control; the moment is marked the same way). 60 s later the battle ends. Every 0.5 s `fall_sample`:
every unit's place, men, health, CCO `MoralePercent`, `MoraleState`, `MoraleGreatestEffect`,
`IsAlive`, `PercentCasualtiesRecently`, `PercentHpLostRecently`, routing / shattered / wavering / in
melee, role (`lord`, `fight`, `idle`). `lord_fall` marks the moment with the lord's row before it; a
fight unit out of melee for 5 s gets its attack again (`fall_reissue`).

The series and the table:

```bash
.venv/Scripts/python -m tools.nn.lord_fall plan     # 15 battles
.venv/Scripts/python -m tools.nn.lord_fall run      # build + launch each (launch.ps1), then the table
.venv/Scripts/python -m tools.nn.lord_fall          # the table from build/lord-fall/runs
```

The table: each unit's morale change from the last sample before the moment to +1 ... +60 s in points
(MoralePercent x leadership, as in [measurements](../training/measurements.md#morale-events-and-the-army-collapse)),
by faction, treatment and role; `net` subtracts the control's same role; the share routed within
10 / 60 s, the idle units' health lost (undead crumbling), the strongest morale effect after the moment.
15 battles (2 `kill`, 2 `rout`, 1 control a faction, ~2 min each with loading): the idle units' morale
is flat, so the database's -16 points (0.27-0.46 MoralePercent) against the aura's -4 shows on 4 units;
the control is needed only for the melee's drift. A borderline result: add battles to that cell
(`--kill`, `--rout`, `--factions`).

**The result** (15 battles, `build/lord-fall/analysis.json`; per sample: `build/lorddeath/timeline.py`).
The lord **killed**: the idle units show "general died recently", -3 / -7 / -13.5 / -16 points 1 / 2 /
5 / 10 s later, then a flat -16; at 45.5-46 s the effect turns into "general dead", and from ~50 s to
the recording's end (60 s) the units stand at -10. Empire, Skaven and Vampire Counts alike. The ramp is
the morale step itself (1 point or 15 % of the gap a 0.5 s tick), not a gradual effect. The fighting
units within the aura: -6 / -10 / -17.5 / -21...-25 (the aura and the melee on top). The lord **routed**
(Empire, Skaven): the idle units lose nothing, the fighting ones about the aura (-0.5...-6.8 at 10 s net
of the control); "general fled" (-16 in the database) does not show while the lord is on the field (he did not
leave the map within 60 s). **Vampire Counts'** routed lord
crumbles (-12 % health a second), dies at 8-8.5 s, and the army gets the death's shock from 8.5 s. No
unit routed within 60 s.

<a id="lord_duel"></a>

## lord_duel — the network's lord against a lord under one order

The question: what the network's extra moves and orders cost in a duel of lords. The `lord-duel`
build target is the same entry [nn_arena](#nn_arena) in its own folder `build/lord-duel/` (its battles
never mix with the arena's and the gate's). `tools/nn/lord_duel.py` writes the arena: two lords of one
type only (Empire Generals or Skaven Warlords), mirrored, 100 m apart. The network commands our lord
through the companion (`--own-ai net`, as in the [gate](../launch/gate.md)); the script takes theirs
(`enemy_ai` `scripted`) and gives it one order to attack the nearest enemy (`scripted_order`, `why`
`start`); again only when the engine lost it: out of melee and without a target for 3 s (`why` `lost`).
The control is both lords so (`--own-ai scripted`): the trade of two plain lords.

```bash
.venv/Scripts/python -m tools.nn.lord_duel plan
.venv/Scripts/python -m tools.nn.lord_duel run --checkpoint build/nn-train/<label>/m15.pt
.venv/Scripts/python -m tools.nn.lord_duel          # the table from build/lord-duel/runs
```

`run` builds and launches every battle: the network's through [watch.ps1](../launch/watch.md)
`-Target lord-duel -NoBuild` with the companion, the controls through `launch.ps1`. The default plan is
12 battles: for each lord type 4 network battles (attacking and defending in turn: the defender wins on
the 900 s timeout) and 2 controls. The table per battle: both lords' health at the end, the trade (the
enemy's loss - ours: higher is better for the network), the winner, the time to contact, the network's
orders to its lord (a minute, by kind), the abilities used, the metres both walked, the script's
re-issued orders; then the mean by lord type and mode.

**The escort variant** (`--variant escort`, build `--duel-variant escort`): each lord also has two infantry
units of his faction (Empire spearmen, clanrat spearmen) 40 m to his sides, mirrored armies. The network
commands its whole side; the scripted side (`scripted_targets` `like` in [nn_arena](#nn_arena)): the lord
attacks the enemy lord, the infantry the nearest enemy infantry (the nearest enemy when none is left). The
control is both sides so. The question: does the network's lord fight the enemy lord or switch to the
infantry, and what does it cost. The table adds: both lords' switches of the engine's target
(`unit:current_target`) a minute, the switches of the network's attack target for its lord a minute, the
share of the lord's melee seconds with the enemy lord as his target, and the side trade (the enemy units'
mean health lost - ours).

```bash
.venv/Scripts/python -m tools.nn.lord_duel run --variant escort --checkpoint build/nn-train/<label>/m15.pt
```

12 battles (4 network and 2 controls a lord type), ~2.5 min each with loading: ~30-35 min.

<a id="lord_ai"></a>

## lord_ai — a plain lord against a lord under the game's AI

The question: does the game AI's lord hit harder than a plain lord of the same type, and why: stats
(level, skills, difficulty), abilities or behaviour. The `lord-ai` build target is the entry
[nn_arena](#nn_arena) in its own folder `build/lord-ai/`; the arena is the [lord duel](#lord_duel)'s (two
lords of one type, mirrored, 100 m apart). Our lord is under script: one order to attack their lord, no
abilities. Theirs is led by the game's battle AI (`--duel-enemy game`, no `enemy_ai` in the config) or,
the control, by the same script (`--duel-enemy scripted`); the scripted-v-scripted battles of
`build/lord-duel/runs` are controls too. Normal difficulty (the launcher), the True Sight mod as always.
The config has `observe` and `cards`: both sides record `nn_effects`, `nn_ability_ready` and `nn_card`
([observer_adapter](telemetry.md#observer_adapter)), so the AI lord's card (attack, defence, damage,
armour, charge, morale, rank, experience) is read directly.

```bash
.venv/Scripts/python -m tools.nn.lord_ai plan
.venv/Scripts/python -m tools.nn.lord_ai run --ai 4 --control 2
.venv/Scripts/python -m tools.nn.lord_ai          # the table from build/lord-ai/runs and the build/lord-duel/runs controls
```

The default plan is 12 battles: for each lord type 4 against the game's AI (ours attacking and defending
in turn) and 2 controls. Duel seconds: both lords in melee, neither routing. The rate on a lord: his
health lost in duel seconds over their length, %/s; 'fresh': the first 30 duel seconds. Per battle: their
lord's rate on ours and ours on theirs (all and fresh), the winner, how often their lord left melee and
came back, the share of seconds ours has his rear threatened, their abilities (when, seconds after
contact), their effects, both lords' first card and the changes of theirs. Per lord type: the pooled rate
against the AI and in the control, the ratio AI / control with a 95 % bootstrap interval over battles.
An ability's use: its phase appears in the active effects (`can_perform_special_ability` stays true on a
unit of the game's AI) or, for ours, its ready turns true -> false. Writes `build/lord-ai/analysis.json`.
Results: [the game AI's lord in a duel](../game/game-ai.md#the-game-ais-lord-in-a-duel).

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

Build: `python -m tools.build nn-arena --own-ai attack|defend|hold|net --timeout 900 [--arena NAME]`.
`--timeout 900` is a battle limit of 900 s of game time, as in the records of 30.09.2026;
without it the build gives 600 s.
Scenario `nn_arena.xml` is written by `tools/nn/scenario.py` from `config/nn/arena.json`: the
empty flat map MP Crossroads (flat), each side a lord, 4 spearmen and 2 archers of the Empire,
the armies' fronts 350 m apart. With `--arena NAME` — a named arena from `config/nn/arenas.json`:
each side has its own army and faction ([measurements](../training/measurements.md)). The script makes no battle decisions: side 2 is always led by
the game's general battle AI, side 1 by CA's planner ([planner_adapter](orders.md)). The script
repeats the planner's task, sends rallied and idle units back into battle (below) and records
the battle. The side that wins on timeout defends:

- `attack` — the planner attacks (the order repeated every 15 s), the game's AI defends;
- `defend` — the planner defends where the army stands, the game's AI attacks;
- `hold` — no planner: our units get no orders and stand (a target), the game's AI attacks;
- `net` — the network commands our side: every `--decide-ms` (1 s) the state goes to the companion
  outside the game, its orders come back and are given to our units ([bridge](bridge.md),
  [watching the network](../launch/watch.md)); the game's AI attacks. Events `nn_orders`,
  `nn_miss`; `nn_sample` is recorded as in the other modes.
- `scripted` (the `lord-duel` target, the control) — each unit of ours under script with one order to
  attack the nearest enemy; `enemy_ai` `scripted` takes side 2 from the game's AI the same way ([lord duel](#lord_duel));
  with `scripted_targets` `like` a lord attacks the nearest enemy lord, the others the nearest non-lord unit;
  without `enemy_ai` side 2 stays the game's AI ([the lord against the game's AI](#lord_ai));
- `human` (the `human` build target) — a human commands our side, the script gives no orders; the
  recording adds [observer_adapter](telemetry.md#observer_adapter); the game's AI attacks
  ([a battle played by a human](../launch/run.md#a-battle-played-by-a-human)).

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
| `sv` | `unit:strategic_value()`: the game's strength estimate of the unit now (from 02.10.2026) |
| `pcr`, `phr` | CCO `PercentCasualtiesRecently`, `PercentHpLostRecently`: men / HP lost in the last 4 s |
| `mge` | CCO `MoraleGreatestEffect`: localised text of the effect weighing most on morale now (missing when empty) |
| `sk` | Skirmish mode on now (`unit:is_behaviour_active('skirmish')`); only for units that have it. With `config.skirmish = 'off'` (build `--skirmish off`) the bridges turn it off ([bridge](bridge.md#skirmish-mode)) |

The sample itself also carries `bop` — CCO `BattleRoot.BalanceOfPowerPercent`, the top bar for the
player's alliance — and `bop_side`, that alliance as our side number (`bm:get_player_alliance_num()`,
also in `ready` with the CCO `BattleRoot.PlayerAllianceContext.Id` as `player_alliance_cco`). Each of
these is read under `pcall`: a value the game does not give is left out. They are for the simulator's
army collapse and morale rules ([measurements](../training/measurements.md#morale-events-and-the-army-collapse))
and are not in the companion's state.

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
