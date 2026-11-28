# detour_escape: units held for minutes beside 'cannot stand' ground (28.11.2026)

Switch `terrain.passable.detour.escape` (sim/config/sim.json; `on` 0 = off, bit for bit as before), sim/detour.py
`_escape_on`, `add_escape_state`, `spoil`; rules of the simulator's own path search, not game numbers. Raw rows and
scripts: tmp/stalls/ (slog.py the per-second logger, stage.py / drills.py the runs, table.py, trace.py, synth.py).

## The defect

64 general stage-0 battles (net.stage0.specs.general on ksl_plains_01, both sides the stage's scripts human / nearest /
hold_shoot, 600 s, CPU, seed 2811; tmp/aiappr/stage.py's set). Every second, each unit whose `blk_s` grew (a step the
stop of battle.py:1715-1724 reverted) is classed: (a) routing, (b) in melee or a rectangle gap <= 2 m to an enemy,
(e) the goal on closed ground, (c) a move / withdraw follower, (d) an attack follower, (f) other. A run (consecutive
refused seconds of a unit) takes the class of most of its seconds.

The (c) / (d) cause, in every trace: the unit's centre stands 0.04-0.56 m (by the detour's clearance) from closed
ground; each plan finds the straight line to the goal clear (no corners, `pa_n` 0, no 'not found'), the stop reverts
the step, `spoil` drops the way, the next plan is the same. Two pieces of detour.py:
- `_thresholds` (l.522-525): the start's leg keeps `min(r, F(start)) - tol_m` - negative below 0.5 m, so every line
  passes (the straight check l.659);
- the clearance is a node-point distance read bilinearly (l.71-77, 477-485, 450-459) while the stop reads the node at
  floor (ground.py:182-186): inside a closed node it reads up to sqrt 2 m (an isolated node, near its far corner); the
  largest of 200k points sampled inside Kislev's closed nodes is 1.398 m (tmp/stalls/check_align.py) - so lines from
  F(start) ~0.5-1.9 m still cross closed nodes.

Traces (off; battle frame, 2 steps a second both refused, no way planned):

| battle / unit | seconds | x, z | order | goal (distance) | clearance | enemy |
|---|---|---|---|---|---|---|
| 63 / u10 (AI side, infantry) | 79-388, 391-600 | -42.0, 74.4 (still) | move | 0.1, 22.9 (67 m), later others | 0.04 | 226 -> 178 m |
| 3 / u3 (our side, infantry) | 126-324 | 141.5, 155.9 (still) | move | 114.6, 190.9 (44 m) -> -30.7, 230.4 (188 m) | 0.05 | 102 -> 205 m |
| 16 / u3 (our side, infantry) | 211-302 | 150.1, 200.5 (still) | move | 117.5, 216.9 (36 m) | 0.56 | 294 -> 227 m |
| 62 / u12 (AI side, infantry) | 89-600 | 135.0, -12.0 (still) | move to 299 s, then attack | target 273 -> 433 m | 0.04 | 273 -> 388 m |

Routers are outside the detour (battle.py:1561, 1638: the straight step and the stop): 58 runs >= 5 s, median 61 s,
of which 55 >= 10 s, 28 of those within 100 m of an enemy, 18 at the map's edge band.

## The rule (on)

A unit whose step just before was reverted by the stop (`spoil` keeps that step's clock in `es_rt`; a refusal of an
older step - the detour off meanwhile - is not taken) and that follows (move / withdraw order, or closing on its
target, out of melee) - or routs, with `routers` 1 - takes an escape point: of 16 headings at 2 m, those whose 8
sub-steps of 0.25 m all land on free nodes (ground.blocked) inside the field less 1 m (a router: its exit square, the
battle's own bounds st.bounds, less 1 m), the one most along its goal's way + 1.0 x the cosine to the slide's heading
+ 0.01 x the clearance there (<= 50 m). The slide's heading is set by an escape taken without one and lives 30 s from
then; a later escape does not renew it. It walks there (no way, no plan) until within 0.5 m, a new refusal, 4 s, or it
stops being eligible. A follower's escapes are bounded: an escape asked while its best distance to its goal has not
improved by 2 m since the last improvement counts; at 10 it holds (stands, no plan) until its goal moves by more than
the detour's replan tolerance (max(goal_m, replan_share x its distance)), when the count starts anew.

## Lines (written before the runs)

| line | result |
|---|---|
| off = main ea1403f | tmp/aiappr/ident.py (6 generated battles x 1000 steps): `156b0dcd625fc637`; the 64 map battles off: 6,388 logged rows, the unit table and all 64 trades equal to a sim/ of ea1403f |
| on: (c) + (d) runs >= 10 s | 0 battles (off: 7) |
| on: units missing the fight | 0 (off: 4) |
| on, routers 1: battles with a run >= 5 s | 0, longest 2 s (off: 41, longest 512 s) |
| on + ai_game.approach, ridge 16 x 8 | contact 128 of 128, 104 / 123 / 132 s; enemy lord within 60 m of his infantry at contact 128, in melee within 60 s 128 |
| synthetic checks (tmp/stalls/synth.py) | below |
| cost (v1 of the rule) | the 64 battles side by side, one thread each: off 684.9 s, on 693.6 s (+1.3 %; the logger in both) |
| sim/tests | pass (2 of 2) |

Runs >= 5 s by the run's class (unit-seconds of those runs / runs / battles / median / longest, s):

| class | off | on (routers 1) | on, routers 0 (v1) |
|---|---|---|---|
| (a) routing | 4,767 / 58 / 35 / 61 / 305 | 0 | 4,604 / 62 / 37 / 43.5 / 305 |
| (c) march | 1,059 / 9 / 7 / 92 / 311 | 0 | 0 |
| (d) attack | 512 / 1 / 1 / 512 / 512 (211 of its seconds march, 301 attack) | 0 | 0 |
| (b), (e), (f) | 0 | 0 | 0 |
| all | 6,338 / 68 / 41 | 0 | 4,604 / 62 / 37 |

Harm before the first melee: (c) / (d) refused 1,583 of 221,826 unit-seconds (0.71 %) off, 76 of 220,173 (0.035 %) on.
The drills' own scenes (net/drill/reexam.py builders, lords off, the teacher at the helm; the flat ones have no stop),
v1 of the rule: (c) / (d) runs >= 10 s in 0 of 128 T0-1 and 0 of 64 O0-1 scenes off and on; router runs >= 10 s in
15 of 128 and 12 of 64 off, 0 and 0 on (tmp/stalls/out/t01_fon, o01_fon).

The final trade of the 64 battles on (routers 1) differs in 31, mean |change| 0.021 (max 0.23), signed -0.002 +-
0.007.

Battle 61 unit 3 (our side, infantry; its march point beside the closed band of the map's southern edge, no way):
off it stood 30 s and fought at 407 s; with the unbounded escape (v1) it slid ~65 m along the wall and back and never
fought; now 10 escapes 211-231 s without getting nearer, then it holds; a new goal at 373 s, one refusal, it fights at
407 s as off.

Synthetic checks of `_escape_on` (1 m nodes, the stop's floor rule, refusals recorded by `spoil`, 60 s walks unless
said): a unit 0.5 m inside the field's edge with the pull across a closed column - no escape, never outside; with room
- 7 selections across 14 latched ticks, none beyond the 1 m margin, past the column; a closed 20 m column across the
way - no two-point oscillation, 7 selections, past it at 13 s (the end of that step), at the goal (without the heading
term it cycled between two points; with 0.25 between the column's ends every 24 s); an endless wall - slides 13.5 m
one way, holds from 15 s; a one-node-wide dead end with the goal beyond its end - 10 refusals, 9 selections, holds
from 24 s, no move and no refusal in the last 20 of 120 s; a unit whose 1.5 m step lands free beside a closed node -
untouched; an isolated closed node, the unit at its far corner - no step and no escape point on it, at the goal; a
closed sub-step before a free endpoint - that heading rejected; a refusal two steps old - not taken; a router 0.5 m
inside a 500 m exit square (params.map_half 1020) - every point beyond 499 m rejected.

## Known

- Routers slide slowly along a patch (a refusal, 2 m, a refusal: ~30 m in 300 s on the central patch) and for ever
  along the map's edge band, where the game's routers leave the field; routers have no bound.
- Fix A (the clearance read in the stop's frame: node-centre origin, less 0.854 m - the bilinear reading inside a
  closed node then has the limit (2 + sqrt 2) / 4 - 0.854 = -0.0005 m, so <= 0 up to rounding; -0.016 m the largest
  of 200k sampled points) removes the root of the bad legs but shifts every planned way by half a node: all 64 trades
  change (mean |change| 0.057, signed +0.0005 +- 0.010) - not built.
- A goal with no way costs a follower 10 escapes, then it stands until its goal moves.
