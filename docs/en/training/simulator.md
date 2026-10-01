# Battle simulator

[← Back](README.md) · [Documentation](../README.md) › [Data for training](README.md) › Battle simulator · [Русский](../../ru/training/simulator.md)

Step 3 of the path ([network model](network.md)): a simulator of the game's battle in which
the network will train. It plays thousands of battles at once on the GPU. Each unit is one
object (men, health, place, facing, morale, fatigue, projectiles), not every soldier. The
numbers come from the [unit passports](units.md) and the [game's rules](../game/database.md);
where the game behaves differently, they are calibrated to the [measurements](measurements.md).
Version 1, 30.09.2026: the seven v1 units, a flat empty map.

## How to run it

```bash
bash tools/nn/dock.sh tools.nn.sim.check                     # all checks against the game (CPU and GPU)
bash tools/nn/dock.sh tools.nn.sim.check --device cuda --only battles
.venv/Scripts/python -m pytest -q tests/tools/test_sim_layout.py   # without torch
```

- The code is `tools/nn/sim/` (PyTorch). It needs torch, which is in the `snake-ai-trainer`
  container; the project's `.venv` has none, so `tests/tools/test_sim.py` is skipped there.
  The image has no pytest: to run the tests there, put a copy of pytest (from `.venv`) on
  `PYTHONPATH` and run `python -m pytest -q tests/tools/test_sim.py`.
- The check writes `build/nn-sim/check.json` (not in Git).

```python
from tools.nn.sim import battle, replay, scenario

st = scenario.build([scenario.from_arena("whole_emp_v_skv", "attack")] * 1024, device="cuda")
battle.run(st, replay.nearest_attack)        # policy(state) -> orders, until every battle ends
st.winner, st.t, st.observation()            # [B], [B], dict of [B, N]
```

## What the simulator holds

```mermaid
flowchart LR
  pass["config/nn/units.json<br/>passports"] --> params["params.py"]
  rules["config/nn/game_rules.json<br/>the game's rules"] --> params
  cal["config/nn/sim.json<br/>calibration"] --> params
  params --> scen["scenario.py<br/>armies → state"]
  scen --> step["battle.py: one step"]
  orders["orders.py<br/>orders of the network"] --> step
  step --> obs["state.py<br/>state → network"]
```

**The state** (`tools/nn/sim/state.py`, the one source of truth): tensors `[B, N]` — B battles,
N unit slots of both sides (side 1 in the first half, side 2 in the second, a side's lord
first; `side` = 0 is an empty slot). Three groups:

- *observed* — the same fields and names as a recording's `nn_sample`
  ([what a recording holds](README.md#what-a-recording-holds)): `x`, `z`, `b`, `men`, `hp`,
  `mp`, `ms`, `r`, `s`, `w`, `m`, `mv`, `f`, `a`, `fire`, `target` (the recording's `t`),
  `fat` (the fatigue state 0–5), `k`, `ox`, `oz`, `lf`, `rf`, `bf`, `vis`; the battle time
  is `t` `[B]`. So recorded battles and the simulator feed the network the same way;
- *static* — the passport: men, health, speeds, attack, defence, weapon, armour, shield,
  leadership, projectiles, range, reload, cost;
- *internal* — what only the simulator needs: morale and fatigue points, timers.

**Orders** (`tools/nn/sim/orders.py`): per unit and decision step `kind` ∈ hold (0), move (1),
attack (2), withdraw (3), keep (4: no new order, the one in force goes on; a unit with no order
holds); the point `x`, `z` for move and withdraw (the unit's centre); `target` — the enemy's
slot for attack; `run` — run or walk. All `[B, N]`. A move order to a unit in melee does not
take it out: that is what withdraw is for (it breaks off, and the enemies in contact strike its
back).

## One step (0.5 s)

1. Take the orders (routing units ignore them).
2. Contacts: the units' rectangles (front × depth of the ranks the men fill) touch.
3. Blows in melee and shots.
4. Health, men, kills.
5. Morale: wavering, rout, rally, shattering.
6. Fatigue.
7. Movement: to the order's point or the target, fleeing for routers; the edge of the map.
8. Is the battle over: a side with no standing unit loses; at 60 minutes the defender wins.

## Mechanics and their sources

"DB" — the game's rules or a passport; "measured" — [measurements](measurements.md) and
recordings; "calibrated" — a number fitted so the simulator repeats the game (all of them are in
`config/nn/sim.json` with the reason).

| Mechanic | How | Source |
|---|---|---|
| Speed | walk, run, acceleration, deceleration of the passport; routing units run at 0.9 of the run | DB; the routing speed measured (median 0.80–0.96) |
| Formation | front of the ordered width, ranks 1.5 m apart; a lord is a circle of his radius | measured: 120 spearmen, 30 m, 6 ranks |
| Contact | edges within 1 m (2 m more for those already fighting) | measured: centre distance at the first contact |
| Facing | a moving unit faces where it goes, but a step of less than 10 m to its point goes without turning; a formation in melee turns at most 2° a second (a lord turns at once) | measured: infantry in melee turns 1°/s (median; mean 2.3), a free unit struck in the flank turns 8° in 5 s (median) |
| Order point | the game records the front's centre; the simulator goes to the unit's centre, half a depth behind | measured (spearmen 3.3–4.8 m, slaves 6.5–6.9 m) |
| Men fighting | 0.75 of the files in contact; a unit shares out to each side of its formation (front, left, right, back) no more than that side holds; at most 8 around a lord | calibrated; per side — measured: a unit already fighting hits a newcomer on its flank 2.2× the rule in the first 15 s (with one shared front the simulator gave 1.2×); 8 — measured ([melee](../game/units/melee.md)) |
| Hit chance | 35 + 0.1 × (attack − defence), within 8–90 %; defence ×0.6 from the flank, ×0.3 from the rear; the defence lost counts 2.0× the rule from the flank, 0.25× from the rear (against a lord: the rule) | DB numbers; the weights calibrated (see below and [flanks](#flanks-rear-and-charges-in-whole-battles)) |
| Damage of a hit | armour-piercing + base × (1 − 0.75 × armour/100), no more than a man's health | DB (armour stops a random 50–100 %) |
| Time between blows | `attack_interval_s` of the passport | DB |
| A lord's blow | hits up to `splash` (4) men | DB; agrees with the measured 0.36 kills a second |
| Charge | the charge bonus to attack and damage, fading over 13 s; a unit that meets the enemy running hits ×(1 + 1.5 × its speed share) for those 13 s; one that did not charge brings its men in over 20 s. Bracing: a unit with `charge_reflection` (spearmen, clanrats) standing still (under 0.5 m/s) meets an infantry charge within `bracing_attack_angle` (80°) of its front as a charge of the same speed | DB (13 s, 80°, the attribute); 1.5 fitted to the first 15 s of the whole battles and the pairs, 20 s to the pairs; bracing measured (whole battles: a braced unit charged head-on loses 0.8× what its charger loses) |
| Men lost | blows that do not kill wound: men share = max(1 − g × (1 − health share), health share), g = (hits to kill)^−0.5 | calibrated on men against health in the pairs |
| Shooting | from standing only; first shot 3.3 s (arrows) / 4.3 s (sling) after halting; a shot per man every 11.0 / 11.5 s; range from the formation's edge | measured |
| Hits | 0.42 arrows, 0.47 sling at the edge of range; ×1.29 at 70 m, ×1.12 at 90 m | measured; the distance factor from the range test |
| Shield, resistance | a shield blocks its chance within 60° of the front; missile resistance of the passport | DB |
| A lord as a target | 0.43 of the unit rule | measured: ~33k shots at lords out of melee for the whole flight (General by arrows ~0.5, by sling ~0.37, Warlord by arrows ~0.32) |
| Friendly fire | of the hits aimed at a unit in melee, 0.26 (arrows) / 0.56 (sling) land on the shooter's own units in contact with it, split by their men | measured: the HP those units lose beyond the melee rule while their own side shoots (1.5k + 1.3k seconds) |
| Spill | of the hits aimed at a unit, each unit of the target's side out of melee takes 0.115 within 30 m, 0.034 at 30–60 m, 0.015 at 60–90 m | measured: HP of units nobody shoots at, next to a unit that is shot (15.6k seconds) |
| Target at will | the order's target if in range, else the nearest standing enemy | as the game ([missile damage](../game/units/missile-damage.md)) |
| Morale | points: leadership + effects; MoralePercent = points / leadership; moves 1 point or 15 % of the gap per 0.5 s | DB; the step measured (+2 points a second in every recording) |
| Morale effects | lord within 70 m +4; lord died −16 then −10; neighbour within 120 m +5; casualties −2…−74; recent casualties −6…−80 (last 30 s, of the whole health); winning / losing the melee +3/+6/+8, −3/−8 (damage ratio 1.5 / 2.5 / 4); attacked in the flank −1, rear −2; flanks exposed (an enemy threatens the left, right or rear: `lf`, `rf`, `bf`) −3, two or more −6; routing friends −3 each; routing enemies +2.5 each; under fire −5; very tired −2, exhausted −6; a stronger enemy within 70 m −3 | DB points; window, ratios calibrated; attacked in the flank / rear measured ([flanks](#flanks-rear-and-charges-in-whole-battles)) |
| Faction | Skaven +6 points at the start | measured: before contact they stand 6 higher than the Empire in the same place |
| States | wavering below 16 points, rout at 0, shattered at the third rout, no new rout within 10 s of a rally | DB |
| Rally | while no standing enemy is within 90 m the router regains 2 points a second; rallies at MoralePercent 0.23 | measured: 0.23 and 90 m (365 rallies); 2 points calibrated (median rally 44 s) |
| Fatigue | charge +34, melee +19, shooting +18, running +4, walking −1, standing −7, ×5 a second; states by the database thresholds | DB; ×5 fitted to 1315 recorded changes of state |
| Map | a square ±1020 m; a routing unit that crosses the edge leaves the battle | measured |
| Visibility | everything is visible (`vis`, kept for later) | a flat empty map |

**Why the hit chance is almost flat.** By the database rule the four pairs would need 13 to 43
men striking at once, and a lord ~38 ([measurements](measurements.md)). With a flat ~35 % the
same losses need 12–17 men on a 30 m front in every pair and ~8 around a lord: one number fits
all. So attack − defence counts at 0.1 of the rule. The flank and rear, which the pairs do not
test, are fitted to the whole battles (below).

## Flanks, rear and charges in whole battles

Measured 01.10.2026 in the 28 whole battles (18 mirror, 10 Empire-Skaven), and in the
simulator's open-loop replays of them by the same code (`build/nn-sim/flank/flank.py`, not in
Git). A contact: an enemy standing in melee whose formation edge (the simulator's rectangles at
the recorded places, bearings and men) is within 3 m; its sector is the angle of its centre off
the unit's facing (front within 60°, rear beyond 120°), as the simulator counts it. HP lost is
set against the simulator's frontal melee rule for the same contact.

| Infantry in melee, one contact, not shot at | The game | Simulator before | Now |
|---|---:|---:|---:|
| Seconds with the worst contact front / flank / rear | 58 / 31 / 11 % | 78 / 18 / 4 % | 72 / 23 / 6 % |
| HP lost from the flank, × from the front (10–90 % over runs) | 1.74 (1.56–1.89) | 1.20 | 1.55 |
| HP lost from the rear, × from the front | 1.31 (1.19–1.42) | 2.04 | 1.50 |
| Morale points while fought from the flank / rear (regression*) | −1.0 / −1.8 | −1.0 / −14.5 | −0.8 / −4.0 |
| … one / two or more of `lf`, `rf`, `bf` set | −3.9 / −8.1 | — | −4.3 / −7.0 |

\* Points − leadership against health lost (and its square), flank, rear, 2+ contacts, the
exposed flags, under fire; infantry melee seconds (the game 30k, the simulator 97k).

New contacts (infantry i struck by infantry j): HP i loses in the first 15 s over HP j loses.

| | Contacts in the game | The game | Before | Now |
|---|---:|---:|---:|---:|
| j charges, i stands, from the front | 102 | 0.82 | 2.69 | 1.44 |
| j charges, i runs at it too | 104 | 0.90 | 0.98 | 0.91 |
| j charges i's flank or rear, i free | 206 | 1.11 | 2.65 | 1.65 |
| j charges i's flank or rear, i already fighting | 24 | 1.30 | 3.30 | 2.22 |
| j walks into i's flank or rear, i already fighting | 77 | 1.23 | 2.00 | 1.34 |
| Morale points of i 15 s after it is charged standing | 102 | −6.5 | −17.5 | −11.3 |

- **In the game a charge buys little.** A unit charged while standing loses no more than its
  charger (0.82); braced spearmen (`charge_reflection`, 87 of the 102) 0.80, units without it
  about the same (1.02, 15 contacts). The simulator gave the charger 2.7×, which is why in it
  every unit left standing when charged lost. Now: bracing, and the charge's impact 1.5 instead
  of 2.5 (the pairs keep within 20 %: 51 of 54).
- **In the game formations do not turn round in melee** (1°/s): an enemy on the flank or rear
  stays there. In the simulator a unit turned to its nearest opponent at once, so a flank attack
  on a free unit lasted one step. Now it turns at 2°/s in melee.
- **The flank costs more than the rear by this count** (1.74 against 1.31), against the
  database's defence ×0.6 / ×0.3. The rear seconds are probably often a lagging recorded
  bearing; the simulator is fitted to the measured numbers (flank_slope 2.0, rear_slope 0.25).
- **"Attacked in the flank / rear" is small in the recordings**: −1 / −2 points, not the
  database's −6 / −14 (the simulator with the database's points shows −14.5 for the rear).
  "Flanks exposed" is there as in the database: −3.9 / −8.1 against −3 / −6.
- Still stronger than in the game: a charge into the flank or rear (1.65–2.22 against 1.1–1.3),
  and the morale drop after a charge (−11 against −6.5). `lf` / `rf` / `bf` are set 1.3–3× as
  often as the game's flags (the game's are noisy: 0.3–0.7 with an enemy within 60 m on that side).

Tactic scan against `nearest` (the training's scripts, `build/nn-sim/flank/scan.py`: 48 battles
per army, role and side; wins of the tactic; Empire / Skaven = the tactic plays that army in the
Empire-Skaven battle):

| Tactic | Mirror: before / now | Empire: before / now | Skaven: before / now |
|---|---:|---:|---:|
| `nearest` itself | 47 / 49 % | 60 / 26 % | 58 / 82 % |
| all on one target | 47 / 32 % | 5 / 1 % | 43 / 65 % |
| archers stand and shoot | 0 / 1 % | 1 / 1 % | 77 / 88 % |
| the lord waits 2 minutes | 11 / 18 % | 3 / 1 % | 80 / 80 % |
| attack at a walk | 0 / 0 % | 0 / 0 % | 0 / 0 % |
| enemies already engaged first | 36 / 32 % | 17 / 3 % | 24 / 46 % |
| a reserve waits until the enemy is engaged | 6 / 3 % | 0 / 0 % | 1 / 7 % |
| `hold_shoot` | 19 / 34 % | 3 / 5 % | 48 / 70 % |
| **flank**: melee units go for an enemy already fighting ours, via a point beside its flank | **61 / 69 %** | 27 / 17 % | 32 / 79 % |
| flank, with a reserve | 6 / 15 % | 0 / 1 % | 18 / 51 % |

Now the Skaven win the Empire-Skaven battle (`nearest` against `nearest`: 26 % for the Empire,
82 % for the Skaven; in the game the Skaven won all 10 recorded ones). Flanking beats `nearest`
in the mirror (69 %) and is no longer punished on the other armies (Skaven 79 % against
`nearest`'s own 82 %; before 32 % against 58 %). Waiting (a reserve, a walk) still loses: the
side that waits fights outnumbered.

## Checks against the game

`python -m tools.nn.sim.check`: every recorded run is replayed in the simulator from the
recorded start with the recorded orders (open-loop: `tools/nn/sim/replay.py`: a target fought
or shot → attack it; in melee without a recorded target → attack the nearest enemy (CA's planner
leaves the target empty in ~70 % of its melee seconds, the game's AI in ~6 %); otherwise → move
to the order's point), written down once a second like a recording and measured by the same code
as the game (`tools/nn/measure.py`). Only the battles of CA's planner against the game's AI
count (not the network's own runs). A pair or shooting replay runs on its last recorded orders
until a unit routs; a whole battle stops when its recording ends and is compared then (if it
is not over, the side with more health left in standing units counts as the winner). Results of
30.09.2026 (third version, the same night):

### Mechanics: 51 of 54 numbers within 20 %

| Case | The game | The simulator |
|---|---|---|
| Spearmen — slaves: fight, s | 232 (208–252) | 260 |
| … slaves lose HP/s / spearmen | 22.7 / 12.4 | 23.3 / 14.1 |
| … slaves waver / rout, s | 195 / 232 | 181 / 260 |
| Spearmen — clanrats: fight, s | 275 (219–353) | 290 |
| … spearmen lose / clanrats HP/s | 25.7 / 20.6 | 22.1 / 19.5 |
| … spearmen waver / rout, s | 251 / 275 | 269 / 290 |
| General — clanrats: fight, s | 348 (340–357) | 335 |
| … the General / clanrats lose HP/s | 8.2 / 21.8 | 6.9 / 22.9 |
| … clanrats waver / rout, s | 254 / 348 | 254 / 335 |
| Warlord — spearmen: fight, s | 309 (264–338) | 264 |
| … the Warlord / spearmen lose HP/s | 5.5 / 21.2 | 5.4 / 25.3 |
| … spearmen waver / rout, s | 277 / 309 | 229 / 264 |
| Winner of each pair | 12 of 12 | 12 of 12 |
| Archers → slaves: reload, s / hit rate / HP/s | 11.0 / 0.42 / 66 | 11.6 / 0.43 / 62 |
| … the first shot after halting, s | 3.3 | 3.7 |
| … the target wavers / routs after the first shot, s | 39 / 71 | 41 / 74 |
| Slingers → spearmen: reload, s / hit rate / HP/s | 11.5 / 0.47 / 36 | 11.5 / 0.47 / 37 |
| … the target wavers / routs, s | 154 / 177 | 160 / 183 |

Outside 20 % (01.10.2026, with the flanks, bracing and turning): the HP lost in the first 15 s
of contact (the charge) in three places: the clanrats against the General (−28 %) and against
the spearmen (−33 %), the Warlord (−22 %) — the charge is noisy in the game too (before: four
places, −28 %… +107 %).

### Whole battles: the same winner in 18 of 26 (69 %)

| | Battles decided | Same winner: v1 | v2 (friendly fire, spill) | v3 (stop at the recording's end) | now (flanks) |
|---|---:|---:|---:|---:|---:|
| Empire against Skaven | 10 | 10 | 9 | 8 | 8 |
| Mirror arena (Empire against Empire) | 16 | 11 | 8 | 9 | 10 |

Each recorded battle is played 8 times from starts moved by up to 2 m; the simulator's winner
is the majority's. Two mirror runs ended on our 900 s limit without a winner and are not counted.

| | The game | v1 | v2 | v3 | now |
|---|---|---|---|---|---|
| Empire HP lost 60 / 120 / 180 s after the first contact | 0.25 / 0.41 / 0.53 | 0.27 / 0.48 / 0.61 | 0.27 / 0.45 / 0.58 | 0.27 / 0.45 / 0.58 | 0.29 / 0.49 / 0.63 |
| Skaven HP lost 60 / 120 / 180 s after the first contact | 0.21 / 0.34 / 0.42 | 0.14 / 0.23 / 0.29 | 0.19 / 0.32 / 0.41 | 0.19 / 0.32 / 0.41 | 0.20 / 0.34 / 0.44 |
| Mirror: HP lost by side 1 / side 2 at the end | 0.74 / 0.75 | 0.85 / 0.73 | 0.88 / 0.76 | 0.81 / 0.68 | 0.83 / 0.72 |
| Routs / rallies a battle | 22.5 / 13.0 | 24.4 / 15.5 | 29.2 / 18.8 | 21.7 / 14.6 | 22.2 / 14.8 |
| A rally takes (median), s | 44 | 47 | 44 | 47 | 44 |
| A battle lasts (mean), s | 613 | 716 | 737 | 590 (stopped at the recording's end) | 579 |
| Not over when the recording ends | — | — | — | 80 % | 72 % |

**Why v1 leaned to the Skaven.** Replaying the melee rule on the recorded seconds (the recorded
places, men and contacts) shows that in the whole battles the Skaven infantry lost 2–3× what the
rule gives, the Empire's about 1×, while in the pairs both ~1×. Two missile effects the pairs do
not have explain it: *friendly fire* — Skaven infantry in melee lose 3.2× the rule in the seconds
their own slingers shoot at the enemy they fight, 1.5× otherwise (Empire infantry in the mirror
2.2× and 1.15×); and *spill* — shots at a unit hit its neighbours (an infantry unit in a whole
battle loses ~1.3× per shot aimed at it what the lone target of the shooting arena does). With
both, the casualty curves of both sides match the game within ~10 %. The mirror battles did not
get better (see "What is missing").

**The network's battles against the game's AI** (the gate: generated armies, `own_ai` "net"),
both sides' recorded orders replayed, are reported apart: the same winner in 4 of 6
(01.10.2026). The attacker of a recorded battle comes from its manifest's roles
(`scenario.attacker_of`); a run the network played had side 2 as the attacker before.

### Speed

| Where | Battles at once | Battles a second |
|---|---:|---:|
| GPU, RTX 5070 Ti (`torch.compile`) | 4096 | ~900 (v1 without friendly fire and spill: 1200–1560) |
| CPU in the container (no compile) | 1024 | 8 |

A battle here is the Empire-Skaven battle (40 slots) to its end, 430–620 s of game time. On
the GPU the step is compiled by `torch.compile` (about 10× faster); the container has no C++
compiler for compiling on the CPU.

## What is missing

- **The mirror battles**: in the game the defender wins 15 of 16; the simulator gets 9 of 16.
  Its side 1 (CA's planner) loses too much (0.81 of its health by the recording's end against
  0.75): its archers get caught in melee for 70 s a battle (21 s in the game).
- **Fights break off in the game, not in the simulator.** In the mirror and Empire-Skaven
  battles a unit in melee leaves it (for 4 s or more, both standing) 0.013 times a second
  (infantry), 0.030 (lords), 0.033 (missile units); melee spells are short (median 17 s, the
  General's 9 s; in the simulator 21–112 s). In the one-against-one pairs it almost never
  happens (2 times in 4.9k seconds). A move order is not the cause: infantry with a move order
  10 m or more away from the enemies break off as often as those told to stay (0.013 against
  0.011 a second; they drift ~2 m/s with the fight and lose 1.2× more health); only lords (0.066
  against 0.025) and missile units (0.038 against 0.027) break off more. So the simulator keeps
  "a move order does not take a unit out of melee". A break-off at the measured rates, with the
  database's 10 s of immunity (`melee_breakoff_total_immunity_secs`), was tried and dropped: it
  broke the pairs (fights twice as long) and did not help the mirror. What triggers it in the
  game is not known.
- **Simulated battles end later**: 72 % are not over when their recording ends (80 % before the flanks).
- **Lords in melee lose too much**: 16 HP/s against the game's 12.4 in the whole battles, 11.5
  against 8.2 in the network's gate battles. With one enemy unit in contact it is close (gate 6.1
  against 6.6), but the simulator's lords spend more time with 2–3 units on them and then lose
  15–21 HP/s (the game 8–16). Under fire out of melee they match (9.5 against 10.3 HP/s).
- **Replay is open-loop**: the recorded orders do not react to a battle that went differently.
- **Fatigue** matches the recorded state exactly in 43 % of the samples (off by 0.86 of a
  state on average). What fatigue does to attack, defence and speed is not in the database and
  is not modelled.
- **Not modelled**: terrain, a turn rate out of melee (a moving unit faces where it goes at
  once), cavalry, monsters, magic, flying, artillery, abilities, experience ranks, the scaled
  "strong enemy near" (only −3).
- `vis` is always true: line of sight is not modelled.

## Surprises

- A lord loses ~0.43 of the unit rule (his armour, shield, resistance) per projectile aimed at
  him while he is out of melee for the whole flight (~33k shots). v1 had 0.9: its 2-second loss
  windows overlapped and counted each loss twice.
- Missile units hurt their own side: of the hits aimed at an enemy fighting their own infantry,
  0.26 (arrows) and 0.56 (sling) land on their own men. The Skaven slingers shooting over their
  line are why the Skaven lose more in the whole battles than the pairs suggest.
- The arena's close-range damage per arrow (~25 HP at 34 m) is more than an arrow's whole
  damage (19): melee losses are mixed in; the simulator takes the distance factor from the range
  test instead.
- Routing units run slower than their run: the Empire's 0.80–0.86, the Skaven's 0.85–0.96.
