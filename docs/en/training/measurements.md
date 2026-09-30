# Measurements in the game

[← Back](README.md) · [Documentation](../README.md) › [Data for training](README.md) › Measurements in the game · [Русский](../../ru/training/measurements.md)

Step 2 of the battle simulator ([network model](network.md)): numbers from real
battles of the game that the simulator will be checked against. The units are the
seven v1 units ([unit passports](units.md)). All battles on 30.09.2026: game
v9.0.1, build 50381, the empty flat map MP Crossroads (flat), Normal difficulty,
speed ×20, one battle per game launch. 28 launches, all with `battle_difficulty` = 1
and the player's preferences restored.

## How to get them

```bash
.venv/Scripts/python -m tools.build nn-arena --arena pair_spear_v_slave --own-ai attack --timeout 900
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target nn-arena
.venv/Scripts/python -m tools.nn.measure       # → build/nn-measure/targets.json
```

- The arenas are in `config/nn/arenas.json` ([how to record a battle](README.md#different-armies)).
- `tools/nn/measure.py` takes every fair run of the named arenas and writes
  `build/nn-measure/targets.json` (not in Git): per battle and a summary per arena
  (number of battles, mean, minimum, maximum). Tests: `tests/tools/test_nn_arena.py`.
- Our side is CA's planner (`attack`) or stands without orders (`hold`); the
  other side is the game's AI. The runs are `build/nn-arena/runs/20260930-181821`
  … `20260930-184908` (not in Git).

| Experiment | Arena | Battles | Runs |
|---|---|---:|---|
| Empire spearmen against skavenslave spearmen | `pair_spear_v_slave` | 3 | `181821`, `181939`, `182026` |
| Empire spearmen against clanrat spearmen | `pair_spear_v_clanrat` | 3 | `182117`, `182208`, `182257` |
| Empire General against clanrat spearmen | `pair_general_v_clanrat` | 3 | `182351`, `182446`, `182541` |
| Skaven Warlord against Empire spearmen | `pair_warlord_v_spear` | 3 | `182636`, `182730`, `182820` |
| Empire archers shoot skavenslave spearmen | `missile_archers_v_slave` | 3 | `182915`, `183143`, `183223` |
| Skavenslave slingers shoot Empire spearmen | `missile_slingers_v_spear` | 3 | `182956`, `183305`, `183352` |
| Whole battle, the Empire is our side | `whole_emp_v_skv` | 5 | `183439` … `184732` |
| Whole battle, the Skaven are our side | `whole_skv_v_emp` | 5 | `183555` … `184908` |

The tables below give the mean over the battles and the spread (minimum–maximum).

## Melee: one unit against one

The units start 150 m apart; our planner attacks, the game's AI defends. Counted
from the first contact (`is_in_melee` of either) to the rout of one unit. The
first 15 s of contact are the charge; the rest is the steady fight.

| Pair (ours — the game's AI) | Won | Fight, s | Losses in the steady fight, men/s | HP/s | HP lost in the charge |
|---|---|---:|---|---|---|
| Spearmen — slaves | spearmen, 3 of 3 | 232 (208–252) | spearmen 0.15 (0.13–0.16), slaves 0.41 (0.38–0.46) | 12.4 (11.9–12.8) and 22.7 (21.2–24.3) | 84 and 675 |
| Spearmen — clanrats | clanrats, 3 of 3 | 275 (219–353) | spearmen 0.36 (0.29–0.42), clanrats 0.28 (0.25–0.32) | 25.7 (20.4–29.5) and 20.6 (18.0–23.0) | 567 and 764 |
| General — clanrats | General, 3 of 3 | 348 (340–357) | General 0, clanrats 0.36 (0.35–0.37) | 8.2 (7.9–8.6) and 21.8 (21.4–22.1) | 38 and 500 |
| Warlord — spearmen | Warlord, 3 of 3 | 309 (264–338) | Warlord 0, spearmen 0.31 (0.28–0.36) | 5.5 (4.6–6.2) and 21.2 (19.2–25.0) | 45 and 459 |

- **The lords survive with room to spare:** the General ends with 1224–1429 HP of
  4068, the Warlord with 1997–2868.
- **A lord hits several.** The General kills clanrats in "events": 0.25 events a
  second, 1–4 men in an event (166 times one, 66 two, 25 three, 2 four). 0.25 a
  second is exactly one blow per 4 s, and up to 4 targets matches `splash` in the
  passport. The Warlord: 0.23 events a second, 1–3 men.

### Rout and morale

| Who routed | Began to waver, s after contact | Routed, s |
|---|---:|---:|
| Slaves (against spearmen) | 195 (171–211) | 232 (208–252) |
| Empire spearmen (against clanrats) | 251 (217–310) | 275 (219–353) |
| Clanrats (against the General) | 254 (244–259) | 348 (340–357) |
| Empire spearmen (against the Warlord) | 277 (215–311) | 309 (264–338) |

- The loser's morale (`MoralePercent`) falls almost evenly: the slaves' from 1.17
  to ~0.8 in the first 30 s (the charge), then ~0.1 per 30–40 s; wavering at about
  0.2–0.3, rout at about 0. The winner stays near 1.0 (the lords 0.6–0.8 by the
  end). Curves every 10 s are in `targets.json` (`morale_every_10s`).
- All 12 pairs ended with one unit routing, never "to the last man".

### How many fight at once

Counting by the database rule (hit chance 35 + attack − defence within 8–90 %,
armour stops on average 75 % of its value, a blow every `attack_interval_s` of the
passport; [melee](../game/units/melee.md)), the recorded losses need this many men
hitting at once:

| Who hits → whom | Men hitting | Share of the unit |
|---|---:|---:|
| Empire spearmen → slaves | 13 (12–14) | 11 % |
| Slaves → Empire spearmen | 43 (41–44) | 24 % |
| Empire spearmen → clanrats | 18 (15–20) | 15 % |
| Clanrats → Empire spearmen | 31 (24–35) | 19 % |
| Clanrats → General | 38 (37–40) | — |
| Empire spearmen → Warlord | 32 (27–36) | — |
| General → clanrats | 2.2 targets per blow | — |
| Warlord → spearmen | 2.4 targets per blow | — |

This is derived from the rule, not measured directly: the records do not say
which rule is right.

## Shooting at a unit that stands

The target is our unit without orders (`--own-ai hold`), the shooters are the
game's AI, which attacks. The units start 250 m apart. Distance is between the
units' centres.

| | Archers → slaves | Slingers → Empire spearmen |
|---|---|---|
| First shot, s from the start | 36 (35–36) | 37 (37–38) |
| Distance of the first shot, m | 134 (133–135) | 120 (119–121) |
| First shot after halting, s | 3.3 (3–4) | 4.3 (4–5) |
| Shots per man per second | 0.091 (0.086–0.095) | 0.087 (0.086–0.088) |
| Reload that follows, s (passport) | 11.0 (10.6–11.7) (10) | 11.5 (11.3–11.6) (9) |
| Target HP per shot | 8.0 (7.4–8.5) | 3.0 (2.9–3.2) |
| Target men per 100 shots | 10.8 (10.3–11.4) | 3.0 (2.9–3.1) |
| Hit rate (derived) | 0.42 (0.39–0.45) | 0.47 (0.46–0.49) |
| Target HP per second | 66 (57–71) | 36 (35–39) |
| Projectiles used | 33 % (30–36 %) | 70 % (67–72 %) |
| Target wavers, s after the first shot | 39 (35–44) | 154 (145–159) |
| Target routs, s | 71 (63–81) | 177 (166–183) |

- **The game's AI shoots from the edge of its range** and comes no closer: every
  shot left from 118–135 m. So damage by distance is not measured (see below).
- **The rate is below the reload:** ~0.09 shots per man per second — the reload
  in practice is 11–11.5 s, for the sling 28 % longer than the passport's 9 s. The
  archers: 11 s against 10 s. On the range on 28.09.2026 the archers shot 0.1, in
  whole arena battles 0.073 ([missile damage](../game/units/missile-damage.md)).
- **Hit rate** is HP per shot divided by a hit's damage by the database rule (the
  slaves have no armour: 17 + 2 = 19; the spearmen armour 30:
  1 + 7 × (1 − 0.75 × 0.3) ≈ 6.4). Extra damage on a man already dying lowers it
  (the slaves have 50 HP — by roughly a tenth).
- **Against the earlier measurement:** on the range the archers took 8–10 HP per
  arrow from Empire spearmen (armour 30) at 105–120 m. Here, from slaves without
  armour, 7.4–8.5 at ~125 m to the front rank — not more. Why unarmoured slaves
  take as much is not checked.
- The target never got into melee: it routed and the battle ended.

## Whole battles: Empire against Skaven

Equal budget: the Empire 2500 (General, 4 spearmen, 2 archers — 7 units, 661 men),
the Skaven 2575 (Warlord, 2 clanrat spearmen, 4 skavenslave spearmen, 4
skavenslave slingers — 11 units, 1601 men). The Skaven line has 6 spear units
(clanrats in the centre), the slingers 35 m behind them, the Warlord 60 m. Gap
350 m, battle limit 1200 s.

| Who attacks | Battles | Skaven won | Battle, s | First contact, s |
|---|---:|---:|---|---|
| Empire | 5 | 5 | 627 (488–859) | 97–112 |
| Skaven | 5 | 5 | 835 (560–1047) | 75–364 |
| All | 10 | **10** | 731 (488–1047) | |

| Run | Ours | Our order | Attacker | Battle, s | Empire: men at the end, HP lost | Skaven: men at the end, HP lost |
|---|---|---|---|---:|---|---|
| `183439` | Empire | attack | Empire | 488 | 235, 79 % | 811, 64 % |
| `183733` | Empire | defend | Skaven | 1047 | 165, 84 % | 410, 70 % |
| `184100` | Empire | attack | Empire | 513 | 270, 79 % | 880, 65 % |
| `184351` | Empire | defend | Skaven | 805 | 286, 81 % | 936, 56 % |
| `184732` | Empire | attack | Empire | 552 | 277, 76 % | 827, 64 % |
| `183555` | Skaven | attack | Skaven | 840 | 83, 91 % | 732, 66 % |
| `183925` | Skaven | defend | Empire | 859 | 160, 87 % | 514, 64 % |
| `184220` | Skaven | attack | Skaven | 560 | 256, 81 % | 973, 62 % |
| `184546` | Skaven | defend | Empire | 723 | 299, 79 % | 639, 61 % |
| `184908` | Skaven | attack | Skaven | 924 | 187, 87 % | 671, 58 % |

- **The Skaven won all 10**, in both roles and under both AIs (CA's planner and
  the game's AI). An equal budget by cost does not make the armies equal: the
  Skaven have 2.4 times as many men.
- **Rout and rally.** In every battle all 7 Empire units routed; 3–6 rallied. Of
  the Skaven 5–10 of 11 routed, and nearly all rallied (4–10). Shattered by the
  end: the Empire 0–5, the Skaven 0–6.
- **Losses over time** (the army's share of HP every 30 s) are in `targets.json`
  (`hp_share_every_30s`, `men_every_30s`). In the first 3 minutes after contact
  the Empire loses 46–65 % of its HP (mean 53 %), the Skaven 32–52 % (mean 42 %).
- When the Skaven attack and the Empire defends under the planner, the first
  contact comes after ~6 min (363–364 s): the game's AI takes long to close in.

## Acceleration and charge (from the pairs)

Speed between the one-second records before contact:

| Unit | Top, m/s | Run in the passport | Last 2 s before contact | Charge in the passport | To 90 % of the top, s |
|---|---:|---:|---:|---:|---:|
| Empire spearmen | 3.3–3.4 | 3.0 | 3.0–3.2 | 3.8 | 1–11 (mostly 3–5) |
| Clanrat spearmen | 4.0–4.3 | 4.2 | 4.1–5.1 | 4.8 | 3–16 |
| General | 3.4 | 3.4 | 1.9–2.3 | 4.1 | 2 |
| Warlord | 4.0 | 4.0 | 2.9–4.5 | 4.7 | 2–3 |

- The run matches the passport for the lords and the clanrats. The Empire
  spearmen run ~10 % faster than the passport's 3.0 m/s.
- The charge speed (above the run) shows only here and there (clanrats 5.1,
  Warlord 4.5); one-second records cannot single it out.
- Full run is reached 1–3 s after starting to move (at a 1 s record step); this
  agrees with `acceleration` 2 m/s² in the passport. The long run-ups (up to 16 s)
  are the game's AI walking first and running later.

## Not measured

| What | Why |
|---|---|
| Arrow and sling damage by distance | The game's AI shoots only from the edge of its range (118–135 m); needs an experiment at set distances, like the range `archer-range --range-mode damage` |
| Whether the loser of a pair rallies | A one-unit-against-one battle ends as soon as a unit routs; rallies are seen only in whole battles |
| Time between blows (`attack_interval_s`: 5.7 s for the Empire spear against 4.0 in general) | Losses cannot separate it from the number of men hitting; a lord's 0.25 "events" a second agree with 4 s, but the gaps between them are mostly 2–4 s |
| Charge speed and acceleration finer than 1 s | One record per second; needs frequent position polling |
| Projectile accuracy and `calibration_*` | Only the hit rate at the edge of range (0.42 and 0.47); how it changes with distance is not seen |
| Shields | The targets in the shooting experiments have no shields |
| How many men really fight | Derived from the database rule (above), not measured |

## Surprises

- The Skaven with an equal budget won all 10 whole battles.
- The database rule does not fit for lords: to take 8 HP/s from the General the
  rule needs ~38 clanrats hitting at once, and that many cannot stand around one
  figure ([melee](../game/units/melee.md): up to ~8 reach a lord). So a lord is
  hit harder than the rule gives — for example armour or the hit chance work
  differently.
- The shares of men hitting differ between the two sides of one pair (Empire
  spearmen 11 %, slaves 24 %), while the earlier analysis gave 13–18 % for all.
- The reload in practice is longer than the passport's: the sling 11.5 s instead
  of 9.
