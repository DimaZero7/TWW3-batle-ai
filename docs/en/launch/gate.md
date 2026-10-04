# Checking the network in the game

[← Back](README.md) · [Documentation](../README.md) · [Watching the network](watch.md) · [Русский](../../ru/launch/gate.md)

One command: N real battles where our side is commanded by a trained network (a checkpoint) and
the other by the game's AI at fair Normal difficulty. The armies come from the
[army generator](../training/armies.md) on evaluation seeds the network never saw in training. The
network attacks and defends in turn, in swapped pairs: every seed twice, the network on either army
([fair metrics](../training/training.md#network-evaluation-fair-metrics)). The gate is passed with
**4 battles (2 pairs), at least 3 wins**. It is
a quick check before the night one ([readiness](../training/network.md#readiness)), not that check.

## Launch

Docker Desktop running, the game closed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/gate.ps1 -Checkpoint build/nn-train/latest.pt
```

What happens in each battle:

1. `python -m tools.build nn-arena --own-ai net --army-seed <seed> --own-role attack|defend --speed 20
   --timeout 3600`: a generated battle with our side under the network ([build](build.md#options)).
   The battle file goes to `build/nn-arena/random_<seed>.xml`; `scenarios/` is not changed.
   The second battle of a pair adds `--army-swap`: our side gets the generator's enemy army and the
   game's AI its own army (`random_<seed>_swap.xml`; the run's `army.swap` is true).
2. [watch.ps1](watch.md) `-NoBuild -LingerSeconds 0`: the companion in the container with this
   checkpoint, one battle per game launch through [launch.ps1](run.md): Normal difficulty
   (`battle_difficulty 1`), the user's preferences restored byte for byte, only our files removed.
3. A battle without an outcome (the game crashed while loading and the like) is run once more
   (`-Retries 1`).

At the end: `build/nn-gate/<time>/summary.json` and a table. Exit code: 0 passed, 1 not passed,
2 the gate stopped (for example, the game was already open).

| Parameter | Default | What it does |
|---|---|---|
| `-Checkpoint` | `build/nn-train/latest.pt` | The network's weights; a path inside the repository (the container sees it as `/repo`) |
| `-Battles` | 4 | How many battles |
| `-Offset` | 0 | Skip this many battles of the plan: another set of battles |
| `-Speed 1\|3\|10\|20` | 20 | Battle speed |
| `-TimeoutModelSeconds` | 0, as the simulator | Battle limit, s of battle time; 0 is `battle_limit_s` of `config/nn/sim.json` (3600) |
| `-DecideMs` | 1000 | Battle time between two decisions of the network, ms |
| `-Greedy` | off | The most likely order; by default orders are sampled, as in training and in the simulator check (`tools/nn/train/evaluate.py`) |
| `-Retries` | 1 | Reruns of a battle without an outcome |
| `-Symmetric` | off | 4 battles a seed (`gate.SYMMETRIC`): each army attacks and defends under either side (below) |

One battle of the plan: `-Battles 1 -Offset 3` is the 4th (the largest armies).

## Which battles

`tools/nn/gate.py`, `python -m tools.nn.gate plan --battles 4`:

- Seeds come from the game-check block `1 000 900 000 … 1 000 999 999`: the last 100 000 of the
  generator's `EVAL_SEEDS`. Training takes `TRAIN_SEEDS`, so the network never sees these battles.
  The lower part of `EVAL_SEEDS` is left to the simulator check.
- Battles come in swapped pairs: battles 1 and 2 are one seed, the network on the generator's own
  army, then on its enemy army; the same army attacks in both, so the network attacks in battles
  1, 3, … and defends in 2, 4, … A pair is won both, split or lost both: a faction matchup that
  decides single battles weighs on both battles of a pair alike, so a pair won both says more about
  the network than two single wins. No script baseline in the game (the game's AI against itself
  would cost a launch per battle).
- `-Symmetric` (`plan --symmetric`): in the plain plan the generator's own army always
  attacks, so the network defends only with the other army and a matchup's roles never turn round.
  Symmetric blocks play a seed 4 times: own army attacks, enemy army defends, own army defends,
  enemy army attacks (pairs 2k+1 and 2k+2; in the second pair the generator's enemy army attacks).
  Use a multiple of 4 battles.
- The plan walks the block in order and takes a pair's seed by the size of the generator's own army
  (units besides the lord), in turn 1-4, 10-14, 5-9, 15-19 (`PAIR_BINS`). So even a short plan has
  small and large armies.

The first 4 battles (with the current pools; they follow `config/nn/pools.json`):

| Battle | Pair | Seed | Our army | The game AI's army | Network |
|---|---|---|---|---|---|
| 1 | 1 | 1000900000 | Empire, lord + 1 | Empire, lord + 1 | attacks |
| 2 | 1, swapped | 1000900000 | Empire, lord + 1 | Empire, lord + 1 | defends |
| 3 | 2 | 1000900007 | Skaven, lord + 10 | Skaven, lord + 9 | attacks |
| 4 | 2, swapped | 1000900007 | Skaven, lord + 9 | Skaven, lord + 10 | defends |

If a network version is chosen by these battles, they stop being an independent check: take
another set (`-Offset`) for the final check.

## A battle's outcome

The outcome is the game's `result` event (`src/entries/nn_arena.lua`):

| `status` | Winner |
|---|---|
| `completed` | side `winner` (1 the network, 2 the game's AI) |
| `timeout` | time is out: the defender (the game's rule: the defender wins on time) |
| `stalled` | nobody took damage for 10 minutes of battle: the defender, as if time were out |
| `deadline`, `incomplete`, no result | no outcome; not a win |

A battle not at Normal difficulty (`battle_difficulty` in `launch.json` other than 1) does not count.

## What the gate writes

`build/nn-gate/<time>/`:

| File | What |
|---|---|
| `battles.json` | the plan and each battle's run folder (`build/nn-arena/runs/<time>/`) |
| `summary.json` | wins, losses, no outcome, `passed`, `fair`, `preferences_restored` and a row per battle |
| `summary.json`: `by_faction` | wins and battles by our faction and role and by matchup, ours first (EMP-SKV), battles without an outcome counted; also a line under the table |
| `summary.json`: `pairs` | the swapped pairs: shares won both / split / lost both, `pair_score` (won both − lost both), incomplete pairs (a battle without an outcome or not played), and per pair (`each`) its seed, outcomes and result; also the table's last line |
| `summary.json`: `pair_gold` | the pairs' gold balance ([fair metrics](../training/training.md#network-evaluation-fair-metrics)): ours minus the game AI's on the same armies / budget, the exchange factor, the weak army's destroyed / lost in our hands vs the game AI's; a battle's `gold` (lost per side from the units' end state × passport cost, destroyed, budget, trade, margin); a line under the table |
| `summary.json`: `liveliness` | measured only ([liveliness](../training/training.md#liveliness)): our network's order changes, attack-target switches, A→B→A flips, move jitter and twitching out of melee per role, from the given orders (`nn_orders`); both sides' own target switches every second (`nn_sample`): the game's AI as the reference band (median [min, max] of the battles); a battle's `lively` (the counts); two lines under the table |

`pairs.each` and separate pair lines include `battles`, `pair_gold` and `verdict`: 2–0 means `net stronger`;
0–2 means `game AI stronger — analyse`; 1–1 means `armies decide, even` for gold from −0.10 to +0.10
(upper bound excluded), `armies decide, we trade better` at ≥ +0.10,
or `armies decide, we trade worse — analyse` below −0.10 (a soft loss). One pair's gold is
(the network's destroyed gold in both battles − its lost gold in both battles) / budget; positive means better trades.
An incomplete pair or a split without gold has verdict `incomplete`. The line `analyse: battles N, M, ...`
and JSON list `analyse` include both battles of every pair whose verdict says `analyse`; otherwise the line is `analyse: none`.

A battle's row: seed, factions and army templates, budget (B, each side's budget and cost), units per side (lord included), role,
who won and how, battle length, men at the start and at the end, units still standing, the
network's counters (`nn`: decisions, answers, misses, orders given, `keep`, bad files), Lua errors,
difficulty, preferences restored.

The table:

```
 # pair       seed  factions   units   role  winner       how battle s    men left nn moves/miss/orders
 4    - 1000900008   EMP-SKV   19v20 defend game_ai completed    567.6    866/2996           568/0/4306
```

`pair` is the pair and `s` its swapped battle (`-` in a gate planned before the pairs); `units` counts the lord; `men left` is the network's / the game AI's men.

## A trial battle

A trial battle: `gate.ps1 -Battles 1 -Offset 3` (battle 4 of the plan), ×20, checkpoint
`build/nn-train/latest.pt` (12 updates, barely trained), sampled orders. Run
`build/nn-arena/runs/20260930-224052`, gate folder `build/nn-gate/20260930-224050`.

| What | Result |
|---|---|
| Armies | Empire: lord + 18 (spearmen, archers), 1981 men; Skaven: lord + 19 (clanrats, slaves, slingers), 3041 men; budget 6318 |
| Decisions | 568 in 567.6 s of battle, every one answered, no misses, no bad files |
| From writing the state to the orders given | real time: median 43 ms, at most 199 ms |
| Orders given | 4306 (move 1020, withdraw 905, attack 1567, hold 814), plus 3089 `keep` |
| Carried out | for 1919 of 1925 move and withdraw orders the game's order point (`ordered_position`) is the network's point (median difference 0 m) |
| Battle | the game's AI won: the network kept 866 men and no unit standing, the Skaven 2996 and all 20. An untrained network does not win |
| Speed | 567.6 s of battle in 94 s of real time (×6 at ×20: 39 units). The battle's deadline (840 s) allows ×5 and slower |
| Fairness and cleanup | `battle_difficulty` 1, the user's preferences restored; no Lua errors; the game closed, the companion's container stopped, our pack, mod list and exchange files removed; `scenarios/` unchanged |

One game launch.
