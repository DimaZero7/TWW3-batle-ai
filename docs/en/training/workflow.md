# The working process: a chain step, the run card, metric profiles, the gap card, leftovers, speed

[← Back](README.md) · [Documentation](../README.md) › [Data for training](README.md) › Workflow · [Русский](../../ru/training/workflow.md)

How a step of the training chain is run and judged, and the small tools that take the chores
out of it (`tools/ops`). The standing rules every agent follows are in the repository's
`CLAUDE.md`; a brief adds only the task.

```mermaid
flowchart LR
  l["leftovers<br/>nothing running?"] --> s["step<br/>the command of the step"] --> r["bash … > log<br/>wait.sh on the log"] --> c["card<br/>the numbers, the change"] --> d["decide:<br/>keep / repeat / fix"] --> l
```

## One cycle

| What | How | Instead of |
|---|---|---|
| Is anything left from the last step? | `.venv/Scripts/python -m tools.ops.leftovers` — our containers, the GPU lock, wait loops, the game, the GPU's load; exit 1 when something is left; `--kill` ends wait loops older than 10 min, `--unlock` removes a stale lock, `--stop NAME` stops one of our containers | `docker ps`, `ls build/*.lock`, `ps` by hand, and the misses: nine containers at once (training 2x slower), a trap that deleted another job's lock |
| The next step's command | `.venv/Scripts/python -m tools.ops.step --label n4 --from n3_consolidate [--minutes 25] [--set drills=0.1] [--drop teach-normal] [--write build/steps/n4.sh] [-- run.py options]` | sed-editing the previous step's script (a wrong critic or reference goes unnoticed) |
| Waiting for a run or a test | `bash tools/ops/wait.sh -t 5400 build/steps/n4.log '^written:\|Traceback'` — a timeout, case-insensitive, exit 2 on time | `until grep …; do sleep 60; done` — ran for hours when `FAILED` came in capitals |
| A simulator or drill change is ready, the previous step still trains | `.venv/Scripts/python -m tools.ops.baselines --build <main checkout>/build --run` from the worktree with the change — the new version's references on the CPU, ~8 min ([the cache](#the-baselines-cache-and-the-canary)) | the step waited 25–50 min for them at its "before" evaluation |
| Judging the step | `.venv/Scripts/python -m tools.ops.card build/nn-train/test5/n4 --prev build/nn-train/test5/n3_consolidate` | reading ~200 lines of `trend.md` for ten numbers |
| Computing only what is needed | `--profile drills,transfer` of `tools.ops.step`, `test5` and `tools.nn.gate summary` ([metric profiles](#metric-profiles)) | every evaluation computed everything: drills, transfer, liveliness, capacity |
| Where the simulator differs from the game after a gate | `DOCK_NAME=agent-gap bash tools/nn/dock.sh tools.ops.gapcard build/nn-gate/<time>` ([gap card](#game-vs-sim-gap-card-toolsopsgapcardpy)) | one-off replay scripts in `build/…` for every question |

## The chain step (`tools/ops/step.py`)

The step takes the previous label and derives what the chain needs
([the chain's settings](training.md#the-chains-settings)): its last `m<minute>.pt` (or the one
named, `--from n3/m15`) is the `--init` and the `--reference`, its run's `runs/test5_<label>/latest.pt`
the `--critic-init`; the standing options come from `config/train-chain.json` (`--set key=value`
overrides one, `--drop key` removes one, options after `--` are appended as given; `run.py` takes the
last value of a repeated option). The container is `orch-<label>`; `--gpu-duty 0.9` sits in the
options file. The tool prints the two-line script (or writes it with `--write`), the init and
critic it chose, whether the references' cache (the script baselines and the drill check scripts)
will hit, and the expected wall time: training + (points + 2) evaluations of ~3.3 min (+ ~10 min on a
cache miss; ahead of it: [`tools.ops.baselines`](#the-baselines-cache-and-the-canary)). It runs
nothing: the orchestrator starts the script in the background and waits on its log. `--profile`
(default: the options file's `test5.profile`, else `full`) goes to `test5`: which metrics each
evaluation computes ([metric profiles](#metric-profiles)).

The options file is outside `config/nn` on purpose: that folder is part of the simulator's
version (below), and an edit there would make the baselines miss.

## The run card (`tools/ops/card.py`)

One table per step, a column per evaluation point: the rating (logit, higher is better), pair
gold per opponent (ours minus the opponent's with the same armies, / budget; positive: we trade
better), the exchange vs `ai_like`, each drill's win rate with the skilled script's beside, each
drill's transfer to normal battles (APPLIED share — higher is better; MISTAKE share — lower is
better; network / `ai_like`), the teachers' shares, own lord dead, the kinds hold / move / attack,
missile time in melee, timeouts, the evaluation's seconds (a cache miss shows here: ~500–600 s
instead of ~190), the KL distance from the start. The last column is the change from the first
point to the last, marked `*` beyond the noise (two standard errors where the number has one:
rating, pair gold, drill wins, lord deaths; |change| ≥ 0.05 otherwise). Below: anomalies (a rating
fall beyond noise between two points, the move share or lord deaths ×1.5, timeouts above 5 %, a
drill's win falling ≥ 0.15), the capacity verdict, the training's minutes and updates. `--prev`
adds the previous step's end point and the change from it; `--json` the data.

The line `computed:` under the table names each point's metric profiles (`full` for older
evaluations): a "-" in a row its profile did not compute means "not computed", not zero. The row
"own unit-time exhausted" is the share of own units' time exhausted, "own moves running far from the fight"
the run share of move orders far from the fight (profile `fatigue`; ~100 % in the gates' games).

It reads `report.json`; while the run goes, `before.json` and the `eval_m<minute>.json` written so
far (the fair metrics computed with `skill.py`, numpy), so the card of a running step is a command
away.

## Metric profiles

Every evaluation used to compute everything. Now a small **mandatory set** is always computed and
the rest by profiles chosen for the run's question (`tools/nn/train/profiles.py`). What a metric
means does not depend on the profile: the profile only says whether it is computed.

| Where | Always (mandatory) | Profiles (`--profile a,b`; `full` all, `mandatory` none) |
|---|---|---|
| `test5` (and `tools.ops.step --profile`) | the rating, pair gold against each script (with the script baselines), own lord deaths, wins and gold, order kinds | `behaviour` (alias `shooters`: missile units in melee, flanks, piles, abilities), `liveliness`, `fatigue` (the share of own units' time by fatigue state; the run share of move orders far from the fight), `transfer`, `drills`, `capacity`; default `full` |
| `python -m tools.nn.gate summary <folder> --profile …` | outcomes, pairs, pair gold, own lord deaths (from the recordings) | `liveliness` (default), `routs`, `lords`, `fatigue`, `activity`, `shooters`, `wrap` — each recording's numbers (as in the gap card) |
| `tools.ops.gapcard` | win, trade, own and enemy gold lost, own lord death, battle length | `routs`, `lords`, `fatigue`, `activity`, `shooters`, `wrap`; default `full` |

The teachers add what they read: `--drill-teach auto` adds `drills`, an adaptive `--teach-normal`
adds `transfer` (`test5` prints what it added). The chain runs both teachers now, so
`--profile mandatory` there gives `mandatory+transfer+drills`. Every evaluation keeps its
`profile`, `report.json` too; the run card shows it.

The cost of one evaluation (CPU, 8 cores, `s6_fatigue/m20`, 64 battles per opponent = 192 of the
network's battles, 32 per drill; the scripts' references left out — in a real run they are a cache
file read): `mandatory` 470 s, `full` 1888 s, 4 times longer. The difference is the drills' battles
and the behaviour, liveliness and transfer counters at every simulator step. On the GPU the counters
are compiled and weigh less (`full` there is ~200 s at 512 battles per opponent); not measured there,
the GPU is training.

## Game-vs-sim gap card (`tools/ops/gapcard.py`)

After a gate one command says where the simulator differs from the game on the same battles:

    DOCK_NAME=agent-gap bash tools/nn/dock.sh tools.ops.gapcard build/nn-gate/<time> [--profile fatigue,lords] [--copies 8]
    .venv/Scripts/python -m tools.ops.gapcard build/nn-gate/<time> --from-json [--profile routs]   # reprint, no simulator

Each gate battle starts in the simulator from its recorded start (places, bearings, men, widths,
factions — `scenario.from_recording`): side 1 is the gate's checkpoint (live, sampling its orders
when the gate did), side 2 the `ai_like` script, at the game's cadence (a decision a second, the
orders ~0.36 s late), `--copies` copies (default 8; the copies differ by the start's 2 m jitter and
the sampled orders). Every copy is recorded once a second as the game records (`check.Recorder`),
and the game's recording and the copies are measured by the same code (`tools/nn/battle_metrics.py`)
up to the game battle's end time (the result and the length: the whole battle). The replay is
`tools/nn/train/gapsim.py` (the closed loop from the analyses `build/fat/opus/gate_check`,
`build/pos/opus`, `build/gap4/astra`; their probes — perturbed numbers, recorded-order replay,
observation tweaks — stay there).

The table: a row per metric, a column per battle "game / sim (mean of the copies)", then over all
battles the game's and the simulator's means and their difference. A mark when the difference is
beyond the noise: in one battle `!` when |game − sim| > max(2 × the copies' spread × √(1 + 1/n),
threshold); over all battles `*` when |difference of the means| > max(2 × √Σ(spread² × (1 + 1/n)) / k,
threshold). The copies' spread is the noise of one battle, the game's single battle is counted with
it. The threshold is the smallest difference that matters (a share and the trade 0.05, a win 0.25,
the length 30 s, routs 0.15 per unit, the line lap 15 m, abilities 0.5, the first ability 30 s, a lord's health loss
0.002 a second). Under the table the marked rows in words. The battles' numbers go to
`<gate folder>/gapcard.json`.

| Profile | What is measured |
|---|---|
| mandatory | win (1 / 0); trade = (enemy gold lost − ours) / budget; own and enemy gold lost / that army's cost (as the gate counts: dead, shattered or gone whole, routing half of what is left); own lord dead (health or men 0, or shattered); battle length |
| `routs` | rout onsets per unit, own / enemy |
| `lords` | a lord's health lost per second of his melee: ours ("lost") and the enemy's ("dealt"); enemy lord dead; our lord's ability uses (count, first use time; in the game the bridge's `nn_ability` events) |
| `fatigue` | the share of the units on the field tired or worse in the bins 0–120 / 120–240 / 240–360 / 360+ s; exhausted over the whole window; own / enemy. The run share: of our units' time under a move / withdraw order far from the fight (before the battle's first melee and no standing enemy within 150 m, about a bow's range), the share that runs (the orders in force: in the game the bridge's given `nn_orders`, in the simulator `order_kind` / `order_run`; in `test5` the same rule in `behaviour.Fatigue`) |
| `activity` | the share of standing units' time in melee / firing / moving / still; own / enemy |
| `shooters` | missile units' (not the lord's) time in melee, own / enemy |
| `wrap` | the share of melee entries from the rear (a melee unit, not the lord nor a missile unit, going into melee at 120° or more off its enemy's facing; the enemy: its target when that one is in melee, else the nearest standing enemy), own / enemy; the share of the side's units' melee time (not the lord's) with an enemy behind (the rear flag `bf`), own / enemy; how far the enemy's melee line laps ours at the first melee, both ends summed, m (across the line between the two lines' centres) |

Gate `20261005-161910` (`s6_fatigue/m20`, 4 battles, 8 copies, 246 s of simulation on 8 cores,
~4.5 min with the container), game / sim mean, marked beyond noise: wins 0.00 / 0.78; trade −0.39 /
+0.10; own gold lost 0.96 / 0.67 of the army, the enemy's 0.57 / 0.76; own lord dead 0.75 / 0.25;
routs per unit own 1.52 / 0.92, enemy 0.58 / 1.24; own time in melee 0.50 / 0.38; missile units in
melee own 0.25 / 0.10, enemy 0.19 / 0.02; tired or worse at 240–360 s 0.97 / 0.80 (own) and 0.96 /
0.79 (enemy); exhausted own 0.20 / 0.14. Matching: the lords' health loss per second of melee
(0.0056 / 0.0059 ours, 0.0034 / 0.0036 the enemy's), abilities (2.75 / 2.50 a battle, the first at
171 / 151 s), the firing and moving shares, running far from the fight (1.00 / 1.00). So in the
simulator the enemy routs twice as often and missile units are hardly ever caught in melee.

## The baselines' cache and the canary

The fair metrics compare the network with the script playing itself on the same battles
([test protocol](training.md#test-protocol-test5)), and the drill evaluation compares it with the
drill's two check scripts (naive and skilled). These script battles ("references") are cached in
`build/nn-train/baselines/`:

| File | What | Version: a hash of |
|---|---|---|
| `<opponent>_19u_3600s_<version>.json` | `ai_like`, `nearest`, `hold_shoot` against themselves, 256 pairs each | the files a script battle depends on: `tools/nn/train/version.py` `VERSION_FILES` (the simulator, the armies, the scripts, the reward, the model, `config/nn`) |
| `drill_<drill>_128_broad<share>[_embed<share>]_<version>.json` | `kiting`, `counter`, `hold_fire`: naive and skilled, 128 battles each | the simulator version and `tools/nn/train/drills` (`version.drill_version`) |

A change to any of those files misses the cache.

**A miss is played on the CPU.** Before its first evaluation `test5` starts the missing files
(`tools/nn/train/refs.py`): each script's baseline and each drill check script is a process of its own
with one thread (9 jobs, the longest first) while the GPU plays the network's own battles; the
evaluation waits for the files where it reads them. The battles step compacted — only those still
running: on the CPU a step costs in proportion to the battles it steps, and the battles' lengths differ
a lot (kiting: median ~600 s, 10 % to the 3600 s limit). Measured on version `080d6157e23f`, 8 cores:
8 min 6 s from the container's start (ai_like 461 s, hold_shoot 408, kiting 410 + 477, hold_fire 389 +
325, nearest 212, counter 38 + 43). In game mode (5 cores): ~10 min estimated.

**Ahead, so the training never waits.** `.venv/Scripts/python -m tools.ops.baselines` prints the versions
and what is missing; `--run` plays it in the container `orch-baselines` (8 cores, no GPU; wait with
`bash tools/ops/wait.sh` on its log). When a simulator change is ready in an agent's worktree while
the previous step still trains, from that worktree:
`.venv/Scripts/python -m tools.ops.baselines --build <main checkout>/build --run` — the files land under
the new version (the version is a hash of the files' contents, the commit gives the same one) and the
next step finds them. `tools.ops.step` says whether the cache will miss (drills included) and names
this command.

**One file, one process.** While a process plays a reference file, a marker `<file>.computing` lies next to
it; a thread of that process touches it every 30 s. Another `test5` or `tools.ops.baselines` that finds a
live marker waits for the file instead of playing it a second time (`test5` used to repeat
`orch-baselines`' work). A marker untouched for 150 s is a crashed process's: it is taken over. If the
process with the marker stops without writing the file, the evaluation plays it itself.

**The canary** (`test5 --baseline-canary 32`, the chain's default; `refs.py` the same): on a miss a
script's baseline plays its first 32 pairs first; when they come out identical in every field (winner,
gold lost, start, budget, factions, attacker) to an older version's file, that file is adopted under
the new version — a change that cannot touch a script battle (the network, a reward term) costs ~1 min
(ai_like: 59 s instead of 461). The 32 pairs come from the whole baseline's batch
(`script_battles(first=…)`): the start places' jitter is drawn over the whole batch
(`randomise.apply`), and the same pairs in a batch of their own are other battles. The CPU repeats
itself exactly (two runs: the same numbers), so the canary compares files made on the CPU; it never
adopts an earlier version's GPU file. The drills have no canary.
`python -m tools.nn.train.version` says the simulator version and whether its files exist (on the
host without torch, the same hash as the container).

**The noise of a recompute.** The rating and the pair gold do not use the baselines. A script's
baseline enters only the network's edge over the script (`baseline`: `gold_adv`, `win_adv`). In the
overall cell every pair is the script's battle from both sides, which sums to exactly 0 trade and 0.5
wins, so the overall cell does not depend on the baseline file (CPU and GPU: equal to 4 digits on
`s6_fatigue/before.json`). The cells by role and faction depend on the realisation: the CPU instead of
the GPU (the same armies, 83 % the same winners, not one battle exactly the same) moved them by
0.2–0.8 standard errors at the median, 2.2 at most (`nearest` attacking: 0.106 → 0.079, error 0.014).
That is the baseline's own noise, no larger than before: the script trade's spread over battles 0.35 /
0.30 / 0.27 against 0.34 / 0.30 / 0.28 on the GPU; any recompute after a simulator change moves these
cells the same way. The drill check scripts, CPU against GPU: kiting naive 0.742 / 0.672 wins, skilled
0.828 / 0.805; hold_fire 0.203 / 0.164 and 0.867 / 0.844; counter 1.0 / 1.0 and 0.992 / 1.0; the trade
within 0.02. The drill teacher takes the gap skilled − network, so its share moves by ~0.01.

## Step speed

A 20-minute step with a baseline-cache hit took ~28–30 min of wall time, an evaluation a median ~3.3 min
(245–515 s). What was done (measurements and scripts: `build/speed1`). A check step (6 min from
`s11b_noteach/m20.pt`, the chain's options): "before" 181 s (with compiling), the minute-3 evaluation 107 s,
the "after" 123 s, against 167–177 s with the old code on the same network. A faster rollout and bank do not
shorten a step (it is capped in minutes): they give more updates in the same minutes.

| What | How now | Gain |
|---|---|---|
| The "before" evaluation is not repeated | A chain step starts from the previous step's last network, whose last evaluation (`after.json`) exists already. `test5` looks in the folder of `--init` (`after.json`, `eval_m*.json`, `before.json`) for an evaluation of the same key and takes it as the "before" (field `reused`: from where; without the training's blocks: `distance`, `teach`, `teach_auto`, `teach_normal`). The key (`test5.eval_key`): a hash of the network's weights, the code of the simulator, the drills and the evaluation itself (`version.eval_version`, `EVAL_FILES`), the battles, the profile, the cadence, the drills' frame shares. `--fresh-before` plays it again | 166–355 s a step |
| TF32 from the start | `run.train` turned TF32 on after the "before" evaluation, and the compiled graphs are guarded on that switch: the first evaluation after training compiled everything again (`TORCH_LOGS=recompiles`: `GLOBAL_STATE changed: allow_tf32`). Now `test5` turns TF32 on before the first evaluation | the first evaluation after training compiles nothing again (+17 s on a warm compile cache here, 60–150 s in the audit) |
| The drills' evaluation steps the running battles only | The drills' evaluation (3 × 128 battles) stepped its whole batch until the last battle ended (up to 3600 s): 90–120 s of a ~175 s evaluation. Now the batch shrinks to the 64 running battles and the shrunk batch's decision is replayed as one CUDA graph, as in the main evaluation | drills 66–70 → 55 s, the whole evaluation 112–120 → 106 s (one network, one process) |
| A teacher at share 0 | The teacher's script labelled every decision (~20 % of the rollout), also at share 0, when no label reaches the training. Now at share 0 the script runs on every 8th decision only (`rollout.TEACH_PROBE`): the log's agreement is still measured, on a smaller sample | rollout 5.8 → 4.7 s an update at shares 0: ~6 % more updates in the same minutes |
| The next bank of armies built ahead | A bank (2048 battles and the drills' battles) took 33–49 s every 5 min while the GPU waited (logs `s10_lord`–`s11b_noteach`: 3–4 times a step, 110–140 s). Now the next bank is built in a thread while training goes on (an evaluation inside the step waits for a build in progress: it captures CUDA graphs); `--no-bank-ahead`: as before | instead of 33–49 s idle, ~18 s of two slower updates (the thread shares the Python interpreter with the training) |

Small ones: `docker stop` used to leave the GPU lock behind (a container's first process without a handler
does not get SIGTERM and is killed after 10 s); now `test5` exits normally on SIGTERM and frees the lock.
`DOCK_CPUS=N bash tools/nn/dock.sh …` also sets `OMP_NUM_THREADS` / `MKL_NUM_THREADS` = N: a 6-core
container ran torch with a thread per host core.

The evaluation's numbers do not change: the "before" from the previous step is the same network on the
same battles (a fresh evaluation would differ only by the sampled orders: two evaluations of one
`s10_lord/m20.pt` gave a rating of 0.76 and 0.64, 30 % of the battles identical). The drills' compaction
changes only the sampled orders of the battles still running (on the CPU with greedy orders the shrunk
and the whole batch give the same battles, `tests/tools/test_nn_eval_pairs.py`).

`evaluate.TRACE = []` records at every check of the batch (step, running battles, batch size, time): how
the evaluation's time was measured.

## Tried and rejected

- Intermediate compaction sizes in the main evaluation (1024, 512, 256, 128 and 64 instead of 64 alone):
  95 s instead of 56 s on the same network (`s11b_noteach/m20.pt`). Every new size is a new CUDA graph and
  a new compilation of the order assembly (its shape depends on how many of the network's units are on
  each side, which changes in almost every evaluation); the whole 1536-battle batch steps at ~40 ms a
  decision and gets down to 64 running battles in 30–40 s: there is little to save.

- The references on the GPU inside the "before" evaluation: the baselines ~5 min, the drill check
  scripts ~25–37 min (uncompiled and not compacted: a batch of 128 battles stepped until its last one
  ended); a "before" with a miss took 1858–3112 s instead of ~200, and the training waited.
- The canary on a batch of the first 32 pairs alone: the start places' jitter is drawn over the whole
  batch, and in a batch of 32 these are other battles — the canary never matched (6 misses of 6, each
  ~25–50 min).
- A process per drill (naive and skilled in one): 12 min 49 s for everything, kiting alone 762 s. A
  process per drill script: 8 min 6 s.
