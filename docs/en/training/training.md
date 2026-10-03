# Training the network

[← Back](README.md) · [Documentation](../README.md) › [Data for training](README.md) › Training · [Русский](../../ru/training/training.md)

Step 3 of the path ([network model](network.md)): PPO training of the [network](model.md) in the
[battle simulator](simulator.md) on [random armies](armies.md), against itself, its past versions,
scripted opponents and drills. Code: `tools/nn/train/` (PyTorch, runs in the `snake-ai-trainer`
container through `tools/nn/dock.sh`). This page says how it works now; what was tried and dropped
is at the end ([tried and rejected](#tried-and-rejected)).

## How to run it

```bash
# the standard test of a change (below): before -> training -> after, in build/nn-train/test5/<label>/
DOCK_NAME=t2-test5 bash tools/nn/dock.sh tools.nn.train.test5 --label mychange [-- run.py options]
# a trend run of the chain: 15 minutes from the last network, the full evaluation every 5 minutes
DOCK_NAME=orch-x bash tools/nn/dock.sh tools.nn.train.test5 --label x --init <last m*.pt> \
  --updates 0 --minutes 15 --every 5 -- <the chain's options, below>
# a plain run: PPO, then the final evaluation and replays
bash tools/nn/dock.sh tools.nn.train.run --name r1 --minutes 30 --init <checkpoint>
# evaluation alone: random battles of EVAL_SEEDS per opponent, in swapped pairs with the script baseline
bash tools/nn/dock.sh tools.nn.train.evaluate --checkpoint build/nn-train/latest.pt --generated 512 \
  --opponents ai_like,nearest,hold_shoot
bash tools/nn/dock.sh tools.nn.train.checkpoint                  # write an untrained random.pt
```

The first steps compile for 1–3 minutes (seconds once a run of the same shapes compiled them:
[training speed](#training-speed)); training time does not count them.

### Options of `run.py`

| Group | Options (default) |
|---|---|
| run | `--name` (folder under `build/nn-train/runs/`), `--init` (start checkpoint; default `random.pt`), `--minutes` (5; wall time of training), `--updates` (0; if set, train that many updates and `--minutes` only caps the time; schedules follow the updates), `--seed`, `--device`, `--no-eval`, `--print-every` (5), `--snapshot-every` (20) |
| battles | `--battles` (1024 at once), `--steps` (64 decisions per update), `--max-units` (19 units a side), `--small share:units` (that share of every bank with at most `units` a side), `--bank` (2048 ready battles), `--bank-refresh` (5 min), `--limit` (3600 s) |
| cadence | `--decide-s` (1.0 s of battle between decisions), `--order-latency` (0.36 s): [decisions](#decisions) |
| PPO | `--lr` (1e-4), `--gamma` (0.9997 per 0.5 s), `--epochs` (1), `--minibatch` (4096 decisions), `--adv-norm` (`batch` or `role`), `--critic-warmup` (0 updates that train the critic alone), `--critic-init` (the critic from another checkpoint) |
| exploration | `--entropy` (0.01) and `--entropy-end` (linear over the run), `--entropy-target` (0: off), `--entropy-max` (0.1), `--entropy-rate` (1.25) |
| leash | `--anchor` (0: KL weight to the reference) and `--anchor-end`, `--reference` (default `--init`), `--anchor-roll` (0 s: seconds of training between renewals of the reference) |
| reward | `--gold` (1.0), `--rout-share` (0.5), `--lord` (0.3), `--lord-rout` (0), `--idle` (2e-4), `--idle-tau` (150 s), `--idle-pause` (30 s), `--idle-step` (0.5), `--idle-cap` (20), `--idle-rate` (0), `--idle-window` (30 s), `--order-cost` (0.001), `--retarget` (0.003): [reward](#reward) |
| opponents | `--mix` (json shares), `--pool` (8 past versions), `--pool-extra` (more checkpoints for the pool), `--eval-past`, `--eval-generated` (512) |
| drills | `--drills` (0: share of the battles), `--drill-weights` (json; default the `READY` drills equally), `--drill-bank` (256 per drill): [drills](#drills) |

### The chain's settings

Training goes as a chain: each run starts from the last checkpoint of the previous one (its
`m<minute>.pt`, with the critic of its `runs/test5_<label>/latest.pt`). The options the chain
passes after `--`:

```bash
--critic-init <previous runs/test5_<label>/latest.pt> --critic-warmup 3 \
--reference <init> --anchor 0.03 --anchor-end 0.03 --anchor-roll 600 --adv-norm role \
--idle-rate 0.05 --lord-rout 0.5 --entropy-target 0.05 --entropy-max 0.03 \
--drills 0.1 --drill-weights '{"counter": 1}' \
--mix '{"self": 0.05, "past": 0.1, "nearest": 0.35, "hold_shoot": 0.15, "hold": 0.05, "ai_like": 0.3}'
```

The reasons: a small KL to the network's own start that rolls forward every 10 minutes (training
without any leash collapses: [tried and rejected](#no-leash-and-a-leash-only-for-new-networks));
advantages normalised per role (the attacker's, wider with its idle cost, no longer outweigh the
defender's); the progress clock at 0.05 of the budget a minute (the network sees it:
[model](model.md)); a shattered lord counted as dead; drills at 10 % (30 % made the small
network forget: [drills](#drills)).

## What it writes

Everything goes to `build/nn-train/` (not in Git):

| File | What |
|---|---|
| `random.pt` | the untrained network (random weights), the untrained opponent |
| `latest.pt` | the network of the last `run.py` (the companion's default) |
| `runs/<name>/latest.pt`, `best.pt` | the run's last network; the one with the best window of training battles against the scripts (the worst opponent-role win rate with ≥ 20 battles, over ≥ 5 of them) |
| `runs/<name>/pool/v*.pt` | past versions: the opponents of self-play |
| `runs/<name>/log.jsonl` | one line per update: losses, entropy, KL, the entropy and KL weights in force, `start_kl` (distance from the start), `anchor_rolls`, reward and the same by term a minute of battle per role (`reward_parts`), the critic's quality (`ev`, per role), win rate by opponent and role, order changes, order kinds, lord deaths, target switches, ability uses, speed |
| `runs/<name>/eval.json`, `replays/*/` | the final evaluation; battles of the final network written like the game's recordings |
| `test5/<label>/` | `report.json`, `before.json`, `after.json`, a trend's `m<minute>.pt`, `eval_m<minute>.json`, `trend.md` |
| `baselines/<opponent>_<units>u_<limit>s_<version>.json` | the script baselines of the evaluation ([fair metrics](#network-evaluation-fair-metrics)) |

**A checkpoint** (`tools/nn/train/checkpoint.py`) is one file, a dict saved by `torch.save`:
`format` (2), `kinds` (the order kinds in code order), `preset`, `config` (the network's sizes: the
actor is rebuilt from it), `actor` (all the game needs), `critic` (training only), `meta` (with the
`cadence` it was trained at). `load_policy(path)` gives the actor ready to play; the
[companion](../apps/bridge.md) loads it so.

**Replays** are folders like a recorded run: `manifest.json` and `events.jsonl` with an `nn_sample`
every second and the `result`. `tools.nn.gamedata.load(folder)` reads them as a game battle.

## How a training step goes

```mermaid
flowchart LR
  obs["Observe both sides<br/>1024 battles"] --> act["Learner, past version<br/>and scripts give orders"]
  act --> step["2 simulator steps of 0.5 s<br/>(the networks' orders land<br/>0.36 s late; scripts every step)"]
  step --> rew["Reward per side<br/>(summed over the steps)"]
  rew --> reset["Finished battles take a new battle<br/>from the bank, new numbers"]
  reset --> obs
  rew --> buf["64 decisions collected<br/>(a chunk of 64 s)"]
  buf --> ppo["PPO update<br/>actor through the chunk + critic"]
```

Modules: `scenes.py` (where battles come from: a bank of ready starts), `league.py` (who plays whom,
the pool), `opponents.py` (the scripts), `drills/` (drill frames, their banks and checks),
`randomise.py`, `reward.py`, `rollout.py` (the battles and one decision), `cadence.py` (how often
the networks decide), `ppo.py`, `evaluate.py` (evaluation and replays), `skill.py` (the rating),
`behaviour.py` (behaviour and liveliness), `capacity.py`, `matchups.py`, `checkpoint.py`, `run.py`,
`test5.py`.

## Battles and opponents

**Battles.** Random armies of the [army generator](armies.md): equal budget, a lord and up to
`--max-units` units a side, half the bank with side 1 attacking. Training seeds only; the bank
(`--bank` battles) is renewed every `--bank-refresh` minutes; evaluation uses `EVAL_SEEDS`, never
trained on. `--small 0.35:6` keeps 35 % of every bank at ≤ 6 units a side. A finished battle takes
a new one from the bank with new randomised numbers. The learner plays side 1 in some battles and
side 2 in others: every army in both roles. The network sees the map as the game shows it (the
Crossroads minimap frame, as the companion), not the simulator's square. The fixed arenas
(`config/nn/arenas.json`) remain only as scenes of `evaluate --per-scene`.

**Domain randomisation** (`randomise.py`): per battle, damage, speed and leadership are multiplied
by 1 ± 15 % (the same for both sides) and by 1 ± 5 % per side; start places move by up to 5 m. The
network does not see the factors.

**Opponents** (`league.py`; `--mix` shares; the default `MIX` in brackets, the chain's above):

| Opponent | Default share | What it does |
|---|---:|---|
| `self` | 10 % | the learner on both sides; both give training data |
| `past` | 15 % | one network from the pool (`--pool` 8, the untrained one always in it, `--pool-extra` more), drawn again every 2 updates |
| `ai_like` | 40 % | modelled on the game's AI ([below](#the-opponent-ai_like)) |
| `nearest` | 20 % | every unit attacks the nearest standing enemy, running |
| `hold_shoot` | 10 % | holds and shoots; a melee unit counter-charges an enemy within 80 m; as the attacker everyone attacks after 5 minutes |
| `hold` | 5 % | holds (shoots at will, fights back); met only as the defender |
| `drill_<name>` | `--drills` | a drill's enemy on that drill's battles ([drills](#drills)) |

Within each opponent's share the layout cycles the learner's side, so every opponent meets every
army in both roles. The game's AI takes no part in training: it stays an independent check
([network model](network.md#readiness)).

### The opponent `ai_like`

`tools/nn/train/opponents.py`, numbers in `Line`, fitted to the game's AI in 110 recorded
network-vs-game-AI gate battles ("pool"; the analysis scripts in `build/ail/`, not in Git):

| Rule | Number | Source |
|---|---|---|
| the attacker's line advances together at a run (a point 30 m ahead); a unit more than 15 m ahead of its line's rearmost free melee unit waits (the march goes at the slowest unit's pace); after the side's first fight nobody waits | run, 15 m | pool: the game's attacking line marched at 2.8–2.9 m/s, 16–17 m deep |
| a defender with no missile units advances too | — | the gate battles |
| once own units have fought, the free units go forward and join enemies within 150 m, missile units up to their range | 150 m | pool: after the first contact the game's units stand idle 1.6 % (melee) / 8.7 % (missile) of the time |
| the attacker charges from | 95 m | pool: an attacking melee unit's first target at 87 m median (q10–q90 68–105 m) |
| the defender counter-charges from | 100 m | pool: first targets at ~100 m |
| an enemy that wavers is charged from | 150 m | ours |
| missile units halt at a share of range and shoot; the enemy lord first when in range, in melee too | 0.9 | gate battles: with our lord in range the game's missile units shot him 89 % of their firing seconds |
| missile units withdraw 50 m from an enemy melee unit within | 50 m | pool: a free missile unit moves away from a closing melee unit in 52–58 % of the seconds within 40 m, 42 % at 40–60 m |
| a missile unit caught in melee breaks off (withdraw) for its first | 15 s | pool: the game's missile units in melee move 52 % of their first 5 s |
| the lord stays behind his line's centre, never charges first: goes in when the line does, at his target within 90 m, or at an enemy within 50 m that already fights; withdraws below 30 % health | 20 m, 90 m | pool: 20–22 m behind the centre before contact, his first target at 90 m |
| melee targets: score = distance − 20 m if the enemy is in melee (+30 m back if the unit stands within 40 m of an own fight) + 25 m for the enemy lord + 6 m for each other own unit already on it (up to 3) − 10 m × cos(the side's forward, the enemy) + 15 m if it wavers − 27 m × a fixed Gumbel draw per (battle row, unit, enemy); the lowest wins, a new one must be 15 m better | see left | a conditional logit fitted on the game AI's 3 397 new targets of melee units out of melee (`build/ail/choice.py`); hysteresis ours |
| a free unit goes for an enemy already fighting via a point 12 m beside its flank | 12 m | the simulator's tactic scan |

The random part is an integer hash of (battle row, unit slot, enemy slot): the same every step,
nothing to replay. Fitted against the game on the same 8 gate battle starts (side 2's behaviour;
game AI / `ai_like` with these rules): missile units' time in melee 0.11 / 0.10; melee target
distance (median) 91 / 93 m, the nearest enemy taken 0.48 / 0.49; new melee targets a battle 27 /
33; idle after contact 0.045 / 0.031; the lord 22 / 27 m behind the centre. The simulator's
prediction of the network's in-game trade on those battles: −0.245 (the game −0.333). Left:
`ai_like` piles two or more units on one of ours 0.26 of the time (the game 0.10), and its lord
goes in sooner (6 s after the first contact against 12 s).

## Decisions

**Cadence: a decision a second, the orders 0.36 s late, as in the game** (`cadence.py`;
`--decide-s 1 --order-latency 0.36`, the same in `run.py`, `test5` and `evaluate.py`). The
companion decides once a second of battle time (the bridge's `decide_ms` 1000) and its orders reach
the units later: in the gate recordings (`nn_orders` `wait_model_ms`, 9251 decisions) median 0.3 s,
mean 0.36 s, p10–p90 0.2–0.6 s of battle time.

- The simulator's step stays 0.5 s (the game's morale tick); one decision (`Battles.step`) plays
  `decide_s / 0.5` = 2 of them.
- The networks' orders (the learner's and the past version's) land after the latency, rounded to
  whole steps at random so that its mean is 0.36 s: one step late with probability 0.72, at once
  otherwise; until then the orders in force go on (`KEEP`), as in the game between ticks.
- The scripts (`nearest`, `ai_like`, the drills' enemies) give orders every simulator step: they
  stand in for the game's AI, which has no tick of ours.
- The observation and its memory are taken only at decisions, as the companion reads the state
  once a second; the GRU steps once a decision.
- The reward of a decision is the sum of its steps'; a battle that ends within a decision ends it.
- `--steps` 64 decisions per update = 64 s of battle a chunk.
- `--gamma` (0.9997) and GAE's λ (0.95) are given per 0.5 s; per decision they are γ^(decide_s / 0.5)
  (0.99940 at 1 s) and λ^(decide_s / 0.5) (0.9025): the horizon in seconds (~28 min) stays.
- Evaluation decides as training does; its per-step counts (behaviour, drill metrics, recordings)
  still see every simulator step. `--decide-s 0.5 --order-latency 0` gives a decision every step
  with the orders at once (numbers of the two cadences are not comparable).

A 10-minute battle is ~600 decisions. Every checkpoint plays at either cadence.

**The order `keep`.** The network may give a unit no new order: `keep` (kind code 4) leaves the
order in force; a unit with no order holds. A real change is a new kind, a new attack target, or a
move point more than 10 m from the one in force; `keep` and the same order again cost nothing.

**Masks.** Dead, routing and padding units give no loss; invalid orders are masked by the network
itself (attack without a visible enemy, orders to a unit that cannot take them).

**Time limit 60 minutes**, as in the game: the defender wins at it. The network sees no time limit
(no input refers to it: a campaign battle may have none).

## Reward

`reward.py`, per side. The live terms:

| Term | Weight (option) | Why |
|---|---:|---|
| win / loss | ±1 | the goal; at the time limit the defender wins and the attacker gets the normal −1 |
| (enemy gold destroyed − own gold lost) / budget, per step | 1.0 (`--gold`) | a steady signal long before the end, by what the units are worth |
| the enemy lord fell − own lord fell, per step | 0.3 (`--lord`) | in the game a lord's fall breaks the army (3 of 4 lost gate battles ended within 3 s of our lord shattering); his gold alone does not show it. With `--lord-rout` r > 0 a shattered lord counts as dead and a routing one as r of a death, given back on a rally (the chain: 0.5) |
| the attacker's idle cost, per 0.5 s step | 2e-4 × m (`--idle`) | the time limit an hour away is hardly seen at γ 0.9997 (0.9997^7200 ≈ 0.11); this is seen every step. The defender never pays |
| each real order change of a unit, divided by the side's units | 0.001 (`--order-cost`) | against jitter |
| each switch of an attack to another target while the old one still stands, divided by the side's units | 0.003 (`--retarget`) | hysteresis: units dithering between two enemies |

The win, gold and lord terms are zero-sum (side 2 gets minus side 1's). The shaping terms are
potential differences: they do not change which outcome is best. No style terms yet: the faction
characters are placeholders.

**Gold lost** (`reward.gold_lost`, `gold_sides`, `budget`). A unit is worth its `multiplayer_cost`
(`config/nn/units.json`); the gold it has lost is that cost × the worst share of it lost so far:

| The unit | Share lost now | Why |
|---|---|---|
| fighting | its health lost (`1 − hp_abs / hp0`) | for a unit of many men health and men fall together; for a single entity (a lord) health shows the damage before its death |
| dead, gone off the map, shattered | 1 | it never comes back |
| routing (not shattered) | health lost + `rout_share` (0.5) × what is left | out of the fight now, but it may rally |

**A loss counts once.** The state keeps each unit's worst share lost so far (`lost_worst`, stored
by `rollout.Battles` after every simulator step, also in `evaluate.script_battles` and
`drills/verify.py`); gold lost = cost × that worst share. A rally gives nothing back; a rout after a
rally adds nothing until the unit loses more than at its worst (routing at half health: 0.75;
rallied: still 0.75; hit to 60 % lost and routing again: 0.8, +0.05). Without it, a defender unit
that routed, rallied and routed again counted as new damage each time: in 89–92 % of battles, ~7 %
of the damage the progress clock counted (up to 22 %), with false progress shown in 38–59 % of
battles. One rule for the trade, the idle clock, the gold metrics and the network's progress inputs
(the companion runs the same code on the game's states). The budget is the mean of the two armies'
starting cost (the same for both sides, so the trade is exactly zero-sum).

**The attacker's idle cost** (`reward.idle_cost`, `idle_scale`): the attacker pays 2e-4 × m every
step, whichever of its units are busy (progress only: a reserve, a second line or a lord kept back
cost nothing while the army as a whole makes progress); m depends on its damage:

- before its first damage: m = exp(t / 150 s) − 1 (`--idle-tau`): almost nothing for the first
  minutes (time to deploy and march), then sharply more;
- after it: 0 while it deals damage; after 30 s without damage (`--idle-pause`), k = ⌊seconds since
  the last damage / 30 s⌋ and m = exp(0.5 k) − 1 (`--idle-step`): a step up every 30 s, faster than
  the first curve and without its grace; new damage sets it back to 0;
- at most 20 (`--idle-cap`): 0.004 a step, 0.48 a minute.

What counts as damage: with `--idle-rate` 0 any health the defender loses (`reward.struck`); with
`--idle-rate` x > 0 (the chain: 0.05) only a damage rate of at least x — the defender's gold lost, a
share of the budget a minute, an exponential mean over `--idle-window` 30 s (`reward.hit_rate`).
0.05 a minute (the whole budget in 20 minutes) tells a fight from scratches: with the army fighting
the median rate is 0.10–0.19 and ≥ 0.05 in 82–95 % of decisions; one or two units skirmishing out of
6+, median 0.04. `rollout.Battles.last_hit` keeps the battle time of the last damage per battle (−1
before the first). The network sees this clock (the rate / 0.05, reached yet, seconds since:
`observation.PROGRESS`, [model](model.md)); it matches the reward only at 0.05, 30 s and a rout
share of 0.5 (`rollout.Battles` warns otherwise).

| No damage from the start | a step | so far |
|---|---:|---:|
| 1 min | 0.0001 | 0.006 |
| 3 min | 0.0005 | 0.07 |
| 5 min | 0.0013 | 0.26 |
| 10 min | 0.004 (the cap from 7.6 min) | 2.2 |
| 15 min | 0.004 | 4.6 |

| A pause after damage | a step from then | the pause so far |
|---|---:|---:|
| 30 s | 0.00013 | 0.000 |
| 60 s | 0.00034 | 0.008 |
| 90 s | 0.0007 | 0.03 |
| 120 s | 0.0013 | 0.07 |
| 180 s | 0.0038 | 0.28 |

Against the outcome (±1): 3 minutes of deploying and marching cost 0.07; an attacker that has dealt
no damage by minute 10 has paid 2.2, more than a lost fight, so standing on is worse than attacking
and losing; every further minute is −0.48. Short pauses in the fight cost little (90 s: 0.03).

**Order costs.** With 0.001 a unit that changes its order every decision (1 s) costs its side ~0.6
over a 10-minute battle; one change every 5 s ~0.12. The costs restrain flipping: without them the
A→B→A flips rose by 60–80 % ([tried and rejected](#removing-the-order-costs)).

`log.jsonl` and the console show the reward by term a minute of battle per learner role
(`reward_parts`: `trade`, `lord`, `end`, `idle`, `orders`).

## PPO

`ppo.py`, MAPPO style: each own unit is an agent with its own probability ratio; all units of a
side share the side's advantage; the critic sees the whole field (and is never shipped).

| Setting | Value | Why |
|---|---|---|
| discount γ | 0.9997 per 0.5 s (0.99940 per 1 s decision) | a horizon of ~28 min: a win 10 minutes away still counts 0.7 |
| GAE λ | 0.95 per 0.5 s (0.9025 per decision) | the usual |
| clip | 0.2 | the usual |
| learning rate | 1e-4 (`test5` and the chain: 1.5e-4), Adam (eps 1e-5) | |
| epochs, minibatch | 1, 4096 decisions of whole chunks (fewer for battles of more than 22 slots: `run.sized`) | the update costs more than the battles; the memory of a 16 GB card |
| stop at KL | 0.05 per unit | a guard against a too large step |
| entropy | on the order kind only; `--entropy` → `--entropy-end` linearly; with `--entropy-target` a floor: the weight × `--entropy-rate` every update while the kind's entropy is below the target, back down to the schedule above it, at most `--entropy-max` | a bonus on the move point's 128 bins would pay for moving |
| KL to a reference | `--anchor` → `--anchor-end` on the order kind and target; the reference is `--reference` (default `--init`); `--anchor-roll` S: every S seconds of training the reference becomes the current actor (`anchor_rolls` in the log) | a leash that bounds drift within a window, not over the whole run |
| advantage normalisation | `batch`, or `role`: the attacking and defending rows apart | the attacker's advantage is 2–3 times as wide |
| value loss, gradient norm | 0.5; actor and critic clipped apart, each to 0.5 | clipped together, a large critic gradient shrank the actor's step below Adam's eps: the actor did not train at all |
| critic warm-up | `--critic-warmup` N: the first N updates train the critic alone | a fresh or foreign critic does not fit the actor's values |

- **Memory (GRU) through time.** A rollout is a chunk of 64 decisions of every battle; the update
  runs the actor over each chunk from the memory stored when the chunk began, emptied where a new
  battle begins (recurrent PPO, R2D2's stored state). Only the GRU steps one by one.
- **Events in the input** ([inputs](model.md)): per unit how lately it fought in melee and routed
  (enemies: while seen); own and enemy lord slain and how lately; the damage timers and the
  attacker's progress clock.
- **Distance from the start.** `ppo.distance`: the mean per-unit KL (kind and target) of the actor
  to the network the run started from, on one minibatch of every update (`start_kl` in the log). A
  distance that grows while the rating rises means the search works; a flat rating with a stalled
  distance, a leash too short.

## Lord abilities

The network decides itself when its lord uses an ability, like a player
([model](model.md#abilities-each-unit-each-of-its-ability-slots), [bridge](../apps/bridge.md)):

- the learner and the past version choose abilities (`heads.sample(..., abilities=True)`); the
  rollout keeps `Action.ability`, the update counts it in the log-probability;
- only the scripted opponents' lords fire by the simulator's game-AI rule: `Battles.by_rule`
  (`tools/nn/sim/abilities.py` `set_rule`), set again after every restart, because a battle from
  the bank brings `scenario.build`'s default (side 2 by the rule);
- a stored decision keeps only the slots' state (5 numbers) and the battle's bank row;
  `rollout.full_obs` puts the passports back in the update (the whole input would be ~3 GB for 1024
  battles × 64 decisions); the slice of the stored state is a copy;
- the critic's warm-up updates run the actor without a graph (memory);
- `run.py` logs `abilities_per_battle`.

## Drills

Curriculum battles that teach one skill each (`tools/nn/train/drills/`). A drill is a **frame** — a
condition — and every battle inside it is generated from a seed: random rosters and unit types that
satisfy the condition, sizes, distances, the place and bearing of the whole battle on the map
(within 700 m of the centre), and which side we play (half side 1, half side 2). The drill's enemy
is a script (a league opponent `drill_<name>`); the network plays it on that drill's battles only.
Drills change *which battles* the network plays: there is no imitation term. Each drill keeps a
`skilled` script, so an annealed imitation term per drill could use it as the teacher.

Every drill battle runs under the standard battle limit, as every training battle: the network sees
no time limit, so a drill must be won or lost by the fight, not by the clock.

**Verification before a drill trains anything** (`drills/verify.py`): two scripts of ours play
≥ 256 generated battles of the drill against its enemy, with the training's randomised numbers; the
naive script must lose clearly (win rate ≤ 0.25), the script that uses the skill must win clearly
(≥ 0.75), else the frame is retuned. Only drills that pass are in `drills.READY`, the default of
`--drill-weights`.

```bash
bash tools/nn/dock.sh tools.nn.train.drills.verify --drill pincer,kiting --battles 256 --show 4
# training: 10 % of the battles are drills, all of them counter
... tools.nn.train.run ... --drills 0.1 --drill-weights '{"counter": 1}'
```

How it plugs in: `league.with_drills` takes `--drills` of the mix from the other opponents in
proportion and shares it by `--drill-weights`; `drills/source.py` `Mixed` builds one bank of the
generated battles and every drill's battles (`--drill-bank` per drill, renewed with the bank), and a
drill row restarts as a battle of its drill with our side = the side the learner plays in that row.
In the training log the drills appear as opponents (`drill_counter/attack`, ...).

| drill | frame | enemy | naive (win / gold trade) | skilled (win / gold trade) | status |
|---|---|---|---|---|---|
| `counter` | Empire v Empire, two pairs in random order along the line, 110–200 m apart, 20–60 m between the pairs: our X opposite its counter C, our Z (which beats C) opposite Y; (X, Z, C, Y) from the same-faction pairings run 2 v 2 in the simulator; we attack | holds; a unit whose opponent broke presses the nearest of ours | `nearest`: 0.215 / +0.06 (X breaks on its counter, which joins the other fight) | each unit on the enemy it beats (passport matchup): 0.867 / +0.22 | `READY`; on the evaluation seeds the skilled script attacks its correct target 98 % of the time, naive 53 % |
| `hold_fire` | our infantry unit in contact with the enemy's lord; 2–3 of our shooters 70–95 m behind it (Empire archers, Skaven Night Runners or slave slingers); free enemy missile units 65–95 m beyond our shooters' line; we attack | melee units attack the nearest, missile units hold | our shooters focus the lord in the melee: 0.066 / −0.59 | shoot the free enemies; with none left step out of range of the melee and hold: 0.848 / +0.37 | `READY` (fire into the melee around a lord costs us 1.5–3.5× what it costs him; into an ordinary unit it pays) |
| `kiting` | 1–2 Skaven Night Runners (run 5.4 m/s, sling 140 m) against Empire melee infantry (run 3.0 m/s); we attack | every unit runs at our nearest missile unit | `hold`, stand and shoot: 0.059 / −0.24 | shoot while the nearest chaser is far, run back when it comes near, halt and shoot again (each halt gives a volley): 1.000 / +0.86 | `READY` (needs the volley rule of the simulator) |
| `pincer` | 2–3 Empire greatswords holding 120–180 m apart; two of ours per greatsword in a column opposite it, 140–240 m away; we attack | `hold` | `nearest`: 0.223 (the pair piles on the front) | front pin + flank: 0.922 | needs the pending flank rule ([simulator](simulator.md#pending-changes)): with the current simulator both 0.00; with the flank rule 0.281 / 0.922 (naive above 0.25); not in `READY` |
| `defend` | Skaven defenders (a line + 4 missile units) against L + 2 stronger attackers playing `ai_like` | `ai_like` | marching out | holding the line and shooting | built, being tuned; not in `READY` |
| `reserve` | a reserve that must intercept flankers instead of joining the main melee | scripted flankers | joins the melee | intercepts | module written, not in `NAMES` (not loaded) |

`READY` = `kiting`, `counter`, `hold_fire`. The chain trains `counter` at 10 % of the battles:
with it the network learned the counter-pick (drill win 0.40 → 0.65–0.73, time on the correct
target 0.60 → 0.72, first correct order 44 → 26–40 s). At 30 % the same skill came at a cost:
capacity verdict "widen-candidate", 17 older numbers dropped beyond noise, the overall rating
−0.41 → −0.60.

**Drill metrics** (`drills/metrics.py`): per battle, over our standing units' seconds, the share
attacking the correct target / a bad one (its hard counter) / another enemy, the other order kinds,
the share in melee, and the median battle time a unit's attack order first goes to its correct
target; the same over the last 100 s before the limit. `test5` evaluates every `READY` drill
(`--drill-eval`, 128 battles each on `DRILL_EVAL_SEEDS`): win rate, gold trade and these metrics,
with the two check scripts on the same battles beside them, in the report and in `trend.md`.

## Evaluation

`evaluate.py`: whole battles to the end against each opponent, the learner's side alternating (both
roles), orders sampled as in training; `hold` only as the defender. `--generated N`: N battles of
`EVAL_SEEDS` per opponent (in swapped pairs by default, `--no-pairs` once each; with the script
baseline, `--no-baseline` without); `--per-scene N`: the fixed arenas; `--opponents`, `--against`
(the network behind `past`), `--together` (all opponents in one batch), `--greedy`, `--out`. Per
opponent and role: win rate, gold destroyed and lost, timeouts, own / enemy lord dead, order kinds,
target switches, the behaviour and liveliness metrics.

**Only the running battles.** An evaluation runs until its longest battle ends. Once at most 64
battles are left, the batch shrinks to them (`rollout.Battles.narrow`: padded with ended battles,
which stay frozen; `evaluate.BUCKETS`) and the shrunk batch's decision is replayed as one CUDA graph
(`evaluate.Graphed`). The same seeds give the same battles; the graph's replay gives bit for bit
the battles of stepping. `play(compact=False, cuda_graph=False)` steps the whole batch.

### Test protocol (`test5`)

`tools/nn/train/test5.py`: the standard short test of a training change. Every change is closed
with it, so tests can be compared with each other.

1. **Before.** The starting network (`--init`) plays the evaluation: `--eval` (512) battles of random
   armies from `EVAL_SEEDS` (up to 19 units a side) per opponent against `ai_like`, `nearest` and
   `hold_shoot`: 256 seeds, each played twice, the network on either side, with the script
   baselines of the same battles (cached) and the `READY` drills. All in one batch, the same battles
   every time.
2. **Training.** 36 PPO updates (`--updates`; ~5 minutes on a free GPU) with the current code and
   `test5.PROTOCOL` (`--small 0.35:6 --critic-warmup 3 --lr 1.5e-4 --entropy 0.003 --entropy-end
   0.001 --anchor 0.06 --anchor-end 0.03`, `long19` in the pool, `--snapshot-every 10 --no-eval`).
   Options after `--` go to `run.py` and override them; `report.json` keeps them all (`protocol`,
   `options`, `train_args`). A fixed number of updates, not minutes: on a shared GPU an update can
   take minutes. `build/nn-train/latest.pt` is not touched.
3. **After.** The trained network plays the same evaluation.
4. **The report**, printed and written to `build/nn-train/test5/<label>/report.json` (with
   `before.json` and `after.json`).

A **trend run** (`--updates 0 --minutes M --every K`) evaluates every K minutes of training (their
time not counted), keeps each network (`m<minute>.pt`, the last with its critic) and evaluation
(`eval_m<minute>.json`) and writes the table minute 0 / K / … / M (`trend.md`, `report.json`).
`--before PATH` reuses a "before" evaluation (only while the simulator has not changed);
`--report-only` rebuilds the report of a finished label without the GPU; `--prev-report` names the
previous iteration for the capacity block.

The report's blocks: skill (the [fair metrics](#network-evaluation-fair-metrics), with the distance
from the start: `start_kl`, the anchor KL, the anchor rolls), drills, win rate by opponent and role,
by faction and matchup (with the gold exchange ratio; `matchups.py`), behaviour, liveliness, and
capacity and forgetting.

| Behaviour metric | What is counted (the learner's units; `behaviour.py`) |
|---|---|
| enemy gold destroyed, own gold lost / battle; gold exchange ratio | the reward's gold at the end of the battle; the ratio is destroyed / lost over all battles of the cell |
| own lord dead | share of battles that ended with the own lord dead |
| missile s in melee / battle, missile time in melee | seconds missile units (not a lord) stand in melee, per battle and as a share of their standing time |
| own melee hit flank/rear | share of own melee unit-seconds struck in the flank or rear (the simulator's `flank_hit`) |
| decisions with a pile | share of decisions with more than 2 own units on one enemy while another enemy strikes an own unit in the flank or rear |
| own melee into flank/rear | share of own melee unit-seconds striking a standing enemy from its flank or rear |
| target switches / min, order changes / min, order kinds | per standing unit, per opponent |
| ability uses / battle, timeouts | own abilities started; share of battles at the time limit |

Heavy GPU jobs take turns: the test waits while `build/gpu-train.lock` exists (polled every 30 s),
then holds it (its label and start time) until it ends, also on failure; a test started within 60 s
of a release (`build/gpu-train.released`) waits out those 60 s first. `DOCK_NAME` names the
container. Noise: at 256 battles a role a win rate moves by ±3 points (one standard error); 36
updates move the network little, so the test shows whether a change breaks something and where the
behaviour goes, not a final strength.

### Network evaluation: fair metrics

The factions are unequal, and that is fine: the game is rock-paper-scissors. But in Empire–Skaven
battles the faction decides most outcomes (at equal gold the network won 59–79 % as the Skaven
against the Empire and 24–50 % as the Empire against the Skaven; mirrors ~45–55 %), so a win rate
says more about the matchups drawn than about the network. `skill.py` (plain numpy) takes the
matchup out:

| Measure | How it is counted |
|---|---|
| **Swapped pairs**, `pairs` | Every generated evaluation battle is played twice on the same armies: the network on side 1, then on side 2. The same army attacks in both, so the network's role swaps too. A pair is won both / split / lost both; `pair_score` = P(won both) − P(lost both). `hold` has no pairs. |
| **Advantage over the script**, `baseline` | The same battles played by the opponent script against itself: the matchup's natural edge. Per battle the network minus the script, same armies, same side: the win and the gold trade ((enemy gold destroyed − own gold lost) / budget), per matchup and overall (`win_adv`, `gold_adv`). |
| **Rating**, `skill` | One ridge logistic (Bradley–Terry) fit over all battles: P(win) = σ(r_opponent + edge(ours, theirs) + a · (+1 attack, −1 defend)), r = skill_net − skill_opponent in logits; the faction edge antisymmetric; a Gaussian prior with sd 3; the 95 % interval from the inverse Hessian. `overall` = the mean of the opponents' ratings: **the single trend number**. 0 is even with that opponent; +0.4 ≈ 60 %. |
| **Margin**, `margin` | The winner's gold left / its starting gold, + when we win, − when we lose; per opponent and per matchup. |
| **Pair gold**, `pair_gold` | In a pair both pairs of hands get the same two armies. `pair_gold` = (enemy gold destroyed by us in both battles − by the opponent in both) / budget, the mean ± 95 % over the pairs; `exchange`: the same as a factor; `weak` / `strong`: the army we did worse / better with, in our hands vs the opponent's. In the [gate](../launch/gate.md) the gold comes from each unit's end state × its passport cost, counted as the reward counts it. |

Pairs are the default for generated battles; the baseline is on in `test5` and in
`python -m tools.nn.train.evaluate`, off in `run.py`'s final evaluation, never with `--small`. The
result keeps the per-battle lists (`battles`) for a refit. `scenes.Generated` lays the batch out in
blocks of four battles `[p, q, p, q]` (two pairs; the network on side 1, then on side 2), cycling
through the opponents (`paired_order`). The baseline cache
(`build/nn-train/baselines/<opponent>_<units>u_<limit>s_<version>.json`) is keyed by a hash of what a
script battle depends on (`evaluate.VERSION_FILES`: the simulator, the army generator, the scripts,
the reward, `config/nn`), so a change there plays the baselines again.

Why the rating: 10 % of the Skaven's gold (budget ×1.0 against ×0.9, the same network, 384 battles
per opponent) flips the cross matchups — a matchup's win rate moves by up to 0.37, the network's win
rate on one faction by 0.16 — while the fitted faction edge takes it (−0.70 → +0.72) and the overall
rating moves by 0.03 (its interval ±0.12).

### Liveliness

Units that change orders and targets too often look artificial. Measured only (no reward), per
opponent and role, in every evaluation (`behaviour.py` Tracker; `test5`'s block "liveliness") and in
the gate from the game's recordings (`tools/nn/gate.py`):

| Number | What |
|---|---|
| order changes / unit-min | a new kind, a new attack target or a move point more than 10 m away (as the order cost counts them) |
| attack-target switches / unit-min | a new attack target while the old one still stands |
| flips A→B→A / unit-min | a change back to the order before the previous one within 10 s |
| move-point jitter, m | the mean distance between successive points of a unit that keeps moving; re-points per moving minute |
| changes out of melee / unit-min; units twitching | the changes of a standing unit out of melee; the share of units with 30 s or more out of melee that change orders there 6 times a minute or more |
| own target switches / unit-min (1 s) | the unit's current target changing between whole seconds, as the game samples it; also the opponent's units (in the game the game's AI is the reference band) |

In the game the companion gives the network the simulator's order point, so the network recognises
its own order in force: 4.4 order changes a unit-minute, target switches as the game AI's
([bridge](../apps/bridge.md)).

### Capacity and forgetting

When to widen the network: `capacity.py`, at the end of every `test5` run (block "capacity and
forgetting", `report.json` `capacity`), or on a finished folder:
`python -m tools.nn.train.capacity build/nn-train/test5/<label> [--prev <folder or report.json>]`.
Nothing is played.

- Forgetting: the final evaluation against the previous iteration's (`--prev-report`, else the
  folder of `--init`, else the label with its number one less) and against minute 0: per opponent
  the pair score and pair gold, per matchup the win rate and gold trade, per our faction and role the
  win rate. Same battles are matched (opponent, seed, side) and the noise is the paired difference's
  error. A drop of 5 points or more beyond 2.58 standard errors is "forgetting", within the noise
  "drop?". Against the previous iteration it counts only with the same simulator version.
- The training log: explained variance, value and policy loss (a plateau: the late half's slope
  within 2 errors of 0), entropy, grad norms (`grad_norm` the actor's, `grad_norm_critic`), KL, clip.
- Verdict: **widen-candidate** when old numbers drop beyond noise while others rise beyond noise
  and the overall rating does not (interference: new skill at the cost of old); **watch** for
  forgetting with a rising rating, forgetting with nothing rising (the settings first), drops within
  the noise, a plateau, a falling critic, an entropy collapse, a rising grad norm; else **ok**.

## Training speed

RTX 5070 Ti, 1024 battles, up to 19 units a side, 64 decisions an update, at the game's cadence:
~9 900 s of battle a second of training; warm-up 13 s once compiled before (cold: 160–170 s); the
`test5` evaluation with 128 battles per opponent and 64 per drill ~81 s. How:

- **Compiled bookkeeping** (`rollout.fast()`, CUDA only): the scripts and the assembly of orders,
  the order costs, the rewards and their counters, `reward.measure`, the actor's decision (forward,
  sampling, orders) and the critic's values in collection, the log-probability, and the behaviour
  tracker. Inside a compiled graph the ability encoder and head take every slot and mask the
  unowned; eager, they pick the owned slots with `nonzero`.
- **No reads of GPU values in the hot loop:** `torch.distributions` without argument validation,
  battle and timeout counters kept on the GPU, order placement by index tensors made once, kind
  counts by comparison.
- **The memory through the chunk** (`TokenMemory.scan`): the norm, the masks and the residual of the
  GRU run on all 64 steps at once, only the cell steps one by one; `unbind` instead of `x[t]`.
- **The attention bias** is made once, aligned for the attention kernel.
- **The observation** is compiled (8 ms → 0.6 ms for 1024 battles).
- **The compile cache:** `tools/nn/dock.sh` keeps `torch.compile`'s kernels in the Docker volume
  `tww3-torch-cache`.
- **One decision a second** plays two simulator steps for one observation and one network pass
  (5 700 → 9 900 s of battle a second against a decision every step).
- **Evaluation** shrinks to the running battles with a CUDA graph (above): a 1536-battle evaluation
  180 s → 65 s.

In the update, the GRU's 64 steps and the attention's float32 backward remain the main cost. TF32 is
on (`allow_tf32`).

## What is missing

- A wider network: the `small` preset (0.84 M) shows interference (new skills at the cost of old).
  The plan is ×2 by weight surgery (new units with zero outgoing weights: the same output at the
  start) and training on with the usual self-anchor.
- Unit-anchored orders (attack enemy N from the flank, stand behind own unit M, cover its flank):
  the move point is a 16 × 8 grid relative to the enemy direction, so flanking a given enemy needs
  the network to compute the cell itself.
- The `pincer`, `defend` and `reserve` drills; the flank rule in the simulator for `pincer`.
- `ai_like` still piles more and sends its lord in sooner than the game's AI.
- More factions and unit kinds in the army generator; style rewards from the faction character.

## Tried and rejected

What was tried, the result in numbers, and why it is not used. Git keeps the code of each.

### Two networks, attack and defence

Two networks instead of one, each twice as wide (3.2 M weights against 0.84 M), warm-started from
the scripts. The attack network came out at the level of the single network (vs `ai_like` 44–54 %,
vs `nearest` 28–42 %), with 98 % attack orders and three times as many piles. The defence network
failed: after 10 / 20 / 30 minutes of PPO 22 / 17 → 34 / 28 → 14 / 6 % on held-out armies (vs
`ai_like` / `nearest`), against the single network's 64 / 42 %: it grew on its own battles and did
not carry over to new armies. The only plus: the attacker's pressure no longer leaked into defence.
The small network copies both teachers as accurately as the wide one (kind 98.5 %, target 98 %), so
size was not the limit. One network with the role as an input.

### No leash, and a leash only for new networks

The rule "the KL leash to the script copy is only for a network started from random weights" was
tried: without any leash (`--anchor 0`) training collapsed every time. 10 minutes: vs `ai_like`
54 / 66 → 28 / 47 % (attack / defend), `nearest` 40 / 48 → 21 / 21 %, lord deaths ×2, 52 % of the
attacks on `hold_shoot` ran out the hour. An hour (411 updates): vs `ai_like` 0.44 / 0.50 → 0.07 /
0.16, gold exchange 0.94 → 0.68, "hold" when attacking 2 % → 57 %, battles up to 40 minutes, 0 of 4
in the game. Two causes: (1) a loophole in the idle cost of the time (waived while any one unit
fought: the network kept 1–2 units skirmishing while the rest stood), now closed by the progress
clock; (2) a PPO step is nearly all noise: the cosine between the policy gradients of independent
batches was 0.065, with random advantages 0.073. Without a leash a steady bias wins over hundreds of
updates. The collapsed network could not be brought back: made to attack as often as a healthy one,
it lost even worse (2–6 % wins): its targets and coordination were unlearned. A leash to its own
copy renewed every 10 updates (`--reference self`) held almost nothing (KL to it 0.0002–0.027) and
was removed. Used now: a small KL (0.03) to the network the run started from, rolled forward every
600 s (`--anchor-roll`).

### Warm start by imitating the scripts (`bcmix`)

Before PPO the network copied `nearest` (and later `nearest` + `ai_like`) by behaviour cloning
(99–100 % of kinds, 98–99 % of targets), and a KL leash held it near the copy. From random weights
PPO alone did not learn to attack in 10 minutes (0 % wins as the attacker), so the copy did get an
army that fights. But the copy of the mixture was weaker than either teacher (31 / 36 % vs
`ai_like`, 28 / 24 % vs `nearest`: the network told the teacher from its own orders in force), and
the habits it copied were the scripts' dumb ones: the `nearest` copy played 100 % attack orders,
sent its lord in alone, ran 350 m into prepared missile fire and never withdrew — 1 of 4 against
the game's AI. `imitate.py` is deleted; networks continue from the chain, and a new wider network
is grown from the current one by weight surgery, not by imitation.

### Per-unit credit

Each unit got its own return beside the side's, a second critic head valued it, and the unit's
advantage (centred over the side's units each decision) was added to the side's: A_i = A_side +
c × A_unit_i. Three designs:

| Design | What a unit's own reward was | Result |
|---|---|---|
| own trade + shaped terms | the gold it destroyed (HP it dealt × its target's gold per HP) − the gold it lost; −2e-4 when struck in the flank / a missile unit in melee / piling, +2e-4 striking a flank; half its neighbours' | c 0.6: "hold" 0.70–0.85 of decisions within 36 updates, half the wins lost, 66–77 % of attacks on `hold_shoot` ran out the hour. "Destroyed" paid missile units 7.6× their part (the kill spill on neighbours counted at the target's HP per man) |
| contribution + `shirk` | the enemy's gold loss split among the units engaging it by the HP each dealt; −2e-4 for a melee unit standing still out of melee while its side fights | the best 25-update probe (c 0.5 → 0) collapsed in a full run: move orders 24 % → 70 %, own lord dead 0.32 → 0.60, rating −0.25 → −1.05. Walking about was free and paid above a fighting unit |
| contribution + `shirk` with closing | standing by charged unless the unit closes on the enemy at ≥ 1 m/s | c 0.5: units stood and walked (in melee 0.36 of the time against 0.70), pair gold vs `nearest` / `ai_like` −0.68 / −0.59 against −0.07 / −0.06 at c 0; c 0.25: "hold" doubled. The order was 0 > 0.25 > 0.5 on every column |

Why: a fighting unit takes losses, and in a losing trade its own reward sits below its fellows'
mean; any state out of the fight that costs nothing is above it. A unit's own account pays
self-preservation as long as its losses count against it and its share of a win does not; each
charge on one way of staying out (standing, walking) moved the units to another. The per-unit lord
terms (`lord_lead`, `lord_exposed`, `lord_fall`) went with it. The code is deleted; a future design
starts from scratch, with no gain from staying out of the fight.

### Charging units that stand by (`shirk`, side `shirk`)

A side-level cost for its melee units standing out of melee while the side fights with an enemy
within 300 m (`--shirk-side 2e-3`, also at credit 0): idle melee 0.096 → 0.069 of the army's cost,
"hold" 0.46 → 0.38 in a 25-update probe, no gain on top of anything else. The run with it was
stopped: it contradicts positional play — a reserve, a flank guard, a defender holding its line
wait legitimately (the `reserve` drill depends on it). The idle cost by army progress covers the
attacker without charging waiting units.

### The idle cost by the share of the army, and its march exemption

- **The attacker pays × the share of its army (by cost) that neither fights nor shoots**
  (`--idle-share 1`) closed the skirmisher loophole but charged every waiting unit — a reserve, a
  second line, the lord kept back — and pushed the lord into melee (lord deaths 0.28 → 0.43).
  Against it, the progress-only cost (15 minutes each from the same start): rating −0.31 / −0.23,
  pair gold vs `ai_like` −0.025 / −0.010, `nearest` −0.058 / −0.038, `hold_shoot` −0.057 / −0.010.
  Progress-only kept.
- **No idle cost while closing in on the enemy** (`--close-speed`): marching is not idling, the idea
  went. In the 30-minute run with it 4–6 % of the attacks on `ai_like` / `hold_shoot` ran out the
  hour (none before); an exemption for moving can be gamed by walking about without engaging (the
  same hole `shirk` had). The cost counts while marching; the exponential start (3 minutes ≈ 0.07)
  leaves the time to deploy and march.
- **`--idle-rate` 0.02** instead of 0.05: not run. It would lower the pressure while the "hold"
  share was rising (0.30 → 0.55 after the progress inputs).

### Removing the order costs

`--order-cost 0 --retarget 0` (15 minutes against the same run with the costs): the quality was
equal, but the A→B→A flips against `nearest` and `hold_shoot` were 60–80 % higher at the end. The
costs stay.

### Other reward terms replaced

- **The health and standing terms** (`hp`, `standing`, 0.5 each): replaced by the gold trade, which
  holds both; a health share counts a lord as much as a unit of cheap infantry.
- **The attacker's extra loss at the time limit** (−1.5 instead of −1): timeouts were ~0 in
  practice, and the idle cost already presses the attacker. A timeout is the normal −1.
- **A late tempo cost** (0.0003 a decision past 600 s whatever the attacker did): the idle cost
  alone presses the attacker.

### Exploration tricks

- **The entropy floor as exploration** (target 0.1, ceiling 0.1): the kind's entropy rose 0.009 →
  0.10 within ~20 updates, but no new kind was learnt (hold, withdraw, keep stayed ≤ 0.01, move fell
  0.14 → 0.06); the randomness was churn: order changes a minute 2.6 → 5.2–6.8, struck in the flank
  0.42 → 0.47, rating −0.06 → −0.16. The chain keeps only a mild floor (target 0.05, ceiling 0.03)
  against a collapse of the kind.
- **Softening a collapsed actor** (`--kind-temperature`, dividing the kind logits once): a softened
  `nearest` copy collapsed back to 0.98 attack within 3 minutes; a network stuck in "hold", softened,
  attacked more but won 2–6 %. Removed.
- **An entropy bonus on a collapsed policy** (0.01 → 0.002): at 1.000 attack its gradient vanishes;
  nothing changed.

### Inputs about the battle's end

A time-limit input (`t / 3600`, or any "time left") is forbidden: the network must not know when
the battle ends — a campaign battle may have no limit, and a drill decided by the clock teaches
nothing. Only the time elapsed is given (a fine clock for the first minutes). Dropping `t / 3600`
changed 0.1 % of the greedy choices of trained networks.

### Per-drill time limits

The first drills had their own short limits (e.g. 300 s for `counter`). The network learned nothing
from them (8 % drills for 15 minutes: drill win 0.18 → 0.20; 30 % for 25 minutes: 0.21 → 0.20,
first order on the correct target 146 s against the skilled script's 0.5 s): the naive play lost
only on the clock it cannot see — at 900 s it won 1.00. Rebuilt to be decided by the fight under the
standard limit, the `counter` drill was learned (above).

### The Skaven at 0.8 of the budget

At equal gold the Skaven won all 10 whole battles in the game, so the generator gave them 0.8 of
the Empire's budget; in the simulator they then lost ~90 %. Back to 1.0: the faction imbalance is
handled by the fair metrics, not by gold, and the Empire got shielded units
([random armies](armies.md)).

### Speed tricks not kept

bf16 autocast in the update was slower (5.06 s against 4.62 s) and moved the log-probabilities from
the rollout's by ~2·10⁻³ (TF32: 2·10⁻⁵), as much as a real update's KL; compiling the actor's blocks
and the critic for the update gave nothing; the GRU's input weights hoisted out of the loop were
slower; a graph for any batch size (`dynamic=True`) needs a C++ compiler the container lacks.

### Evaluations during the run

`--eval-every` with a `best_eval.pt` chosen by them: the evaluations fell after the first 10
minutes of a 50-minute run and the chosen network won 1 of 8 in the game. The `test5` trend run
does this job.

## Tests

`tests/tools/test_nn_train.py` (torch; skipped in `.venv`, run in the container): GAE, clipping, the
entropy floor; the reward (win and gold and lord differences, the time limit as a lost fight, gold
lost by the worst share, zero-sum, a rout-rally-rout counted once in the trade and the clock, the
idle cost before and after damage, a marching attacker pays, the defender never pays, progress only,
the damage rate and `--idle-rate`, `struck`, the terms adding up to the step and to the logged
reward, order changes and retargets, a lord's death and `--lord-rout`); schedules; the layout
(`ai_like` in both roles, `hold` only defends, every scene from both sides); randomisation; the
behaviour facts and liveliness; `keep`; the memory through a chunk equals step by step; generated
battles restarting with the setup following; `ai_like` (the line, the counter-charge, the lord shot
in range, missile units stepping back and breaking off, the target choice, the lord not charging
alone); the small share of a bank; masks; symmetric rewards; restarts; the attacker's last damage and
progress seen by both sides and by the companion; abilities (only scripted sides by the rule, the
stored input, PPO trains the head); a training step; the actor's step not shrunk by a huge critic
loss; continuing from older checkpoints; `--critic-init`, per-role advantages, `--anchor-roll` and
`start_kl`; checkpoints; replays readable by `gamedata`; evaluation (every battle counted, several
opponents in one batch); `test5`'s report, trend, liveliness and faction blocks, and the GPU lock.
`tests/tools/test_nn_eval_pairs.py` (torch): the pairs' layout, a paired evaluation, the baselines and
their cache, the version hash, a shrunk batch playing its battles as the whole one does.
`tests/tools/test_nn_skill.py`, `test_nn_capacity.py`, `test_nn_matchups.py` (numpy): the rating,
pairs, margins, pair gold, forgetting and the verdict, matchups. `tests/tools/test_nn_cadence.py`:
the cadence. `tests/tools/test_nn_drills.py` and `test_nn_drill_*.py`: the drill framework, frames,
scripts and metrics. `tests/tools/test_nn_gate.py`: the gate's pairs and liveliness from recordings.
