# The game AI's student

[← Back](README.md) · [Documentation](../README.md) › [Data for training](README.md) › The game AI's student · [Русский](../../ru/training/student.md)

The student is a network of the same v2 build as ours ([network input and build](model.md)), taught by imitation of
the game's AI on the recordings of its battles. Why: an opponent in the pool of past versions and a twin instead of
the hand-written `ai_like` script (no more improving it by hand), later the base of a "copy of the user" (fine-tuned on
his battles). Code: `tools/nn/student/`.

## In short

- **Data:** the game AI against the game AI (`build/observe/aivai_ok.txt`, both sides) and the AI's side of the exam
  (the network against the AI, 3 starts). Converted by the same converter as [learning by
  observation](training.md#learning-by-observation) (`tools/nn/observe/convert.py`), which already takes both sides of
  an AI-vs-AI battle.
- **A label for every second** of every unit that takes orders: the AI's new order (kind, attack target, 25-m cell,
  run) or "leave it" (`keep`) when the AI gave nothing new. So the student learns both *what* to order and *when*.
- **The split goes by start** (the army seed), not by second: the check runs on battles the student never saw a
  second of. The exam (~400 battles on 3 starts) is a check of its own and does not go into training (the student
  would learn 3 starts by heart).
- **The student's checkpoint is an ordinary network checkpoint**: the pool of past versions takes it with no new
  code (`run.py --pool-extra build/student/<name>/best.pt`).

## Data

| What | How many | Where |
|---|---|---|
| AI against AI (both sides) | 135 battles, 270 sides | `build/observe/data` (79 battles) + `build/student/data` (56, converted for the student) |
| Exam (the AI's side) | 399 battles on 3 starts | `build/observe/data` |
| A human's battles | not taken | — |

New AI-vs-AI recordings are converted into a folder of their own, not into `build/observe/data` (training's
`--observe-dir` reads that whole folder):

```bash
python -m tools.nn.observe.convert --list <list of runs> --out build/student/data --workers 6
```

`dataset.discover` finds the battles in both folders, takes the army seed from the run's manifest and splits: 15 % of
the AI-vs-AI starts are the check (`val`, the same in every run: a hash of the seed), the rest is training; the exam
apart (`--exam-train <seeds>` puts chosen exam starts into training).

## Labels

- **A new order**, as the converter reads it ([the reading rules](training.md#learning-by-observation)): a new engine
  target → attack; a moved order point → hold or move to a cell; "move running" and "withdraw" are one game command,
  the label is the set of both answers (the loss takes the log of the sum of both probabilities). The reading's
  accuracy on the network's battles: kind 89 %, target 93 %, point 100 % within 12.5 m, run 99.9 %.
- **`keep`**: a second where the unit takes orders but the AI gave nothing new. ~93 % of them (~88 % in the exam).
- **A repeat is `keep` too** (`dataset.repeats`): an order of the same kind, with the same attack target, to the same
  cell and with the same run as the unit's last order. The AI often "clicks" again - the point moves by a couple of
  metres in the same cell; in the network's language that is "leave it".
- **The commitment** (v2: with a new order the unit keeps it 2 / 4 / 8 / 16 s, [commit.py](model.md)), from the pause
  to the unit's next order: the longest duration that fits into it (no next order: 16 s).
- **The commitment is replayed over the battle** (`commit.inputs` / `apply`, as in the simulator), so the inputs
  "free" and "seconds left" in training are those the student sees in battle. A held unit's second teaches nothing
  (its only choice is `keep`). An AI order 1 s after the last one falls on a held unit (the network cannot hold less
  than 2 s): it is dropped and counted. Over the 212 training sides: 31 379 AI orders read, 26 % of them repeats (made
  `keep`), of the rest 12 % fell on a held unit.
- Abilities, formation and width are not taught (the recording does not read them): the ability head stays as the
  starting weights have it.

## Training

```bash
# from S12's weights
bash tools/nn/dock.sh tools.nn.student.train --name s12 --init build/v2/itS12.pt --epochs 8
# from scratch (S12's build, fresh weights)
bash tools/nn/dock.sh tools.nn.student.train --name scratch --scratch build/v2/itS12.pt --epochs 8
```

- **Streams:** B battle sides run side by side (`--B`, 16), each from the battle's start in chunks of T seconds
  (`--T`, 32); the memory (GRU) is carried from chunk to chunk (no gradient across the border) and starts empty where
  a new battle begins.
- **Loss:** cross-entropy over the chain's heads (kind → target → sector → cell → commitment, run), as
  [observation's](training.md#learning-by-observation), plus `keep`; the parts after the kind under the right parts
  before them. A new order's weight `--new-weight` (1: the honest share, the student gives orders in battle as often
  as the AI).
- AdamW, step 1e-4 from S12's weights and 3e-4 from scratch, gradient clip 1.
- After every epoch: the check on `val` and 30 exam sides, checkpoints `latest.pt` and `best.pt` (the lowest loss on
  `val`), the log `log.jsonl` in `build/student/<name>/`. `--minutes`: a time limit.

## Evaluation

```bash
bash tools/nn/dock.sh tools.nn.student.eval --checkpoint build/student/s12/best.pt --split val,exam --sim 16
```

- **Guessing** on `val` and the exam: over the AI's new orders - the kind (the most likely of hold / move / attack /
  withdraw), the attack target, the 100-m sector (and it or a neighbour), the 25-m cell within the right sector, run,
  the commitment, the point's error in metres (the most likely sector, then its cell). "When": the student's chance of
  a new order where the AI gave one against where it did not.
- **Simple baselines** on the same seconds: "repeat" (the unit's last order again), "nearest" (attack the nearest seen
  enemy, its cell for a move, running), "majority" (move, running).
- **Play in the simulator** (`--sim N`): N random battles an opponent in swapped pairs (`evaluate.play`) against
  `nearest` and `ai_like` - wins, gold, orders a minute, order kinds, the share of battles with contact.

## The student as an opponent

The pool of past versions (`league.Pool`) reads a checkpoint (`format 2`, `config`, `actor`) and builds the network
from its `config`; the student is saved the same way (`checkpoint.save`, no critic). So:

```bash
python -m tools.nn.train.run ... --pool-extra build/student/s12/best.pt
```

The student enters the pool with the highest `q` and then lives by the pool's rules (it leaves once the network
beats it steadily). Checked by a test (`tests/tools/test_nn_student_torch.py`): the pool takes the student's
checkpoint, the network is built from it.

## Results

A CPU run from S12's weights: 10 epochs in 2 h 52 min (6 cores, 212 training sides, 106 starts), repeats as
`keep`. Checked on all 58 `val` sides (29 starts, AI-vs-AI battles never seen) and 60 exam sides (3 starts).
Guessing - over the AI's new orders, more is better.

| | student, `val` | "repeat" | "nearest" | "majority" | student, exam | "repeat", exam |
|---|---:|---:|---:|---:|---:|---:|
| order kind | **0.76** | 0.50 | 0.35 | 0.55 | **0.70** | 0.64 |
| attack target | **0.53** | 0.01 | 0.45 | — | 0.29 | 0.00 (nearest 0.31) |
| 100-m sector | 0.43 | 0.43 | 0.37 | — | 0.33 | **0.40** |
| point within 25 m | 0.22 | **0.46** | 0.27 | — | 0.13 | **0.40** |
| run | 0.97 | 0.75 | 0.97 | 0.97 | 0.97 | 0.86 |
| chance of an order: where the AI gave one / where not | 0.31 / 0.03 | — | — | — | 0.24 / 0.05 | — |

Play in the simulator (32 battles an opponent, swapped pairs): it wins **0.41** against `nearest` (attacking 0.44,
defending 0.38; gold trade -0.08) and **0.25** against `ai_like` (0.38 / 0.12; -0.22). Contact in every battle, the
first at 60-67 s, ~1.7 orders a minute a unit (keep 96 % of the decisions).

What it shows:
- **The student guesses the kind and the target better than every simple rule**, run as they do.
- **The point worse than "repeat"**: with the repeats as keep the gap is smaller (0.22 against 0.46; it was 0.16
  against 0.69), but it stays. The point's median error is 71 m - about a neighbouring sector.
- **About even against `nearest`**, it loses to `ai_like`, defending above all. Sensible (it goes into battle, picks
  targets), but not stronger than the script.
- The exam is harder: against the network the AI gives twice as many orders (10 % of the seconds against 5 %), other
  targets.
- The short 20-min runs (1.6 epochs, before the repeats were keep): from scratch - kind 0.78, target 0.44, wins 0.19
  / 0.06; from S12's weights - 0.80 / 0.54 / 0.31 / 0.25. Starting from S12's weights is better.
