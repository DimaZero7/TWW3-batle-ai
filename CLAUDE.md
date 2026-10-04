# Working rules for Claude Code in this repository

A battle AI for Total War: WARHAMMER III: a PyTorch simulator (`tools/nn/sim`), PPO training
(`tools/nn/train`), drills with an automatic teacher, a Lua bridge + Docker companion that plays
the real game, in-game gates. Docs: `docs/ru` (master) and `docs/en`, same paths. One
ORCHESTRATOR session decides, launches runs and commits; SUBAGENTS implement from a written
brief. Everything below holds for every agent without being repeated in the brief; the brief
adds only the task, its files, its numbers and its acceptance check.

## Resources (the GPU and Docker are shared with the training run)

- ONE container per task in total, helpers included: `DOCK_NAME=<role>-<label> bash tools/nn/dock.sh ...`
  (`agent-`, `t-` for agents; `orch-` for the chain). Nine containers at once halved a training run.
- Nothing on the GPU while a training run or an in-game gate is going. Check first:
  `.venv/Scripts/python -m tools.ops.leftovers` (our containers, the GPU lock, wait loops, the GPU).
- The GPU lock `build/gpu-train.lock` belongs to the job that wrote it (test5 and run.py take it
  themselves). Never create it by hand, never delete one you did not write, no `trap ... rm lock`.
- An agent's GPU run is a smoke run: a few minutes (`test5 --updates N`, `--eval 64`, `--minutes 3`),
  never a chain step. Verification runs once; no re-checking a result on two numbers.
- Waiting: `bash tools/ops/wait.sh -t SECONDS FILE REGEX` (case-insensitive, a timeout, exit 2).
  No hand-written `until grep ...; do sleep` loops: two of them ran for hours on `passed|failed`
  (pytest prints FAILED in capitals).

## Scope

- Change only what the brief names. The simulator (`tools/nn/sim`, `config/nn`), the reward, the
  scripts (`opponents.py`), the evaluation protocol (`test5.py`, `evaluate.py`) and the bridge are
  frozen unless the brief says otherwise; a needed change there is a proposal in the report.
- Any change under `tools/nn/train/version.py` VERSION_FILES changes the simulator version: the next
  "before" evaluation plays the script baselines again (~25 min) unless the canary adopts the old
  file (`test5 --baseline-canary`). Batch such changes; say in the report that the version changed.
- Numbers come from the game's database (`tools/nn/gamedb.py` -> `config/nn/game_rules.json`,
  `docs/*/game/`) before any guess; the game is played at Normal difficulty only; a new network
  input only from data the game already gives, never the time limit or time-to-end.
- Do not commit, do not push; the orchestrator commits. No task / backlog / idea files. Analysis
  output (json, csv) never goes into Git — only png/svg/md. No report `.md` files: the report is
  the final message.

## Code, tests, docs

- Small modules with clear contracts; every new rule gets a test at once (functions, module cases
  on properties, wiring with spies; engine checks only on game updates).
- Host tests: `.venv/Scripts/python -m pytest -q` (torch tests skip). Torch tests, only when no
  training runs: `DOCK_NAME=t-tests bash tools/nn/dock.sh pytest tests/tools -q -x`.
- Docs say how things work NOW: update the section in place (ru first, then en, same paths,
  navigation line, links that exist), failures go to "Tried and rejected" with numbers, no dated
  notes. A new page: both languages, a row in the hub, `python -m tools.docs.index_doc`.

## Reports (the orchestrator reads them, the user reads the orchestrator)

- First line: the task restated in your words and, for a metric, its direction (rating logit:
  higher is better; pair gold: positive = we trade better than the opponent with the same armies;
  drill TRANSFER "applied share" higher is better, "mistake share" lower is better; teacher share =
  the share of battles labelled). A brief misread this way inverted a conclusion once.
- Real wall time of what you ran; what changed (files); what you verified and how; what you did
  not do. Under ~300 words unless asked. No emojis.

## The orchestrator's tools (`tools/ops`, docs/en/training/workflow.md)

- A chain step: `.venv/Scripts/python -m tools.ops.step --label <new> --from <prev label>[/m15]
  [--minutes 25] [--set drills=0.1] [--drop teach-normal] [--write build/steps/<new>.sh] [-- run.py opts]`
  — options from `config/train-chain.json`, says whether the baseline cache hits and the wall time.
- The run card: `.venv/Scripts/python -m tools.ops.card build/nn-train/test5/<label> --prev <prev folder>`
  (works on a running folder too).
- Leftovers: `.venv/Scripts/python -m tools.ops.leftovers [--kill] [--unlock] [--stop NAME]`.
- Memory for the orchestrator lives outside the repo (the memory dir); rules of the process live here.
