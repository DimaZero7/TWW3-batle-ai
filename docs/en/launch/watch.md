# Watching the network in battle

[← Back](README.md) · [Documentation](../README.md) · [Running](run.md) · [Русский](../../ru/launch/watch.md)

A real battle where our side (side 1) is commanded by the network and side 2 by the game's AI. The
network runs in the companion (`tools/nn/companion/`) in the training container; the game and the
companion talk through two files ([bridge](../apps/bridge.md)). With the default untrained
`random.pt` its orders are random, but they go the whole way: game → network → orders → our units.

## Launch

One command (Docker Desktop must be running, the game closed):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/watch.ps1
```

What it does:

1. Builds the arena with our side under the network: `python -m tools.build nn-arena --own-ai net
   --speed 1` (normal speed, a decision every second).
2. Starts the companion in the `snake-ai-trainer` container (the repo at `/repo`, the game folder at
   `/game`). Its lines appear in the same window: one per decision.
3. Runs the battle through [launch.ps1](run.md): fair Normal difficulty, the user's preferences put
   back byte for byte, only our files removed.
4. The battle ends by itself (a side wins, 10 minutes of battle, or a stall). The game stays open
   30 s more to see the end, or until you close it. Closing the game earlier is fine too.
5. Stops the companion, removes the exchange files from the game folder, copies the companion's
   log into the run folder.

Options:

| Option | Default | What it does |
|---|---|---|
| `-Speed 1\|3\|10\|20` | 1 | Battle speed. The script holds it: fast-forward in the game is set back |
| `-DecideMs` | 1000 | Battle time between decisions, ms (250–5000) |
| `-Checkpoint` | `build/nn-train/random.pt` if present | The network's weights (a checkpoint of [training](../training/README.md)); without one a fresh untrained network |
| `-Greedy` | off | The most likely order instead of a random draw: the untrained network then usually sends everyone at one enemy |
| `-Arena` | `arena` | A named arena from `config/nn/arenas.json` |
| `-TimeoutModelSeconds` | 600 | Battle limit, s of battle time |
| `-LingerSeconds` | 30 | How long the game stays open after the result |
| `-NoBuild` | off | Use the net build already in `build/nn-arena/` |
| `-EnemyAi game\|ai_like` | `game` | The enemy: the game's AI or the simulator's script in the companion (build `--enemy-ai`; with `-NoBuild` the build's, a mismatch is an error); the script's log `companion_enemy.jsonl` ([bridge](../apps/bridge.md#the-enemy-under-the-simulators-script)) |
| `-Target nn-arena\|lord-duel` | `nn-arena` | Whose build and runs folder; `lord-duel` ([lord duel](../apps/entries.md#lord_duel)) only with `-NoBuild`, the companion's log goes to `build/lord-duel/companion/` |

A quick check: `-Speed 20 -LingerSeconds 0`.

The 8-battle set against `ai_like` instead of the game's AI: `build/steps/watch_set8.ps1 -Ckpt build/v2/itN.pt -EnemyAi ai_like`.

## What the companion prints

```
move   61  t   60.0 s  think   4.0 ms  turn  10.8 ms  | hold 1 attack 5 withdraw 1 | lord attack spear_1, ...
```

Move number, battle time, the network's time (`think`), the time from seeing the state to writing
the answer (`turn`), how many units got each kind of order, the first orders.

## Results

`build/nn-arena/runs/<time>/`: as for every arena run (`events.jsonl`, `status.json`,
`launch.json`), plus `companion.jsonl` — every decision of the companion with its orders. In
`events.jsonl`: `nn_orders` (orders given, the answer's wait), `nn_miss` (no answer in time), and in
`result` the counters `nn_*` ([bridge](../apps/bridge.md#events)).

## A check in the game

Run `20260930-204047`: ×20, arena 7 against 7 of the Empire, the untrained network
(`build/nn-train/random.pt`, sampling), Normal difficulty, the preferences put back.

| What | Result |
|---|---|
| Decisions | 261 over 260.6 s of battle (29 s of real time); all 261 answered, no misses, no broken files |
| Files | renamed every time (`nn_write_mode: renamed`): the game's Lua has `os.rename` and `os.remove` |
| The network's time | median 4.8 ms (small network, 4 CPU threads) |
| From the state written to the orders given | real time: median 37 ms, 90% ≤ 43 ms, at most 122 ms |
| Orders given | 1502 (only changed ones are given again) |
| Were they carried out | for 674 of 679 move/withdraw orders the game's own ordered point (`ordered_position`) is the network's point (median difference 0 m); our units were moving in 77% of the samples and each went 360–730 m |
| Battle | the game's AI won (534 men left against 375): random orders do not win |
| Lua errors | none; the game closed, our pack and the exchange files removed |

One launch (`20260930-203835`) crashed the game while it was loading, before our
script ran (no events at all; the crash report shows only the game's code). The same build ran fine
the next time. The cause is not known; the companion now looks for files 10 times a second while
nothing is happening (loading, menus) instead of 200.

At normal speed the answer's wait is the companion's turn (~10–20 ms) plus up to 100 ms until the
game reads the file: well within a decision of 1 s. Not measured at ×1 in a separate run.
