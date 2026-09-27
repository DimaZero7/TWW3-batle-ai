# ai

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/ai.md)

The home of **our AI**: observation in, decision out. No engine objects,
only plain tables, so everything is testable without the game.
Code: `src/apps/ai/`.

## contract

- `profile(name, speed?)` — army profile:

  | Profile | Units | Commands per step | Roster | Speed |
  |---|---|---|---|---|
  | `empire-7-v1` | 7 | 14 | lord, 4 spearmen, 2 archers | ×20 |
  | `empire-15-v1` | 15 | 30 | lord, 8 spearmen, 6 archers | ×7 or ×20 |

  One decision every 1000 ms, a budget of 1,000,000 instructions per call.
- Duel contract: `policy.decide(observation) → action, reason`, `action` is
  `'attack'` or `'wait'`, `reason` is a string for the log.

## services

- `duel_observation(unit, target, order, now_ms)` — builds the observation
  from read values: `routing`, `shattered`, `men`, `target_valid`,
  `target_visible`, `in_melee`, `idle`, `has_order`, `since_order_ms`.
- `decide_duel(policy, observation)` — calls the policy and checks the answer.

## policies

| File | Version | Behaviour |
|---|---|---|
| `forced_melee.lua` | `forced-melee-v2` | Attack the visible target; repeat the order after 5 s idle |
| `delayed_melee.lua` | `forced-melee-v3-delay` | Same, but waits the first 10 s — shows a new version was picked up |

## Adding a policy

1. Create `src/apps/ai/policies/<name>.lua` with a `version` field and a
   `decide` function.
2. Wire it in an entry (`require('apps.ai.policies.<name>')`) or place the
   file as `tww3_bai_policy.lua` in the game folder: the duel re-reads it
   before each battle without rebuilding the pack.
3. Write a `decide` test in `tests/apps/ai/`.

An external policy file runs with full Lua access. Load third-party
policies through the [sandbox](sandbox.md).
