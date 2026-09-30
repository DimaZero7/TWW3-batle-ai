# Attacker and defender in a battle from a scenario file

[← Back](README.md) · [Game knowledge](README.md) · [Русский](../../ru/game/battle-roles.md) · [The game's AI](game-ai.md) · [Deployment](units/deployment.md)

Who attacks and who defends in a battle from our own scenario file, and how to
set it.

**In short.** In a battle started from our own scenario file (`game_startup_mode
battle <file>`), without the tag below `alliance:is_attacker()` is `true` for both
sides and the enemy's game AI attacks. With this tag in `<battle_description>`

```xml
<timeout_winning_alliance_index>1</timeout_winning_alliance_index>
```

(the alliance that wins when the battle's time runs out) `is_attacker()` gives an
attacker (alliance 0) and a defender (alliance 1), and the game AI of alliance 1
stays where it is. The neural-network arena writes this tag into its battle:
`tools/nn/scenario.py` creates `scenarios/nn_arena.xml`, and the defender is the
side that wins on timeout.

## Measurement (28.09.2026)

The Moorlands Route, `catchment_03`. Alliance 0 — a lord, 6 spearmen and 8 archers
in the west, under the script; alliance 1 — a lord, 8 spearmen and 6 archers in the
east, the game's AI, not controlled by the script. x20. The roles are read at the
start of the battle (`battle.read_roles`, event `roles`); the positions of alliance
1's units are recorded every tick.

| Battle file | `is_attacker()` | Alliance 1 |
|---|---|---|
| unchanged (deployment areas — map halves, X ∓270, 400 × 260 m) | both `true` | from 515 m to 72 m in 4 min of game time, down to 37 m in 7 min |
| without `deployment_area` | both `true` | movement not recorded |
| `<type>land_normal</type>` instead of `classic` | both `true` | from 386 m to 83 m in 2 min of game time |
| `<timeout_winning_alliance_index>1</timeout_winning_alliance_index>` | alliance 0 attacker, alliance 1 defender | 8 min of game time at 550 m, not a single move after deployment |

Deployment areas and the battle type did not change the roles.

## Not checked

- "Defender = the timeout winner" is an observation of this measurement; CA's documentation does not say so.
- Roles in a custom battle from the menu and in a campaign battle.
