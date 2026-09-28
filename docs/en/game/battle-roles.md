# Attacker and defender in a battle from a scenario file

[← Back](README.md) · [Documentation](../README.md) › [Game knowledge](README.md) › Battle roles · [Русский](../../ru/game/battle-roles.md)

**In short.** In a battle started from our own scenario file (`game_startup_mode
battle <file>`), without the tag below `alliance:is_attacker()` is `true` for both
sides and the enemy's game AI attacks. With this tag in `<battle_description>`

```xml
<timeout_winning_alliance_index>1</timeout_winning_alliance_index>
```

(the alliance that wins when the battle's time runs out) `is_attacker()` gives an
attacker (alliance 0) and a defender (alliance 1), and the game AI of alliance 1
stays where it is. `tools/sim/formation.py` writes the tag into all our battles.

## Measurement (28.09.2026)

The Moorlands Route, `catchment_03`; alliance 0 scripted, alliance 1 the game's AI; x20.

| Battle file | `is_attacker()` | Alliance 1 |
|---|---|---|
| unchanged (deployment areas — map halves) | both `true` | from 515 m to 72 m in 4 min of game time |
| without `deployment_area` | both `true` | not recorded |
| `<type>land_normal</type>` instead of `classic` | both `true` | from 386 m to 83 m in 2 min |
| `<timeout_winning_alliance_index>1</timeout_winning_alliance_index>` | 0 attacker, 1 defender | 8 min at 550 m, not a single move after deployment |

Deployment areas and the battle type did not change the roles.

## Not checked

- "Defender = the timeout winner" is an observation of this measurement; CA's documentation does not say so.
- Roles in a custom battle from the menu and in a campaign battle.
