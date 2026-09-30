# orders

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/orders.md)

Unit orders: which engine calls give them, how to hand units to the game's own
AI and which facings the engine holds. Code: `src/apps/orders/`.

## adapter — verified calls

Only recipes from [commands verified in battle](../game/units/commands.md):

| Function | Engine calls |
|---|---|
| `take_control(army, unit)` | `create_unit_controller`, `add_units`, `take_control` |
| `release(uc)` | `release_control` |
| `prepare_melee(uc, unit)` | `fire_at_will(false)`, disable `skirmish`, `melee(true)` |
| `move`, `move_formation`, `rotate`, `halt` | `goto_location`, `goto_location_angle_width`, `rotate`, `halt` |
| `attack_melee(uc, enemy)` | `melee(true)`, `attack_unit(enemy, false, true)` |
| `attack_ranged(uc, enemy)` | `melee(false)`, `fire_at_will(false)`, `attack_unit(enemy, true, false)` |
| `set_fire_at_will(uc, on)` | `fire_at_will(on)` |
| `stop_firing(uc)` | `halt()` + `fire_at_will(false)` |
| `set_guard(uc, unit, on)` | `change_behaviour_active('defend', on)` |
| `withdraw(uc)` | `withdraw(true)` |
| `use_ability_on_self(uc, unit, key)` | `perform_special_ability(key, unit)` |
| `teleport(uc, p, bearing, width)` | `teleport_to_location` |

## planner_adapter — hand units to the game's AI

`hand_over(bm, name, alliance, units, enemies)` gives units to the game's
tactical AI instead of our orders. It uses CA's `script_ai_planner`
(`lib_battle_script_ai_planner.lua`) — the mechanism generated battles use
for AI armies. It wraps `alliance:create_ai_unit_planner()`; without the
library the adapter calls that planner directly. Returns
`{mode, attack(), defend(centre, radius), check_rallies(), check_idle(ms), release()}`:
`attack()` means "attack the enemy force" and is re-issued by the entry. Used by
[ai_vs_ai](entries.md#ai_vs_ai) and [nn_arena](entries.md#nn_arena).

**What CA's planner does not do by itself, and how we mend it** (arena records, 30.09.2026):

- `check_rallies()` — a unit that routed and rallied is no longer led by the planner: without
  this it stood idle to the end (the game's AI leads its rallied units back). Such a unit is
  taken out of the planner, put back, and the last order is repeated 0.6 s later.
- `check_idle(ms)` — a unit stands idle under arrows: not moving, not in melee, no target,
  losing health. Our units did so 15-75% of the time they were shot at, the game's AI 0-3%.
  After 6 s of it — back into the planner with the order; if 6 s later it again stands under
  fire — a planner of its own that attacks the nearest enemy. At most once in 15 s a unit.
  Tested in `tests/apps/orders/test_planner_adapter.py`.

### The game AI with an objective (27.09.2026)

In battle `script_ai_planner` has, among others: `attack_force`, `attack_unit`,
`defend_force`, `defend_position`, `move_to_position`, `move_to_force`,
`rush_position`, `set_patrol_defend_radius`, `release`. `defend_position(position,
radius)` is verified: the game AI holds the place
([how it stands](../../../research/analysis/enemy-layout/README.md), in Russian).

**The battle file sets who attacks and who defends** (28.09.2026): the defender is the
alliance that wins on timeout, `<timeout_winning_alliance_index>` in
`<battle_description>`. Without this tag the engine counts both sides as attackers and
the game AI always attacks; with it `is_attacker()` gives an attacker and a defender, and
the game AI as defender stays where it is (8 game minutes, not one move). Deployment zones
do not change the roles (map halves and a battle without zones were checked).
`tools/nn/scenario.py` writes this tag into the arena's scenario: the game's AI defends when
our side attacks, and our side defends when it defends.
Every variant and measurement: [attacker and defender](../game/battle-roles.md).

## facing — facings the engine holds

`apps.orders.facing` (pure): the engine holds a unit's facing in 64 sectors of
5.625° and sets it to the sector's middle
([measurement](../game/units/commands.md#unit-facing-64-sectors)). `snap(b)` is
the facing the engine will hold; `command(f)` is what to send; `teleport` and
`move_formation` send it themselves.

## Worth knowing

- An accepted call is not proof of execution. Check the result through unit
  state ([units](units.md)).
- `fire_at_will(false)` does **not** cancel an explicit fire order: add `halt()`.
- `defend` stops pursuit; it is not entrenchment.
- `withdraw` works only when the XML allows withdrawal.
- There is no instant stop; units finish a few metres from the order point.
