# orders

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/orders.md)

Unit commands: which exist, how they are validated and which engine calls
execute them. Code: `src/apps/orders/`.

## contract — command shape

| Action | Fields |
|---|---|
| `move` | `unit_id`, `x`, `z`, `run?`; with a movement contract also `facing_deg`, `width_m` |
| `attack` | `unit_id`, `target_id`, `mode = 'melee' \| 'ranged'` |
| `guard` | `unit_id`, `enabled` |
| `halt` | `unit_id` |

`validate(commands, own, enemies, limit, movement_contract?)` rejects the
whole set if any command is wrong:

- a dense array no longer than `limit`, no unknown fields;
- the unit is own and alive; at most one motion (`move`/`attack`/`halt`) and
  one `guard` per unit;
- the attack target is **visible**; `ranged` only for archers with ammo;
- `facing_deg` / `width_m` only with `formation-move-v1` or
  `deployment-march-formation-v1`, width within `units.contract`.

## adapter — verified calls

Only recipes from [commands verified in battle](../game/units/commands.md):

| Function | Engine calls |
|---|---|
| `take_control(army, unit)` | `create_unit_controller`, `add_units`, `take_control` |
| `prepare_melee(uc, unit)` | `fire_at_will(false)`, disable `skirmish`, `melee(true)` |
| `move`, `move_formation`, `rotate`, `halt` | `goto_location`, `goto_location_angle_width`, `rotate`, `halt` |
| `attack_melee(uc, enemy)` | `melee(true)`, `attack_unit(enemy, false, true)` |
| `attack_ranged(uc, enemy)` | `melee(false)`, `fire_at_will(false)`, `attack_unit(enemy, true, false)` |
| `stop_firing(uc)` | `halt()` + `fire_at_will(false)` |
| `set_guard(uc, unit, on)` | `change_behaviour_active('defend', on)` |
| `withdraw(uc)` | `withdraw(true)` |
| `use_ability_on_self(uc, unit, key)` | `perform_special_ability(key, unit)` |
| `teleport(uc, p, bearing, width)` | `teleport_to_location` |
| `apply(command, ctx)` | Executes one validated command |

## planner_adapter — hand units to the game's AI

`hand_over(bm, name, alliance, units, enemies)` gives units to the game's
tactical AI instead of our orders. It uses CA's `script_ai_planner`
(`lib_battle_script_ai_planner.lua`) — the mechanism generated battles use
for AI armies. It wraps `alliance:create_ai_unit_planner()`; without the
library the adapter calls that planner directly. Returns
`{mode, attack(), release()}`: `attack()` means "attack the enemy force" and
is re-issued by the entry.

### The game AI with an objective (27.09.2026)

In battle `script_ai_planner` has, among others: `attack_force`, `attack_unit`,
`defend_force`, `defend_position`, `move_to_position`, `move_to_force`,
`rush_position`, `set_patrol_defend_radius`, `release`. `defend_position(position,
radius)` is verified: the game AI holds the place. In our XML battles the engine
counts both sides as attackers; without an objective the game AI attacks.

## Worth knowing

- An accepted call is not proof of execution. Check the result through unit
  state ([units](units.md)).
- `fire_at_will(false)` does **not** cancel an explicit fire order: add `halt()`.
- `defend` stops pursuit; it is not entrenchment.
- `withdraw` works only when the XML allows withdrawal.
- There is no instant stop; units finish a few metres from the order point.
