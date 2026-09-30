# Starting deployment

[← Back](README.md) · [Units](README.md) · [Русский](../../../ru/game/units/deployment.md) · [Attacker and defender](../battle-roles.md)

Verified on 26 September 2026 in five controlled diagnostic launches on [The Moorlands Route](../maps/moorlands-route.md), using 15 Empire units per side (General, eight shieldless Spearmen, six Archers), rank 1, XP 0, Ultra. A diagnostic script placed the units of both sides. All owned games were closed. The fifth run completed deployment and 10 simulated combat seconds at x7. [Evidence and limitations](../../../../research/evidence/units/deployment-20260926/README.md).

## How a script places units

During the `Deployment` phase the script takes control of a unit and moves it with `controller:teleport_to_location(position, facing_deg, width_m)`: an X/Z point, a facing and a formation width. Ground Y comes from the terrain. In the project this is `apps.orders.adapter.teleport` ([orders](../../apps/orders.md)). In these runs both sides' plans were checked before the first placement, and real positions and distances between own units after it; then deployment ended and the battle started.

The final diagnostic applied 30 placements during Deployment, verified both sides before combat, then confirmed control of all 30 units and x7. Minimum native distances between same-side bounding boxes were **48.464 m / 43.029 m**. The largest difference between requested anchor and native main-squad position was **9.418 m**. This is an observed difference, not a universal placement-error limit.

The allowed test rectangles are declared in the scenario XML: centres X=-270/+270, Z=0, width 400 m, depth 260 m, opposite orientations. Native XML export retained those rectangles. The zones do not set the attacker and defender roles (28.09.2026): `<timeout_winning_alliance_index>` does — [details](../battle-roles.md). This is a known test-zone source, **not a discovered general Lua getter for a campaign deployment area**. A source for campaign zones has not been found yet. Map radar bounds and reachability alone do not define a deployment zone.

## Commands and readouts differ

- `is_valid_for_deployment()` returned false for spawned units that were later successfully repositioned. It is not a coordinate-in-zone test.
- Script-control flags were false in the same callback as `take_control()`; all 30 were confirmed after 200 simulated ms in the tested sequence. Do not assume immediate acknowledgement.
- The requested anchor, native main-squad position, requested facing, `bearing()`, `ordered_bearing()` and `ordered_width()` are separate values. The engine can adjust a formation. A 5° equality assumption rejected an earlier valid spatial placement with a 7.91° bearing difference. The engine holds a unit's facing in 64 sectors — [details](commands.md#unit-facing-64-sectors).
- A single General has no formation ranks: width parameter 5 m produced an ordered-width readout about 1.4 m. Infantry requested width, ordered width and lateral entity-centre span also differed.
- `ManList.At(i).Position` did not reliably follow the second army's new native positions in this setup. The second General's point still matched its initial XML spawn while its native position matched the requested plan. Stale or differently synchronized UI data is a hypothesis; the cause was not proven. Do not mirror/shift these values or reveal enemies to make them look correct.

The checks covered the **native unit position** and native distances between own bounding boxes. Complete physical bounds of every soldier, obstacle clearance and the legality of any placement remain unverified. CCO entity positions are diagnostics only.

API references: [CA unitcontroller reference](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_unitcontroller.html#function:battle_unitcontroller:teleport_to_location), [unit readouts](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_unit.html), [legacy official XML specification](https://wiki.totalwar.com/w/Battle_XML_Documentation.html). Legacy map dimensions in the XML manual are not WH3 map measurements.
