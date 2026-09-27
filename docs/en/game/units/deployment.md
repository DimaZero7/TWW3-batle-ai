# Starting deployment

[Units](README.md) · [Русский](../../../ru/game/units/deployment.md) · [Current policy contract](../../apps/deployment.md)

Verified on 26 September 2026 in five controlled diagnostic launches on [The Moorlands Route](../maps/moorlands-route.md), using 15 Empire units per side (General, eight shieldless Spearmen, six Archers), rank 1, XP 0, Ultra. All owned games were closed. The fifth run completed deployment and 10 simulated combat seconds at x7; these were synthetic test policies, not a fighter competition. [Evidence and limitations](../../../../research/evidence/units/deployment-20260926/README.md).

## Current contract and historical scope

New generation-2 placement uses [deployment-placement-v2](../../apps/deployment.md). The five runs and numbers below describe the historical v1 adapter. Its reservation circles, 72 m derived minimum and 35 m reference inset were artificial restrictions and are withdrawn. The new seven-attempt evidence verifies close/edge references with sampled native pair separation, without a requested/native delta veto. It does not prove whole-soldier bounds or a perfect wall. General fighter integration remains pending.

## What the historical runs verified

Each policy runs its own `deploy` function **inside the Lua mod**. It selects its units' X/Z positions, requested facing and width. The common adapter checks both complete plans before placing any unit, then issues `controller:teleport_to_location(position, facing_deg, width_m)` only during `Deployment`. Ground Y comes from the terrain reader. After native position and own-unit separation checks, the mod ends deployment and starts ordinary policy decisions. No external agent chooses the formation or advises a running policy.

The final diagnostic applied 30 placements during Deployment, verified both sides before combat, then confirmed control of all 30 units and x7. Minimum native distances between same-side bounding boxes were **48.464 m / 43.029 m**. The largest difference between requested anchor and native main-squad position was **9.418 m**. This is an observed difference, not a universal placement-error limit.

The allowed test rectangles are declared in the scenario XML: centres X=-270/+270, Z=0, width 400 m, depth 260 m, opposite orientations. Native XML export retained those rectangles. This is a known test-zone source, **not a discovered general Lua getter for a campaign deployment area**. A campaign zone provider remains unimplemented. Map radar bounds and reachability alone do not define a deployment zone.

## Commands and readouts differ

- `is_valid_for_deployment()` returned false for spawned units that were later successfully repositioned. It is not a coordinate-in-zone test.
- Script-control flags were false in the same callback as `take_control()`; all 30 were confirmed after 200 simulated ms in the tested sequence. Do not assume immediate acknowledgement.
- The requested anchor, native main-squad position, requested facing, `bearing()`, `ordered_bearing()` and `ordered_width()` are separate values. The engine can adjust a formation. A 5° equality assumption rejected an earlier valid spatial placement with a 7.91° bearing difference.
- A single General has no formation ranks: width parameter 5 m produced an ordered-width readout about 1.4 m. Infantry requested width, ordered width and lateral entity-centre span also differed.
- `ManList.At(i).Position` did not reliably follow the second army's new native positions in this setup. The second General's point still matched its initial XML spawn while its native position matched the policy plan. Stale or differently synchronized UI data is a hypothesis; the cause was not proven. Do not mirror/shift these values or reveal enemies to make them look correct.

The historical v1 checks validated the **native reference position**, a declared conservative reservation and native distances between own bounding boxes; that reservation requirement is withdrawn for generation 2. Complete physical bounds of every soldier, obstacle clearance and universal deployment legality remain unverified. CCO entity positions are retained as diagnostics, not authority for that claim. The adapter never substitutes a tactical placement when a required check fails.

API references: [CA unitcontroller reference](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_unitcontroller.html#function:battle_unitcontroller:teleport_to_location), [unit readouts](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_unit.html), [legacy official XML specification](https://wiki.totalwar.com/w/Battle_XML_Documentation.html). Legacy map dimensions in the XML manual are not WH3 map measurements.
