# Code modules

[← Back](../README.md) · [Documentation](../README.md) › Modules · [Русский](../../ru/apps/README.md)

Every `src/apps` module — what it does, why, how. How modules serve the AI tree — [the tree](../tree/README.md), by level — [module map](../tree/modules.md).

<!-- generated:docs:index -->
- [ai](ai.md) — The home of **our AI**: observation in, decision out
- [alignment](alignment.md) — **Tree nodes:** [Alignment](../tree/align.md)
- [approach](approach.md) — **Tree nodes:** [Approach to the window](../tree/approach.md)
- [assessment](assessment.md) — Force assessment at the start of a battle
- [battle](battle.md) — Battle context: sides, armies, units and roles
- [battlefield](battlefield.md) — The battlefield between the two main groups
- [core](core.md) — Shared helpers with no engine access and no dependency on other apps
- [deployment](deployment.md) — Initial placement under the `deployment-placement-v2` contract
- [Entry points](entries.md) — An entry wires apps into a specific battle: it subscribes to phases, runs the tick, calls adapters and services and writes telemetry
- [formation](formation.md) — **Tree nodes:** [Deployment](../tree/deploy.md) · [Formation on the map](../tree/map_fit.md) · [Formation keeps the window](../tree/formation_window.md)
- [intel](intel.md) — What a side knows about enemies: whether one is visible now and where it was last seen
- [logistics — queues past an obstacle](logistics.md) — **Tree nodes:** [Logistics past an obstacle](../tree/logistics.md)
- [map](map.md) — Map reading: the minimap (radar) frame, cell grid, height, ground, cell clearance, buildings and structures
- [mask](mask.md) — "Can we stand here?" mask over the [battlefield](battlefield.md): a grid in the
- [missile](missile.md) — Missile damage: HP a second a shooting block takes off its target, and a step of a missile exchange for the simulation
- [navigation](navigation.md) — Point reachability for specific units
- [observation](observation.md) — **One side's view** — the only thing an AI side may receive
- [orders](orders.md) — Unit commands: which exist, how they are validated and which engine calls execute them
- [plan](plan.md) — **Tree nodes:** [Deployment](../tree/deploy.md)
- [reach](reach.md) — **Tree nodes:** [Formation keeps the window](../tree/formation_window.md) · [Approach to the window](../tree/approach.md)
- [sandbox](sandbox.md) — Loads API v1 policies into a restricted environment
- [strategy](strategy.md) — Strategy choice
- [tactics](tactics.md) — **Tree nodes:** [Approach to the window](../tree/approach.md) · [Detour out of their reach](../tree/safe_detour.md) · [Stop under fire](../tree/under_fire_stop.…
- [telemetry](telemetry.md) — Recording what happened in a battle
- [tree](tree.md) — The AI's tree as data: the trunk of phases in order, branches that improve a phase, and for every branch its **baseline** (what happens without it)
- [units](units.md) — Own unit state, missile range and shared formation width rules
- [vision](vision.md) — How we see the enemy
<!-- /generated -->
