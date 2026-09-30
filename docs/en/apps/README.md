# Code modules

[← Back](../README.md) · [Documentation](../README.md) › Modules · [Русский](../../ru/apps/README.md)

Every `src/apps` module — what it does, why, how; and the `src/entries` entry points that wire the modules into probes and battle recorders. The overview: [list of apps](../architecture/apps.md).

<!-- generated:docs:index -->
- [battle](battle.md) — Battle context: sides, armies, units and roles
- [bridge](bridge.md) — The bridge to the network: our side of a real battle is commanded by the network running in the companion, a program outside the game (`tools/nn/companion/`)
- [core](core.md) — Shared helpers with no engine access and no dependency on other apps
- [Entry points](entries.md) — An entry wires apps into a specific battle: it subscribes to phases, runs the tick, calls adapters and services and writes telemetry
- [intel](intel.md) — What a side knows about enemies: whether one is visible now and where it was last seen
- [map](map.md) — Map reading: the minimap (radar) frame, cell grid, height, ground, cell clearance, buildings and structures
- [navigation](navigation.md) — Point reachability for specific units
- [observation](observation.md) — **One side's view** — the only thing an AI side may receive
- [orders](orders.md) — Unit orders: which engine calls give them, how to hand units to the game's own AI and which facings the engine holds
- [telemetry](telemetry.md) — Recording what happened in a battle
- [units](units.md) — Own unit state, missile range, the unit card and its soldiers' positions
<!-- /generated -->
