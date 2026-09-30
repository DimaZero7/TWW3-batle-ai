# Battle map catalogue

[← Back](../README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › Map catalogue · [Русский](../../../ru/game/maps/README.md)

Cards for specific official battle maps. The [map collection guide](../map/README.md) describes methods; this catalogue stores map identities, confirmed properties and links to reusable measurements.

<!-- generated:docs:index -->
- [MP Crossroads (flat) — an empty flat map](crossroads-flat.md) — A flat field with no obstacles in the middle
- [The Moorlands Route — Empire grasslands](moorlands-route.md) — The project's first field map: its terrain and objects were captured, and units, visibility and deployment were checked on it
<!-- /generated -->

A map identity includes the database key, terrain resource, catchment, tile upgrade and scenario conditions. A shared terrain resource can contain different battlefields. Do not reuse another catchment's grid or infer dimensions from zero database defaults.

Each card separates database facts, live measurements and unknowns. New verified captures retain scenario/source hashes, game version when available, coordinate frame, grid step, measurement duration and limits. Height, forest, water and object layers link to their evidence rather than duplicating collection instructions. Only the ground goes here: army positions and battle logs stay outside the catalogue.

A card saves research, but an AI — and a future network — must read the ground in battle and work on other maps. A card does not authorize hard-coded routes, enemy positions or hidden observations.
