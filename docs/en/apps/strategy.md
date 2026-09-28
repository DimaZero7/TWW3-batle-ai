# strategy

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/strategy.md)

Strategy choice. Pure: a catalogue of strategies, each with conditions, a
score, unit roles and the formation layout it asks for. Reads only
[assessment](assessment.md) features; knows nothing about geometry.
Code: `src/apps/strategy/`.

| Function | What it does |
|---|---|
| `CATALOG` | Strategies; now one: `wall_and_arc`, six conditions, layout `line_and_blocks` |
| `select(features)` | Checks every strategy; best score among those that fit; returns the key (or `none`) and every candidate with its conditions |
| `assign_roles(entry, units, category)` | Unit roles: `wall`, `arc`, `lord`, `other` |
| `decide(features, units, category)` | Decision: strategy, layout, its parameters, roles, candidates |
| `contract.check_decision(d, units)` | Contract with the formation: a layout and a known role for every unit |
