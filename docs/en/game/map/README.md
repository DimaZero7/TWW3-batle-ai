# Map information: working guide

[← Back](../README.md) · [Project](../../../../README.md) · **English** | [Русский](../../../ru/game/map/README.md)

This is the current instruction for collecting **map information only**.
It covers methods already exercised in the installed WH3 build. Unit state,
combat orders, neural networks and future terrain/navigation hypotheses are outside this guide.

## Read in order

1. [Coordinates and metres](coordinates.md).
2. [Recovering the radar frame](boundaries.md).
3. [Sampling the surface](sampling.md).
4. [Terrain heights and height maps](heights.md).
5. [Forest zones and tree placements](vegetation.md).
6. [Water and ground modifiers](water-ground.md).
7. [Slopes and height differences](slopes.md).
8. [Physical objects and structure effects](objects.md).
9. [Point reachability and verified fords](passages.md).
10. [Bridge identification and tested height layers](bridges.md).
11. [Using the Lua module and capture tools](tools.md).
12. [Evidence, pictures and timing](evidence.md).

## What we obtain

| Output | Meaning |
|---|---|
| Radar rectangle in world coordinates | A measured reference frame; exact movement boundary remains unproven |
| Terrain height at X/Z | One numeric height value, not a full 3D scene |
| Ground type at a sampled point | A string such as `grass`, `forest` or `rock` |
| Forest mask at 3 m | Cell-center `forest` classifications; not a measured unit-effect mask |
| Selected tree placements for the tested Kislev scenario | Extracted from resources and aligned to terrain; not live Lua enumeration or a collision mask |
| Area restriction flag | `is_area_clear` for a specified rectangle; not a full pathfinding result |
| Our cell grid | A derived representation with explicit metre spacing, origin and dimensions |
| Shallow/deep water masks | Ground strings sampled at cell centres; no water depth measurement |
| Estimated slopes and neighbour differences | Geometry derived from saved heights; no engine penalty formula |
| Native physical-object list | Positions, names and state; no guaranteed complete collision geometry |
| Per-unit point reachability | Checked after deployment; not a returned path or straight-segment test |
| Three Cathay ford connections | Derived from the sampled grid and crossed by cavalry and infantry |
| Bridge object and CCO fields | One bridge identified; traversal and deck geometry not established |

## Workflow

Read the radar transform in the loaded battle → verify the four corners → choose
cell size → sample each cell at its local terrain height → write raw data and
metadata → check completeness → draw and interpret the result.

**Do not label the radar rectangle as a proven playable boundary. Do not turn
a negative area query into “every part of this cell is impassable.”**

Working landscape: vanilla Kislev Plains, `wh3_main_macro_ksl_plains_01`, a fixed
classic test scenario. Current recommendation: 2 m cells; 1 m remains available
for finer one-time capture. Neither recommendation approves a neural-network schema.

New drafts go to `tmp/`. New confirmed results need raw evidence, hashes, conditions
and limits. The Russian page is the master; the English one mirrors it with the same facts.
Do not add research lists or untested API candidates to this guide or its index.

The field-feature checks preserve negative results too: no runtime `scrub` mask,
bridge traversal, general narrow-pass width or active structure aura
has been validated. Siege and settlement tests remain outside the current scope.

[Specific map cards](../maps/README.md) store reusable map data; this guide remains the source for collection methods.
