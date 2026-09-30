# Research archive

[← Back](../README.md) · [Documentation](../README.md) · [Русский](../../ru/research/README.md)

The history of how the knowledge behind the code was obtained. Reports keep
their research stage, so they may mention old paths, module names and
decisions that changed later. Current behaviour: [apps](../architecture/apps.md)
and [game knowledge](../game/README.md).

The reports below exist **in Russian only**; each row gives the main result.

## Map data (2026-09-25)

Section hub (Russian): [map data research](../../ru/research/map-data/README.md).

| Report | Main result |
|---|---|
| [Map data catalogue](../../ru/research/map-data/catalog.md) | Which battlefield data sources exist and which were verified |
| [MP Crossroads: boundaries without launch](../../ru/research/map-data/crossroads-boundaries.md) | Boundary fields exist in files but are unset; edges cannot be recovered from them |
| [MP Crossroads: live check](../../ru/research/map-data/crossroads-live-check.md) | Height, ground and `is_area_clear` work; XML export gives no boundaries |
| [MP Crossroads: large areas](../../ru/research/map-data/crossroads-large-area-check.md) | Queries up to 8 × 8 km answer but reveal no edge coordinates |
| [MP Crossroads: dense grid](../../ru/research/map-data/crossroads-grid-check.md) | No closed frame in `is_area_clear`; a height feature near ±1000 m |
| [Internet research on edges](../../ru/research/map-data/map-edges-internet-research.md) | Best route: a frame from minimap coordinates |
| [MP Crossroads: minimap frame](../../ru/research/map-data/crossroads-radar-check.md) | `BattleRadarPosition` works; a 1536 × 1536 m frame recovered |
| [Deployment zone as scale](../../ru/research/map-data/deployment-scale-check.md) | Deployment zone size from a test XML is not the map scale |
| [Kislev plains](../../ru/research/map-data/kislev-plains-check.md) | Frame and surface grid on an official landscape, cavalry route |
| [Kislev: 5/3/2/1 m cells](../../ru/research/map-data/kislev-scales-check.md) | Cost and size of collection at different grid steps |

## Launch

Section hub (Russian): [launch research](../../ru/research/launch/README.md).

| Report | Main result |
|---|---|
| [Loading times](../../ru/research/launch/loading-times.md) | ~83 s to the first battle, ~11 s per rematch; a series in one process saves time |
| [Required True Sight mod](../../ru/research/launch/dependencies.md) | How the mod is pinned and loaded into every battle |
| [A series of battles in one process](../../ru/research/launch/test-stand.md) | A scripted rematch through `button_rematch` and `dialogue_box`: 3 battles in a row, x25–28 at a requested x40 |

## Measurement analyses (27–28.09.2026)

| Analysis | Main result |
|---|---|
| [Moorlands Route: obstacles](../../../research/analysis/moorlands-obstacles/README.md) | 10 large obstacle groups, no narrow gaps between rocks; `is_area_clear` hardly sees fences |
| [Moorlands Route: the hamlet](../../../research/analysis/hamlet/README.md) | For the engine the hamlet is one solid block; an order into it is silently dropped; jams come from own units |
| [How the game's AI stands in defence](../../../research/analysis/enemy-layout/README.md) | Lines with 2–6 m gaps, ≈ 20 m between lines, archers in the second line |

## Archive files

The [research/](../../../research/) folder at the repository root:

| Path | Contents |
|---|---|
| `research/evidence/map/` | Map field captures: grids, masks, objects, bridges, fords, forest, slopes |
| `research/evidence/map-data/` | Evidence for the reports above: XML, JSONL, CSV, images |
| `research/evidence/maps/moorlands-route/` | The full Moorlands Route package |
| `research/evidence/units/`, `units-catalog/` | Unit experiments: state, range, commands, deployment |
| `research/analysis/` | Measurement analyses: write-ups and pictures; data stays local |
| `research/probes/` | Minimal Lua probes for state and range |
| `research/scripts/` | Research Python/Lua scripts and the previous launchers |
| `research/scripts/legacy-lua/` | The original Lua modules before the move to `src/apps` |
| `research/kit-manifest.json` | Origin and SHA-256 of every file of the source set |

**Git holds only images (`.png`, `.svg`) and descriptions (`.md`) from
`research/evidence` and `research/analysis`.** Raw data — `.csv`/`.gz` grids, `.npz` arrays, `.jsonl`
logs, `.json` summaries, experiment XML — lives only on the local disk (see
`.gitignore`). The docs mark it as "local archive". Its hashes are kept in
`research/kit-manifest.json`, so a copy can be verified. It can only be
recollected by running the game, so keep a backup of the folder.

Scripts in `research/scripts` refer to the old layout (`src/`, `tools/`,
`.tools/`) and run only after fixing paths; working versions are in `tools/`.
