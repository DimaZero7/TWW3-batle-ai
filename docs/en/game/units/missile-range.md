# Missile attack range

[Units](README.md) · [Русский](../../../ru/game/units/missile-range.md) · [Visibility](../../apps/intel.md)

**Verified 2026-09-26:** Lua can read missile range for own units, a separate same-alliance army, and currently visible enemies. This covers ranged and hybrid infantry. It does not measure vision radius. **Enemy visibility must be checked before reading range or querying a pair.** The [information rule](../../apps/intel.md) is mandatory.

## Values and meanings

| Readout | Type / unit | Measured interpretation |
|---|---|---|
| `unit:missile_range()` | number, metres | Native current missile-range readout; became zero after forced ammunition depletion |
| Card `scalar_missile_range` | numeric `Value`, `DisplayedValue`, `ValueBase` | Card stat; retained the weapon range with empty ammunition |
| `source:unit_in_range(target)` | boolean | Engine pair-range result; **not** a guarantee of firing or a missile weapon |
| `source:unit_distance(target)` | number, metres | Native distance accounting for unit bounding boxes, distinct from centre distance |
| `centre_distance_xz_m` | number, metres | Our planar distance between current unit positions; not the engine range predicate |

Read the card through `CcoBattleUnit.UnitDetailsContext.StatList`: enumerate rows, find `Key == 'scalar_missile_range'`, then read its values. Do not hardcode an index or parse a translated label. An absent row is unknown, not zero. All three card values agreed here; modified-range buffs, upgrades, alternative weapons and arbitrary units were not tested.

| Fixture | With ammunition: native / card | Forced empty: native / card |
|---|---:|---:|
| Empire Archers, `wh2_dlc13_emp_inf_archers_0` | 130 / 130 | 0 / 130 |
| Free Company Militia, `wh_dlc04_emp_inf_free_company_militia_0` | 90 / 90 | 0 / 90 |
| Shieldless Spearmen, `wh_main_emp_inf_spearmen_0` | 0 / absent | Not applicable |

Numbers are fixture observations, not constants to embed in faction policies. The installed DB identifies Free Company Militia as `hybrid_melee` AI usage with a pistol weapon. Same-alliance army 2 and visible enemy instances returned matching baseline values. This does not establish networked ally ownership or exact player-UI parity.

Successful `controller:melee(true)` and `melee(false)` commands left range at 130/90. This verifies the commanded-mode experiment, not an independently read mode state. `is_in_melee()` is unit-wide; it cannot identify contact with the selected target. Forced `set_current_ammo_unary(0)` produced zero ammo and native range, while the card retained its number; restoring ammunition restored native range. **Natural exhaustion was not tested.**

Halted units could have positive ammunition and an in-range target while `IsFiringMissiles=false`. Explicit attack orders then produced firing and ammunition consumption. Do not replace firing, visibility, ammunition or line-of-fire checks with the range boolean. In 47 permitted Spearmen samples, zero bounding-box distance produced `unit_in_range=true` despite native range zero: the predicate alone does not establish shooting capability.

## Geometry

Examples from the final run after 20 simulated seconds of settling; sources reported stationary and not firing:

| Source | Native range | Centre XZ distance | Native unit distance | In range |
|---|---:|---:|---:|---|
| Archers | 130 | 138.956 | 126.527 | true |
| Hybrid | 90 | 98.459 | 87.089 | true |
| Archers, farther target | 130 | 148.935 | 136.621 | false |
| Hybrid, farther target | 90 | 109.990 | 97.101 | false |

Across 7,636 permitted pair samples, the boolean matched `unit_distance <= missile_range`; a centre-distance comparison disagreed 354 times. This is a fixture result, **not a universal replacement formula**. Requested widths 10/30/60 m and bearings 0/90/180/270 degrees were exercised, but actual positions/formations shifted after teleporting. Use measured values, not stage names or requested coordinates. No exact firing-sector polygon, minimum range, terrain obstruction, elevation correction or human range-circle geometry was recovered.

`Exact examples and JSONL line numbers` (local archive: `research/evidence/units/missile-range-20260926/measured-examples.json`). Public line numbers refer to decompressed permitted logs; corresponding privileged raw lines remain in operator storage.

## Trusted reader and anti-disclosure checks

[src/units/range.lua](../../../../src/apps/units/) is a separate optional reader, **not connected to policy API v1**. Its internal call is `observe(source, target_or_nil, context)`. Only a trusted adapter may hold these game objects and provide `observer_alliance`, a native-registry-backed `is_friendly`, optional CCO provider and observation time. A policy cannot claim that an enemy is friendly.

The reader checks both source and target before any range, CCO, pair-distance or position read. Own/allied membership is verified by the adapter; enemies require native alliance visibility exactly `true`. Hidden, failed or unknown visibility blocks every sensitive call. No error includes raw engine text. Source-only range can be requested internally with no target. When a supplied target is hidden, the entire pair reading is withheld, including the source range.

Output contains plain `sensors` with `missile_range_m`, card values and, for a pair, `unit_in_range`, `unit_distance_m`, `centre_distance_xz_m`. Each is `{status='known', value=...}` or `{status='unknown', reason=...}`; preserve known zero/false. No engine objects or callable pair-query function reach the policy. The reader has no memory: `observed_ms` timestamps an attempt, not a refreshed last-seen value. Any future memory layer must preserve old values/times while hidden.

In 759 withheld live records there were **zero** calls to sensitive methods. Tests cover hidden source, hidden target, two enemies, source-only reads, visibility errors/nil/non-booleans, missing relationship and observer, and visible→hidden→visible. This is tested filtering, not an OS sandbox or certification of continuous engine visibility.

## Evidence and reproduction

Three completed launches, 11,235 observation records: `range` 3,900; `fog` 2,040; `boundary` 5,295. WH3 v9.0.0 build 50218.4334952, vanilla DB, neutral diagnostic script only, minimum graphics, Ultra, ×20. Maps: `chokepoint_badlands_river`, then `wh3_main_macro_ksl_plains_01/catchment_04`. Two native armies in alliance 1; one in alliance 2. Teleports, controlled orders and forced ammo changes are diagnostic tools, not policy permissions. All three owned games closed; temporary installed files removed.

`Summary` (local archive: `research/evidence/units/missile-range-20260926/summary.json`) · `Archive hashes` (local archive: `research/evidence/units/missile-range-20260926/archive.json`) · `DB extract` (local archive: `research/evidence/units/missile-range-20260926/database.json`) · `Research sources` (local archive: `research/evidence/units/missile-range-20260926/research.json`).

The public archive contains exact scripts/XML/module snapshots, permitted logs, lifecycle records and own-unit diagnostic fields. Privileged raw logs and raw analyses are not published. Author: battle-operator; publication reviewer: coordinator. Six range tests and six existing state tests passed. To verify hashes and reconstruct each exact pack without launching:

```text
python -X utf8 tools/unit-range/replay.py range
python -X utf8 tools/unit-range/replay.py fog
python -X utf8 tools/unit-range/replay.py boundary
python -X utf8 -m unittest discover -s tests -p test_unit_range.py
```

Optional `--output <new-pack-path>` saves a reconstructed pack and refuses overwrite. It neither installs nor launches. A future diagnostic run requires operator ownership and a new evidence directory. API reference: [CA unit reference mirror](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_unit.html), [CA CCO reference mirror](https://chadvandy.github.io/tw_modding_resources/WH3/cco/documentation.html).
