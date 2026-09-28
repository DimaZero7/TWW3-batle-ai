# Formation on the map

[← Back](README.md) · [Documentation](../README.md) › [AI tree](README.md) › Formation on the map · [Русский](../../ru/tree/map_fit.md)

A branch of deployment: the formation does not stand on rocks. On a rock the
engine sets a unit crooked (up to 18 m off, facing 172 deg instead of 90 — measured 28.09.2026).

## Place in the tree

<!-- generated:tree:here:map_fit -->
```mermaid
flowchart BT
  classDef done fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  classDef done_b_h fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-dasharray:5 4,stroke-width:4px
  base["Base<br/>18 modules"]:::done
  deploy["1 · Deployment<br/><i>Strategic</i>"]:::done
  map_fit["Formation on the map<br/>map_fit"]:::done_b_h
  base --> deploy
  map_fit -.- deploy
```
<!-- /generated -->

## Card

<!-- generated:tree:card:map_fit -->
| | |
|---|---|
| What it does | When the formation does not fit on the rocks: another width, a shift sideways and back, a turn by a sector |
| Starts when | a captured map (the mask) is given |
| Without it | the formation stands where asked; the map is not considered |
| Switch | `map_fit` (on) |
| Kind, level | branch, Strategic |
| Grows from | [Deployment](deploy.md) |
| Modules | [formation](../apps/formation.md), [mask](../apps/mask.md) |
| Status | done |
<!-- /generated -->

## How it works

```mermaid
flowchart LR
  p["formation where asked"] --> q{"every unit<br/>fits?"}
  q -- yes --> ok["as planned"]
  q -- no --> w{"another width<br/>(6 best options)"}
  w -- fits --> ok2["another width"]
  w -- no --> s{"shift: sideways up to 90 m,<br/>back up to 60 m,<br/>6 m steps"}
  s -- found --> ok3["shifted"]
  s -- no --> t{"turn by 1–2 sectors<br/>(±5.6, ±11.25 deg)"}
  t -- found --> ok4["turned"]
  t -- no --> no["as planned,<br/>marked 'no place'"]
```

## Example

The test battle: part of the formation would stand on a rock — it moved 18 m back and fits.

![Formation on the map](../../assets/tree/test_halves.png)

## Checks

- `tests/apps/formation/test_fit.py`; branch off = the formation without the map: `tests/tools/test_tree_branches.py`;
- battle 28.09.2026: 15 units 0.5–1.0 m from the plan, none on a rock.

Modules: [formation](../apps/formation.md) · [mask](../apps/mask.md).
