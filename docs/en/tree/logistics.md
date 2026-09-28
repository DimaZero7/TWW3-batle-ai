# Logistics past an obstacle

[← Back](README.md) · [Documentation](../README.md) › [AI tree](README.md) › Logistics · [Русский](../../ru/tree/logistics.md)

A branch of the approach: when a step walks round an obstacle, units pass it in
queues, not in a crowd. **Off** by default — the engine walks round by itself.

## Place in the tree

<!-- generated:tree:here:logistics -->
```mermaid
flowchart BT
  classDef done fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  classDef done_b_h fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-dasharray:5 4,stroke-width:4px
  base["Base<br/>18 modules"]:::done
  deploy["1 · Deployment<br/><i>Strategic</i>"]:::done
  approach["2 · Approach to the window<br/><i>Tactical</i>"]:::done
  logistics["Logistics past an obstacle<br/>logistics"]:::done_b_h
  base --> deploy
  deploy --> approach
  logistics -.- approach
```
<!-- /generated -->

## Card

<!-- generated:tree:card:logistics -->
| | |
|---|---|
| What it does | Queues past an obstacle: who goes round, which side, in which order and width |
| Starts when | a step walks round an obstacle |
| Without it | the engine walks the units round an obstacle by itself |
| Switch | `logistics` (off) |
| Kind, level | branch, Tactical |
| Grows from | [Approach to the window](approach.md) |
| Modules | [logistics](../apps/logistics.md), [mask](../apps/mask.md) |
| Status | done |
<!-- /generated -->

## How it works

```mermaid
flowchart LR
  d["a step round<br/>an obstacle"] --> band["band: obstacle + 5 m,<br/>who crosses it"]
  band --> side["sides: who goes<br/>left, who right"]
  side --> q["a queue per lane,<br/>no overtaking"]
  q --> n{"unit wider<br/>than the gap?"}
  n -- yes --> narrow["narrow in place"]
  n -- no --> go["go when the one<br/>ahead has moved on"]
  narrow --> go
```

## Example

A long wall at a rock (16 units): queues on both sides, no crowding.

![Logistics at a rock](../../../research/analysis/logistics/rock_march_wide/walk_1.png)

In motion, with the queue and all at once side by side: `python -m tools.viewer logistics rock_march_wide` — [battle viewer](../launch/viewer.md#round-an-obstacle-in-queues).

## Checks

- `tests/apps/logistics/`; battle 27.09.2026: 5 kinds of ground, Skaven too; backlog #18.

Modules: [logistics](../apps/logistics.md) · [mask](../apps/mask.md).
