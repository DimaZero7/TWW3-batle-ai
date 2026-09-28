# Phase 1 · Deployment

[← Back](README.md) · [Documentation](../README.md) › [AI tree](README.md) › Deployment · [Русский](../../ru/tree/deploy.md)

The first phase and the tree's root: from the armies pick a strategy and place the formation.

## Place in the tree

<!-- generated:tree:here:deploy -->
```mermaid
flowchart BT
  classDef done fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  classDef done_h fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-width:4px
  classDef done_b fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-dasharray:5 4
  base["Base<br/>18 modules"]:::done
  deploy["1 · Deployment<br/><i>Strategic</i>"]:::done_h
  map_fit["Formation on the map<br/>map_fit"]:::done_b
  formation_window["Formation keeps the window<br/>formation_window"]:::done_b
  approach["2 · Approach to the window<br/><i>Tactical</i>"]:::done
  base --> deploy
  map_fit -.- deploy
  formation_window -.- deploy
  deploy --> approach
```
<!-- /generated -->

## Card

<!-- generated:tree:card:deploy -->
| | |
|---|---|
| What it does | Weighs the armies, picks a strategy and places the formation: a spear wall, archers in two echelons, the lord in the centre |
| Starts when | the battle starts |
| Without it | the tree's root, cannot be off |
| Switch | — |
| Kind, level | phase, Strategic |
| Grows from | — |
| Modules | [plan](../apps/plan.md), [assessment](../apps/assessment.md), [strategy](../apps/strategy.md), [formation](../apps/formation.md) |
| Status | done |
<!-- /generated -->

## How it works

```mermaid
flowchart LR
  in["our and their units<br/>(roster)"] --> as["assessment<br/>weigh the armies"]
  as --> st{"strategy<br/>wall and arc<br/>fits?"}
  st -- no --> none["no_strategy<br/>(an error in battle)"]
  st -- yes --> fo["formation<br/>every wall width,<br/>archer width and rows"]
  fo --> rules{"square archer blocks,<br/>wall >= 5 m,<br/>reach >= 80 m"}
  rules --> best["the thickest wall<br/>that passed"]
  best -. branch .-> win["formation keeps the window"]
  best -. branch .-> fit["formation on the map"]
  best --> out["every unit's place:<br/>wall, 2 echelons,<br/>lord in the centre"]
```

The lord stands in the centre, in a 10 m passage between the halves of the first
archer row: his job is to bind the armoured enemy lord, and from the centre every
flank is closer.

```mermaid
flowchart TB
  subgraph wall["spear wall · 6 x 30 m, front 175 m, 9.3 m deep"]
    w1[" "] --- w2[" "] --- w3[" "] --- w4[" "] --- w5[" "] --- w6[" "]
  end
  subgraph e1["first echelon · 4 squares 20 x 18 m"]
    a1["archers"] --- a2["archers"] --- L(("lord")) --- a3["archers"] --- a4["archers"]
  end
  subgraph e2["second echelon"]
    b1["archers"] --- b2["archers"] --- b3["archers"] --- b4["archers"]
  end
  wall --- e1 --- e2
```

## Example

The test army (a lord, 6 spearmen, 8 archers) on the test battle's map (labels in Russian):

![The test army deployed](../../assets/tree/test_halves.png)

| What | Value |
|---|---|
| Wall | 6 x 30 m: front 175 m, 9.3 m deep |
| First echelon | its middle 21.3 m behind the wall's front |
| Lord's passage | 10 m planned, 11.7 m by the soldiers in battle |
| Lord | 0.1 m from the plan in battle; the way to a flank 101 m, clear |

## Checks

- `tests/apps/formation/` — 565 mixes of 2 to 20 units; `tests/apps/strategy/`, `tests/apps/plan/`;
- golden simulator runs (`tests/golden/`);
- battle 28.09.2026: all 15 units 0.5–2.4 m from the plan, held 60 s.

Modules: [plan](../apps/plan.md) · [assessment](../apps/assessment.md) · [strategy](../apps/strategy.md) · [formation](../apps/formation.md).
