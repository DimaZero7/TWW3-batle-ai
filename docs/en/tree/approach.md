# Phase 2 · Approach to the window

[← Back](README.md) · [Documentation](../README.md) › [AI tree](README.md) › Approach to the window · [Русский](../../ru/tree/approach.md)

Step up and stand in the **window**: our first echelon reaches their infantry,
their archers do not reach our wall. The trunk `tactics` decides — the same in battle and in the simulation.

## Place in the tree

<!-- generated:tree:here:approach -->
```mermaid
flowchart BT
  classDef done fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  classDef done_h fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-width:4px
  classDef done_b fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-dasharray:5 4
  classDef started fill:#FAEEDA,stroke:#854F0B,color:#412402
  base["Base<br/>18 modules"]:::done
  deploy["1 · Deployment<br/><i>Strategic</i>"]:::done
  approach["2 · Approach to the window<br/><i>Tactical</i>"]:::done_h
  align["Alignment<br/>align"]:::done_b
  logistics["Logistics past an obstacle<br/>logistics"]:::done_b
  safe_detour["Detour out of their reach<br/>safe_detour"]:::done_b
  under_fire_stop["Stop under fire<br/>under_fire_stop"]:::done_b
  fire["3 · Fire from the window<br/><i>Combat</i>"]:::started
  base --> deploy
  deploy --> approach
  align -.- approach
  logistics -.- approach
  safe_detour -.- approach
  under_fire_stop -.- approach
  approach --> fire
```
<!-- /generated -->

## Card

<!-- generated:tree:card:approach -->
| | |
|---|---|
| What it does | In 50 m steps to the window: our archers reach them, theirs do not reach us; one action at a time; the formation is carried whole, a rock in the way — a step aside |
| Starts when | the army is deployed, the enemy is seen |
| Without it | the army stands where it was deployed |
| Switch | `approach` (on) |
| Kind, level | phase, Tactical |
| Grows from | [Deployment](deploy.md) |
| Modules | [tactics](../apps/tactics.md), [approach](../apps/approach.md), [reach](../apps/reach.md), [battlefield](../apps/battlefield.md), [vision](../apps/vision.md), [mask](../apps/mask.md) |
| Status | done |
<!-- /generated -->

## How it works

```mermaid
flowchart TB
  v["view of the battle: our and<br/>their blocks, field, mask"] --> w["reach: the window<br/>(pair by pair)"]
  w --> s["stop line and step <= 50 m"]
  s --> f{"under fire?"}
  f -- yes --> hold1["stand<br/>(branch stop under fire)"]
  f -- no --> al{"turned > 10 deg<br/>or aside > 15 m?"}
  al -- yes --> align["align<br/>(branch alignment)"]
  al -- no --> arr{"at the stop line?"}
  arr -- yes --> hold2["stand: phase 3"]
  arr -- no --> path{"the formation fits<br/>where the step ends?"}
  path -- yes --> step["step"]
  path -- no --> past["a place past the obstacle<br/>(branch detour)"]
  past -- found --> step
  past -- no --> blocked["stop: no place"]
```

```mermaid
sequenceDiagram
  participant G as game / simulator
  participant T as tactics (trunk)
  G->>T: view of the battle
  T-->>G: intent: step 50 m
  G->>G: orders, the army walks
  G->>T: units stopped (finish)
  G->>T: new view
  T-->>G: stand: at the stop line
```

```mermaid
flowchart LR
  far["beyond 104 m:<br/>nobody reaches"] --> win["96–104 m: WINDOW<br/>we reach, they do not"] --> close["closer than 96 m:<br/>their archers hit the wall"]
  style win fill:#E1F5EE,stroke:#0F6E56
  style close fill:#FAECE7,stroke:#993C1D
```

## Example

Simulation of the test armies in the open (the enemy stands as the game's AI did in the
test battle): 8 steps, stands 100.7 m from their front, window 96.7–104.7 m; 60 s of fire — us 0, them −254.

![Approach to the window](../../assets/tree/window_open.png)

In battle (x20): the window from the real blocks — ours reach from 100 m, their archers
from 95 m; in the window, 59 s — us 0, them −96…−149. The game's AI does not stand
still: it re-forms as we come, and its spearmen charge after about 100 s (phases 3–5).

Watch this battle in motion, the game and the simulation side by side: [battle viewer](../launch/viewer.md).

## Checks

- `tests/apps/reach/`, `tests/apps/approach/`, `tests/apps/tactics/`; golden runs (15 armies);
- branch off = the army stands where deployed: `tests/tools/test_tree_branches.py`;
- battle: `window_game` (4 battles 28.09.2026).

Modules: [tactics](../apps/tactics.md) · [approach](../apps/approach.md) · [reach](../apps/reach.md) · [battlefield](../apps/battlefield.md) · [vision](../apps/vision.md) · [mask](../apps/mask.md).
