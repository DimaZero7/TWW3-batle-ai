# The AI tree

[← Back](../README.md) · [Documentation](../README.md) › AI tree · [Русский](../../ru/tree/README.md)

Everything our battle AI does is one tree. The **trunk** is the battle's phases in
order: the next starts when its condition holds. **Branches** improve a phase: each
says what happens without it and can be switched off without breaking the trunk.
Read it **bottom up**: from the base (engine and data) to the battle.

## The whole tree

<!-- generated:tree:diagram -->
```mermaid
flowchart BT
  classDef done fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  classDef done_b fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-dasharray:5 4
  classDef started fill:#FAEEDA,stroke:#854F0B,color:#412402
  classDef planned fill:#F1EFE8,stroke:#888780,color:#444441
  classDef planned_b fill:#F1EFE8,stroke:#888780,color:#444441,stroke-dasharray:5 4
  base["Base<br/>18 modules"]:::done
  deploy["1 · Deployment<br/><i>Strategic</i>"]:::done
  map_fit["Formation on the map<br/>map_fit"]:::done_b
  formation_window["Formation keeps the window<br/>formation_window"]:::done_b
  approach["2 · Approach to the window<br/><i>Tactical</i>"]:::done
  align["Alignment<br/>align"]:::done_b
  logistics["Logistics past an obstacle<br/>logistics"]:::done_b
  safe_detour["Detour out of their reach<br/>safe_detour"]:::done_b
  under_fire_stop["Stop under fire<br/>under_fire_stop"]:::done_b
  fire["3 · Fire from the window<br/><i>Combat</i>"]:::started
  their_archers["4 · Their archers hit our wall<br/><i>Combat</i>"]:::started
  infantry_attack["5 · Infantry attacks<br/><i>Combat</i>"]:::planned
  echelon_step["6 · Echelon to their archers<br/><i>Combat</i>"]:::planned
  breakthroughs["7 · Breakthroughs<br/><i>Combat</i>"]:::planned
  finish["8 · Finish and withdrawal<br/><i>Combat</i>"]:::planned
  lord_vs_lord["Lord against lord"]:::planned_b
  base --> deploy
  map_fit -.- deploy
  formation_window -.- deploy
  deploy --> approach
  align -.- approach
  logistics -.- approach
  safe_detour -.- approach
  under_fire_stop -.- approach
  approach --> fire
  fire --> their_archers
  their_archers --> infantry_attack
  infantry_attack --> echelon_step
  echelon_step --> breakthroughs
  breakthroughs --> finish
  lord_vs_lord -.- fire
```
<!-- /generated -->

Green — done, yellow — started, grey — planned. A solid arrow up is the trunk's
next phase; a dashed line is a branch (a dashed frame — it can be switched off).
Phases 3–8 and "lord against lord" come from the battle theory (Russian only).

## All nodes

<!-- generated:tree:table -->
| Node | Kind | Level | Status | Switch | Without it | Modules |
|---|---|---|---|---|---|---|
| [Deployment](deploy.md) | phase | Strategic | done | — | the tree's root, cannot be off | [plan](../apps/plan.md), [assessment](../apps/assessment.md), [strategy](../apps/strategy.md), [formation](../apps/formation.md) |
| [Formation on the map](map_fit.md) | branch | Strategic | done | `map_fit` (on) | the formation stands where asked; the map is not considered | [formation](../apps/formation.md), [mask](../apps/mask.md) |
| [Formation keeps the window](formation_window.md) | branch | Strategic | done | `formation_window` (on) | the thickest wall; the window is not checked | [formation](../apps/formation.md), [reach](../apps/reach.md) |
| [Approach to the window](approach.md) | phase | Tactical | done | `approach` (on) | the army stands where it was deployed | [tactics](../apps/tactics.md), [approach](../apps/approach.md), [reach](../apps/reach.md), [battlefield](../apps/battlefield.md), [vision](../apps/vision.md), [mask](../apps/mask.md) |
| [Alignment](align.md) | branch | Tactical | done | `align` (on) | no alignment: the army steps along its own facing | [alignment](../apps/alignment.md), [battlefield](../apps/battlefield.md), [vision](../apps/vision.md) |
| [Logistics past an obstacle](logistics.md) | branch | Tactical | done | `logistics` (off) | the engine walks the units round an obstacle by itself | [logistics](../apps/logistics.md), [mask](../apps/mask.md) |
| [Detour out of their reach](safe_detour.md) | branch | Tactical | done | `safe_detour` (on) | a place past an obstacle is searched up to 20 m short of their front | [tactics](../apps/tactics.md), [approach](../apps/approach.md), [reach](../apps/reach.md) |
| [Stop under fire](under_fire_stop.md) | branch | Tactical | done | `under_fire_stop` (on) | one action at a time even under fire | [tactics](../apps/tactics.md), [approach](../apps/approach.md) |
| [Fire from the window](combat.md#3) | phase | Combat | started | comes with the node | — | [missile](../apps/missile.md), [reach](../apps/reach.md) |
| [Their archers hit our wall](combat.md#4) | phase | Combat | started | comes with the node | — | [tactics](../apps/tactics.md) |
| [Infantry attacks](combat.md#5) | phase | Combat | planned | comes with the node | — | — |
| [Echelon to their archers](combat.md#6) | phase | Combat | planned | comes with the node | — | — |
| [Breakthroughs](combat.md#7) | phase | Combat | planned | comes with the node | — | — |
| [Finish and withdrawal](combat.md#8) | phase | Combat | planned | comes with the node | — | — |
| [Lord against lord](combat.md#lord) | parallel branch | Combat | planned | comes with the node | — | — |
<!-- /generated -->

## Switching nodes on and off

The army's `tree` field (`config/armies/*.json`). A node switched off switches off
everything that grows from it and every later phase; its parent does what the
"Without it" column says.

```json
{"tree": {"align": false, "logistics": true}}
```

```mermaid
flowchart BT
  classDef done fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  classDef off fill:#F1EFE8,stroke:#888780,color:#888780,stroke-dasharray:5 4
  d["1 · Deployment"]:::done --> a["2 · Approach"]:::done
  a -.- al["Alignment<br/>off"]:::off
  a -.- lg["Logistics<br/>on"]:::done
  a -.- sd["Detour out of their reach"]:::done
```

`{"approach": false}`: the army deploys and stands; alignment, logistics and
everything after phase 2 do not run. The first phase cannot be switched off.

## How we know the tree does not break

```mermaid
flowchart LR
  code["a change in a module"] --> g["golden simulator runs<br/>15 armies"]
  code --> b["branch off =<br/>its baseline"]
  code --> ar["architecture:<br/>levels, size, time"]
  g & b & ar --> ok{"all passed?"}
  ok -- yes --> commit["commit"]
  ok -- no --> fix["you see which<br/>battle changed"]
```

More: [tests](../testing/tests.md); modules by level — [module map](modules.md);
how to keep these docs — [rules](../architecture/documentation.md).

## Pages

<!-- generated:docs:index -->
- [Phase 1 · Deployment](deploy.md) — The first phase and the tree's root: from the armies pick a strategy and place the formation
- [Formation on the map](map_fit.md) — A branch of deployment: the formation does not stand on rocks
- [Formation keeps the window](formation_window.md) — A branch of deployment: the wall no thicker than the window allows
- [Phase 2 · Approach to the window](approach.md) — Step up and stand in the **window**: our first echelon reaches their infantry, their archers do not reach our wall
- [Alignment](align.md) — A branch of the approach: stand square opposite the enemy's main group so the wall faces their front
- [Logistics past an obstacle](logistics.md) — A branch of the approach: when a step walks round an obstacle, units pass it in queues, not in a crowd
- [Detour out of their reach](safe_detour.md) — A branch of the approach: when the formation does not fit where a step ends, a place past the obstacle is searched **only where their shooters do not reach**
- [Stop under fire](under_fire_stop.md) — A branch of the approach, the only exception to "one action at a time": if a unit of ours is under fire, the manoeuvre is broken off and nothing new is started …
- [Combat level · phases 3–8](combat.md) — The phases after we stand in the window
- [Module map](modules.md) — Which code modules sit on which level and who uses whom
<!-- /generated -->
