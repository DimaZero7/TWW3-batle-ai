# Detour out of their reach

[← Back](README.md) · [Documentation](../README.md) › [AI tree](README.md) › Detour out of their reach · [Русский](../../ru/tree/safe_detour.md)

A branch of the approach: when the formation does not fit where a step ends, a
place past the obstacle is searched **only where their shooters do not reach**.

## Place in the tree

<!-- generated:tree:here:safe_detour -->
```mermaid
flowchart BT
  classDef done fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  classDef done_b_h fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-dasharray:5 4,stroke-width:4px
  base["Base<br/>18 modules"]:::done
  deploy["1 · Deployment<br/><i>Strategic</i>"]:::done
  approach["2 · Approach to the window<br/><i>Tactical</i>"]:::done
  safe_detour["Detour out of their reach<br/>safe_detour"]:::done_b_h
  base --> deploy
  deploy --> approach
  safe_detour -.- approach
```
<!-- /generated -->

## Card

<!-- generated:tree:card:safe_detour -->
| | |
|---|---|
| What it does | A place past an obstacle is searched only out of their shooters' reach |
| Starts when | the formation does not fit where the step ends |
| Without it | a place past an obstacle is searched up to 20 m short of their front |
| Switch | `safe_detour` (on) |
| Kind, level | branch, Tactical |
| Grows from | [Approach to the window](approach.md) |
| Modules | [tactics](../apps/tactics.md), [approach](../apps/approach.md), [reach](../apps/reach.md) |
| Status | done |
<!-- /generated -->

## How it works

```mermaid
flowchart LR
  s["the formation does not fit<br/>where the step ends"] --> search["search further along<br/>the axis, every 3 m"]
  search --> lim{"no further than<br/>their reach and 20 m<br/>short of their front"}
  lim -- found --> step["step there<br/>(the engine walks round)"]
  lim -- no --> stop["stop: no place"]
```

## Example

A rock across the way, close to them (`rock_attack`):

| With the branch: stop before the rock | Without: past the rock, under their archers |
|---|---|
| ![With](../../assets/tree/rock_attack.png) | ![Without](../../assets/tree/rock_attack_no_safe_detour.png) |

## Checks

- branch off = the old behaviour (past the rock): `tests/tools/test_tree_branches.py`; golden `rock_attack`, `rock_attack_wide`.

Modules: [tactics](../apps/tactics.md) · [approach](../apps/approach.md) · [reach](../apps/reach.md).
