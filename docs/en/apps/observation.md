# observation

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/observation.md)

**One side's view** — the only thing an AI side may receive.
Code: `src/apps/observation/`.

## Two summary modes

| Mode | Built by | Contents | Purpose |
|---|---|---|---|
| Side view | `observation.adapter.observe_side` | Own units in full (68 fields); for enemies — visibility, position **only while visible**, last known position; range of own shooters (withheld for invisible targets) | AI input, "no hidden unit leaks" check |
| Full summary | `telemetry.sampler_adapter.sample` | All units of both sides incl. hidden: positions, health, morale, visibility per side | Ground truth for checks, post-battle analysis. **Never an AI input** |

## adapter

`observe_side(ctx)`: `ctx = {side, alliance, own, enemies, memory, cco, now_ms, shooters}`
→ `{side, time_ms, own = {name → readings}, enemies = {name → intel record}, range = {"shooter>target" → readings}}`.

## services

- `check(view)` — self-check on every build; errors if an enemy record has a
  field other than `id, visibility, sampled_ms, current, last_seen,
  position_status`, if an invisible enemy has a current position, or if range
  to an invisible enemy is not withheld.
- `visibility_counts(view)` — how many enemies are visible / hidden.

## The rule for orders

A side's orders also stay within what it sees:

- it may attack only an enemy that is visible now;
- only units with missile weapons and ammunition may shoot;
- it may order only its own living units;
- one unit gets at most one movement order at a time.

This used to be checked by the `orders/contract.lua` module. It was removed with
our AI in the reset of 30.09.2026, and the remaining code has no such check: battle
decisions are now made by the game's AI and CA's planner, and the probes give
only orders set in advance. A future AI must
check its orders against this rule.

## Leak check

`python -m tools.analysis.unit_readout <run> --view side --side 1` compares the
side view with the full summary as ground truth: enemy record fields, current
position of hidden units, range to hidden units, "last seen" being a real
earlier sighting, visibility agreement. Result on 2026-09-27 (run `20260927-142521`):
**14 / 14 for each side**, no leaks. Mechanics measurements: [visibility](../game/units/visibility.md).
