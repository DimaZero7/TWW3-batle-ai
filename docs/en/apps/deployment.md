# deployment

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/deployment.md)

Initial placement under the `deployment-placement-v2` contract.
Code: `src/apps/deployment/`. Measurements: [deployment](../game/units/deployment.md).

## Plan

An array `{unit_id, x, z, facing_deg, width_m}` — exactly one entry per own
unit. Width is never turned into a "reservation circle": no footprint is
inferred from it.

## contract

| Function | What it does |
|---|---|
| `zone(z)` | Validates the XML zone: centre, orthonormal `axis_u`/`axis_v`, half sizes |
| `inside(p, zone, tolerance?)` | Point inside the rectangular zone |
| `context(roster, zone, copy, 'deployment-placement-v2')` | Context for a policy: roster, zone, verification limits |
| `validate(plan, context)` | Complete plan, own units without duplicates, finite numbers, `facing_deg` 0..360, width, point in zone |

## services — the transaction

1. `prepare(policies, contexts, get_phase, copy)` — both sides return plans;
   both are validated **before** anything is applied.
2. `apply(plans, get_phase, place)` — placement; every step checks the phase
   is still `Deployment`.
3. `verify(plan, context, measure, pair_distance, get_phase, previous?)` —
   checks one engine sample. Returns `{valid, stable, errors, units, pairs}`.
   Error codes: `ordered_anchor_mismatch` (> 0.1 m), `native_reference_outside_zone`,
   `native_position_unstable` (> 0.25 m between samples), `native_contact_unresolved`
   (pairs closer than 1 m) and more. Re-sample every second until two stable
   samples in a row or 10 s.

## adapter

Callbacks for `services` built from verified calls: `place_callback`
(teleport), `measure_callback` (position, ordered position, bearing, width),
`pair_distance_callback` (`unit_distance`).

> The adapter is built from the calls used in the deployment measurements
> but has not been run in game within this project yet.

The former v1 contract with reservation circles is only in the
[archive](../../../research/scripts/legacy-lua/runtime/deployment.lua).
