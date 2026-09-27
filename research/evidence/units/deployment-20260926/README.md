# Deployment evidence — 26 September 2026

[English guide](../../en/deployment.md) · [Русское описание](../../ru/deployment.md)

Author/operator: battle-operator. Publication reviewer: coordinator. Five controlled diagnostics, synthetic public test policies only; no fighter submission or competitive strategy is included. WH3 v9.0.0 build 50218.4334952, official Moorlands Route, 15 Empire units per side, minimum graphics and Ultra. The final attempt completed 10 simulated combat seconds at x7. All five owned processes were closed. This evidence supports an engineering mechanism, not complete deployment geometry, campaign integration or competitive admission.

| Attempt | Result |
|---|---|
| 01 | Stopped before placement: spawned unit's `is_valid_for_deployment()` was false |
| 02 | Stopped before placement: immediate script-control flag was false after requesting control |
| 03 | All placements applied; an unverified equality between entity span and requested width blocked combat |
| 04 | Spatial checks passed; an invented 5° bearing-equality tolerance blocked combat |
| 05 | Both plans verified before combat, 30 controlled units, x7, diagnostic completed |

`acceptance.json` identifies the final pack/audit and exact accepted checks. `handoff.json` preserves outcomes, measurements, known limits and timing for every attempt. `sources.json` records API references. `attempt-*/build/` retains exact manifest-listed source, synthetic policies, scenario, tooling and tests; manifests include the original pack hash. Packs are not installed by reading this archive. `attempt-*/run/` contains JSONL compressed losslessly with gzip, native XML exports and lifecycle/timing records. Failures are retained rather than rewritten. `checks/` holds successful final preparation outputs: 20 Python/Lua tests plus the PowerShell incremental-reader check. `hashes.json` covers the archived artifacts (this explanatory index is outside that inventory).

The final raw audit is safe to publish here because it contains only these explicit synthetic fixtures. **This is not permission to publish combined fighter logs or deployment plans.** Actual competition logs remain private per the information contract; each fighter receives only its own permitted view.

The UI/CCO positions from the second army are diagnostic readouts with unverified freshness. Native reference/reservation containment and same-side bounding-box distances were checked; full soldier bounds were not. The ordered parameters and resulting formation readouts must remain separate.

To inspect JSONL, use Python `gzip.open(path, 'rt', encoding='utf-8')`. Verify SHA-256 against `hashes.json` before analysis. Re-running a historical attempt requires a new explicit diagnostic assignment and a new output identity; do not overwrite these files or reuse an exhausted launch budget.
