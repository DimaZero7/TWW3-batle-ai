# Moorlands Route: terrain-only handoff

Captured 2026-09-26 with the existing map reader and native objects module. No fighter policy was loaded and no battle between fighters was run. This directory is the shareable export; `../capture/` holds operator source evidence and diagnostic-unit observations and is not the fighter export.

- `summary.json`: validation, identity, counts, limits and original source hashes.
- `terrain-grid.csv`: 116,964 unique surface records, without unit reachability columns.
- `height`, `ground-code`, `forest`, `water-ground`, `clear`, `inside`: matrices in CSV and NPY. All use `[iz,ix]`, south-to-north rows, west-to-east columns. Step 3 m; original grid origin (-512,-512).
- `ground-legend.json`: native ground string codes. Forest=1 means a forest ground sample. Water-ground=1 means shallow_water or deep_water, not measured depth.
- 116,281 centres are inside the radar frame; 683 last-row/column centres are outside. Their classified values are -1 and heights NaN in the matrices. The original CSV retains their explicit inside_radar=0 samples. Do not count these as known inside-frame terrain.
- `objects.json`: 904 native object records, 887 origins inside the frame. No unit state, ownership or object health is exported. Origins are not collision footprints or a complete tree list.
- `requested-terrain.xml`: exact requested resource/catchment_03, no requested tile upgrade. `loaded-terrain.xml`: actual exported resource, resolved tile position (0.453125,0.40625) and engine-added default upgrades. Engine export does not retain the catchment label; do not invent an independent resource-level catchment validation.
- `timing.json`: surface capture 0.87599945068359 s by Lua os.clock; diagnostic reachability 2.3789978027344 s, not included in shared terrain matrices; complete successful launch/capture 35.5434067 s. These are one-run timings, not universal performance guarantees.
- `cleanup.json`: owned PID 9788 closed; saved outputs verified; installed capture files removed. No remaining WH3 process found in the final check.
- `overview.png`: inspected height/ground/object visualization. Colour never proves passability.

Inside-frame height range: 516.490967–624.861023 m. Ground counts: grass 79,379; forest 30,225; mud 3,687; sharp_stones 2,990. No shallow_water/deep_water centre samples; this does not exclude small unsampled water features. Radar bounds do not prove movement walls. No policies or hidden combat information are included.

The first attempt crashed before producing a grid. The same scenario without UTF-8 BOM completed; this suggests a serialization issue, not a proven crash root cause. Its evidence remains in `../failed-bom-attempt/`.

Source hashes and file hashes are in summary.json and hashes.json. The coordinator gates publication. Both fighters must review the shared grid before the coordinator authorizes integration.
