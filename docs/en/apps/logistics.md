# logistics — queues past an obstacle

[← Back](README.md) · [Documentation](../README.md) · [Русский](../../ru/apps/logistics.md)

**Tree nodes:** [Logistics past an obstacle](../tree/logistics.md)

[Russian version](../../ru/apps/logistics.md) (full rules and results)

`src/apps/logistics/services.lua`, pure. When an approach step walks round an
obstacle, it plans who goes round (the obstacle grown by 5 m touches the unit's
strip along the movement), which side (a closed side sends everyone round the
other), in what order (front row first, one lane is a queue without overtaking,
no crossing between sides) and width (own; narrower only when wider than the
gap, reformed in place before walking), and when each unit sets off (when the
one ahead has walked as far as planned). Identical units in one lane swap
places so the first out takes the farthest place. The group keeps one command;
each unit gets at most four orders. Switched on by `"logistics": true` in the
army, off with `--engine-only`; the lowest branch, nothing below it.

In battle (5 terrains, 13–16 units, Empire and Skaven): peak crowding 1.5–7
times lower than the engine alone, the detour 2–22% longer
([analysis, Russian](../../../research/analysis/logistics/README.md)).
