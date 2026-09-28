# battlefield

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/battlefield.md)

The battlefield between the two main groups. Pure: it **only computes**; the
next task (aligning the army) uses it. Code: `src/apps/battlefield/`.

The axis runs from the centre of our main group to the centre of theirs
([vision](vision.md)), so the picture is symmetric about it. Along the axis:
our back and front lines, the gap, their front and back lines. The field is the
rectangle from our back line to theirs, as wide as the wider reach either side
plus a 40 m manoeuvre margin (user's decision; 200 m was too much).

| Function | What it does |
|---|---|
| `frame({own, enemy}, params?)` | The field; `own`/`enemy` = `{points, centre?}` of each main group |
| `to_frame(field, p)` / `to_world(field, along, across)` | Map point ↔ battlefield frame |
| `middle(points)` | Middle of points |

Output: `status` (`ok` / `contact`), `origin`, `bearing`, `centres_m`, `gap_m`,
`own`/`enemy` extents, `half_width_m`, `margin_m`, `corners`. Checked by tests
(straight and diagonal armies, contact, six real game AI layouts), in the
simulation and in battle ([results](../../../research/analysis/battlefield/README.md), in Russian).

Watch it in motion: [battle viewer](../launch/viewer.md), layer “battlefield” (preset `modules`).
