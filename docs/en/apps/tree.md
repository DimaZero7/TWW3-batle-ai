# tree

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/tree.md)

The AI's tree as data: the trunk of phases in order, branches that improve a phase, and for every
branch its **baseline** (what happens without it). A node switched off switches off its subtree and
every later phase; the parent does its baseline. Switches: the army's `tree` field
(`{"align": false, "logistics": true}`); unknown names are an error; the first phase cannot be off.
Nodes: `deploy` (phase 1) with `map_fit`, `formation_window`; `approach` (phase 2) with `align`,
`logistics` (off by default), `safe_detour`, `under_fire_stop`. Code: `src/apps/tree/`.
