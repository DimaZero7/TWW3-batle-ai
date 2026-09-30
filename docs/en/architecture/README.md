# Architecture

[← Back](../README.md) · [Documentation](../README.md) › Architecture · [Русский](../../ru/architecture/README.md)

Project structure and rules: apps, layers and dependencies, errors, documentation.

<!-- generated:docs:index -->
- [List of apps](apps.md) — The base code: apps in the game, entry points and tools outside the game
- [How the documentation is built and kept](documentation.md) — Rules (user's decision 28.09.2026), checked by `tests/docs/`
- [Errors and unknown values](error_handling.md) — In battle, a script error must never pass silently: the battle would go on without our script and the record or measurement would be wrong
- [Project structure and layers](overview.md) — Code is split into **apps** by domain, logic lives in **services**, and entry points only wire them together
<!-- /generated -->
