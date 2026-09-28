# Architecture

[← Back](../README.md) · [Documentation](../README.md) › Architecture · [Русский](../../ru/architecture/README.md)

Project structure and rules. The AI's working drafts are in Russian only.

<!-- generated:docs:index -->
- [List of apps](apps.md) — The code was moved from the `lua-knowledge-kit` research set
- [How the documentation is built and kept](documentation.md) — Rules (user's decision 28.09.2026), checked by `tests/docs/`
- [Errors and unknown values](error_handling.md) — In battle, a script error must never silently turn into vanilla AI behaviour
- [Project structure and layers](overview.md) — The project follows the photo-fixing layout: code is split into **apps** by domain, logic lives in **services**, and entry points only wire them together
<!-- /generated -->
