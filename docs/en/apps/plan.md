# plan

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/plan.md)

**Tree nodes:** [Deployment](../tree/deploy.md)

Start-of-battle plan: [assessment](assessment.md) → [strategy](strategy.md) →
[formation](formation.md). It only calls the modules in order and passes the
data along. `start(input, modules?)` returns `{status, features, decision,
formation}`; `status = 'no_strategy'` when nothing fits (no formation is
built). `modules` are replaced by spies in tests. The same chain runs in
battle (`entries.formation_probe`) and in the simulation (`tools/sim/formation.py`).
