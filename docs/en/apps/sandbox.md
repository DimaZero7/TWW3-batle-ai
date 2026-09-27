# sandbox

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/sandbox.md)

Loads API v1 policies into a restricted environment. Needed when running
someone else's policy code. Code: `src/apps/sandbox/services.lua`.

## Policy API v1

```lua
return {
    api_version = 1,
    create = function(context) return state end,
    step = function(state, observation) return commands end,   -- see orders.contract
    -- optional: deployment_version = 2, deployment_contract = 'deployment-placement-v2',
    --           deploy = function(state, context) return plan end
}
```

## Functions

| Function | What it does |
|---|---|
| `load(source, context, profile)` | Loads source text (not bytecode), returns `{deploy, step}` |
| `bounded(fn, ...)` | Call with a budget of 1,000,000 instructions |
| `copy(data)` | Deep copy of plain data: no metatables or cycles, depth < 24, ≤ 4096 keys per table |
| `validate(commands, own, enemies, profile, contract?)` | `orders.contract.validate` with the profile's command limit |

A policy sees only `pairs`, `ipairs`, `next`, `type`, `tonumber`,
`tostring`, `assert`, `error`, `select` and parts of `math`, `string`,
`table`. Input and output data are copied.

## Known limits

This is **not a hardened sandbox**:

- string methods stay reachable through the shared string metatable, e.g.
  `('x'):rep(n)`; a C function call counts as one instruction, so memory can
  be exhausted this way;
- `copy` does not limit reuse of one subtable (an acyclic graph) and runs
  outside the instruction budget.

This does not matter for our own policies. Close these gaps before running
third-party code.
