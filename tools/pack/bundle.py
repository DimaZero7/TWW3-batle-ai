"""Bundles src/ Lua modules into one battle script.

Modules use plain `require('apps.core.json')`. The bundle registers every
module reachable from the entry as a factory and passes its own loader as
the local `require`, so the game's package system is not involved.
"""
import json
import re
from pathlib import Path

REQUIRE = re.compile(r"""\brequire\s*\(\s*['"]([A-Za-z0-9_.]+)['"]\s*\)""")

LOADER = """\
local __modules, __loaded = {}, {}
local function __require(name)
    local loaded = __loaded[name]
    if loaded ~= nil then return loaded end
    local factory = __modules[name]
    if factory == nil then error('module not bundled: ' .. tostring(name), 2) end
    local result = factory(__require)
    if result == nil then result = true end
    __loaded[name] = result
    return result
end
"""


def module_path(src, name):
    return Path(src, *name.split(".")).with_suffix(".lua")


def collect(src, entry):
    """Returns [(name, source)] in dependency-first order."""
    ordered, state = [], {}

    def visit(name, chain):
        if state.get(name) == "done":
            return
        if state.get(name) == "active":
            raise ValueError("circular require: " + " -> ".join(chain + [name]))
        path = module_path(src, name)
        if not path.is_file():
            raise FileNotFoundError(f"module {name} not found at {path}")
        state[name] = "active"
        source = path.read_text(encoding="utf-8")
        for dependency in REQUIRE.findall(source):
            visit(dependency, chain + [name])
        state[name] = "done"
        ordered.append((name, source))

    visit(entry, [])
    return ordered


def lua_literal(value):
    """Python config value -> Lua literal (dict/list/str/number/bool)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, dict):
        parts = [f"[{json.dumps(str(k))}] = {lua_literal(v)}" for k, v in sorted(value.items())]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "{" + ", ".join(lua_literal(v) for v in value) + "}"
    raise TypeError(f"unsupported config value: {value!r}")


def bundle(src, entry, config, header=""):
    """Returns (script text, [module names])."""
    modules = collect(src, entry)
    parts = [header, LOADER]
    for name, source in modules:
        parts.append(f"__modules[{json.dumps(name)}] = function(require)\n{source}\nend\n")
    parts.append(
        "if bm then\n"
        f"    __require({json.dumps(entry)}).main(bm, {lua_literal(config)},\n"
        "        {common = common, battle_vector = battle_vector})\n"
        "end\n")
    return "".join(parts), [name for name, _ in modules]
