"""Lua 5.1 runtime (tests, simulation) where `require('apps.x.y')` loads src/apps/x/y.lua.

Lua's own file loader cannot open non-ASCII Windows paths, so sources are
read by Python and handed to a loader registered in package.loaders.
"""
from pathlib import Path

from lupa.lua51 import LuaRuntime

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _read_module(name):
    path = SRC.joinpath(*name.split(".")).with_suffix(".lua")
    return path.read_text(encoding="utf-8") if path.is_file() else None


def new_runtime():
    lua = LuaRuntime(unpack_returned_tuples=True)
    install = lua.eval("""
        function(read)
            table.insert(package.loaders, 2, function(name)
                local source = read(name)
                if not source then return '\\n\\tno module ' .. name .. ' in src/' end
                return assert(loadstring(source, '@src/' .. name:gsub('%.', '/') .. '.lua'))
            end)
        end
    """)
    install(_read_module)
    return lua


def load(lua, module):
    return lua.eval(f"require('{module}')")
