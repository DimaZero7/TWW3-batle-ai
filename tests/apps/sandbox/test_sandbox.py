"""apps.sandbox.services: policy loading, restricted globals and budget."""
import pytest

from tests.lua_runtime import load, new_runtime

POLICY = """
return {
    api_version = 1,
    create = function(context) return {steps = 0} end,
    step = function(state, observation)
        return {{unit_id = 'lord', action = 'halt'}}
    end,
}
"""


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().sandbox = load(runtime, "apps.sandbox.services")
    return runtime


def run_policy(lua, source):
    return lua.eval("function(src) return sandbox.load(src, {}, 'empire-7-v1') end")(source)


class TestSandbox:
    def test_step_returns_copied_commands(self, lua):
        policy = run_policy(lua, POLICY)
        commands = policy.step(lua.table())
        assert commands[1].action == "halt"

    def test_globals_are_restricted(self, lua):
        source = "local x = os and os.time(); return {api_version=1, create=function() return {} end, step=function() return {} end}"
        with pytest.raises(Exception, match="attempt to index global 'os'"):
            run_policy(lua, source.replace("os and os.time()", "os.time()"))

    def test_instruction_budget(self, lua):
        source = "while true do end"
        with pytest.raises(Exception, match="policy instruction budget"):
            run_policy(lua, source)

    def test_bytecode_rejected(self, lua):
        with pytest.raises(Exception, match="source text required"):
            run_policy(lua, "\x1bLua")

    def test_copy_rejects_cycles_and_metatables(self, lua):
        lua.execute("""
            local t = {}; t.self = t
            assert(not pcall(sandbox.copy, t))
            assert(not pcall(sandbox.copy, setmetatable({}, {})))
            assert(not pcall(sandbox.copy, {x = 0/0}))
        """)
