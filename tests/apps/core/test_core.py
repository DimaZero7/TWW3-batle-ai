"""apps.core: value reads, JSON encoding, error codes and guards."""
import json

import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    g = runtime.globals()
    g.value = load(runtime, "apps.core.value")
    g.json = load(runtime, "apps.core.json")
    g.errors = load(runtime, "apps.core.errors")
    return runtime


class TestValue:
    def test_read_keeps_zero_and_false_known(self, lua):
        lua.execute("""
            assert(value.read(function() return 0 end, 'number').value == 0)
            local r = value.read(function() return false end, 'boolean')
            assert(r.status == 'known' and r.value == false)
        """)

    def test_exception_nil_type_and_nonfinite_are_unknown(self, lua):
        lua.execute("""
            assert(value.read(function() error('x') end, 'number').reason == 'read_error')
            assert(value.read(function() return nil end, 'number').reason == 'nil')
            assert(value.read(function() return 'a' end, 'number').reason == 'type_string')
            assert(value.read(function() return 'a' end, 'number', 'invalid_type').reason == 'invalid_type')
            assert(value.read(function() return 0/0 end, 'number').reason == 'nonfinite')
            assert(value.read(function() return math.huge end, 'number').reason == 'nonfinite')
        """)

    def test_vector_copies_plain_table(self, lua):
        lua.execute("""
            local p = {get_x=function() return 1 end, get_y=function() return 2 end, get_z=function() return 3 end}
            local r = value.vector(function() return p end)
            assert(r.value.x == 1 and r.value.z == 3 and r.value.get_x == nil)
            assert(value.vector(function() return {} end).reason == 'invalid_vector')
        """)


class TestJson:
    def test_flat_row_is_sorted_and_escaped(self, lua):
        text = lua.eval("json.encode({b = 1, a = 'x\"y\\n', c = true})")
        assert text == '{"a":"x\\"y\\n","b":1,"c":true}'
        assert json.loads(text) == {"a": 'x"y\n', "b": 1, "c": True}

    def test_nested_arrays_objects_and_nonfinite(self, lua):
        text = lua.eval("json.encode({list = {1, 2, {k = 0/0}}, empty = {}})")
        assert json.loads(text) == {"empty": {}, "list": [1, 2, {"k": None}]}


class TestErrors:
    def test_raise_carries_code(self, lua):
        lua.execute("""
            local ok, message = pcall(errors.raise, 'BAD_INPUT', 'no unit')
            assert(not ok and errors.code(message) == 'BAD_INPUT')
            assert(message:find('no unit', 1, true))
        """)

    def test_guard_reports_once_and_stops(self, lua):
        lua.execute("""
            local reports, stopped = {}, false
            local f = errors.guard(function() error('boom') end,
                function(e) reports[#reports + 1] = e; stopped = true end,
                function() return stopped end)
            f(); f()
            assert(#reports == 1 and reports[1]:find('boom', 1, true))
        """)
