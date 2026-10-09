"""Lua runtime for tests: see tools/lua_runtime.py. Without lupa (the training container) every test module that
imports this one is skipped."""
import pytest

pytest.importorskip("lupa", reason="the Lua runtime (lupa): on the host, not in the training container")

from tools.lua_runtime import ROOT, SRC, load, new_runtime  # noqa: E402, F401
