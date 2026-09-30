"""apps.bridge: the orders file the companion writes, as the game reads it (levels 1 and 2)."""
import pytest

from tests.lua_runtime import load, new_runtime
from tools.nn.companion import exchange


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().bridge = load(runtime, "apps.bridge.services")
    return runtime


def parse(lua, text):
    doc, reason = lua.eval("function(t) local d, r = bridge.parse_orders(t); return d, r end")(text)
    return doc, reason


def test_the_companions_file_is_read_whole(lua):
    orders = [{"unit": "own_lord", "kind": "move", "x": -12.5, "z": 40.0, "run": True},
              {"unit": "own_spear_1", "kind": "attack", "target": "enemy_spear_2", "run": False},
              {"unit": "own_spear_2", "kind": "withdraw", "x": 1.0, "z": -2.0, "run": True},
              {"unit": "own_archer_1", "kind": "hold"}, {"unit": "own_archer_2", "kind": "keep"}]
    doc, _ = parse(lua, exchange.orders_text("b-1", 7, orders, 12.25))
    assert doc.move == 7 and doc.batch == "b-1" and doc.think_ms == 12.2 and doc.count == 5
    o = doc.orders
    assert (o.own_lord.kind, o.own_lord.x, o.own_lord.z, o.own_lord.run) == ("move", -12.5, 40.0, True)
    assert (o.own_spear_1.kind, o.own_spear_1.target, o.own_spear_1.run) == ("attack", "enemy_spear_2", False)
    assert (o.own_spear_2.kind, o.own_spear_2.run) == ("withdraw", True)
    assert o.own_archer_1.kind == "hold" and o.own_archer_2.kind == "keep"


@pytest.mark.parametrize("text, reason", [
    ("", "empty"),
    ("something else\nend\n", "unknown format"),
    (exchange.FORMAT + "\nmove 3\nbatch b\nunit own_lord hold\n", "incomplete"),
    (exchange.FORMAT + "\nmove 3\nbatch b\nunit own_lord dance\nend\n", "bad kind: dance"),
    (exchange.FORMAT + "\nmove 3\nbatch b\nunit own_lord move 1 nan 1\nend\n", "bad point for own_lord"),
    (exchange.FORMAT + "\nbatch b\nend\n", "no move or batch"),
])
def test_a_broken_or_half_written_file_is_not_taken(lua, text, reason):
    doc, why = parse(lua, text)
    assert doc is None and why == reason


def test_an_order_is_given_again_only_when_it_changed(lua):
    changed = lua.eval("function(a, b) return bridge.changed(a, b) end")
    t = lua.table_from
    move = t({"kind": "move", "x": 0, "z": 0, "run": True})
    assert changed(None, move)
    assert not changed(move, t({"kind": "move", "x": 3, "z": 3, "run": True}))     # 4.2 m: the same
    assert changed(move, t({"kind": "move", "x": 6, "z": 0, "run": True}))
    assert changed(move, t({"kind": "move", "x": 0, "z": 0, "run": False}))
    attack = t({"kind": "attack", "target": "enemy_1", "run": True})
    assert not changed(attack, t({"kind": "attack", "target": "enemy_1", "run": True}))
    assert changed(attack, t({"kind": "attack", "target": "enemy_2", "run": True}))
    assert changed(attack, t({"kind": "hold", "run": False}))


def test_real_time_from_os_clock(lua):
    real_ms = lua.eval("function(a, b) return bridge.real_ms(a, b) end")
    assert real_ms(1.5, 1.5234) == 23 and real_ms(None, 2) is None
