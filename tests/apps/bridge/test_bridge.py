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


def test_a_shooter_that_stands_idle_under_an_attack_order_is_freed_and_later_takes_its_target(lua):
    duty = lua.eval("function(d, me, tg) return bridge.missile_duty(d, me, tg) end")
    t = lua.table_from
    release, free_min = lua.eval("bridge.RELEASE_AFTER"), lua.eval("bridge.FREE_MIN")
    idle = t({"men": 90, "a": 1800, "fire": False, "mv": False, "m": False})
    firing = t({"men": 90, "a": 1800, "fire": True})
    in_melee = t({"men": 100, "m": True})
    free = t({"men": 100, "m": False})
    d = t({})
    assert [duty(d, idle, in_melee) for _ in range(release - 1)] == [None] * (release - 1)
    assert duty(d, firing, in_melee) is None and d.idle == 0         # shooting: the count starts again
    for _ in range(release - 1):
        duty(d, idle, in_melee)
    assert duty(d, idle, in_melee) == "release" and d.free
    # Free: the target in melee keeps it free; out of melee it is taken again, after FREE_MIN decisions.
    assert [duty(d, firing, in_melee) for _ in range(free_min + 2)] == [None] * (free_min + 2)
    assert duty(d, firing, free) == "resume" and not d.free
    d = t({})
    for _ in range(free_min):
        assert duty(d, t({"men": 90, "a": 1800, "mv": True}), free) is None   # walking to range is not idle
        assert duty(d, t({"men": 90, "a": 0}), free) is None                  # no ammunition: not a shooter
        assert duty(d, t({"men": 90, "a": 1800, "r": True}), free) is None    # routing
    assert not d.free


def test_a_shooter_freed_by_the_bridge_does_not_resume_on_a_broken_target(lua):
    duty = lua.eval("function(d, me, tg) return bridge.missile_duty(d, me, tg) end")
    t = lua.table_from
    d = t({"free": True, "free_for": 100})
    me = t({"men": 90, "a": 10})
    assert duty(d, me, t({"men": 0})) is None
    assert duty(d, me, t({"men": 50, "r": True})) is None
    assert duty(d, me, None) is None and d.free


def test_an_attack_on_a_target_in_melee_within_range_is_fire_at_will(lua):
    freely = lua.eval("function(me, tg, r) return bridge.fire_freely(me, tg, r) end")
    t = lua.table_from
    me = t({"men": 90, "x": 0, "z": 0})
    assert freely(me, t({"men": 100, "m": True, "x": 100, "z": 50}), 150)          # 112 m
    assert not freely(me, t({"men": 100, "m": True, "x": 200, "z": 0}), 150)       # out of range: walk up
    assert not freely(me, t({"men": 100, "m": False, "x": 100, "z": 0}), 150)      # a free target: explicit
    assert not freely(me, t({"men": 100, "m": True, "x": 100, "z": 0}), None)      # range unknown
    assert not freely(me, None, 150)


def test_the_idle_count_goes_on_across_the_networks_new_targets(lua):
    # The adapter keeps the duty table across attack orders; missile_duty never restarts it itself.
    duty = lua.eval("function(d, me, tg) return bridge.missile_duty(d, me, tg) end")
    t = lua.table_from
    idle = t({"men": 90, "a": 1800})
    d = t({})
    out = [duty(d, idle, t({"men": 100, "m": i % 2 == 0})) for i in range(lua.eval("bridge.RELEASE_AFTER"))]
    assert out[-1] == "release"
