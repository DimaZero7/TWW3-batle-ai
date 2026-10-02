"""The bridge's abilities (src/apps/bridge): the orders file's ability lines, using an ability in the
fake battle (tests/entries/fake_battle.lua), the uses and active effects in the state document."""
import json
from pathlib import Path

import pytest

from tests.lua_runtime import new_runtime

FAKE = (Path(__file__).parent / "fake_battle.lua").read_text(encoding="utf-8")
SYG, FS = "wh_main_character_abilities_stand_your_ground", "wh_main_character_abilities_foe_seeker"


@pytest.fixture
def lua(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = new_runtime()
    runtime.globals().fake = runtime.execute(FAKE)
    return runtime


def events(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_ability_lines_are_parsed_and_bad_ones_refused(lua):
    lua.execute("S = require('apps.bridge.services')")
    doc = lua.eval(f"S.parse_orders('tww3_bai_nn_orders 1\\nmove 2\\nbatch b\\nunit own_lord keep\\n"
                   f"ability own_lord {SYG}\\nend\\n')")
    assert doc.abilities[1].unit == "own_lord" and doc.abilities[1].key == SYG and doc.count == 1
    doc, reason = lua.eval("S.parse_orders('tww3_bai_nn_orders 1\\nmove 2\\nbatch b\\nability own_lord\\nend\\n')")
    assert doc is None and "ability" in reason
    plain = lua.eval("S.parse_orders('tww3_bai_nn_orders 1\\nmove 2\\nbatch b\\nunit own_lord hold\\nend\\n')")
    assert len(list(plain.abilities.values())) == 0


def test_active_effects_are_read_from_the_card(lua):
    lua.execute("""
        S = require('apps.bridge.services')
        CARD = {['ActiveEffectList.Size'] = 2, ['ActiveEffectList.At(0).PhaseRecordContext.Key'] = 'a',
                ['ActiveEffectList.At(1).PhaseRecordContext.Key'] = 'b'}
    """)
    assert list(lua.eval("S.active_effects(function(f) return CARD[f] end)").values()) == ["a", "b"]
    assert lua.eval("S.active_effects(function(f) return nil end)") is None
    assert len(list(lua.eval("S.active_effects(function(f) return f == 'ActiveEffectList.Size' and 0 or nil end)")
                    .values())) == 0


SETUP = """
    own = {fake.unit('own_lord', 'lord', -235, 0), fake.unit('own_spear_1', 'spears', -175, 0)}
    enemy = {fake.unit('enemy_lord', 'lord', 235, 0), fake.unit('enemy_spear_1', 'spears', 175, 0)}
    own[1].abilities = {[SYG] = true, [FS] = false}
    bm = fake.manager({own, enemy})
    local function slots(side)
        return {{script_name = side .. '_lord', slot = 'lord', key = 'lord'},
            {script_name = side .. '_spear_1', slot = 'spear_1', key = 'spears'}}
    end
    CONFIG = {build = 'test', speed = 20, tick_ms = 1000, deadline_s = 600, stall_ms = 600000,
        timeout_ms = 600000, defend_radius_m = 150, units = {own = slots('own'), enemy = slots('enemy')},
        own_ai = 'net', enemy_role = 'attack', decide_ms = 1000, poll_ms = 100,
        factions = {own = 'wh_main_emp_empire', enemy = 'wh2_main_skv_skaven'}}
    GLOBALS = {common = fake.common, battle_vector = fake.vector_type}
    fake.cco['uid_enemy_lord'] = {['ActiveEffectList.Size'] = 1,
        ['ActiveEffectList.At(0).PhaseRecordContext.Key'] = 'wh_main_character_abilities_deadly_onslaught'}
"""


def test_the_lord_uses_an_ability_once_and_the_state_tells_when(lua, tmp_path):
    from tools.nn.companion import exchange
    lua.globals().SYG, lua.globals().FS = SYG, FS
    lua.execute(SETUP + """
        STATE = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
        bm:pump()
    """)
    doc = exchange.read_state(tmp_path / exchange.STATE)
    fx = {u["n"]: u.get("fx") for u in doc["units"]}
    assert fx["enemy_lord"] == ["wh_main_character_abilities_deadly_onslaught"] and fx["own_lord"] is None
    assert doc["abilities_used"] == {}
    keep = [{"unit": "own_lord", "kind": "keep"}, {"unit": "own_spear_1", "kind": "keep"}]
    uses = [{"unit": "own_lord", "key": SYG}, {"unit": "own_lord", "key": FS}, {"unit": "enemy_lord", "key": SYG}]
    exchange.write_atomic(tmp_path / exchange.ORDERS, exchange.orders_text(doc["batch"], 1, keep, None, uses))
    lua.execute("bm:tick(100)")
    assert list(lua.eval("bm.orders").values()) == [f"ability own_lord {SYG} on own_lord"]
    lua.execute("for _ = 1, 9 do bm:tick(100) end")             # the next decision: move 2
    doc = exchange.read_state(tmp_path / exchange.STATE)
    assert doc["move"] == 2 and doc["abilities_used"] == {"own_lord": {SYG: 100}}
    exchange.write_atomic(tmp_path / exchange.ORDERS, exchange.orders_text(doc["batch"], 2, keep, None, uses[:1]))
    lua.execute("""
        bm:tick(100)                                            -- on recharge now: refused
        own[1].routing = true
        own[1].abilities[FS] = true
    """)
    for _ in range(10):
        lua.execute("bm:tick(100)")
    doc = exchange.read_state(tmp_path / exchange.STATE)
    exchange.write_atomic(tmp_path / exchange.ORDERS,
                          exchange.orders_text(doc["batch"], doc["move"], keep, None, uses[1:2]))
    lua.execute("""
        bm:tick(100)                                            -- a routing lord uses nothing
        bm.outcome, bm.winner = true, 2
        for _ = 1, 10 do bm:tick(100) end
        assert(STATE.finished, 'the battle did not finish')
    """)
    assert list(lua.eval("bm.orders").values()) == [f"ability own_lord {SYG} on own_lord"]
    rows = events(tmp_path / "tww3_bai_events.jsonl")
    assert "error" not in [r["event"] for r in rows]
    used = [(r["move"], r["u"], r["key"], r["status"]) for r in rows if r["event"] == "nn_ability"]
    assert used == [(1, "own_lord", SYG, "used"), (1, "own_lord", FS, "not_ready"),
                    (1, "enemy_lord", SYG, "unknown_unit"), (2, "own_lord", SYG, "not_ready"),
                    (doc["move"], "own_lord", FS, "down")]
    result = rows[-1]
    assert result["nn_abilities_used"] == 1 and result["nn_abilities_refused"] == 4
    ready = [(r["u"], r["key"], r["ready"]) for r in rows if r["event"] == "nn_ability_ready"]
    assert ready == [("own_lord", FS, False), ("own_lord", SYG, True), ("own_lord", SYG, False),
                     ("own_lord", FS, True)]                     # changes only, once per decision
    fx = [(r["u"], r["fx"]) for r in rows if r["event"] == "nn_effects"]
    assert ("enemy_lord", ["wh_main_character_abilities_deadly_onslaught"]) in fx
    assert len([u for u, _ in fx if u == "enemy_lord"]) == 1


def test_a_held_shooter_is_aimed_at_the_nearest_enemy_in_range(lua, tmp_path):
    from tools.nn.companion import exchange
    lua.globals().SYG, lua.globals().FS = SYG, FS
    lua.execute(SETUP + """
        own[2].ammo, own[2].range = 1000, 120
        enemy[2].pos = fake.vector_type.new()
        enemy[2].pos:set_x(-60)
        STATE = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
        bm:pump()
    """)
    doc = exchange.read_state(tmp_path / exchange.STATE)
    hold = [{"unit": "own_lord", "kind": "keep"}, {"unit": "own_spear_1", "kind": "hold"}]
    exchange.write_atomic(tmp_path / exchange.ORDERS, exchange.orders_text(doc["batch"], 1, hold))
    lua.execute("bm:tick(100)")
    assert list(lua.eval("bm.orders").values())[-1] == "halt"
    lua.execute("for _ = 1, 9 do bm:tick(100) end")             # the next decision aims it
    assert list(lua.eval("bm.orders").values())[-1] == "attack enemy_spear_1"
    assert lua.eval("own[2].attack_args.run") is False
    lua.execute("""
        bm.outcome, bm.winner = true, 2
        for _ = 1, 10 do bm:tick(100) end
    """)
    rows = events(tmp_path / "tww3_bai_events.jsonl")
    assert [(r["u"], r["action"], r["tg"]) for r in rows if r["event"] == "nn_hold"] == [
        ("own_spear_1", "aim", "enemy_spear_1")]
    assert rows[-1]["nn_hold_aims"] == 1 and "error" not in [r["event"] for r in rows]


def test_an_attack_the_engine_dropped_is_given_again(lua, tmp_path):
    from tools.nn.companion import exchange
    lua.globals().SYG, lua.globals().FS = SYG, FS
    lua.execute(SETUP + """
        STATE = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
        bm:pump()
    """)
    doc = exchange.read_state(tmp_path / exchange.STATE)
    attack = [{"unit": "own_lord", "kind": "attack", "target": "enemy_spear_1", "run": True},
              {"unit": "own_spear_1", "kind": "keep"}]
    exchange.write_atomic(tmp_path / exchange.ORDERS, exchange.orders_text(doc["batch"], 1, attack))
    lua.execute("bm:tick(100)")
    given = len(list(lua.eval("bm.orders").values()))
    after = lua.eval("require('apps.bridge.services').STALL_AFTER")
    lua.execute(f"for _ = 1, {10 * after} do bm:tick(100) end")    # walking to it: nothing new
    assert len(list(lua.eval("bm.orders").values())) == given
    lua.execute("own[1].moving = false")                            # stands: the engine dropped the order
    lua.execute(f"for _ = 1, {10 * after} do bm:tick(100) end")
    orders = list(lua.eval("bm.orders").values())
    assert len(orders) == given + 1 and orders[-1] == orders[given - 1]
    lua.execute("""
        bm.outcome, bm.winner = true, 2
        for _ = 1, 10 do bm:tick(100) end
    """)
    rows = events(tmp_path / "tww3_bai_events.jsonl")
    assert [(r["u"], r["k"], r["tg"]) for r in rows if r["event"] == "nn_stall"] == [
        ("own_lord", "attack", "enemy_spear_1")]
    assert rows[-1]["nn_stalls"] == 1 and "error" not in [r["event"] for r in rows]
