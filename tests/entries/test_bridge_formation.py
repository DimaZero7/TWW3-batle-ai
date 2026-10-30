"""The formation's orders through the real bridge (src/apps/bridge/adapter.lua) and the fake engine: a hold, a move or
a withdraw with a facing and / or a width (the executor's formation heads, tools/nn/model/chain.py cfg.form) becomes one
goto_location_angle_width - at the unit's place for a hold, at the point for a move; the part not given from the unit
(its bearing / the way to its point, its ordered width); re-given unchanged it is not given again."""
from tests.entries.test_bridge_abilities import SETUP, SYG, FS, events, lua  # noqa: F401
from tools.nn.companion import exchange


def start(lua, tmp_path):
    lua.globals().SYG, lua.globals().FS = SYG, FS
    lua.execute(SETUP + """
        STATE = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
        bm:pump()
    """)


def answer(lua, tmp_path, orders):
    doc = exchange.read_state(tmp_path / exchange.STATE)
    exchange.write_atomic(tmp_path / exchange.ORDERS, exchange.orders_text(doc["batch"], doc["move"], orders))
    lua.execute("bm:tick(100)")


def args(lua, i=2):
    a = lua.eval(f"own[{i}].formation_args")
    return None if a is None else {k: a[k] for k in ("x", "z", "angle", "width", "run")}


def test_a_hold_with_a_facing_turns_in_place_and_a_move_with_a_width_arrives_formed(lua, tmp_path):
    start(lua, tmp_path)
    answer(lua, tmp_path, [{"unit": "own_spear_1", "kind": "hold", "face": 90.0, "width": 40.0}])
    a = args(lua)
    assert a == {"x": -175, "z": 0, "angle": 92.8125, "width": 40.0, "run": False}   # the engine's 64-sector facing
    assert "halt" not in list(lua.eval("bm.orders").values())
    lua.execute("own[2].formation_args = nil")
    lua.execute("for _ = 1, 9 do bm:tick(100) end")
    answer(lua, tmp_path, [{"unit": "own_spear_1", "kind": "hold", "face": 90.0, "width": 40.0}])
    assert args(lua) is None                                              # the same order: not given again
    lua.execute("for _ = 1, 9 do bm:tick(100) end")
    answer(lua, tmp_path, [{"unit": "own_spear_1", "kind": "move", "x": -175.0, "z": 100.0, "run": True,
                            "width": 60.0}])
    a = args(lua)
    # facing: the way to the point (+z: bearing 0, the sector's middle 2.81), the width ordered, at a run
    assert a["x"] == -175 and a["z"] == 100 and abs(a["angle"] - 2.8125) < 1e-6 and a["width"] == 60.0 and a["run"]
    lua.execute("for _ = 1, 20 do bm:tick(100) end")
    rows = events(tmp_path / "tww3_bai_events.jsonl")
    assert "error" not in [r["event"] for r in rows]
    formed = [(r["k"], r.get("face"), r.get("width")) for r in rows if r["event"] == "nn_formation"]
    assert formed == [("hold", 90, 40), ("move", None, 60)]


def test_a_hold_without_a_formation_still_halts(lua, tmp_path):
    start(lua, tmp_path)
    answer(lua, tmp_path, [{"unit": "own_spear_1", "kind": "hold"}])
    assert args(lua) is None and "halt" in list(lua.eval("bm.orders").values())
