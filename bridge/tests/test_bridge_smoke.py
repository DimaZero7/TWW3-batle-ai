"""The bridge's smoke test (no game):
1. every entry of bridge/lua bundles and compiles (Lua 5.1, lupa); the chain's battle pack builds from this tree
   (bridge.build nn-arena --army-seed .. --armies v3 --map ksl_plains_01 --deploy strategist --own-ai net, into a temp
   folder) and reads back - the script, the battle file, the required mod;
2. the companion's chain (the operative's and the tacticians' formulas, the unit network - its weights when present,
   else a fresh one) answers a real state document (data/state_m1.json: the base M1 chain battle in the game,
   tmp/clean_bridge_fixture.py) with a complete orders file - every own unit that takes orders a line, 'mode chain',
   'end' - that parses back in Python and in the bridge's own Lua parser (services.parse_orders).
(1) runs on the host (lupa: requirements of the project); (2) needs torch and net (the container)."""
import json
from pathlib import Path

import pytest

from bridge import build, config
from bridge.pack import bundle, pfh5

DATA = Path(__file__).resolve().parent / "data"
ENTRIES = ("nn_arena", "charge_probe", "missile_probe", "morale_probe", "map_capture", "move_probe")
CHAIN_BUILD = ["nn-arena", "--army-seed", "1000702112", "--armies", "v3", "--map", "ksl_plains_01", "--deploy",
               "strategist", "--own-ai", "net", "--speed", "10", "--timeout", "1200"]


def lua_runtime():
    pytest.importorskip("lupa")
    from lupa.lua51 import LuaRuntime
    return LuaRuntime(unpack_returned_tuples=True)


def lua_module(lua, name):
    """A module of bridge/lua with its requires, loaded as the bundle loads it (package.preload)."""
    load = lua.eval("function(src, name) return assert(loadstring(src, '@' .. name)) end")
    preload = lua.globals().package.preload
    for mod, src in bundle.collect(config.LUA, name):
        preload[mod] = load(src, mod)
    return lua.globals().require(name)


def test_every_entry_bundles_and_compiles():
    lua_runtime()
    for entry in ENTRIES:
        script, modules = bundle.bundle(config.LUA, f"entries.{entry}", {"build": "smoke"})
        assert build.check_syntax(script), entry
        assert modules[-1] == f"entries.{entry}"


def test_the_chain_battle_pack_builds(tmp_path, capsys):
    assert build.main(CHAIN_BUILD + ["--out", str(tmp_path)]) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    files, deps = pfh5.read_pack((tmp_path / manifest["pack"]).read_bytes())
    assert list(deps) == [m["pack"] for m in config.required_mods()]           # True Sight, always
    script = files["script\\battle\\mod\\tww3_bai_nn_arena.lua"].decode("utf-8")
    assert f"build {manifest['build']}" in script.splitlines()[0]
    xml = files["script\\battle\\tww3_bai\\nn_arena.xml"].decode("utf-8")
    c = manifest["config"]
    assert c["own_ai"] == "net" and c["deploy"] == "strategist" and c["map_name"] == "ksl_plains_01"
    assert c["decide_ms"] == 1000 and set(c["deploy_zones"]) == {"own", "enemy"}
    assert c["skirmish"] == "off"                                   # as the simulator's network units (parity)
    names = [u["script_name"] for side in ("own", "enemy") for u in c["units"][side]]
    assert names and all(n in xml for n in names)
    assert manifest["syntax_checked"] in (True, False)
    assert "entries.nn_arena" in manifest["modules"] and "apps.bridge.deploy_adapter" in manifest["modules"]


def the_chain():
    """The chain as the companion plays it (loop.load_chain: formulas above the unit network); the unit network's
    weights when present (net/config/tactic.json train.unit_ckpt), else a fresh network (the answer's format is the
    same)."""
    torch = pytest.importorskip("torch")
    from bridge.companion import chain as CH
    from bridge.companion import loop
    from net.tactic import cfg as tcfg
    if config.resolve(tcfg()["train"]["unit_ckpt"]).exists():
        chain, _ = loop.load_chain(op="formula", tactic="formula", greedy=True, task="hold")
        return chain
    from net.unit import net as Nt
    torch.manual_seed(0)
    return CH.Chain(Nt.UnitNet().eval(), task="hold", unit_greedy=True)


def test_the_chain_answers_a_recorded_state(tmp_path):
    from bridge.companion import exchange, loop
    chain = the_chain()
    doc = exchange.read_state(DATA / "state_m1.json")
    assert doc is not None
    seat = loop.Seat(tmp_path / exchange.STATE, tmp_path / exchange.ORDERS, chain)
    lines = []
    orders, think_ms, attempts = loop.answer(seat, doc, lines.append)
    text = (tmp_path / exchange.ORDERS).read_text(encoding="utf-8")
    assert text.splitlines()[0] == exchange.FORMAT and text.rstrip().endswith("end")
    assert "mode chain" in text.splitlines()
    parsed = exchange.parse_orders(text)
    assert parsed["move"] == doc["move"] and parsed["batch"] == doc["batch"]
    up = {u["n"] for u in doc["units"]
          if int(u["side"]) == 1 and (u.get("men") or 0) > 0 and not u.get("r") and not u.get("s")}
    assert up and set(parsed["orders"]) == up                       # every own unit that takes orders, no other
    assert {o["kind"] for o in parsed["orders"].values()} <= {"hold", "move", "attack", "withdraw", "keep"}
    assert all(o["target"].startswith("enemy_") for o in parsed["orders"].values() if o["kind"] == "attack")
    assert lines and lines[0].startswith(f"move {doc['move']:4d}")
    # the bridge reads the same text
    try:
        lua = lua_runtime()
    except pytest.skip.Exception:
        return
    services = lua_module(lua, "apps.bridge.services")
    got, reason = services.parse_orders(text)
    assert got is not None, reason
    assert got.mode == "chain" and int(got.move) == doc["move"] and got.batch == doc["batch"]
    assert int(got.count) == len(parsed["orders"])
    for name, o in parsed["orders"].items():
        assert got.orders[name].kind == o["kind"], name
