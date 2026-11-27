"""One press by our order, the same in the simulator and through the bridge (contract/parity.md "кнопки"; the lord line's
step 0a, tmp/lordline/BRIEF_0a.md Build 1; the outside reviews tmp/astra/skills_review/out/REPORT.md section 2 and
tmp/astra/lord0a_review/out/REPORT.md section 5).

The one slot <-> key map is sim/abilities.press_map; the one press route is bridge/companion/lord_script.LordScript.request
(requested -> called -> took). A lord at (0, -20) with two standing melee allies at (-24, 0) and (24, 0) (31.2 m: inside
35 m); a forced choice asks the faction's support button at seconds 5, 10 (its active phase) and 24 (its recharge) and
nothing else. Trace = (floor of the delivery's battle second, actor, key, the controller's ACTUAL cast target name).

  host (lupa, no torch) - the bridge's own Lua (apps.bridge.adapter: decide writes the state the companion reads, poll
      parses the reply - polled TWICE a move - and calls a recording unit controller); the card shows the phase on the
      lord and the allies < 35 m from the second after the call for the passport's active_s. Cases: Empire / Stand Your
      Ground and Skaven / Rally (one call at 5, the cooldown from the took); no card read (no call, the script FAILED);
      the lord's card alone absent (proved on the recipients: one call at 5); the lord down at 4..7 s (no call at 5; the call at 10)
  container (torch) - the PRODUCTION route: the chain 'tactic_v2 only' (TacticChain.decide with its real Gate, the tactic
      network weights/tactic/attack1/final.pt, BAI_LORD_SCRIPT=T) on the simulator's own state documents, its button
      forced at the same seconds; the reply serialised (exchange.orders_text) and parsed back; the simulator gets the
      press through sim.orders.press_on_landing (net/operative/lower.py Units.orders' landing) at each latency the
      cadence allows (net/common/cadence.Cadence().delay at dt), every request that reaches physics recorded and the
      key decoded from the slot's passport row (ab{k})

  assert sim_trace == lua_trace == [(5, "own_lord", key, "own_lord")]
"""
import math
import os

import pytest

from bridge import config
from bridge.companion import exchange
from bridge.companion import lord_script as LS
from bridge.pack import bundle
from net.cards import abilities as MA

GENERAL, WARLORD = "wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0"
SPEAR, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0"
# the expected keys written out here (not the script's constants)
CASES = {"emp": (GENERAL, SPEAR, "wh_main_character_abilities_stand_your_ground"),
         "skv": (WARLORD, CLANRAT, "wh_main_character_abilities_rally")}
PLACES = {"own_lord": (0.0, -20.0), "own_u1": (-24.0, 0.0), "own_u2": (24.0, 0.0)}
ASK_AT = (5, 10, 24)
SECONDS = 30

HARNESS = r"""
local calls, events = {}, {}
local H = {calls = calls, events = events, now = 0, rows = {}, down = {}}
local function noop() return nil end
function H.unit(name, men, abilities)
    local u = {name = name}
    function u:number_of_men_alive() return men end
    function u:is_routing() return H.down[name] == true end
    function u:is_shattered() return false end
    function u:ammo_left() return 0 end
    function u:missile_range() return 0 end
    function u:can_perform_special_ability(key) return true end
    function u:owned_non_passive_special_abilities() return abilities end
    function u:can_use_behaviour() return false end
    return setmetatable(u, {__index = function() return noop end})
end
local army = {}
function army:create_unit_controller()
    local uc = {}
    function uc:add_units(unit) self.unit = unit end
    function uc:perform_special_ability(key, target)
        calls[#calls + 1] = {t = H.now, actor = self.unit.name, key = key, aim = target and target.name or 'NONE'}
    end
    return setmetatable(uc, {__index = function() return noop end})
end
H.army = army
function H.opts(own, enemies, state_file, orders_file)
    return {army = army, own = own, enemies = enemies, vector = function(x, z) return {x = x, z = z} end,
        rows = function() return H.rows end, meta = {batch = 'parity'},
        emit = function(e, f) events[#events + 1] = {e = e, f = f} end,
        now_ms = function() return H.now end, model_ms = function() return H.now end,
        state_file = state_file, orders_file = orders_file, skirmish = 'game'}
end
return H
"""


def lua_bridge():
    pytest.importorskip("lupa")
    from lupa.lua51 import LuaRuntime
    lua = LuaRuntime(unpack_returned_tuples=True)
    load = lua.eval("function(src, name) return assert(loadstring(src, '@' .. name)) end")
    preload = lua.globals().package.preload
    for mod, src in bundle.collect(config.LUA, "apps.bridge.adapter"):
        preload[mod] = load(src, mod)
    return lua, lua.globals().require("apps.bridge.adapter"), lua.execute(HARNESS)


def card(key, called_s, t, passports):
    """{name: [phase]} the game's card shows at second t (module doc)."""
    p = passports[key]
    out = {n: [] for n in PLACES}
    if called_s is not None and called_s + 1 <= t < called_s + 1 + float(p["active_s"]):
        lx, lz = PLACES["own_lord"]
        for n, (x, z) in PLACES.items():
            if math.hypot(x - lx, z - lz) < float(p["range_m"]) or n == "own_lord":
                out[n] = list(p["phases"])[:1]
    return out


def rows_at(t, lord_key, inf_key, fx, wavering, variant):
    rows = []
    for n, (x, z) in PLACES.items():
        r = {"n": n, "side": 1, "key": lord_key if n == "own_lord" else inf_key, "x": x, "z": z, "b": 0.0,
             "men": 1 if n == "own_lord" else 120, "hp": 1.0, "m": n != "own_lord", "w": wavering and n == "own_u1",
             "r": False, "s": False, "t": "enemy_u1" if n != "own_lord" else "", "v": True, "fx": fx[n]}
        if variant == "no_card" or (n == "own_lord" and variant == "lord_unread"):
            del r["fx"]
        if n == "own_lord" and variant == "down" and 4 <= t <= 7:
            r["r"] = True
        rows.append(r)
    rows.append({"n": "enemy_u1", "side": 2, "key": inf_key, "x": 0.0, "z": 12.0, "b": 180.0, "men": 120, "hp": 1.0,
                 "m": True, "w": False, "r": False, "s": False, "t": "own_u1", "v": True, "fx": []})
    return rows


def lua_trace(case, tmp_path, variant="plain"):
    """The bridge path's trace of one case (module doc) and the script."""
    lord_key, inf_key, key = CASES[case]
    lua, adapter, H = lua_bridge()
    passports = {**MA.load_spells(), **MA.load()}
    own = [{"name": n, "unit": H.unit(n, 1 if n == "own_lord" else 120,
                                      lua.table_from(sorted(LS.LOADOUT[lord_key]) if n == "own_lord" else []))}
           for n in PLACES]
    enemies = [{"name": "enemy_u1", "unit": H.unit("enemy_u1", 120, lua.table_from([]))}]
    state, orders = tmp_path / f"{case}_{variant}_state.json", tmp_path / f"{case}_{variant}_orders.txt"
    handle = adapter.start(H.opts(lua.table_from([lua.table_from(x) for x in own]),
                                  lua.table_from([lua.table_from(x) for x in enemies]), str(state), str(orders)))
    script, called = None, None
    for t in range(SECONDS):
        H.now = t * 1000
        H.down["own_lord"] = variant == "down" and 4 <= t <= 7
        fx = card(key, called, t, passports)
        H.rows = lua.table_from([lua.table_from({k: (lua.table_from(v) if isinstance(v, list) else v)
                                                 for k, v in r.items()})
                                 for r in rows_at(t, lord_key, inf_key, fx, case == "skv", variant)])
        handle.decide()
        doc = exchange.read_state(state)
        assert doc is not None and doc["move"] == t + 1
        if script is None:
            script = LS.LordScript([u["n"] for u in doc["units"]], [u["key"] for u in doc["units"]],
                                   [u["side"] for u in doc["units"]])
        script.follow(doc)
        rows = {u["n"]: u for u in doc["units"]}
        uses = script.request(rows, t, key, "forced", move=doc["move"], recipients=["own_u1", "own_u2"]) \
            if t in ASK_AT else []
        keep = [{"unit": n, "kind": "keep"} for n in PLACES]
        exchange.write_atomic(orders, exchange.orders_text("parity", doc["move"], keep, 1.0, uses, mode="chain"))
        n0 = len(H.calls)
        handle.poll()
        handle.poll()                       # (the same reply polled twice: no second call)
        if len(H.calls) > n0 and called is None:
            called = t
    return [(int(H.calls[i].t // 1000), H.calls[i].actor, H.calls[i].key, H.calls[i].aim)
            for i in range(1, len(H.calls) + 1)], script


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_bridge_presses_once(case, tmp_path):
    """Host: one call at second 5 (none in the active phase at 10 nor in the recharge at 24), the reply polled twice
    called once, the self target the controller got is the lord, the cooldown from the took (108 s / 74 s)."""
    lord_key, _, key = CASES[case]
    trace, script = lua_trace(case, tmp_path)
    assert trace == [(5, "own_lord", key, "own_lord")]
    b = script.btn[key]
    p = MA.load()[key]
    assert b["st"] == "cooldown" and b["cd_end"] == 5 + p["active_s"] + p["recharge_s"] + LS.READY_PAD_S
    assert script.press == sim_slots(lord_key) and not script.failed


def test_an_unreadable_card_never_presses_and_fails(tmp_path):
    """No card readable (the lord's nor the recipients'): no call, the script FAILED; the lord's card alone unread: the
    aura button proved on the recipients' cards - one call, took, not FAILED."""
    trace, script = lua_trace("emp", tmp_path, "no_card")
    assert trace == [] and script.failed
    trace, script = lua_trace("emp", tmp_path, "lord_unread")
    assert trace == [(5, "own_lord", CASES["emp"][2], "own_lord")] and not script.failed
    assert script.btn[CASES["emp"][2]]["st"] == "cooldown"


def test_a_down_lord_is_not_pressed(tmp_path):
    trace, script = lua_trace("emp", tmp_path, "down")
    assert trace == [(10, "own_lord", CASES["emp"][2], "own_lord")]


def sim_slots(lord_key):
    from net.cards import passport
    from sim import abilities as A
    return A.press_map(lord_key, passport.load(), {**MA.load_spells(), **MA.load()})


def test_the_press_map_fails_closed():
    from sim import abilities as A
    units = {"x": {"abilities": ["wh_main_character_abilities_rally", "no_such_active_key"]}}
    with pytest.raises(ValueError):
        A.press_map("x", units, MA.load())
    assert sim_slots(GENERAL) == {0: "wh_main_character_abilities_foe_seeker",
                                  1: "wh_main_character_abilities_stand_your_ground"}
    assert sim_slots(WARLORD) == {0: "wh2_main_character_abilities_verminous_valour",
                                  1: "wh_main_character_abilities_deadly_onslaught",
                                  2: "wh_main_character_abilities_rally"}


def sim_trace(case, delay):
    """The production route on the simulator (module doc); needs torch, the tactic weights."""
    pytest.importorskip("torch")
    import torch
    from bridge.companion import loop as LP
    from contract.test_tactic_bridge_parity_torch import doc_of_sim
    from sim import battle, scenario
    from sim import orders as O
    from sim.params import load
    P = load()
    lord_key, inf_key, key = CASES[case]
    fac = {"emp": "wh_main_emp_empire", "skv": "wh2_main_skv_skaven"}[case]
    other = {"emp": ("wh2_main_skv_skaven", CLANRAT), "skv": ("wh_main_emp_empire", SPEAR)}[case]
    units = [{"key": lord_key if n == "own_lord" else inf_key, "x": x, "z": z, "b": 0.0,
              **({"general": True} if n == "own_lord" else {})} for n, (x, z) in PLACES.items()]
    st = scenario.build([{"attacker": 2, "sides": {1: {"faction": fac, "units": units},
                                                   2: {"faction": other[0], "units": [{"key": other[1], "x": 0.0,
                                                                                       "z": 600.0, "b": 180.0}]}}}], P)
    ours = [i for i in range(st.N) if int(st.u["side"][0, i]) == 1]
    theirs = [i for i in range(st.N) if int(st.u["side"][0, i]) == 2]
    assert len(ours) == 3 and len(theirs) == 1
    names = {ours[0]: "own_lord", ours[1]: "own_u1", ours[2]: "own_u2", theirs[0]: "enemy_u1"}
    abk = sorted(P.abilities)
    old = os.environ.get("BAI_LORD_SCRIPT")
    os.environ["BAI_LORD_SCRIPT"] = "T"
    try:
        chain, _ = LP.load_chain("formula", "weights/tactic/attack1/final.pt")
    finally:
        os.environ.pop("BAI_LORD_SCRIPT") if old is None else os.environ.__setitem__("BAI_LORD_SCRIPT", old)
    steps = round(1.0 / P.dt)
    out, called = [], {}
    for t in range(SECONDS):
        doc = doc_of_sim(st, 0, t * 1000)
        doc["batch"] = "parity"
        for u in doc["units"]:
            u["n"] = names[int(u["n"][1:])]
        # abilities_used as the bridge writes it: the battle ms of OUR calls - here the start of the physics step the
        # press landed on. Not doc_of_sim's, which is back-computed from the cooldown for the timers' parity and
        # stamps the END of the landing step (sim/abilities.py ticks cd before a fire sets it): a press landing on
        # the second's last step read t + 1000, outside the script's call window [t, t + 1000) - a false retry
        doc["abilities_used"] = {n: dict(v) for n, v in called.items()}
        orders, _, uses = chain.decide(doc)       # (the real chain: seer, tactic network, unit script, Gate, lord script)
        ls = chain.lord_script
        assert ls is not None and not ls.failed, getattr(ls, "failed", None)
        if t == 0:                                # the forced choice: the script's own request at ASK_AT, nothing else
            ls._press = (lambda rows, tt, note, *a, **k: ls.request(rows, tt, key, "forced", note, move=int(tt),
                                                                    recipients=["own_u1", "own_u2"])
                         if int(round(tt)) in ASK_AT else [])
        text = exchange.orders_text("parity", doc["move"], orders, 1.0, uses, mode="chain")
        back = exchange.parse_orders(text)
        intended = O.hold(1, st.N)
        intended.kind[:] = O.KEEP
        for a in back["abilities"]:
            i = next(i for i, n in names.items() if n == a["unit"])
            intended.ability[0, i] = ls.slot_of[a["key"]]           # (self cast: no cast target)
        lands = torch.tensor([delay])
        for j in range(steps):
            o = O.press_on_landing(intended.clone(), intended, lands, j)
            for i in range(st.N):
                k = int(o.ability[0, i])
                if k >= 0:                                            # every request that reaches physics
                    aim = int(o.cast_target[0, i])
                    ak = abk[int(st.u[f"ab{k}"][0, i])]
                    out.append((t, names[i], ak, names.get(aim, names[i]) if aim >= 0 else names[i]))
                    called.setdefault(names[i], {})[ak] = t * 1000 + j * P.dt * 1000
            battle.step(st, o, P)
    return out


def delays():
    from net.common.cadence import Cadence
    from sim.params import load
    whole, frac = Cadence().delay(load().dt)
    return [whole] + ([whole + 1] if frac > 0 else [])


@pytest.mark.parametrize("case", sorted(CASES))
def test_one_press_sim_equals_bridge(case, tmp_path):
    """The single assertion (module doc) at each landing delay the cadence allows. The two halves need lupa (the host)
    and torch (the container) - no one place has both: each half is held to the same written trace where it runs."""
    want = [(5, "own_lord", CASES[case][2], "own_lord")]
    ran = False
    try:
        import lupa  # noqa: F401
        b, _ = lua_trace(case, tmp_path)
        assert b == want
        ran = True
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
    except ImportError:
        if not ran:
            pytest.skip("neither lupa nor torch")
        return
    for d in delays():
        assert sim_trace(case, d) == want, d
