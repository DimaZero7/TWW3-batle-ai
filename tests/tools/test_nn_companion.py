"""tools.nn.companion: the files exchanged with the game (levels 1 and 2, numpy) and one decision of
the companion (torch: skipped in the project's .venv, runs in the training container)."""
import json
import threading

import numpy as np
import pytest

from tools.nn.companion import exchange
from tools.nn.model import observation as ob

LORD, SPEAR = "wh_main_emp_cha_general_0", "wh_main_emp_inf_spearmen_0"


def unit(name, side, key, x, **fields):
    row = {"n": name, "side": side, "key": key, "x": x, "z": 0.0, "b": 90 if side == 1 else 270, "men": 100,
           "hp": 0.9, "mp": 0.8, "ms": 3, "r": False, "s": False, "w": False, "m": False, "mv": False,
           "f": False, "a": 0, "fire": False, "t": "", "fat": "threshold_fresh", "k": 0, "ox": x, "oz": 0.0,
           "lf": False, "rf": False, "bf": False, "v": True}
    row.update(fields)
    return row


def state_doc(move=1, **enemy):
    return {"batch": "b-1", "move": move, "t": 5000, "done": False, "attacker": 2, "decide_ms": 1000,
            "factions": {"own": "wh_main_emp_empire", "enemy": "wh_main_emp_empire"},
            "units": [unit("own_lord", 1, LORD, -235.0), unit("own_spear_1", 1, SPEAR, -175.0, t="enemy_spear_1"),
                      unit("enemy_lord", 2, LORD, 235.0), unit("enemy_spear_1", 2, SPEAR, 175.0, **enemy)]}


def test_a_state_becomes_the_observations_input():
    doc = state_doc(fat="threshold_tired", mp=0.123)
    b = exchange.battle(doc)
    assert b.names == ["own_lord", "own_spear_1", "enemy_lord", "enemy_spear_1"] and b.own == ["own_lord", "own_spear_1"]
    assert b.setup.attacker[0] == 2 and b.setup.keys[0][1] == SPEAR
    s = exchange.arrays(doc, b.names)
    assert s["x"].shape == (1, 4) and s["t"][0] == 5.0
    assert s["target"][0].tolist() == [-1, 3, -1, -1]
    assert s["fat"][0, 3] == 3 and s["vis"].all()
    obs, _ = ob.observe(s, b.setup, 1)
    assert obs.tokens.shape == (1, 4, ob.TOKEN) and obs.ctrl[0].tolist() == [True, True, False, False]
    assert obs.tokens[0, 3, ob.INDEX["morale"]] == 0     # never the enemy's exact morale


def test_an_enemy_not_seen_is_hidden():
    doc = state_doc(v=False)
    b = exchange.battle(doc)
    s = exchange.arrays(doc, b.names)
    assert s["vis"][0].tolist() == [True, True, True, False]
    obs, _ = ob.observe(s, b.setup, 1)
    assert obs.tokens[0, 3, ob.INDEX["visible"]] == 0 and obs.tokens[0, 3, ob.INDEX["men"]] == 0
    assert not obs.target_ok[0, 3]


def test_missing_readings_are_not_numbers():
    doc = state_doc()
    del doc["units"][0]["hp"]
    doc["units"][0]["men"] = None
    s = exchange.arrays(doc, [u["n"] for u in doc["units"]])
    assert np.isnan(s["hp"][0, 0]) and s["men"][0, 0] == 0


def test_orders_of_the_network_become_the_file_and_back():
    names = ["own_lord", "own_spear_1", "enemy_lord", "enemy_spear_1"]
    side = np.array([1, 1, 2, 2])
    kind = np.array([1, 2, 0, 0])
    orders = exchange.orders_list(names, side, kind, np.array([-100.04, 0, 0, 0]), np.array([20.0, 0, 0, 0]),
                                  np.array([-1, 3, -1, -1]), np.array([True, False, False, False]))
    assert orders == [{"unit": "own_lord", "kind": "move", "x": -100.0, "z": 20.0, "run": True},
                      {"unit": "own_spear_1", "kind": "attack", "target": "enemy_spear_1", "run": False}]
    text = exchange.orders_text("b-1", 9, orders, 4.0)
    assert text.endswith("end\n") and exchange.summary(orders) == "move 1 attack 1"
    doc = exchange.parse_orders(text)
    assert doc["move"] == 9 and doc["batch"] == "b-1" and doc["think_ms"] == 4.0
    assert doc["orders"]["own_lord"] == {"kind": "move", "x": -100.0, "z": 20.0, "run": True}
    assert exchange.parse_orders(text[:-4]) is None


def test_keep_is_written_as_keep():
    names, side = ["own_a", "own_b", "enemy_c"], np.array([1, 1, 2])
    out = exchange.orders_list(names, side, np.array([4, 0, 0]), np.zeros(3), np.zeros(3), np.full(3, -1),
                               np.zeros(3, bool))
    assert out == [{"unit": "own_a", "kind": "keep"}, {"unit": "own_b", "kind": "hold"}]
    text = exchange.orders_text("b", 1, out)
    assert "unit own_a keep\n" in text and exchange.summary(out) == "hold 1 keep 1"
    assert exchange.parse_orders(text)["orders"]["own_a"] == {"kind": "keep"}


def test_withdraw_always_runs_and_attack_without_a_target_holds():
    names, side = ["own_a", "enemy_b"], np.array([1, 2])
    out = exchange.orders_list(names, side, np.array([3, 0]), np.array([5.0, 0]), np.array([6.0, 0]),
                               np.array([-1, -1]), np.array([False, False]))
    assert out == [{"unit": "own_a", "kind": "withdraw", "x": 5.0, "z": 6.0, "run": True}]
    out = exchange.orders_list(names, side, np.array([2, 0]), np.zeros(2), np.zeros(2), np.array([-1, -1]),
                               np.zeros(2, bool))
    assert out == [{"unit": "own_a", "kind": "hold"}]


def test_a_half_written_state_is_not_read(tmp_path):
    path = tmp_path / exchange.STATE
    assert exchange.read_state(path) is None
    path.write_text(json.dumps(state_doc())[:-10], encoding="utf-8")
    assert exchange.read_state(path) is None
    exchange.write_atomic(path, json.dumps(state_doc(move=4)))
    assert exchange.read_state(path)["move"] == 4
    assert not (tmp_path / (exchange.STATE + ".tmp")).exists()


def test_write_waits_while_the_file_is_held(tmp_path, monkeypatch):
    path = tmp_path / exchange.ORDERS
    calls = []
    real = exchange.os.replace

    def busy(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("the game is reading it")
        real(src, dst)
    monkeypatch.setattr(exchange.os, "replace", busy)
    assert exchange.write_atomic(path, "x\n", wait_s=0) == 3 and path.read_text() == "x\n"


FS, SYG, HTL = ("wh_main_character_abilities_foe_seeker", "wh_main_character_abilities_stand_your_ground",
                "wh_main_lord_passive_hold_the_line")


def test_ability_timers_from_the_bridges_uses_and_the_active_effects():
    doc = state_doc()
    doc["t"] = 20000
    doc["abilities_used"] = {"own_lord": {SYG: 10000}}               # used 10 s ago: 18 s active, 90 s recharge
    doc["units"][2]["fx"] = ["wh_main_character_abilities_foe_seeker"]   # the enemy General's Foe Seeker shows
    doc["units"][0]["fx"] = [HTL, SYG]                                   # the lord's card: SYG is on
    doc["units"][1]["fx"] = {}                                            # JSON's empty list
    b = exchange.battle(doc)
    assert b.slots[0] == [FS, SYG, HTL] and b.slots[1] == ["", "", ""]
    s = exchange.arrays(doc, b.names, b.slots)
    assert s["ab1_on"][0, 0] == pytest.approx(8) and s["ab1_cd"][0, 0] == pytest.approx(98)
    assert s["ab0_on"][0, 0] == 0 and s["ab0_cd"][0, 0] == 0
    assert s["ab0_on"][0, 2] == 1 and s["ab0_cd"][0, 2] == 1 and s["ab1_on"][0, 2] == 0
    obs, _ = ob.observe(s, b.setup, 1)
    assert obs.abil_ok[0, 0].tolist() == [True, False, False]
    late = json.loads(json.dumps(dict(doc, t=200000)))
    late["units"][0]["fx"] = [HTL]
    assert exchange.arrays(late, b.names, b.slots)["ab1_cd"][0, 0] == 0   # ready again
    late["units"][0]["fx"] = [HTL, SYG]                                   # still on the card: 1 s more
    assert exchange.arrays(late, b.names, b.slots)["ab1_on"][0, 0] == 1
    doc["units"][0]["fx"] = [SYG]                                         # the card shows it: as counted
    assert exchange.arrays(doc, b.names, b.slots)["ab1_on"][0, 0] == pytest.approx(8)
    doc["units"][0]["fx"] = []                                            # used, but not on the card:
    s = exchange.arrays(doc, b.names, b.slots)                            # it did not take, ready again
    assert s["ab1_on"][0, 0] == 0 and s["ab1_cd"][0, 0] == 0
    del doc["units"][0]["fx"]                                             # the card not read: the count
    assert exchange.arrays(doc, b.names, b.slots)["ab1_cd"][0, 0] == pytest.approx(98)


def test_ability_choices_become_lines_and_back():
    names, side = ["own_lord", "own_spear_1", "enemy_lord"], np.array([1, 1, 2])
    slots = [[FS, SYG, HTL], ["", "", ""], [FS, SYG, HTL]]
    uses = exchange.ability_list(names, side, np.array([1, 0, 0]), slots)
    assert uses == [{"unit": "own_lord", "key": SYG}]                    # empty slots and enemies give none
    orders = [{"unit": "own_lord", "kind": "keep"}, {"unit": "own_spear_1", "kind": "hold"}]
    text = exchange.orders_text("b-1", 3, orders, None, uses)
    assert text.endswith(f"ability own_lord {SYG}\nend\n")
    doc = exchange.parse_orders(text)
    assert doc["abilities"] == uses and doc["orders"]["own_lord"] == {"kind": "keep"}
    with pytest.raises(ValueError):
        exchange.orders_text("b-1", 3, orders, None, [{"unit": "own lord", "key": SYG}])


def _game_points_doc():
    """A state as the game writes it: the ordered_position of a held / attacking unit ~10 m from it."""
    doc = state_doc()
    doc["units"].append(unit("own_spear_2", 1, SPEAR, -150.0, z=40.0, ox=-141.0, oz=46.0))
    doc["units"].append(unit("own_spear_3", 1, SPEAR, -120.0, z=-40.0, ox=-60.0, oz=-10.0, r=True, ms=6))
    for u in doc["units"][:2]:
        u["ox"], u["oz"] = u["x"] + 7.0, 7.0
    return doc


def test_order_points_are_the_simulators():
    doc = _game_points_doc()
    names = [u["n"] for u in doc["units"]]
    side = np.array([u["side"] for u in doc["units"]])
    s = exchange.arrays(doc, names)
    given = {"own_spear_1": {"unit": "own_spear_1", "kind": "attack", "target": "enemy_spear_1", "run": True},
             "own_spear_2": {"unit": "own_spear_2", "kind": "move", "x": -100.0, "z": 30.0, "run": False},
             "own_spear_3": {"unit": "own_spear_3", "kind": "hold"}}
    enemy_ox = s["ox"][0, 2:4].copy()
    last = exchange.order_points(s, names, side, given, {"own_spear_3": (-125.0, -45.0)})
    ox, oz = s["ox"][0], s["oz"][0]
    assert (ox[0], oz[0]) == (-235.0, 0.0)                 # no order yet: holds, its own place
    assert (ox[1], oz[1]) == (175.0, 0.0)                  # attack: the target's place
    assert (ox[4], oz[4]) == (-100.0, 30.0)                # move: the point
    assert (ox[5], oz[5]) == (-125.0, -45.0)               # routing: the point it had
    assert (s["ox"][0, 2:4] == enemy_ox).all()             # enemies: as read
    assert last["own_spear_1"] == (175.0, 0.0) and last["own_spear_3"] == (-125.0, -45.0)
    assert s["order_kind"][0].tolist()[:2] == [0, 2] and s["order_kind"][0, 4] == 1    # the order in force (ORDER)
    assert s["order_target"][0, 1] == 3 and s["order_kind"][0, 2] == -1                  # enemies: not known
    obs, _ = ob.observe(s, exchange.battle(doc).setup, 1)
    assert obs.tokens[0, 0, ob.INDEX["has_order"]] == 0 and obs.tokens[0, 1, ob.INDEX["has_order"]] == 1
    assert obs.tokens[0, 1, ob.INDEX["order_fwd"]] == pytest.approx(0.7)   # 350 m ahead / POS, not ~10 m off
    doc["units"][3]["men"] = 0                             # the target is gone: holds, as in the simulator
    s = exchange.arrays(doc, names)
    exchange.order_points(s, names, side, given)
    assert (s["ox"][0, 1], s["oz"][0, 1]) == (-175.0, 0.0)


def test_running_is_the_simulators_speed_not_the_games_run_mode():
    # The game's f (is_moving_fast) is the run mode: on for a unit standing in melee. The network
    # learned the simulator's: moving faster than the walk + 0.3 m/s.
    doc = state_doc()
    for u in doc["units"]:
        u["f"], u["m"] = True, True
    b = exchange.battle(doc)
    assert b.walk[0] == pytest.approx(1.5) and b.walk[1] > 0           # passports' walk speeds
    s = exchange.arrays(doc, b.names)
    prev = exchange.running_by_speed(s, b.walk)
    assert not s["f"][0].any()                                         # the first state: nobody runs
    doc2 = dict(state_doc(), t=6000)
    doc2["units"][1].update(x=-171.0, mv=True, f=True)                 # 4 m in 1 s: runs
    doc2["units"][3].update(x=174.0, mv=True, f=True, m=True)          # 1 m in 1 s, run mode in melee: walks
    s2 = exchange.arrays(doc2, b.names)
    exchange.running_by_speed(s2, b.walk, prev)
    assert s2["f"][0].tolist() == [False, True, False, False]
    obs, _ = ob.observe(s2, b.setup, 1)
    assert obs.tokens[0, 1, ob.INDEX["running"]] == 1 and obs.tokens[0, 3, ob.INDEX["running"]] == 0


def test_own_targets_are_the_simulators_fought_or_shot_enemy():
    # The game's t is the engine's target: an attack order's target while the unit still walks, and
    # often none for a unit fighting under a hold or a move. The network learned the simulator's:
    # a target exactly while the unit fights or shoots.
    doc = state_doc()
    doc["units"].append(unit("own_spear_2", 1, SPEAR, 170.0, m=True))                       # fights, no t
    doc["units"].append(unit("own_spear_3", 1, SPEAR, -100.0, fire=True, t="enemy_lord"))   # shoots
    doc["units"].append(unit("own_spear_4", 1, SPEAR, 180.0, m=True, t="enemy_lord"))       # fights, t kept
    doc["units"].append(unit("own_spear_5", 1, SPEAR, -50.0, fire=True))                    # no t read
    b = exchange.battle(doc)
    s = exchange.arrays(doc, b.names)
    assert s["target"][0].tolist() == [-1, 3, -1, -1, -1, 2, 2, -1]
    exchange.engaged_targets(s, b.side)
    # own_spear_1 under an attack, not fighting: none; own_spear_2 in melee: the nearest enemy (spear at 175)
    assert s["target"][0].tolist() == [-1, -1, -1, -1, 3, 2, 2, -1]
    obs, _ = ob.observe(s, b.setup, 1)
    assert obs.tokens[0, :2, ob.INDEX["has_target"]].tolist() == [0, 0]
    assert obs.tokens[0, 4:7, ob.INDEX["has_target"]].tolist() == [1, 1, 1]
    doc["units"][3]["men"] = 0                     # the engine's target is gone: the nearest present enemy
    s = exchange.arrays(doc, b.names)
    exchange.engaged_targets(s, b.side)
    assert s["target"][0, 6] == 2 and s["target"][0, 4] == 2


def test_orders_in_force_skip_keep_and_units_that_take_none():
    given = {"own_a": {"unit": "own_a", "kind": "move", "x": 1.0, "z": 2.0, "run": True}}
    orders = [{"unit": "own_a", "kind": "keep"}, {"unit": "own_b", "kind": "attack", "target": "enemy_c", "run": True},
              {"unit": "own_c", "kind": "hold"}]
    exchange.remember_orders(given, orders, {"own_a", "own_b"})
    assert given["own_a"]["kind"] == "move" and given["own_b"]["kind"] == "attack" and "own_c" not in given


def test_a_rallied_unit_is_seen_going_on_with_the_order_it_had():
    # The bridge gives a rallied unit the order it had when it broke (as the simulator goes on with
    # it): the companion's filler HOLD for it while it routed is neither in force nor in the file.
    doc = _game_points_doc()
    names = [u["n"] for u in doc["units"]]
    side = np.array([u["side"] for u in doc["units"]])
    given = {"own_spear_1": {"unit": "own_spear_1", "kind": "attack", "target": "enemy_spear_1", "run": True}}
    doc["units"][1]["r"] = True                                  # routing: out, the network's HOLD a filler
    filler = [{"unit": "own_spear_1", "kind": "hold", "out": True}]
    exchange.remember_orders(given, filler, set())
    assert "own_spear_1" not in exchange.parse_orders(exchange.orders_text("b", 1, filler))["orders"]
    s = exchange.arrays(doc, names)
    last = exchange.order_points(s, names, side, given)
    doc["units"][1]["r"], doc["units"][3]["x"] = False, 150.0     # rallied; its target has moved
    s = exchange.arrays(doc, names)
    exchange.order_points(s, names, side, given, last)
    assert given["own_spear_1"]["kind"] == "attack" and (s["ox"][0, 1], s["oz"][0, 1]) == (150.0, 0.0)


class TestCompanion:
    """One decision and the loop, with a fresh untrained actor (torch)."""

    def test_the_brain_sees_its_own_orders_as_the_simulator_does(self):
        from tools.nn.companion import loop, policy
        brain = loop.Brain(policy.fresh(seed=1))
        orders, _, _ = brain.decide(_game_points_doc())
        mine = {o["unit"]: o for o in orders}
        ctrl = {"own_lord", "own_spear_1", "own_spear_2"}                        # not the routing one
        assert set(brain.given) == {n for n in ctrl if mine[n]["kind"] != "keep"}
        assert all(brain.given[n] == mine[n] for n in brain.given)
        given = dict(brain.given)
        captured = {}
        real = loop.ob.observe

        def spy(state, *a, **k):
            captured["ox"], captured["x"] = state["ox"][0].copy(), state["x"][0].copy()
            return real(state, *a, **k)
        loop.ob.observe = spy
        try:
            brain.decide(dict(_game_points_doc(), move=2))
        finally:
            loop.ob.observe = real
        place = {"own_lord": 0, "own_spear_1": 1, "own_spear_2": 4, "enemy_lord": 2, "enemy_spear_1": 3}
        for n, o in given.items():
            i = place[n]
            want = {"hold": captured["x"][i], "move": o.get("x"), "withdraw": o.get("x"),
                    "attack": captured["x"][place.get(o.get("target"), i)]}[o["kind"]]
            assert captured["ox"][i] == pytest.approx(want)
        brain.decide(dict(_game_points_doc(), batch="b-2"))
        assert brain.battle.batch == "b-2"

    @pytest.fixture(autouse=True)
    def need_torch(self):
        pytest.importorskip("torch")

    def test_a_decision_gives_every_own_unit_a_valid_order(self):
        from tools.nn.companion import loop, policy
        brain = loop.Brain(policy.fresh(seed=1))
        orders, think_ms, uses = brain.decide(state_doc())
        assert [o["unit"] for o in orders] == ["own_lord", "own_spear_1"] and think_ms > 0
        assert all(u["unit"] == "own_lord" and u["key"] in brain.battle.slots[0] for u in uses)
        for o in orders:
            assert o["kind"] in exchange.KINDS
            if o["kind"] == "attack":
                assert o["target"].startswith("enemy_")
        memory = brain.h
        brain.decide(state_doc(move=2))
        assert brain.h is not memory
        brain.decide(dict(state_doc(), batch="b-2"))    # a new battle: memory starts afresh
        assert brain.battle.batch == "b-2"

    def test_a_v2_network_keeps_its_commitment_between_the_games_states(self):
        from tools.nn.companion import loop, policy
        brain = loop.Brain(policy.fresh(seed=1, preset="v2"))
        orders, _, _ = brain.decide(state_doc())
        held = {o["unit"] for o in orders if o["kind"] != "keep" and not o.get("out")}
        assert held and brain.commit["until"].max() >= 2.0
        again, _, _ = brain.decide(state_doc(move=2))          # the same moment: nothing happened, all held
        assert all(o["kind"] == "keep" for o in again if o["unit"] in held)
        brain.decide(dict(state_doc(), batch="b-2"))            # a new battle: the commitment starts afresh
        assert brain.battle.batch == "b-2" and float(brain.commit["until"].max()) <= state_doc().get("t", 0) / 1000 + 16

    def test_the_loop_answers_each_new_state_once(self, tmp_path):
        from tools.nn.companion import loop, policy
        exchange.write_atomic(tmp_path / exchange.STATE, json.dumps(state_doc(move=1)))   # an earlier run's
        lines = []
        worker = threading.Thread(target=loop.run, args=(tmp_path, loop.Brain(policy.fresh())),
                                  kwargs={"exit_on_done": True, "out": lines.append, "idle_exit_s": 20,
                                          "log": tmp_path / "companion.jsonl"})
        worker.start()
        doc = dict(state_doc(move=1), batch="b-2")
        exchange.write_atomic(tmp_path / exchange.STATE, json.dumps(doc))
        for _ in range(2000):
            answer = exchange.read_state(tmp_path / exchange.STATE) and (tmp_path / exchange.ORDERS).exists()
            if answer:
                break
            threading.Event().wait(0.005)
        orders = exchange.parse_orders((tmp_path / exchange.ORDERS).read_text(encoding="utf-8"))
        assert orders["batch"] == "b-2" and orders["move"] == 1
        exchange.write_atomic(tmp_path / exchange.STATE, json.dumps(dict(doc, move=2, done=True)))
        worker.join(10)
        assert not worker.is_alive()
        assert len([ln for ln in lines if ln.startswith("move")]) == 1 and "done" in lines[-1]
        logged = [json.loads(ln) for ln in (tmp_path / "companion.jsonl").read_text().splitlines()]
        assert [r["move"] for r in logged] == [1]


def test_units_that_take_no_orders_count_as_out_not_hold():
    orders = [{"unit": "own_a", "kind": "hold"}, {"unit": "own_b", "kind": "hold", "out": True},
              {"unit": "own_c", "kind": "attack", "target": "enemy_a", "run": True}]
    assert exchange.summary(orders) == "hold 1 attack 1 out 1"
    text = exchange.orders_text("b", 1, orders)
    assert "own_b" not in text                                  # no line: the network's HOLD is a filler
    assert exchange.parse_orders(text)["orders"] == {"own_a": {"kind": "hold"},
                                                     "own_c": {"kind": "attack", "target": "enemy_a", "run": True}}
