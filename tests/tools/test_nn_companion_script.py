"""tools.nn.companion.script: the simulator's script (ai_like) commands the enemy side of a real battle (torch:
skipped in the project's .venv, runs in the training container).

The main check: a simulated battle written decision by decision as the bridge writes the game's state; the
companion's script must give side 2 the orders ai_like gives on the simulator's own state."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.companion import exchange  # noqa: E402
from tools.nn.companion.script import ScriptSide  # noqa: E402
from tools.nn.sim import abilities as sim_abilities  # noqa: E402
from tools.nn.sim import battle, orders as O, scenario  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.sim.state import FATIGUE_LEVELS  # noqa: E402
from tools.nn.train import opponents  # noqa: E402


def army_of(seed):
    from tools.nn.armies import generate
    army = scenario.from_arena(generate.battle(seed), own_ai="defend")
    army["sides"][1]["ai"] = False          # our side: the network's (fires by order, here never)
    return army


def doc_of(st, names, slots, keys, sides, batch="b-1", move=1, used=None, layout=None, factions=None):
    """The state document the bridges write, from the simulator's state (row 0)."""
    u = {k: v[0] for k, v in st.u.items()}
    name_of = {int(s): n for n, s in zip(names, slots)}
    rows = []
    for n, s, k, sd in zip(names, slots, keys, sides):
        f = lambda key: float(u[key][s])
        b = lambda key: bool(u[key][s])
        tg = int(u["target"][s])
        rows.append({"n": n, "side": int(sd), "key": k, "x": f("x"), "z": f("z"), "b": f("b"), "men": f("men"),
                     "hp": f("hp"), "mp": f("mp"), "ms": f("ms"), "r": b("r") and not b("s"), "s": b("s"),
                     "w": b("w"), "m": b("m"), "mv": b("mv"), "f": b("f"), "a": f("a"), "fire": b("fire"),
                     "t": name_of.get(tg, "") if tg >= 0 else "", "k": f("k"), "ox": f("ox"), "oz": f("oz"),
                     "fat": FATIGUE_LEVELS[int(u["fat"][s])], "lf": b("lf"), "rf": b("rf"), "bf": b("bf"), "v": True})
    return {"batch": batch, "move": move, "t": float(st.t[0]) * 1000, "done": False, "attacker": int(st.attacker[0]),
            "factions": factions or {}, "units": rows, "abilities_used": used or {}, "control": 2, "layout": layout or []}


def in_force(st, names, slots, sides, side):
    """{name: order} of one side's orders in force in the simulator (exchange's dicts)."""
    u = {k: v[0] for k, v in st.u.items()}
    name_of = {int(s): n for n, s in zip(names, slots)}
    out = {}
    for n, s, sd in zip(names, slots, sides):
        if sd != side:
            continue
        k = O.KINDS[int(u["order_kind"][s])]
        o = {"unit": n, "kind": k}
        if k in ("move", "withdraw"):
            o.update(x=float(u["ox"][s]), z=float(u["oz"][s]), run=bool(u["order_run"][s]))
        elif k == "attack":
            o.update(target=name_of.get(int(u["order_target"][s]), ""), run=bool(u["order_run"][s]))
        out[n] = o
    return out


def play(seed, seconds, check):
    """A battle, ai_like on both sides (side 1 does not fire abilities), a decision a second; check(st, names,
    slots, keys, sides, decision) at each decision before the orders are taken."""
    params = load()
    army = army_of(seed)
    st = scenario.build([army], params)
    slot = scenario.slots(army, st.N // 2)
    units = army["sides"][1]["units"] + army["sides"][2]["units"]
    names = [u["name"] for u in units]
    slots = [slot[n] for n in names]
    keys = [u["key"] for u in units]
    sides = [1] * len(army["sides"][1]["units"]) + [2] * len(army["sides"][2]["units"])
    layout = [{"n": u["name"], "slot": "lord" if u.get("general") else "unit", "width": u.get("width")} for u in units]
    used = {}
    last_cd = {k: st.u[f"ab{k}_cd"][0].clone() for k in range(sim_abilities.SLOTS)}
    state = {"move": 0}
    factions = {"own": army["sides"][1]["faction"], "enemy": army["sides"][2]["faction"]}

    def policy(s):
        # the uses of side 2's abilities since the last decision (a cooldown that went up), as the bridge counts them
        for k in range(sim_abilities.SLOTS):
            cd = s.u[f"ab{k}_cd"][0]
            for n, sl, sd in zip(names, slots, sides):
                a = int(s.u[f"ab{k}"][0, sl])
                if sd == 2 and a >= 0 and float(cd[sl]) > float(last_cd[k][sl]) + 1e-3:
                    used.setdefault(n, {})[sim_abilities.keys(params)[a]] = float(s.t[0]) * 1000
            last_cd[k] = cd.clone()
        state["move"] += 1
        check(s, names, slots, keys, sides, layout, used, state["move"], factions)
        return opponents.ai_like(s)

    battle.run(st, policy, params, until_s=seconds, every_s=1.0, compile=False)
    return st


SEED = 1000900014


def test_the_script_in_the_game_gives_side_2_the_orders_ai_like_gives_in_the_simulator():
    side = ScriptSide("ai_like")
    seen = {"n": 0, "kind": 0, "attack": 0, "target": 0, "point": 0, "run": 0, "moved": 0, "unit_kinds": set(),
            "missed": {}}

    def check(st, names, slots, keys, sides, layout, used, move, factions):
        doc = doc_of(st, names, slots, keys, sides, move=move, used=used, layout=layout, factions=factions)
        others = in_force(st, names, slots, sides, 1)
        orders, _, _ = side.decide(doc, others)
        want = opponents.ai_like(st)
        name_of = {int(s): n for n, s in zip(names, slots)}
        standing = (st.u["men"][0] > 0) & ~st.u["r"][0] & ~st.u["gone"][0]
        for o in orders:
            s = slots[names.index(o["unit"])]
            if not bool(standing[s]):
                assert o.get("out")
                continue
            k = O.KINDS[int(want.kind[0, s])]
            seen["n"] += 1
            seen["kind"] += o["kind"] == k
            seen["unit_kinds"].add(k)
            if o["kind"] != k:
                seen["missed"][(k, o["kind"])] = seen["missed"].get((k, o["kind"]), 0) + 1
            if o["kind"] == k == "attack":
                seen["attack"] += 1
                seen["target"] += o["target"] == name_of[int(want.target[0, s])]
            if o["kind"] == k and k in ("move", "withdraw"):
                seen["moved"] += 1
                seen["point"] += np.hypot(o["x"] - float(want.x[0, s]), o["z"] - float(want.z[0, s])) < 1.0
                seen["run"] += o["run"] == bool(want.run[0, s])

    play(SEED, 240, check)
    n = seen["n"]
    assert n > 1000 and {"move", "attack"} <= seen["unit_kinds"], seen
    # the few differences: the closing speed and melee clock the game's state gives a second apart, not per step
    # (seeds 1000900014 / 1000903669 / 1000901199, 240 s: kinds 99.2 / 100 / 100 %, attack targets 100 / 100 / 99.9 %,
    # move points and runs 100 %; the 24 kinds off: move by the flank wrap against attack)
    assert seen["kind"] / n > 0.97 and seen["target"] / max(seen["attack"], 1) > 0.97, seen
    assert seen["point"] / max(seen["moved"], 1) > 0.95 and seen["run"] / max(seen["moved"], 1) > 0.97, seen


def test_the_lords_ability_fires_by_the_simulators_ai_rule():
    side = ScriptSide("ai_like")
    params = side.params
    ability_keys = sim_abilities.keys(params)
    lords = [k for k, p in params.units.items() if any(
        sim_abilities.row(params, a)[sim_abilities.COL["trigger"]] == sim_abilities.TRIGGERS["melee"]
        for a in sim_abilities.slot_keys(k, params.units, params.abilities) if a)]
    lord = next(k for k in lords if k.startswith("wh_main_emp_cha_general"))
    keys = [a for a in sim_abilities.slot_keys(lord, params.units, params.abilities) if a]
    melee = [a for a in keys if sim_abilities.row(params, a)[sim_abilities.COL["trigger"]] == sim_abilities.TRIGGERS["melee"]
             and (params.sim["abilities"].get("friends_min") or {}).get(a, 0) == 0]
    assert melee, keys
    spear = "wh_main_emp_inf_spearmen_0"

    def unit(n, side_, key, x, **kw):
        row = {"n": n, "side": side_, "key": key, "x": x, "z": 0.0, "b": 90.0 if side_ == 1 else 270.0, "men": 100,
               "hp": 1.0, "mp": 1.0, "ms": 2, "r": False, "s": False, "w": False, "m": False, "mv": False, "f": False,
               "a": 0, "fire": False, "t": "", "k": 0, "ox": x, "oz": 0.0, "fat": "threshold_fresh", "v": True}
        row.update(kw)
        return row

    def doc(move, t_ms, melee_on, used=None):
        units = [unit("own_lord", 1, lord, -5.0, m=melee_on, t="enemy_lord" if melee_on else ""),
                 unit("own_spear_1", 1, spear, -300.0),
                 unit("enemy_lord", 2, lord, 5.0 if melee_on else 300.0, m=melee_on,
                      t="own_lord" if melee_on else ""),
                 unit("enemy_spear_1", 2, spear, 320.0)]
        return {"batch": "b-ab", "move": move, "t": t_ms, "attacker": 2, "units": units,
                "abilities_used": used or {}, "layout": [{"n": "own_lord", "slot": "lord"}, {"n": "enemy_lord", "slot": "lord"}]}

    _, _, uses = side.decide(doc(1, 1000, False))
    assert not [a for a in uses if a["key"] in melee]                       # not in melee: not yet
    _, _, uses = side.decide(doc(2, 2000, True))
    fired = {a["key"] for a in uses if a["unit"] == "enemy_lord"}
    assert set(melee) <= fired and not [a for a in uses if a["unit"] != "enemy_lord"]
    used = {"enemy_lord": {k: 2000 for k in fired}}
    _, _, uses = side.decide(doc(3, 3000, True, used))
    assert not uses                                                          # on recharge now


def test_the_networks_orders_in_force_are_the_scripts_view_of_our_side():
    side = ScriptSide("ai_like")
    spear = "wh_main_emp_inf_spearmen_0"
    rows = [{"n": "own_spear_1", "side": 1, "key": spear, "x": -100.0, "z": 0.0, "b": 90.0, "men": 100, "hp": 1.0},
            {"n": "enemy_spear_1", "side": 2, "key": spear, "x": 100.0, "z": 0.0, "b": 270.0, "men": 100, "hp": 1.0}]
    doc = {"batch": "b-v", "move": 1, "t": 1000, "attacker": 2, "units": rows}
    side.decide(doc, {"own_spear_1": {"unit": "own_spear_1", "kind": "attack", "target": "enemy_spear_1", "run": True}})
    u = side.st.u
    s_own, s_en = side.slot[0], side.slot[1]
    assert int(u["order_kind"][0, s_own]) == O.ATTACK and int(u["order_target"][0, s_own]) == s_en
    assert float(u["ox"][0, s_own]) == 100.0                                 # an attack's point: its target
    assert int(u["order_kind"][0, s_en]) == O.HOLD                           # the script's own: none yet
    doc2 = dict(doc, move=2, t=2000, units=[dict(rows[0], x=-96.0), dict(rows[1], r=True, x=104.0)])
    side.decide(doc2, {})
    assert abs(float(u["vx"][0, s_own]) - 4.0) < 1e-4 and bool(u["r"][0, s_en])
    assert float(u["rout_s"][0, s_en]) == 1.0


def test_the_loop_answers_both_sides_each_move_once(tmp_path):
    import json
    import threading

    from tools.nn.companion import loop, policy
    lord, spear = "wh_main_emp_cha_general_0", "wh_main_emp_inf_spearmen_0"
    rows = [{"n": n, "side": sd, "key": k, "x": x, "z": 0.0, "b": 90.0 if sd == 1 else 270.0, "men": 100, "hp": 1.0,
             "v": True} for n, sd, k, x in (("own_lord", 1, lord, -235.0), ("own_spear_1", 1, spear, -175.0),
                                           ("enemy_lord", 2, lord, 235.0), ("enemy_spear_1", 2, spear, 175.0))]
    doc = {"batch": "b-2", "move": 1, "t": 1000, "done": False, "attacker": 2, "decide_ms": 1000, "units": rows,
           "factions": {"own": "wh_main_emp_empire", "enemy": "wh_main_emp_empire"}}
    lines = []
    worker = threading.Thread(target=loop.run, args=(tmp_path, loop.Brain(policy.fresh())), kwargs={
        "exit_on_done": True, "out": lines.append, "idle_exit_s": 20, "log": tmp_path / "c.jsonl",
        "enemy": ScriptSide("ai_like"), "enemy_log": tmp_path / "c_enemy.jsonl"})
    worker.start()
    exchange.write_atomic(tmp_path / exchange.STATE, json.dumps(doc))
    exchange.write_atomic(tmp_path / exchange.ENEMY_STATE, json.dumps(dict(doc, control=2)))
    for _ in range(4000):
        if (tmp_path / exchange.ORDERS).exists() and (tmp_path / exchange.ENEMY_ORDERS).exists():
            break
        threading.Event().wait(0.005)
    ours = exchange.parse_orders((tmp_path / exchange.ORDERS).read_text(encoding="utf-8"))
    theirs = exchange.parse_orders((tmp_path / exchange.ENEMY_ORDERS).read_text(encoding="utf-8"))
    assert set(ours["orders"]) == {"own_lord", "own_spear_1"}
    assert set(theirs["orders"]) == {"enemy_lord", "enemy_spear_1"} and theirs["move"] == 1
    assert theirs["orders"]["enemy_spear_1"]["kind"] == "move"          # the attacker's line marches
    exchange.write_atomic(tmp_path / exchange.STATE, json.dumps(dict(doc, move=2, done=True)))
    worker.join(10)
    assert not worker.is_alive()
    assert [ln[:6] for ln in lines if "move" in ln[:12]] == ["move  ", "enemy "]
    assert [json.loads(ln)["move"] for ln in (tmp_path / "c_enemy.jsonl").read_text().splitlines()] == [1]
