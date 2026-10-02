"""tools.nn.train.drills.defend: the drill "defend" (docs/en/training/training.md "Drills").

torch is not in the project's .venv: these tests are skipped there and run in the training
container.

Level 2: the frame's battles satisfy the drill's condition (we defend with a melee line and
missile units behind it, the attacker has more melee units, the gap, the limit, known units);
the scripts give valid orders for every unit of a few generated battles and of an ordinary
battle with lords, and a short run of the battles goes without errors.
"""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.train import drills as D  # noqa: E402
from tools.nn.train.drills import defend  # noqa: E402

UNITS = json.load(open("config/nn/units.json", encoding="utf-8"))
UNITS = UNITS.get("units", UNITS)
POOLS = json.load(open("config/nn/pools.json", encoding="utf-8"))["factions"]


def is_missile(key):
    return bool(UNITS[key].get("missile"))


def test_frame_condition():
    for s in range(200):
        desc = defend.frame(np.random.default_rng(s))
        assert desc["attacker"] == 2
        ours, enemy = desc["sides"][1]["units"], desc["sides"][2]["units"]
        for side in (1, 2):
            fac = desc["sides"][side]["faction"]
            pool = {u["key"] for u in POOLS[fac]["units"]}
            for u in desc["sides"][side]["units"]:
                assert u["key"] in UNITS and u["key"] in pool
                assert not u["general"]
        o_line = [u for u in ours if not is_missile(u["key"])]
        o_mis = [u for u in ours if is_missile(u["key"])]
        e_line = [u for u in enemy if not is_missile(u["key"])]
        e_mis = [u for u in enemy if is_missile(u["key"])]
        assert defend.LINE_N[0] <= len(o_line) <= defend.LINE_N[1]
        assert defend.MISSILE_N[0] <= len(o_mis) <= defend.MISSILE_N[1]
        assert len(e_line) > len(o_line) and len(e_mis) <= 1
        assert all(not UNITS[u["key"]]["missile"]["direct"] for u in o_mis)   # they shoot over the line
        # the missile row is behind our line; the lines face each other GAP_M apart
        line_z = np.mean([u["z"] for u in o_line])
        assert all(u["z"] < line_z - defend.BACK_M[0] + 1e-6 for u in o_mis)
        gap = np.mean([u["z"] for u in e_line]) - line_z
        assert defend.GAP_M[0] - 1e-6 <= gap <= defend.GAP_M[1] + 1e-6
        assert all(abs(u["b"]) < 1e-6 for u in ours) and all(abs(u["b"] - 180) < 1e-6 for u in enemy)


def test_battles_both_sides_on_the_map():
    pairs = D.battles(defend.DRILL, range(40))
    assert {p[1] for p in pairs} == {1, 2}
    for desc, ours in pairs:
        assert desc["attacker"] == 3 - ours          # we always defend
        for s in (1, 2):
            for u in desc["sides"][s]["units"]:
                assert math.hypot(u["x"], u["z"]) <= D.MAP_HALF_M * math.sqrt(2) + 1e-6


def check(st, orders):
    O.check(orders, st.N)
    for k in O.FIELDS:
        assert getattr(orders, k).shape == (st.B, st.N)


def test_scripts_run():
    pairs = D.battles(defend.DRILL, range(6))
    descs = [p[0] for p in pairs]
    ours = torch.tensor([p[1] for p in pairs])
    st = scenario.build(descs)
    params = load()
    for script in (defend.naive, defend.skilled):
        for _ in range(3):
            check(st, script(st))
            check(st, defend.enemy(st))
    for _ in range(40):                              # 40 steps with the skilled script on our side
        battle.step(st, D.merged(st, ours, defend.skilled(st), defend.enemy(st)), params, params.dt)
    assert torch.isfinite(st.u["x"]).all()


def test_scripts_on_an_ordinary_battle_with_lords():
    def side(f, sign):
        units = [D.unit(POOLS[f]["lord"]["key"], 0, sign * 200, 0 if sign < 0 else 180, general=True)]
        for i, u in enumerate(POOLS[f]["units"]):
            units.append(D.unit(u["key"], (i - 3) * 40, sign * 160, 0 if sign < 0 else 180, u["width"]))
        return units
    emp, skv = defend.EMPIRE, defend.SKAVEN
    descs = [D.army(side(emp, -1), side(skv, 1), 1, emp, skv), D.army(side(skv, -1), side(emp, 1), 2, skv, emp)]
    st = scenario.build(descs)
    params = load()
    for _ in range(20):
        for script in (defend.naive, defend.skilled, defend.enemy):
            check(st, script(st))
        battle.step(st, defend.enemy(st), params, params.dt)
