"""tools.nn.train.drills: the drills framework (docs/en/training/training.md "Drills").

torch is not in the project's .venv: skipped there, run in the training container. Level 1: the
battle helpers (turn, mirror, the drills' share of the mix). Level 2: the mixed bank gives a drill row
a battle of its drill with our side = the learner's. Level 3: the rollout plays a drill row with the
drill's enemy script, under the standard battle limit.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import scenario  # noqa: E402
from tools.nn.train import drills as D  # noqa: E402
from tools.nn.train import league, rollout  # noqa: E402
from tools.nn.train.drills import source as ds  # noqa: E402

SPEAR = "wh_main_emp_inf_spearmen_0"


def tiny(rng=None):
    ours = [D.unit(SPEAR, *D.local(-60, lat), 0.0, 30) for lat in (-20, 20)]
    enemy = [D.unit(SPEAR, *D.local(60, 0), 180.0, 30)]
    return D.army(ours, enemy, attacker=1, ours_faction="wh_main_emp_empire",
                  enemy_faction="wh_main_emp_empire")


TINY = D.Drill("tiny", tiny, D.nearest, D.hold, D.nearest)


def test_transform_keeps_the_battle_and_turns_bearings():
    rng = np.random.default_rng(3)
    a = tiny()
    d0 = math.dist((a["sides"][1]["units"][0]["x"], a["sides"][1]["units"][0]["z"]),
                   (a["sides"][2]["units"][0]["x"], a["sides"][2]["units"][0]["z"]))
    b0 = a["sides"][2]["units"][0]["b"] - a["sides"][1]["units"][0]["b"]
    D.transform(a, rng)
    d1 = math.dist((a["sides"][1]["units"][0]["x"], a["sides"][1]["units"][0]["z"]),
                   (a["sides"][2]["units"][0]["x"], a["sides"][2]["units"][0]["z"]))
    assert d1 == pytest.approx(d0, abs=1e-6)
    assert (a["sides"][2]["units"][0]["b"] - a["sides"][1]["units"][0]["b"]) % 360 == pytest.approx(b0 % 360)
    for s in (1, 2):
        for u in a["sides"][s]["units"]:
            assert abs(u["x"]) <= D.MAP_HALF_M and abs(u["z"]) <= D.MAP_HALF_M
    # still facing each other: the enemy is ahead of our unit
    o, e = a["sides"][1]["units"][0], a["sides"][2]["units"][0]
    br = math.radians(o["b"])
    assert (e["x"] - o["x"]) * math.sin(br) + (e["z"] - o["z"]) * math.cos(br) > 0


def test_battles_alternate_our_side_and_mirror():
    out = D.battles(TINY, range(4))
    assert [s for _, s in out] == [1, 2, 1, 2]
    a, _ = out[1]
    assert len(a["sides"][2]["units"]) == 2 and a["attacker"] == 2      # ours (2 units, attacking) on side 2


def test_with_drills_shares():
    mix = league.with_drills({"nearest": 0.5, "ai_like": 0.5}, 0.1, {"pincer": 1, "kiting": 3})
    assert sum(mix.values()) == pytest.approx(1.0)
    assert mix[D.opponent("pincer")] == pytest.approx(0.025) and mix[D.opponent("kiting")] == pytest.approx(0.075)
    assert mix["nearest"] == pytest.approx(0.45)
    assert league.with_drills({"nearest": 1.0}, 0.0) == {"nearest": 1.0}


def test_mixed_bank_gives_drill_rows_their_drill_and_side(monkeypatch):
    monkeypatch.setattr(D, "NAMES", ("tiny",))
    monkeypatch.setitem(league.CODE, D.opponent("tiny"), 99)
    lay = league.Layout(np.zeros(8, dtype=int), np.array([1, 2] * 4), np.array([99] * 4 + [league.CODE["nearest"]] * 4))
    src = ds.Mixed(np.arange(4), 3, None, "cpu", lay, per_drill=6, drills={"tiny": TINY})
    idx = src.pick()
    for b in range(4):
        row = int(idx[b])
        assert int(src.group[row]) == 99 and int(src.ours[row]) == lay.learner[b]
    assert all(int(src.group[int(idx[b])]) == 0 for b in range(4, 8))
    assert "limit_s" not in src.bank.state.u                     # no drill time limit: the standard one


def test_rollout_plays_a_drill_row_with_its_enemy_script(monkeypatch):
    monkeypatch.setattr(D, "NAMES", ("tiny",))
    monkeypatch.setitem(league.CODE, D.opponent("tiny"), 99)
    monkeypatch.setattr(D, "load", lambda names=None: {"tiny": TINY})
    lay = league.Layout(np.zeros(2, dtype=int), np.array([1, 2]), np.array([99, 99]))
    src = ds.Mixed([], 0, None, "cpu", lay, per_drill=4, drills={"tiny": TINY}, sequential=True)
    env = rollout.Battles(lay, device="cpu", source=src, compile=False, auto_reset=False)
    assert 99 in env.scripts
    # the drill's enemy (nearest) attacks; our side holds: the enemy's units take ATTACK orders
    o = rollout.assemble_orders(env.st, env.ctrl, tuple(env.scripts.items()), ())
    enemy = env.st.u["side"] == (3 - torch.as_tensor(lay.learner))[:, None]
    present = enemy & (env.st.u["men"] > 0)
    assert bool((o.kind[present] == 2).all())
