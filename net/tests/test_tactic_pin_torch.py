"""Drill T0-2 'pin and strike' - the drill's measure test (tmp/tactic_slice/design_flank_v2.md section 8): each check
counts what its name says.

    the gate        over the pilot's 768 general battles, the K player's first frame through the real level
                    (battle.TacticSide: lower.Tacticians, tactic/zfit zone) is within reissue.MOVE_M and FACE_DEG of
                    each unit's deployed place - 'keep' = 'hold where deployed'
    the walk        on a hand-built tick: pin.pinned, orders.Facts.free, the teacher's cand(i) (one striker a side, the
                    nearest of its side class), the code's masks (no new order to a unit in melee; the cap of two and one
                    a side) and the counts they report
"""
import math

import pytest

torch = pytest.importorskip("torch")

from net.common import reissue
from net.tactic import decide as DC
from net.tactic import orders as OR
from net.tactic import pin as PN
from net.tactic import raw as TR
from net.tactic import rows as RW
from net.tactic import zone as ZN
from net.unit import scripts as SC


def test_first_frames_hold_where_deployed():
    from net.matchup import duel as DU
    from net.stage0 import battle as BA
    from net.tactic.exam_pin import pilot_specs
    from sim.params import load
    P = load()
    bad, n = [], 0
    for seed in range(3):
        sp = pilot_specs(256, seed)
        for c0 in range(0, len(sp), 128):
            part = sp[c0:c0 + 128]
            st, card = DU.build([BA.NAME] * len(part), BA.unit_battles(part), "cpu", P, per_side=BA.PER_SIDE,
                                maps=[s["map"] for s in part])
            side = BA.TacticSide(PN.player("K"))
            side.start(st, P, "cpu", card, None)
            side.tick(st, TR.Seer(P).raw(st))
            w = side.tac.areas()
            u = st.u
            d = torch.sqrt((w["x"][:, :] - u["x"]) ** 2 + (w["z"] - u["z"]) ** 2)
            off = (torch.remainder(w["b"] - u["b"] + 180.0, 360.0) - 180.0).abs()
            mine = side.mine
            assert bool((w["on"] == mine).all()), "every unit of mine has a first frame"
            over = mine & ((d > reissue.MOVE_M + 1e-4) | (off > reissue.FACE_DEG))
            n += int(mine.sum())
            bad += [(seed, c0 + int(b), int(i), round(float(d[b, i]), 2)) for b, i in over.nonzero().tolist()]
    assert n > 3000
    assert not bad, bad[:5]


def _tick(units):
    """A hand-built tick of one group: units [(who, s, f, b, front, depth, m, r, speed)] in slot order (who 'mine' /
    'enemy'); orders in force: hold, no target. Returns the orders.View."""
    K = len(units)
    rows = torch.zeros((1, K, len(RW.ROW)))
    RI = RW.RI
    who = torch.tensor([[RW.OWN_GROUP if u[0] == "mine" else RW.ENEMY_OTHER for u in units]])
    for k, (_, s, f, b, fr, dp, m, r, v) in enumerate(units):
        rows[0, k, RI["right"]], rows[0, k, RI["forward"]] = s / 100.0, f / 100.0
        rows[0, k, RI["front_cos"]], rows[0, k, RI["front_sin"]] = math.cos(math.radians(b)), math.sin(math.radians(b))
        rows[0, k, RI["front"]], rows[0, k, RI["depth"]] = fr / 50.0, dp / 50.0
        rows[0, k, RI["m"]], rows[0, k, RI["r"]] = float(m), float(r)
        rows[0, k, RI["v_forward"]] = v / 5.0
    mine = [k for k, u in enumerate(units) if u[0] == "mine"]
    G = len(mine)
    rr = {"rows": rows, "mask": torch.ones((1, K), dtype=torch.bool), "slot": torch.arange(K)[None], "who": who}
    mine_raw = torch.zeros((1, G, len(TR.RAW)))
    for g, k in enumerate(mine):
        mine_raw[0, g, TR.I["x"]], mine_raw[0, g, TR.I["z"]] = units[k][1], units[k][2]
    tiles = ZN.Tiles(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.tensor([1]), torch.tensor([G]))
    able = torch.tensor([[not units[k][7] for k in mine]])
    ctx = DC.Ctx(tiles, torch.tensor([mine]), able, torch.tensor([mine]), mine_raw, None)
    order = {"skill": torch.zeros((1, G), dtype=torch.long), "tgt": torch.full((1, G), -1),
             "side": torch.full((1, G), -1), "born": torch.zeros((1, G))}
    fx = OR.Facts(rr, ctx, order)
    z = torch.zeros((1, G), dtype=torch.bool)
    return OR.View(rr, ctx, tiles, fx, order, z, fx.g_m.clone(), fx.g_m.clone(), {"first": z, "hole": z, "zone": z},
                   0.0, torch.tensor([0]))


def test_teacher_walk_and_masks():
    # slot 0 the anvil in melee with enemy slot 5 (it faces south, at f 12); slots 1, 2 free on its left (60, 90 m), 3 on
    # its right (70 m), 4 free but an enemy (slot 6) within 8 m of it; enemy 6 not pinned (no melee), enemy 7 routs
    units = [("mine", 0.0, 0.0, 0.0, 32, 10, 1, 0, 0.0),
             ("mine", -60.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", -90.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", 70.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", 0.0, -150.0, 0.0, 32, 10, 0, 0, 0.0),
             ("enemy", 0.0, 12.0, 180.0, 32, 10, 1, 0, 0.0),
             ("enemy", 0.0, -138.0, 180.0, 32, 10, 0, 0, 0.0),
             ("enemy", 30.0, 60.0, 180.0, 32, 10, 0, 1, 0.0)]
    v = _tick(units)
    pin = PN.pinned(v.fx)
    assert pin[0].tolist() == [False] * 5 + [True, False, False]
    assert v.fx.free[0].tolist() == [False, True, True, True, False]
    # the enemy faces south: my unit at s -60 stands on its right side by its own facing
    assert v.fx.g_side[0, 1, 5] == OR.RIGHT and v.fx.g_side[0, 3, 5] == OR.LEFT
    out = PN.player("T")(v)
    assert out["skill"][0].tolist() == [SC.HOLD, SC.ATTACK, SC.HOLD, SC.ATTACK, SC.HOLD]
    assert out["target"][0].tolist() == [-1, 5, -1, 5, -1]
    skill, tgt, side, change, cnt = OR.masks(v, out)
    assert skill.tolist() == out["skill"].tolist() and change[0].tolist() == [False, True, False, True, False]
    assert float(cnt["rejected"].sum()) == 0 and float(cnt["melee_changes"].sum()) == 0
    # a player asking the anvil (in melee) to move on, and a third striker on enemy 5: both masked, counted
    ask = {**out, "skill": out["skill"].clone(), "target": out["target"].clone()}
    ask["skill"][0, 0], ask["target"][0, 0] = SC.ATTACK, 6
    ask["skill"][0, 2], ask["target"][0, 2] = SC.ATTACK, 5
    skill, tgt, side, change, cnt = OR.masks(v, ask)
    assert tgt[0].tolist() == [-1, 5, -1, 5, -1]
    assert float(cnt["melee_blocked"].sum()) == 1 and float(cnt["rejected"].sum()) == 1
    assert float(cnt["melee_changes"].sum()) == 0
    # D: each free unit at the nearest valid open enemy within reach (enemy 6 is 160-173 m from units 1-3; unit 4 is
    # not free, enemy 6 within 8 m of it): 1 and 3 at enemy 5, unit 2 finds its side taken
    out = PN.player("D")(v)
    assert out["target"][0].tolist() == [-1, 5, -1, 5, -1]


def test_second_enemy_walk():
    # cause (b) 'take the second one': slot 0 the anvil in melee with enemy 4 (pinned); enemy 5 runs south at the line
    # (80 m out), enemy 6 runs at unit 2 (25 m from it: inside its own counter's reach); free units 1 (-60, 10), 2
    # (60, 10), 3 (-40, 100); enemy 7 routs
    units = [("mine", 0.0, 0.0, 0.0, 32, 10, 1, 0, 0.0),
             ("mine", -60.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", 60.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", -40.0, 100.0, 0.0, 32, 10, 0, 0, 0.0),
             ("enemy", 0.0, 12.0, 180.0, 32, 10, 1, 0, 0.0),
             ("enemy", 0.0, 80.0, 180.0, 32, 10, 0, 0, -3.0),
             ("enemy", 60.0, 45.0, 180.0, 32, 10, 0, 0, -3.0),
             ("enemy", -40.0, 140.0, 180.0, 32, 10, 0, 1, -3.0)]
    v = _tick(units)
    fx = v.fx
    cm = OR.comes(fx)[0]
    assert cm[0, 5] and cm[2, 6] and not cm[0, 4] and not cm[3, 5], "the rule of coming on the level's rows"
    e, u = PN.second(fx)
    # 5 and 6 both come at the anvil in melee; 4 is in melee, 7 routs
    assert e[0].tolist() == [False] * 5 + [True, True, False]
    gq = [float(fx.gap[0, k, 5]) for k in (0, 2)]
    assert int(u[0, 5]) == (0 if gq[0] <= gq[1] else 2)     # the nearest unit of mine 5 is second at
    assert PN.come_own(fx)[0].tolist() == [False, False, True, False]
    # the first tick (no pin yet under the two-tick wait): unit 3 is the nearest interceptor to 5; unit 1 is not the
    # nearest to 5 or 6 (unit 3 is); unit 2 meets its own (6) under counter; one interceptor an enemy
    p = PN.Teacher(default="config", wait2=True, second_m=150.0)
    out = p(v)
    assert out["target"][0].tolist() == [-1, -1, -1, 5]
    assert out["skill"][0].tolist() == [SC.COUNTER, SC.COUNTER, SC.COUNTER, SC.ATTACK]
    assert p.last_tags["cause"][0].tolist() == [0, 0, 0, 2] and int(p.last_tags["u"][0, 3]) == int(u[0, 5])
    assert p.last_tags["cell"][0, 3] // 5 != 0, "5 is not pinned"
    skill, tgt, side, change, cnt = OR.masks(v, out, p.allow)
    assert tgt[0].tolist() == [-1, -1, -1, 5] and float(cnt["rejected"].sum()) == 0
    assert float(cnt["melee_blocked"].sum()) == 1                   # (the anvil's counter waits: no order in melee)
    # out of the interceptor's reach: nothing
    assert PN.Teacher(default="config", wait2=True, second_m=30.0)(v)["target"][0].tolist() == [-1] * 4
    # the second tick: 4 pinned on two ticks - a pinned target first gives units 1 and 2 (one a side) 4, unit 3 still 5
    p = PN.Teacher(default="config", wait2=True, second_m=150.0, first="pin")
    p(v)
    out = p(v)
    assert out["target"][0].tolist() == [-1, 4, 4, 5]
    assert p.last_tags["cause"][0].tolist() == [0, 1, 1, 2]
    # a second enemy first: unit 1 is still not the nearest to 5 or 6, so the same
    p = PN.Teacher(default="config", wait2=True, second_m=150.0, first="second")
    p(v)
    assert p(v)["target"][0].tolist() == [-1, 4, 4, 5]
    # DH / DP: D restricted to enemies that come at one of my units / at none (4 pinned comes at nobody); in DH unit 3
    # finds 5's side on it taken by unit 1 and goes at 6
    assert PN.player("DH")(v)["target"][0].tolist() == [-1, 5, 6, 6]
    assert PN.player("DP")(v)["target"][0].tolist() == [-1, 4, 4, -1]
