"""Drill T0-2 'pin and strike' - the drill's measure test (tmp/tactic_slice/design_flank_v2.md section 8): each check
counts what its name says.

    the gate        over the pilot's 768 general battles, the K player's first frame through the real level
                    (battle.TacticSide: lower.Tacticians, tactic/zfit zone) is within reissue.MOVE_M and FACE_DEG of
                    each unit's deployed place - 'keep' = 'hold where deployed'
    the walk        on a hand-built tick: pin.pinned, orders.Facts.free, the teacher's cand(i) (one striker a side, the
                    nearest of its side class), the code's masks (no new order to a unit in melee; the cap of two and one
                    a side) and the counts they report
    the labeller    on hand-built ticks: TEACH's options - pinned at touch 8 m, come_own on enemies out of melee and with
                    a reach as a number, the way through a unit of mine (pin.through_own) - each changes the label it
                    names and nothing else
    the scenes      tactic/scenes' yield on a small draw: A1 scenes hold a teacher's order, N1 twins that run past none;
                    every other recipe draws within its bands, at most 8 units a side
    the pointer     on a hand-built tick (tactic/heads.py): head-free units, the walk fed with the teacher's choice
                    (a side taken closes the key for the next unit of that side; holders in the pair numbers), a fresh
                    pointer keeps, the loss over {null} + open keys less don't-care keys (a label on one: weight 0),
                    orders.own() with the side and the age zeroed without a target and the age capped
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


def _tick(units, skill=None):
    """A hand-built tick of one group: units [(who, s, f, b, front, depth, m, r, speed)] in slot order (who 'mine' /
    'enemy'); orders in force: hold (or skill [G]), no target. Returns the orders.View."""
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
    order = {"skill": torch.zeros((1, G), dtype=torch.long) if skill is None else torch.tensor([skill]),
             "tgt": torch.full((1, G), -1),
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
    # the branch test's two masks (pin.Teacher.shut, tactic/replay.py): unit 1 closed to orders - the next free unit of
    # its side (2) takes 5; enemy 5 closed as a target - no striker at all
    p = PN.player("T")
    shut = lambda: {"units": torch.zeros((1, 8), dtype=torch.bool), "targets": torch.zeros((1, 8), dtype=torch.bool)}  # noqa
    p.shut = shut()
    p.shut["units"][0, 1] = True
    assert p(v)["target"][0].tolist() == [-1, -1, 5, 5, -1]
    p.shut = shut()
    p.shut["targets"][0, 5] = True
    assert p(v)["target"][0].tolist() == [-1] * 5
    # D: each free unit at the nearest valid open enemy within orders.reach_m (enemy 6 is 160-173 m from units 1-3;
    # unit 4 is not free, enemy 6 within 8 m of it): 1 and 3 at enemy 5, unit 2 finds 5's side taken and takes enemy 6
    # when it is within the reach (173.2 m centre to centre: inside 240, outside 150)
    out = PN.player("D")(v)
    two = 6 if float(OR.c_orders()["reach_m"]) >= math.hypot(90.0, 148.0) else -1
    assert out["target"][0].tolist() == [-1, 5, two, 5, -1]


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
    # the tight trigger (u in melee only): the same here - 5 and 6 come at the anvil in melee
    assert PN.second(fx, tight=True)[0][0].tolist() == e[0].tolist()
    p = PN.Teacher(default="config", wait2=True, second_m=150.0, tight=True)
    assert p(v)["target"][0].tolist() == [-1, -1, -1, 5]
    # before any melee (the anvil and 4 not fighting): the loose trigger still finds 5 second at unit 2 (6 nearer to
    # it), the tight one finds no second enemy and gives no order
    calm = [(w, s, f, b, fr, dp, 0, r, sp) for (w, s, f, b, fr, dp, m, r, sp) in units]
    vc = _tick(calm)
    assert PN.second(vc.fx)[0][0, 5] and not PN.second(vc.fx, tight=True)[0].any()
    assert PN.Teacher(default="config", wait2=True, second_m=150.0)(vc)["target"][0].tolist() == [-1, -1, -1, 5]
    assert PN.Teacher(default="config", wait2=True, second_m=150.0, tight=True)(vc)["target"][0].tolist() == [-1] * 4
    # DH / DP: D restricted to enemies that come at one of my units / at none (4 pinned comes at nobody); in DH unit 3
    # finds 5's side on it taken by unit 1 and goes at 6
    assert PN.player("DH")(v)["target"][0].tolist() == [-1, 5, 6, 6]
    assert PN.player("DP")(v)["target"][0].tolist() == [-1, 4, 4, -1]


def test_stage1e_options():
    # the scene of test_second_enemy_walk: anvil 0 in melee with 4 (pinned); 5 runs at the line, 6 at unit 2 (inside its
    # counter's reach); free units 1, 2, 3
    units = [("mine", 0.0, 0.0, 0.0, 32, 10, 1, 0, 0.0),
             ("mine", -60.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", 60.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", -40.0, 100.0, 0.0, 32, 10, 0, 0, 0.0),
             ("enemy", 0.0, 12.0, 180.0, 32, 10, 1, 0, 0.0),
             ("enemy", 0.0, 80.0, 180.0, 32, 10, 0, 0, -3.0),
             ("enemy", 60.0, 45.0, 180.0, 32, 10, 0, 0, -3.0),
             ("enemy", -40.0, 140.0, 180.0, 32, 10, 0, 1, -3.0)]
    v = _tick(units)
    # ~come_own on cause (a): unit 2 (6 comes at it within its counter's reach) no longer leaves for the pinned 4
    p = PN.Teacher(default="config", wait2=True, co=True)
    p(v)
    assert p(v)["target"][0].tolist() == [-1, 4, -1, -1]
    p = PN.Teacher(default="config", wait2=True)
    p(v)
    assert p(v)["target"][0].tolist() == [-1, 4, 4, -1]
    # the two-tick wait read off the rows' ticks in melee: 4 two ticks in melee -> struck at once, one tick -> not
    v.fx.mt = torch.tensor([[0, 0, 0, 0, 2, 0, 0, 0]])
    assert PN.Teacher(default="config", wait2=True, wait_mt=True)(v)["target"][0].tolist() == [-1, 4, 4, -1]
    v.fx.mt = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0]])
    assert PN.Teacher(default="config", wait2=True, wait_mt=True)(v)["target"][0].tolist() == [-1] * 4
    # TIu: 5's and 6's nearest unit in their path is not the one fighting - no second enemy (TI's tight trigger has two)
    assert not PN.second(v.fx, nearest_m=150.0)[0].any() and PN.second(v.fx, tight=True)[0][0, 5:7].all()
    # back to the default: an attack that ended (attack, no target) gets counter with back, stays without it
    vb = _tick(units, skill=[SC.HOLD, SC.ATTACK, SC.ATTACK, SC.HOLD])
    out = PN.Teacher(default="config", wait2=True, back=True)(vb)
    assert out["skill"][0].tolist() == [SC.COUNTER] * 4
    out = PN.Teacher(default="config", wait2=True)(vb)
    assert out["skill"][0].tolist() == [SC.COUNTER, SC.ATTACK, SC.ATTACK, SC.COUNTER]
    assert out["target"][0].tolist() == [-1] * 4


def test_nearest_unit_trigger():
    # anvil 0 in melee with 3 (pinned, f 12); 4 runs at the anvil from 60 m - the anvil is the nearest unit in its path;
    # free units 1, 2 80 m to the sides
    units = [("mine", 0.0, 0.0, 0.0, 32, 10, 1, 0, 0.0),
             ("mine", -80.0, 0.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", 80.0, 0.0, 0.0, 32, 10, 0, 0, 0.0),
             ("enemy", 0.0, 12.0, 180.0, 32, 10, 1, 0, 0.0),
             ("enemy", 0.0, 60.0, 180.0, 32, 10, 0, 0, -3.0)]
    v = _tick(units)
    e, u = PN.second(v.fx, nearest_m=150.0)
    assert e[0].tolist() == [False, False, False, False, True] and int(u[0, 4]) == 0
    assert not PN.second(v.fx, nearest_m=40.0)[0].any(), "gap(e, u) within second_m"
    out = PN.player("TIu")(v)
    assert out["target"][0].tolist() == [-1, 4, -1]           # one interceptor (the first of two equally near)
    # the masks on cause (b): unit 1 closed - it keeps its order (the default fill still applies), unit 2 takes 4; enemy 4
    # closed - nobody
    p = PN.player("TIu")
    p.shut = {"units": torch.tensor([[False, True, False, False, False]]), "targets": torch.zeros((1, 5), dtype=torch.bool)}
    out = p(v)
    assert out["target"][0].tolist() == [-1, -1, 4] and int(out["skill"][0, 1]) == SC.COUNTER
    p = PN.player("TIu")                                       # (fresh: a second call would see 3 pinned on two ticks)
    p.shut = {"units": torch.zeros((1, 5), dtype=torch.bool), "targets": torch.tensor([[False, False, False, False, True]])}
    assert p(v)["target"][0].tolist() == [-1, -1, -1]
    t = PN.player("TEACH")                                     # the config's teacher: 'strike the pinned one' (TCw)
    assert t.second_m is None and t.wait2 and t.back and t.co and t.wait_mt   # with back and co; no cause (b)
    # the labeller's options read by TEACH only (config orders.teach): touch 8 m, come_own on enemies out of melee and
    # within 150 m (the game probe own, review J); no through rule (the probe behind: a don't-care tag instead)
    assert t.touch == 8.0 and t.come_free and not t.no_through and t.come_reach == 150.0
    assert PN.player("TCw_back_co").touch is None and not PN.player("TCw_back_co").come_free


def test_labeller_options():
    # anvil 0 in melee with 3 (f 16: edge gap 6 m - pinned at 8 m, not at 4); free units 1 (-60, 10) and 2 (60, 10) on
    # its two sides; enemy 4 fights somewhere (its m on) and runs at unit 2 (edge gap 20 m: within its counter's reach)
    units = [("mine", 0.0, 0.0, 0.0, 32, 10, 1, 0, 0.0),
             ("mine", -60.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", 60.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("enemy", 0.0, 16.0, 180.0, 32, 10, 1, 0, 0.0),
             ("enemy", 60.0, 40.0, 180.0, 32, 10, 1, 0, -2.0)]
    v = _tick(units)
    v.fx.mt = torch.tensor([[1, 0, 0, 2, 2]])
    assert PN.pinned(v.fx)[0].tolist() == [False] * 5, "4 m: the 6 m gap does not touch"
    assert PN.pinned(v.fx, touch=8.0)[0].tolist() == [False, False, False, True, False], "8 m: 3 pinned; 4 runs"
    T = lambda **k: PN.Teacher(default="config", wait2=True, wait_mt=True, co=True, **k)       # noqa: E731
    assert T()(v)["target"][0].tolist() == [-1, -1, -1]
    assert T(touch=8.0)(v)["target"][0].tolist() == [-1, 3, -1], "unit 2 meets the fighting 4 coming at it"
    # come_own on enemies out of melee only: 4 fights, so unit 2 is free for the pinned 3 too
    assert PN.come_own(v.fx)[0].tolist() == [False, False, True]
    assert PN.come_own(v.fx, free=True)[0].tolist() == [False, False, False]
    assert T(touch=8.0, come_free=True)(v)["target"][0].tolist() == [-1, 3, 3]
    # come_own's reach as a number: an enemy (5) out of melee runs at unit 1 from an edge gap of 55 m - beyond the
    # counter's 30 m (the passport's run is 0 here), within 60 m
    v = _tick(units + [("enemy", -60.0, 75.0, 180.0, 32, 10, 0, 0, -2.0)])
    v.fx.mt = torch.tensor([[1, 0, 0, 2, 2, 0]])
    assert PN.come_own(v.fx, free=True)[0].tolist() == [False, False, False]
    assert PN.come_own(v.fx, free=True, reach_m=60.0)[0].tolist() == [False, True, False]
    assert T(touch=8.0, come_free=True, come_reach=60.0)(v)["target"][0].tolist() == [-1, -1, 3]
    # the way through a unit of mine: unit 1 straight behind the anvil (its way to 3's centre crosses the anvil's
    # rectangle before 3's), unit 2 beside (its way meets 3's rectangle first)
    units = [("mine", 0.0, 0.0, 0.0, 32, 10, 1, 0, 0.0),
             ("mine", 0.0, -60.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", 60.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("enemy", 0.0, 12.0, 180.0, 32, 10, 1, 0, 0.0)]
    v = _tick(units)
    v.fx.mt = torch.tensor([[1, 0, 0, 2]])
    thr = PN.through_own(v.fx)[0]
    assert bool(thr[1, 3]) and not bool(thr[2, 3]) and not bool(thr[0, 3])
    assert T()(v)["target"][0].tolist() == [-1, 3, 3]
    assert T(no_through=True)(v)["target"][0].tolist() == [-1, -1, 3], "unit 1 is not free for 3: taught keep"
    # the tag through_own: crossing and under 15 m across the anvil's facing - straight behind tagged; 20 m aside
    # (its way still crosses the anvil's 32 m front) not tagged
    assert bool(PN.through_tag(v.fx, 15.0)[0, 1, 3]) and not bool(PN.through_tag(v.fx, 15.0)[0, 2, 3])
    v = _tick([units[0], ("mine", 20.0, -60.0, 0.0, 32, 10, 0, 0, 0.0)] + units[2:])
    assert bool(PN.through_own(v.fx)[0, 1, 3]) and not bool(PN.through_tag(v.fx, 15.0)[0, 1, 3])


def test_ticks_in_melee():
    from net.matchup import duel as DU
    from net.stage0 import battle as BA
    from net.tactic.exam_pin import pilot_specs
    from sim.params import load
    P = load()
    sp = pilot_specs(4, 0)[:2]
    st, card = DU.build([BA.NAME] * len(sp), BA.unit_battles(sp), "cpu", P, per_side=BA.PER_SIDE,
                        maps=[s["map"] for s in sp])
    side = BA.TacticSide(PN.player("K"))
    side.start(st, P, "cpu", card, None)
    raw = TR.Seer(P).raw(st)
    side.tick(st, raw)
    tac = side.tac
    assert int(tac.mt.abs().sum()) == 0, "0 at tick 0 (nobody in melee)"
    j = int(side.mine[0].nonzero()[0, 0])                       # a unit of mine (a row of its group)
    lord = int((st.u["lord"][0] & (st.u["side"][0] == 1)).nonzero()[0, 0])
    m = raw.clone()
    m[0, j, TR.I["m"]] = 1.0
    m[0, lord, TR.I["m"]] = 1.0                                 # (the lord has no row: hidden)
    for k in (1, 2):
        side.tick(st, m)
        assert int(tac.mt[0, j]) == k and int(tac.mt[0, lord]) == 0
    side.tick(st, raw)
    assert int(tac.mt[0, j]) == 0, "m off -> 0"
    side.tick(st, m)
    assert int(tac.mt[0, j]) == 1
    g = side.wrap.groups0
    tac.apply(g, g.on, torch.zeros_like(g.on), side.wrap.hidden(raw))   # a fresh parcel
    assert int(tac.mt.abs().sum()) == 0


def test_scene_yields():
    # the generator's yield on a small draw (tactic/scenes): A1 scenes, played by TEACH through the level and the unit
    # script, hold a teacher's attack order; their N1 twins whose enemy runs past (plan point) hold none; the first
    # frames lie where the units stand (the layout's snap)
    from types import SimpleNamespace
    from net.tactic import exam_pin as EP
    from net.tactic import scenes as SN
    from sim.params import load
    a1 = SN.scenes("A1", 4, 310000)
    n1 = [s for s in SN.scenes("N1", 24, 310000) if s["n1_plan"] == "point"][:4]
    assert len(n1) == 4 and SN.check_bands(a1 + n1) > 0
    sp = a1 + n1
    for s in sp:
        s["limit"] = 200
    res, got = EP.record_run("TEACH", sp, [2100], len(sp), SimpleNamespace(device="cpu", sim=""), load(), scene=True)
    has = {x["b"] for x in got["labels"]}
    assert sum(1 for b in range(4) if b in has) >= 3, got["labels"]
    assert not any(b in has for b in range(4, 8)), got["labels"]
    assert max(d for d, _ in got["first"].values()) <= reissue.MOVE_M
    # every other recipe draws within its bands and within 8 units a side (generated, not played: its yields are the
    # exam's - exam_pin scenes --check --scene-set S)
    other = [SN.make(r, 910000 + i) for r in ("A2", "A3", "A4", "A5", "A6", "A4L", "A5L", "N2", "E1", "E2", "E3")
             for i in range(2)]
    assert SN.check_bands(other) > 0
    assert all(len(s["mine"]) <= 8 and len(s["theirs"]) <= 8 for s in other)


def test_pointer_walk_loss_and_own():
    # the pointer's walk, loss and own() on the tick of test_teacher_walk_and_masks: anvil 0 in melee with 5 (pinned),
    # free units 1, 2 on 5's right side, 3 on its left; the teacher's choice 'unit 1 attacks 5' fed through the walk
    from net.tactic import heads as HD
    units = [("mine", 0.0, 0.0, 0.0, 32, 10, 1, 0, 0.0),
             ("mine", -60.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", -90.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", 70.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
             ("mine", 0.0, -150.0, 0.0, 32, 10, 0, 0, 0.0),
             ("enemy", 0.0, 12.0, 180.0, 32, 10, 1, 0, 0.0),
             ("enemy", 0.0, -138.0, 180.0, 32, 10, 0, 0, 0.0),
             ("enemy", 30.0, 60.0, 180.0, 32, 10, 0, 1, 0.0)]
    v = _tick(units)
    fx = v.fx
    assert HD.head_free(fx)[0].tolist() == [False, True, True, True, True], "the anvil in melee is not head-free"
    given = torch.tensor([[-1, 5, -1, -1, -1]])
    wi = HD.walk_inputs(fx, given)
    # after unit 1 takes 5 on its right side, unit 2 (the same side) finds 5 closed, unit 3 (the left) open; 7 routs
    assert not bool(wi["open"][0, 2, 5]) and bool(wi["open"][0, 3, 5]) and bool(wi["open"][0, 1, 5])
    assert not bool(wi["open"][0, 1, 7])
    # pair numbers: the holders on 5 at each unit's turn / 2; dist / pos_m; side one-hot (unit 1 on 5's right)
    assert float(wi["pair"][0, 1, 5, 5]) == 0.0 and float(wi["pair"][0, 3, 5, 5]) == 0.5
    assert abs(float(wi["pair"][0, 1, 5, 0]) - float(fx.g_dist[0, 1, 5]) / 100.0) < 1e-6
    assert wi["pair"][0, 1, 5, 2:5].tolist() == [0.0, 0.0, 1.0]
    # a fresh pointer keeps (last layers zero, the null bias): every pick null; with given, the walk is fed but the
    # picks stay the pointer's
    W = 8
    ptr = HD.OrderPointer(W)
    enc = {"cards": torch.randn(1, fx.K, W), "own": torch.zeros(1, fx.G, HD.OWN)}
    out = HD.choose_orders(enc, fx, ptr, given=given)
    assert out["k"][0].tolist() == [-1] * 5 and out["fed"][0].tolist() == given[0].tolist()
    # the loss: cross-entropy at the label over {null} + open keys less the don't-care keys
    lg = torch.full((1, 1, 1 + 4), HD.NEG)
    lg[0, 0, :4] = torch.tensor([1.0, 2.0, 0.5, 3.0])          # null, rows 0, 1, 2 open; row 3 closed
    dc = torch.tensor([[[False, False, True, False]]])          # row 2 a don't-care key: out of the softmax
    got, st = HD.loss(lg, torch.tensor([[1]]), torch.ones(1, 1), dc)
    want = -(0.5 - math.log(math.exp(1.0) + math.exp(2.0) + math.exp(0.5)))    # (label row 1)
    assert abs(float(got) - want) < 1e-5 and st["n"] == 1 and st["out_of_set"] == 0
    got, st = HD.loss(lg, torch.tensor([[2]]), torch.ones(1, 1), dc)     # the label on a don't-care key: weight 0
    assert st["n"] == 0 and st["out_of_set"] == 1 and float(got) == 0.0
    got, st = HD.loss(lg, torch.tensor([[-1]]), torch.zeros(1, 1), dc)   # a dropped tick: weight 0
    assert st["n"] == 0 and float(got) == 0.0
    # own(): the side zeroed with no target, the age 0 with no target and capped at orders.net.age_cap_s
    order = {"skill": torch.tensor([[SC.HOLD, SC.ATTACK, SC.ATTACK, SC.HOLD, SC.HOLD]]),
             "tgt": torch.tensor([[-1, 5, 5, -1, -1]]), "side": torch.tensor([[1, 2, 1, 2, -1]]),
             "born": torch.tensor([[0.0, 590.0, 10.0, 5.0, 0.0]])}
    o = OR.own(order, None, OR.Facts(v.rr, v.ctx, order), 600.0)
    assert o.shape == (1, 5, 16)
    assert o[0, 0, 12:15].tolist() == [0.0, 0.0, 0.0] and o[0, 3, 12:15].tolist() == [0.0, 0.0, 0.0]
    assert o[0, 1, 12:15].tolist() == [0.0, 0.0, 1.0]
    assert [round(float(x), 4) for x in o[0, :, 15]] == [0.0, round(10 / 60, 4), 5.0, 0.0, 0.0]


def test_kinds_n2w_and_n9_comer():
    # review N: n2w (an unheld enemy within the reach, the nearest coming at another idle unit of mine) is don't-care,
    # not taught; n9's key of an enemy that comes at g is don't-care whatever its melee flag
    from net.tactic import tags as TG
    keep = lambda G: {"skill": torch.full((1, G), SC.HOLD), "target": torch.full((1, G), -1)}     # noqa: E731
    # n2w: enemy 2 runs south at my idle unit 1 (60, 10); unit 0 (-60, 60) is level with it (120 m), not come at
    v = _tick([("mine", -60.0, 60.0, 0.0, 32, 10, 0, 0, 0.0),
               ("mine", 60.0, 10.0, 0.0, 32, 10, 0, 0, 0.0),
               ("enemy", 60.0, 60.0, 180.0, 32, 10, 0, 0, -3.0)])
    T = TG.tick(v, keep(2), PN.player("TEACH"))
    assert bool(T["kinds"]["n2w"][0, 0]) and "n2w" not in TG.TAUGHT and bool(T["dc_keys"][0, 0, 2])
    # n9: enemy 2 two ticks in melee, not pinned (none of mine in melee), runs south at unit 0 (0, 0); unit 1 (200, 60)
    # level with it: for 0 its key is don't-care and the tick no n9, for 1 the tick is n9 and the key taught
    v = _tick([("mine", 0.0, 0.0, 0.0, 32, 10, 0, 0, 0.0),
               ("mine", 200.0, 60.0, 0.0, 32, 10, 0, 0, 0.0),
               ("enemy", 0.0, 60.0, 180.0, 32, 10, 1, 0, -3.0)])
    v.fx.mt = torch.tensor([[0, 0, 2]])
    T = TG.tick(v, keep(2), PN.player("TEACH"))
    assert T["kinds"]["n9"][0].tolist() == [False, True] and "n9" in TG.TAUGHT
    assert T["dc_keys"][0, :, 2].tolist() == [True, False]
