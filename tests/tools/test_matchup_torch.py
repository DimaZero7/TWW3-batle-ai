"""The user's matchups in the simulator (tools/nn/matchup) and the army's formula (operative/army.py): the armies and
places, a short battle of our chain against a script with every parcel correct, the army's groups by role and kept,
the group's target to the units, the stand-still rule. Needs torch (the container; the simulator on the CPU)."""
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.matchup import armies as AR  # noqa: E402
from tools.nn.matchup import duel as DU  # noqa: E402
from tools.nn.model import passport as PP  # noqa: E402
from tools.nn.operative import army as AM  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.tactic import train as TT  # noqa: E402

P = load()


def test_the_matchups_armies_and_places():
    zones = AR.zones()
    for name in ("m1", "m2", "m3"):
        b = AR.base(name)
        assert len(b[1]) == 20 and len(b[2]) == 20
        assert sum(u["general"] for u in b[1]) == 1 and sum(u["general"] for u in b[2]) == 1
        for s, alliance in ((1, 0), (2, 1)):
            z = zones[alliance]
            xs = [u["x"] for u in b[s]]
            assert abs(np.mean(xs) - z["x"]) < 120, (name, s)
        a = AR.augment(b, random.Random(3))
        assert 18 <= len(a[1]) <= 20 and any(u["general"] for u in a[1])
    r = AM.roles(np.asarray(PP.table([u["key"] for u in AR.base("m1")[1]])))
    assert r.count("lord") == 1 and r.count("sniper") == 2 and r.count("anti") >= 4


@pytest.fixture(scope="module")
def unit_net():
    return TT.unit_network("cpu")


def test_a_short_battle_of_the_chain_against_a_script(unit_net):
    rnd = random.Random(0)
    battles = [AR.augment(AR.base("m1"), rnd) for _ in range(2)]
    chain = DU.Chain(1, "formula", command="hold")
    rd, st = DU.play("m1", battles, {1: chain, 2: DU.Script(2, "nearest")}, unit_net, P, "cpu", limit_s=31)
    s = st["side1"]
    assert s["parcel_errors"] == 0 and s["parcels"] > 0
    assert s["reforms"] == 0                                     # (the army's groups are kept)
    g = chain.groups
    rl = AM.roles(chain.setup.card[0].numpy())
    for k in range(g.G):
        mem = (g.member[0] == k).nonzero()[:, 0].tolist()
        if mem:
            assert len({rl[i] for i in mem}) == 1                 # one role a group
    # the target to the units: some own unit has a main target named
    assert bool((chain.units.prio[chain.setup.own] >= 0).any())
    for k in ("inside", "walk_m", "fire_share", "order_changes_per_unit"):
        assert k in rd


def test_the_stand_still_rule():
    from tools.nn.operative import lower as LW
    from tools.nn.sim import orders as O
    rnd = random.Random(1)
    st, card = DU.build("m1", [AR.base("m1")], "cpu", P)
    setup = DU._SideSetup(st, P, "cpu", 1, card)
    units = LW.Units(setup, None, still=True)
    u = st.u
    units.area.update({"x": u["x"].clone(), "z": u["z"].clone(), "front": torch.full_like(u["x"], 80.0),
                       "depth": torch.full_like(u["x"], 48.0)})
    o = O.hold(st.B, st.N)
    o.kind[:] = O.MOVE
    o.x[:] = u["x"] + 10.0
    o.z[:] = u["z"]
    inside = torch.ones_like(setup.own)
    out = units.stand_still(o, st, inside)
    assert bool((out.kind[setup.own] == O.HOLD).all()) and bool((out.kind[~setup.own] == O.MOVE).all())
    o.x[:] = u["x"] + 200.0                                      # out of the area: still a move
    out = units.stand_still(o, st, inside)
    assert bool((out.kind == O.MOVE).all())
    del rnd


def test_the_shooter_shoots_its_target():
    """lower.Units.aim_rule: a ranged unit in its area with its main target in range -> ATTACK on it; out of range or a
    melee unit -> the order stays; a withdraw stays."""
    from tools.nn.operative import lower as LW
    from tools.nn.sim import orders as O
    st, card = DU.build("m2", [AR.base("m2")], "cpu", P)
    setup = DU._SideSetup(st, P, "cpu", 1, card)
    units = LW.Units(setup, None, targets=True, aim=True)
    u = st.u
    own = setup.own[0]
    ranged = [i for i in range(st.N) if bool(own[i]) and float(u["range"][0, i]) > 300]       # the mortars
    melee = [i for i in range(st.N) if bool(own[i]) and float(u["range"][0, i]) <= 0 and not bool(u["lord"][0, i])]
    foe = [i for i in range(st.N) if not bool(own[i]) and float(u["run"][0, i]) < 6.0 and not bool(u["lord"][0, i])][0]
    m, w = ranged[0], melee[0]
    u["x"][0, foe], u["z"][0, foe] = u["x"][0, m] + 200.0, u["z"][0, m]     # within the mortar's range (not fast)
    units.prio[0, m] = units.prio[0, w] = foe
    o = O.hold(st.B, st.N)
    inside = torch.ones_like(setup.own)
    out = units.aim_rule(o, st, inside)
    assert int(out.kind[0, m]) == O.ATTACK and int(out.target[0, m]) == foe
    assert int(out.kind[0, w]) == O.HOLD
    o.kind[0, m] = O.WITHDRAW
    assert int(units.aim_rule(o, st, inside).kind[0, m]) == O.WITHDRAW
    o.kind[0, m] = O.HOLD
    u["x"][0, foe] = u["x"][0, m] + 600.0                        # out of range: as the unit network said
    assert int(units.aim_rule(o, st, inside).kind[0, m]) == O.HOLD


def test_snipers_do_not_take_the_enemy_lord():
    """army.plan: the snipers' target is no enemy lord while the enemy's front stands (08.11: their zone stepped onto
    him); direct-fire snipers do not step up past the line (their place is its flank)."""
    from tools.nn.operative import cfg as ocfg
    keys = [u["key"] for u in AR.base("m2")[1]] + [u["key"] for u in AR.base("m2")[2]]
    cards = np.asarray(PP.table(keys))
    rl = AM.roles(cards)
    K = len(keys)
    own = list(range(20))
    foes = list(range(20, K))
    pos = np.zeros((K, 2))
    pos[:20, 0] = -200.0
    pos[20:, 0] = 900.0
    snip = [j for j in own if rl[j] == "sniper"]
    lord = [e for e in foes if rl[e] == "lord"][0]
    pos[lord] = (0.0, 0.0)
    tc = dict(ocfg()["teacher"]["army"])
    out = AM.plan(0, own, foes, pos, cards, np.zeros(K, bool), {}, False, tc, cards[:, 29] * 500.0)
    g = [x for x in out if set(x[0]) == set(snip)][0]
    assert g[1] != lord
    assert g[2] is None or float(g[2][0]) <= -200.0 + float(tc["ahead_m"]) + 1e-6


def test_shooters_guns_and_the_lord_stay_in_their_area():
    """lower.Units.stay_rule: a gun out of its area under an attack goes to the area's centre; a shooter in its area with
    an attack beyond range holds; the lord without a main target does not go at a far enemy; a withdraw stays."""
    from tools.nn.operative import lower as LW
    from tools.nn.sim import orders as O
    st, card = DU.build("m2", [AR.base("m2")], "cpu", P)
    setup = DU._SideSetup(st, P, "cpu", 1, card)
    units = LW.Units(setup, None, targets=True, aim=True)
    u = st.u
    own = setup.own[0]
    gun = [i for i in range(st.N) if bool(own[i]) and float(u["eng0"][0, i]) > 0][0]
    shot = [i for i in range(st.N) if bool(own[i]) and float(u["range"][0, i]) > 0 and float(u["eng0"][0, i]) <= 0][0]
    lord = [i for i in range(st.N) if bool(own[i]) and bool(u["lord"][0, i])][0]
    foe = [i for i in range(st.N) if not bool(own[i])][1]
    u["x"][0, foe] = u["x"][0, gun] + 600.0                      # beyond every range
    units.area["x"][0, gun] = u["x"][0, gun] - 100.0             # the gun out of its area
    o = O.hold(st.B, st.N)
    o.kind[:] = O.ATTACK
    o.target[:] = foe
    o.kind[0, lord + 0] = O.ATTACK
    inside = torch.ones_like(setup.own)
    inside[0, gun] = False
    out = units.stay_rule(o, st, inside)
    assert int(out.kind[0, gun]) == O.MOVE and abs(float(out.x[0, gun] - units.area["x"][0, gun])) < 1e-3
    assert int(out.kind[0, shot]) == O.HOLD
    assert int(out.kind[0, lord]) == O.ATTACK                    # (config lord_guard_m 0: the lord's part off)
    from tools.nn.operative import cfg as ocfg
    old = ocfg().get("lord_guard_m")
    ocfg()["lord_guard_m"] = 30.0
    try:
        assert int(units.stay_rule(o, st, inside).kind[0, lord]) == O.HOLD
        units.prio[0, lord] = foe                                # a main target named: the lord goes
        assert int(units.stay_rule(o, st, inside).kind[0, lord]) == O.ATTACK
    finally:
        ocfg()["lord_guard_m"] = old
    o.kind[0, shot] = O.WITHDRAW
    assert int(units.stay_rule(o, st, inside).kind[0, shot]) == O.WITHDRAW


def test_ai_game_steps_round_cannot_stand_ground():
    """opponents.ai_game (ai_like, Line.step_around): a march point on 'cannot stand' ground of ksl_plains_01 (the patch
    30 m before the AI's M1 centre) goes to a free point beside it; a free point stays."""
    from tools.nn.sim import ground
    from tools.nn.sim import orders as O
    from tools.nn.train import opponents as OP
    st, _ = DU.build("m1", [AR.base("m1")], "cpu", P)
    u = st.u
    kind = torch.full_like(u["x"], O.MOVE, dtype=torch.long)
    x = torch.full_like(u["x"], 188.0)
    z = torch.full_like(u["x"], -32.0)
    tx, tz = torch.full_like(x, 158.0), torch.full_like(z, -31.0)
    assert bool(ground.blocked(u, tx, tz).all())
    nx, nz = OP._around(u, kind, x, z, tx, tz)
    assert not bool(ground.blocked(u, nx, nz).any())
    fx, fz = torch.full_like(x, 120.0), torch.full_like(z, -31.0)
    ox, oz = OP._around(u, kind, x, z, fx, fz)
    assert torch.equal(ox, fx) and torch.equal(oz, fz)


def test_direct_fire_snipers_stand_on_the_flank_and_need_an_open_line():
    """army.plan: direct-fire snipers behind the line's centre get a goal flank_m beside its outermost infantry unit;
    lower.Units.line_closed: an own unit in the band to the target closes the line, one aside does not."""
    from tools.nn.operative import cfg as ocfg
    from tools.nn.operative import lower as LW
    keys = [u["key"] for u in AR.base("m1")[1]] + [u["key"] for u in AR.base("m1")[2]]
    cards = np.asarray(PP.table(keys))
    rl = AM.roles(cards)
    K = len(keys)
    own, foes = list(range(20)), list(range(20, K))
    pos = np.zeros((K, 2))
    pos[:20, 0] = -140.0
    pos[20:, 0] = 100.0                                                       # (in the snipers' range)
    mel = [j for j in own if rl[j] in ("line", "anti")]
    for k, j in enumerate(mel):
        pos[j] = (-100.0, -100.0 + 200.0 * k / max(len(mel) - 1, 1))       # a line 200 m wide across z at x -100
    snip = [j for j in own if rl[j] == "sniper"]
    pos[snip] = (-140.0, 0.0)                                                 # behind its centre
    tc = dict(ocfg()["teacher"]["army"], flank_m=38.0)                        # (config: off since the probe losfriend)
    out = AM.plan(0, own, foes, pos, cards, np.zeros(K, bool), {}, False, tc, cards[:, 29] * 500.0)
    g = [x for x in out if set(x[0]) == set(snip)][0]
    assert g[2] is not None and not g[3]
    assert abs(abs(float(g[2][1])) - (100.0 + float(tc["flank_m"]))) < 1.0 and abs(float(g[2][0]) + 100.0) < 1.0
    st, card = DU.build("m1", [AR.base("m1")], "cpu", P)
    setup = DU._SideSetup(st, P, "cpu", 1, card)
    units = LW.Units(setup, None, targets=True, aim=True)
    u = st.u
    i = [k for k in range(st.N) if bool(setup.own[0, k]) and bool(u["direct"][0, k])][0]
    j = [k for k in range(st.N) if bool(setup.own[0, k]) and k != i][0]
    tgt = [k for k in range(st.N) if not bool(setup.own[0, k])][0]
    others = [k for k in range(st.N) if bool(setup.own[0, k]) and k not in (i, j)]
    u["x"][0, others] = -400.0                                               # (out of the way)
    u["x"][0, i], u["z"][0, i] = -200.0, 0.0
    u["x"][0, tgt], u["z"][0, tgt] = 50.0, 0.0
    u["x"][0, j], u["z"][0, j] = -100.0, 0.0
    t = torch.full((1, st.N), -1, dtype=torch.long)
    t[0, i] = tgt
    assert bool(units.line_closed(st, t)[0, i])
    u["z"][0, j] = 80.0
    assert not bool(units.line_closed(st, t)[0, i])


def test_one_halberd_pair_a_threat():
    """army.plan: a fast enemy near my shooters draws the nearest halberd pair only (per_target 1); one farther than
    chase_ahead_m past my infantry's front draws none."""
    from tools.nn.operative import cfg as ocfg
    keys = [u["key"] for u in AR.base("m1")[1]] + [u["key"] for u in AR.base("m1")[2]]
    cards = np.asarray(PP.table(keys))
    rl = AM.roles(cards)
    K = len(keys)
    own, foes = list(range(20)), list(range(20, K))
    pos = np.zeros((K, 2))
    pos[:20, 0] = -200.0
    pos[:20, 1] = np.linspace(-150, 150, 20)
    pos[20:, 0] = 400.0
    fast = [e for e in foes if rl[e] == "fast"][0]
    sh = [j for j in own if rl[j] in ("shooter", "sniper")][0]
    pos[fast] = pos[sh] + np.array([50.0, 0.0])                               # 50 m before a shooter: a threat
    tc = dict(ocfg()["teacher"]["army"])
    out = AM.plan(0, own, foes, pos, cards, np.zeros(K, bool), {}, False, tc, cards[:, 29] * 500.0)
    on_fast = [x for x in out if rl[x[0][0]] == "anti" and x[1] == fast]
    assert len(on_fast) == 1
    pos[fast] = pos[sh] + np.array([50.0, 0.0])                               # past the front + chase_ahead_m (20)
    tc2 = dict(tc, chase_ahead_m=20.0)
    out = AM.plan(0, own, foes, pos, cards, np.zeros(K, bool), {}, False, tc2, cards[:, 29] * 500.0)
    assert not [x for x in out if rl[x[0][0]] == "anti" and x[1] == fast]


def test_no_attack_on_a_fast_target_going_aside():
    """lower.Units.fast_rule / bad_fast: a fast enemy (giant rats) running across a shooter's line is no attack target
    (the order is a hold); the same coming at the shooter is."""
    from tools.nn.operative import lower as LW
    from tools.nn.sim import orders as O
    st, card = DU.build("m1", [AR.base("m1")], "cpu", P)
    setup = DU._SideSetup(st, P, "cpu", 1, card)
    units = LW.Units(setup, None, targets=True, aim=True)
    u = st.u
    i = [k for k in range(st.N) if bool(setup.own[0, k]) and float(u["range"][0, k]) > 200][0]
    f = [k for k in range(st.N) if not bool(setup.own[0, k]) and float(u["run"][0, k]) >= 6.0][0]
    u["x"][0, i], u["z"][0, i], u["b"][0, i] = -200.0, 0.0, 90.0          # facing +x
    u["x"][0, f], u["z"][0, f] = 0.0, 0.0
    u["vx"][0, f], u["vz"][0, f] = 0.0, 7.0                                # across the line
    o = O.hold(st.B, st.N)
    o.kind[0, i], o.target[0, i] = O.ATTACK, f
    assert int(units.fast_rule(o, st).kind[0, i]) == O.HOLD
    u["vx"][0, f], u["vz"][0, f] = -7.0, 0.0                               # at the shooter
    assert int(units.fast_rule(o, st).kind[0, i]) == O.ATTACK


def test_the_snipers_duel():
    """army.plan: snipers on enemy Jezzails that outrange them (275 against 260), no enemy melee unit near, step past
    the +ahead_m limit to have them in range - duel_in_m (at most duel_ahead_m past the infantry's front); with an
    enemy melee unit near the point they stay within ahead_m."""
    from tools.nn.operative import cfg as ocfg
    keys = [u["key"] for u in AR.base("m1")[1]] + [u["key"] for u in AR.base("m1")[2]]
    cards = np.asarray(PP.table(keys))
    rl = AM.roles(cards)
    K = len(keys)
    own, foes = list(range(20)), list(range(20, K))
    pos = np.zeros((K, 2))
    pos[:20, 0] = -200.0
    pos[:20, 1] = np.linspace(-150, 150, 20)
    pos[20:, 0] = 600.0
    snip = [j for j in own if rl[j] == "sniper"]
    pos[snip] = (-200.0, 0.0)
    jz = [e for e in foes if rl[e] == "sniper"]
    pos[jz] = (90.0, 0.0)                                                      # 290 m: beyond 260, within 275 + 100
    ranges = cards[:, 29] * 500.0
    tc = dict(ocfg()["teacher"]["army"])
    out = AM.plan(0, own, foes, pos, cards, np.zeros(K, bool), {}, False, tc, ranges)
    g = [x for x in out if set(x[0]) == set(snip)][0]
    assert g[1] in jz and g[2] is not None
    assert -200.0 + float(tc["ahead_m"]) < float(g[2][0]) <= -200.0 + float(tc["duel_ahead_m"]) + 1e-6
    assert abs(float(np.hypot(*(pos[g[1]] - g[2]))) - (260.0 - float(tc["duel_in_m"]))) < 1.0
    melee = [e for e in foes if rl[e] in ("line", "anti")][0]
    pos[melee] = (-110.0, 0.0)                                                 # an enemy melee unit by the duel point
    out = AM.plan(0, own, foes, pos, cards, np.zeros(K, bool), {}, False, tc, ranges)
    g = [x for x in out if set(x[0]) == set(snip)][0]
    assert g[2] is None or float(g[2][0]) <= -200.0 + float(tc["ahead_m"]) + 2.0   # (the forward axis tilts a little)
