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
    foe = [i for i in range(st.N) if not bool(own[i])][1]
    m, w = ranged[0], melee[0]
    u["x"][0, foe], u["z"][0, foe] = u["x"][0, m] + 200.0, u["z"][0, m]     # within the mortar's range
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


def test_snipers_step_up_to_their_range():
    """army.plan: a sniper group whose target stands beyond its range by less than range_step_m gets a goal at its
    range less range_margin_m, and does not hold."""
    from tools.nn.operative import cfg as ocfg
    keys = [u["key"] for u in AR.base("m2")[1]] + [u["key"] for u in AR.base("m2")[2]]
    cards = np.asarray(PP.table(keys))
    rl = AM.roles(cards)
    K = len(keys)
    own = list(range(20))
    foes = list(range(20, K))
    pos = np.zeros((K, 2))
    pos[:20, 0] = -200.0
    pos[20:, 0] = 400.0                                          # all the enemy far
    snip = [j for j in own if rl[j] == "sniper"]
    jz = [e for e in foes if rl[e] == "sniper"][0]
    pos[jz] = (100.0, 0.0)                                       # the Jezzails 300 m off the snipers (range 260)
    ranges = cards[:, 29] * 500.0
    tc = dict(ocfg()["teacher"]["army"])
    out = AM.plan(0, own, foes, pos, cards, np.zeros(K, bool), {}, False, tc, ranges)
    g = [x for x in out if set(x[0]) == set(snip)][0]
    rows, seed, goal, holds = g
    assert seed == jz and not holds and goal is not None
    assert abs(float(np.hypot(*(pos[jz] - goal))) - (260.0 - float(tc["range_margin_m"]))) < 1.0
