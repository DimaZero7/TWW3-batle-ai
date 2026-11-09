"""Stage 0's battles in the simulator (net/stage0: battle.py, behave.py, watch.py): the lords behind and out of the
chain, the strategist's placement changed by a behaviour, the named reserve, a short battle with every parcel
correct and the measures, the flank behaviour hitting the flank, Spearman. Needs torch (the container, CPU)."""
import math
import random

import pytest

torch = pytest.importorskip("torch")

from sim import maps as arena_map  # noqa: E402
from sim.params import load  # noqa: E402

pytestmark = pytest.mark.skipif(not (arena_map.MAPS / "ksl_plains_01" / "meta.json").exists(),
                                reason="sim/maps/ksl_plains_01 (the captured map) is not here")
P = load()


@pytest.fixture(scope="module")
def unit_net():
    from net.tactic import train as TT
    return TT.unit_network("cpu")


def _local(zn, x, z):
    from net.stage0 import specs as SP
    return SP.local(zn, x, z)


def test_the_strategist_deploys_and_the_lords_stand_150_m_behind_out_of_every_group():
    from net.stage0 import battle as BA
    from net.stage0 import specs as SP
    sp = SP.generals(6, random.Random(2))
    pr = BA.Prep(sp, P, "cpu")
    assert pr.groups0 is not None and bool((pr.groups0.member[:, 0] == -1).all())
    for b, s in enumerate(sp):
        bt = pr.battles[b]
        for side, zk in ((1, "mine_zone"), (2, "enemy_zone")):
            lord, rest = bt[side][0], bt[side][1:]
            assert lord["general"] and not any(u["general"] for u in rest) and len(rest) == len(s["mine"])
            zn = s[zk]
            fl = _local(zn, lord["x"], lord["z"])[1]
            fm = sum(_local(zn, u["x"], u["z"])[1] for u in rest) / len(rest)
            assert abs((fm - fl) - 150.0) < 1.0
        # my units inside my zone
        for u in bt[1][1:]:
            s_, f_ = _local(s["mine_zone"], u["x"], u["z"])
            assert abs(s_) <= s["mine_zone"]["front"] / 2 + 1 and abs(f_) <= s["mine_zone"]["depth"] / 2 + 1


def test_the_placement_behaviours_move_the_line_to_its_ground():
    from sim import terrain
    from net.stage0 import battle as BA
    from net.stage0 import behave as BH
    from net.stage0 import specs as SP
    sp = SP.drills("c01", 8, random.Random(5))
    h = {}
    for who in ("teacher", "high", "low"):
        pr = BA.Prep(sp, P, "cpu", BH.by_name(who))
        x = torch.tensor([[u["x"] for u in bt[1][1:4]] for bt in pr.battles])
        z = torch.tensor([[u["z"] for u in bt[1][1:4]] for bt in pr.battles])
        h[who] = float(terrain.height(pr.st0.u, x, z).mean())
    assert h["high"] > h["teacher"] - 0.5 and h["high"] > h["low"] + 3.0


def test_the_named_reserve_in_a_general_battle_is_the_rearmost_moved_back():
    from net.stage0 import battle as BA
    from net.stage0 import behave as BH
    from net.stage0 import specs as SP
    rnd = random.Random(9)
    sp = [SP.general(rnd, n=7) for _ in range(4)] + [SP.general(rnd, n=3)]
    pr = BA.Prep(sp, P, "cpu", BH.by_name("res_break"))
    assert pr.reserve[:4].sum(1).tolist() == [2, 2, 2, 2] and int(pr.reserve[4].sum()) == 0
    for b in range(4):
        zn = sp[b]["mine_zone"]
        f = [_local(zn, u["x"], u["z"])[1] for u in pr.battles[b][1][1:]]
        res = [i for i in range(7) if bool(pr.reserve[b, 1 + i])]
        for i in res:
            assert f[i] <= max(f) - 79.0 or f[i] <= -zn["depth"] / 2 + 16.0


def test_a_short_battle_has_correct_parcels_the_lord_holds_and_the_measures(unit_net):
    from net.stage0 import battle as BA
    from net.stage0 import behave as BH
    from net.stage0 import specs as SP
    rnd = random.Random(3)
    sp = [SP.general(rnd, enemy="nearest", n=n) for n in (2, 3, 6, 8)] + [SP.general(rnd, enemy="chain", n=4)]
    for s in sp:
        s["limit"] = 60
    m, st = BA.play(sp, BH.Behaviour(), unit_net, P, "cpu")
    assert st["parcels"] > 0 and st["parcel_errors"] == 0 and st["enemy_parcel_errors"] == 0, st
    for k in ("trade", "flank_hits", "flank_dmg", "gap_m", "dh", "reserve_s", "orders_min", "t_contact", "win"):
        assert m[k].shape == (5,)
    assert float(m["lord_moved_m"].max()) < 3.0
    assert bool((m["dh"][torch.isfinite(m["dh"])] == 0).all()) or any(s["map"] != "flat" for s in sp)


def test_the_flank_behaviour_hits_flanks_more_than_all_in_the_face(unit_net):
    from net.stage0 import battle as BA
    from net.stage0 import behave as BH
    from net.stage0 import specs as SP
    sp = SP.drills("t02", 8, random.Random(11), enemy="nearest")
    for s in sp:
        s["limit"] = 120
        s["map"] = "flat"
    hits = {}
    for who in ("flank5", "infront"):
        torch.manual_seed(0)              # (the same draws for both, whatever test ran before)
        m, _ = BA.play(sp, BH.by_name(who), unit_net, P, "cpu")
        hits[who] = float(m["flank_hits"].sum())
    assert hits["flank5"] > hits["infront"], hits

