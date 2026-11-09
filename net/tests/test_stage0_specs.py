"""Stage 0's battles as plain data (net/stage0/specs.py): the roster, armies equal in gold, the sizes and the
mixes of the general battles, the drills' places."""
import math
import random
from collections import Counter

import pytest

from sim import maps as arena_map

pytestmark = pytest.mark.skipif(not (arena_map.MAPS / "ksl_plains_01" / "meta.json").exists(),
                                reason="sim/maps/ksl_plains_01 (the captured map) is not here")


def test_the_roster_is_infantry_without_the_special_ones():
    import json
    from net import ROOT
    from sim.maps import MAPS
    from net.stage0 import cfg
    pp = json.loads((ROOT / "sim/config/units.json").read_text(encoding="utf-8"))["units"]
    for fac, keys in cfg()["roster"].items():
        assert len(keys) >= 5
        for k in keys:
            assert pp[k]["caste"] == "melee_infantry" and not pp[k].get("missile"), k
            assert ("_emp_" in k) == (fac == "emp")
            for bad in ("flagell", "huntsm", "pusbag", "grinder", "death_runner", "plague", "triad"):
                assert bad not in k


def test_general_battles_are_equal_in_gold_and_mixed_as_the_plan():
    from net.stage0 import cfg
    from net.stage0 import specs as SP
    sp = SP.generals(1000, random.Random(4))
    sizes = Counter(len(s["mine"]) for s in sp)
    assert set(sizes) == set(range(2, 9))
    small = sum(v for k, v in sizes.items() if k <= 4) / len(sp)
    assert abs(small - 0.6) < 0.05
    ok = [abs(s["gold"][1] - s["gold"][0]) / s["gold"][0] <= cfg()["general"]["gold_tol"] + 1e-9 for s in sp]
    assert sum(ok) / len(ok) > 0.97
    assert all(len(s["mine"]) == len(s["theirs"]) for s in sp)
    maps = Counter(s["map"] for s in sp)
    assert abs(maps["flat"] / len(sp) - 0.5) < 0.05
    en = Counter(s["enemy"] for s in sp)
    assert abs(en["ai_game"] / len(sp) - 0.4) < 0.05 and abs(en["chain"] / len(sp) - 0.3) < 0.05
    assert {s["east"] for s in sp} == {True, False}
    assert {(s["fac"][1], s["fac"][2]) for s in sp} == {("emp", "emp"), ("emp", "skv"), ("skv", "emp"), ("skv", "skv")}
    for s in sp[:50]:
        assert s["places_mine"] is None and len(s["places_theirs"]) == len(s["theirs"])
        # the enemy stands on its side of the field, facing mine
        mx, ex = s["mine_zone"]["x"], s["enemy_zone"]["x"]
        assert all((x - mx) * (ex - mx) > 0 for x, _, _ in s["places_theirs"])


def test_the_drills_places():
    from net.stage0 import specs as SP
    rnd = random.Random(7)
    for s in SP.drills("c01", 50, rnd):
        assert 3 <= len(s["mine"]) <= 6 and s["map"] == "ksl_plains_01" and s["enemy"] in ("ai_game", "nearest")
    for s in SP.drills("t02", 50, rnd):
        assert 2 <= len(s["mine"]) <= 3 and len(s["theirs"]) == 2
        zn = s["mine_zone"]
        fm = [SP.local(zn, x, z)[1] for x, z, _ in s["places_mine"]]
        fe = [SP.local(zn, x, z)[1] for x, z, _ in s["places_theirs"]]
        assert min(fe) - max(fm) >= 150.0
        assert all(abs(f) <= zn["depth"] / 2 for f in fm)
    for s in SP.drills("o03", 50, rnd):
        n = len(s["mine"])
        assert 6 <= n <= 8 and len(s["reserve"]) == 2
        zn = s["mine_zone"]
        f = [SP.local(zn, x, z)[1] for x, z, _ in s["places_mine"]]
        line = [f[i] for i in range(n) if i not in s["reserve"]]
        for i in s["reserve"]:
            assert 59.0 <= max(line) - f[i] <= 101.0
            assert abs(f[i]) <= zn["depth"] / 2


def test_rows_lay_units_across_the_zone():
    from net.stage0 import specs as SP
    zn = SP.zone(1)
    pts = SP.rows(zn, 7, dz=10.0)
    loc = [SP.local(zn, x, z) for x, z, _ in pts]
    assert [round(f) for _, f in loc] == [0] * 5 + [-40] * 2
    assert abs(sum(s for s, _ in loc[:5]) / 5 - 10.0) < 1e-6


def test_spearman_with_ties():
    from net.stage0 import stat as STT
    assert STT.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert STT.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert STT.ranks([5, 1, 5, 2]) == [2.5, 0.0, 2.5, 1.0]
    assert math.isnan(STT.spearman([1, 2], [1, 2]))
