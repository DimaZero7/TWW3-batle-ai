"""tools/nn/train/drills/transfer.py and the drills' broad frames (docs/en/training/training.md "Drills",
"Transfer"). torch: the training container.

Level 1: each READY drill has a transfer detector and a broad frame; the clean battles of a seed do not
depend on the broad share. Level 2: the detectors on hand-made states (the situation, the skill applied,
the mistake); the broad frames keep the drill's condition. Level 3: the evaluation returns the transfer
block, the network's and the opponent script's, and test5 shows it.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config, policy  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim import scenario  # noqa: E402
from tools.nn.train import drills as D  # noqa: E402
from tools.nn.train import evaluate, test5  # noqa: E402
from tools.nn.train.drills import counter, hold_fire, kiting  # noqa: E402
from tools.nn.train.drills import transfer as T  # noqa: E402

UNITS = json.loads((Path(__file__).resolve().parents[2] / "config/nn/units.json").read_text(encoding="utf-8"))["units"]
EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"
NR, SWORD = "wh2_main_skv_inf_night_runners_1", "wh_main_emp_inf_swordsmen"
SPEAR, SHIELD, FLAG = counter.SPEAR, counter.SHIELD, counter.FLAG
ARCHERS, GENERAL = "wh2_dlc13_emp_inf_archers_0", "wh_main_emp_cha_general_0"


def _slot(desc, side, k):
    """Slot of the k-th listed unit of a side (scenario.slots: generals first)."""
    st = scenario.build([desc])
    names = sorted(desc["sides"][side]["units"], key=lambda u: not u.get("general"))
    return st, (side - 1) * (st.N // 2) + names.index(desc["sides"][side]["units"][k])


class TestFramework:
    def test_ready_drills_have_a_detector_and_a_broad_frame(self):
        loaded = D.load(list(D.READY))
        for name in D.READY:
            assert loaded[name].transfer is not None and loaded[name].broad is not None
        assert set(T.detectors()) == set(D.READY)

    def test_the_clean_battles_of_a_seed_do_not_depend_on_the_broad_share(self):
        for name in D.READY:
            drill = D.load([name])[name]
            clean = D.battles(drill, range(40), broad=0.0, embed=0.0)
            mixed = D.battles(drill, range(40), broad=0.5, embed=0.0)
            wide = [d for d, _ in mixed if d["broad"]]
            assert 5 < len(wide) < 35
            for (a, sa), (b, sb) in zip(clean, mixed):
                assert sa == sb and not a["broad"]
                if not b["broad"]:
                    assert a == b
            assert all(d["broad"] for d, _ in D.battles(drill, range(8), broad=1.0, embed=0.0))


class TestDetectors:
    def test_kiting_situation_run_back_and_caught(self):
        desc = D.army([D.unit(NR, 0, 0, 0, 30)], [D.unit(SWORD, 0, 40, 180, 30)], 1, SKV, EMP)
        st, me = _slot(desc, 1, 0)
        _, en = _slot(desc, 2, 0)
        u = st.u
        u["vz"][0, en] = -3.0                                           # the swordsmen come at us
        sit, app, mis = kiting.transfer(st)
        assert bool(sit[0, me]) and not bool(app[0, me]) and not bool(mis[0, me])
        u["vz"][0, me] = -4.0                                           # we run back
        sit, app, mis = kiting.transfer(st)
        assert bool(sit[0, me]) and bool(app[0, me])
        u["m"][0, me] = True                                            # caught
        sit, app, mis = kiting.transfer(st)
        assert bool(mis[0, me]) and not bool(app[0, me])
        u["m"][0, me] = False
        u["vz"][0, en] = 0.0                                            # standing still, not in melee: no situation
        assert not bool(kiting.transfer(st)[0][0, me])
        u["z"][0, en] = 100.0                                           # far
        u["vz"][0, en] = -3.0
        assert not bool(kiting.transfer(st)[0][0, me])

    def test_counter_situation_goes_for_its_counter_or_the_better_target(self):
        desc = D.army([D.unit(SPEAR, 0, 0, 0, 30)], [D.unit(FLAG, 0, 80, 180, 30), D.unit(SHIELD, 60, 80, 180, 30)],
                      1, EMP, EMP)
        st, me = _slot(desc, 1, 0)
        _, flag = _slot(desc, 2, 0)
        _, shield = _slot(desc, 2, 1)
        u = st.u
        sit, app, mis = counter.transfer(st)
        assert bool(sit[0, me]) and not bool(app[0, me]) and not bool(mis[0, me])       # no aim yet
        u["order_kind"][0, me], u["order_target"][0, me] = O.ATTACK, flag
        sit, app, mis = counter.transfer(st)
        assert bool(mis[0, me]) and not bool(app[0, me])
        u["order_target"][0, me] = shield
        sit, app, mis = counter.transfer(st)
        assert bool(app[0, me]) and not bool(mis[0, me])
        u["m"][0, shield] = True                                        # the better target is no longer free
        assert not bool(counter.transfer(st)[0][0, me])

    def test_hold_fire_situation_firing_into_the_lords_melee(self):
        ours = [D.unit(SPEAR, 0, 0, 0, 30), D.unit(ARCHERS, 0, -90, 0, 40)]
        desc = D.army(ours, [D.unit(GENERAL, 0, 8, 180, general=True)], 1, EMP, EMP)
        st, inf = _slot(desc, 1, 0)
        _, arch = _slot(desc, 1, 1)
        _, lord = _slot(desc, 2, 0)
        u = st.u
        assert not bool(hold_fire.transfer(st)[0][0, arch])                # not in melee yet
        u["m"][0, inf] = u["m"][0, lord] = True
        sit, app, mis = hold_fire.transfer(st)
        assert bool(sit[0, arch]) and bool(app[0, arch]) and not bool(sit[0, inf])
        u["fire"][0, arch], u["target"][0, arch] = True, lord
        sit, app, mis = hold_fire.transfer(st)
        assert bool(mis[0, arch]) and not bool(app[0, arch])
        u["x"][0, arch] = 500.0                                         # out of range
        assert not bool(hold_fire.transfer(st)[0][0, arch])


class TestBroadFrames:
    def test_kiting_broad_keeps_the_speed_condition_and_varies(self):
        sizes, lords = set(), set()
        for desc, ours in D.battles(kiting.DRILL, range(64), broad=1.0, embed=0.0):
            mine, theirs = desc["sides"][ours]["units"], desc["sides"][3 - ours]["units"]
            kiters = [u for u in mine if (UNITS[u["key"]].get("missile") or {}).get("ammo")]
            run = min(UNITS[u["key"]]["speed"]["run"] for u in kiters)
            for e in theirs:
                assert UNITS[e["key"]]["speed"]["run"] <= run - kiting.MIN_GAP_MS
                assert "unbreakable" not in UNITS[e["key"]]["attributes"]
            assert all(abs(u["x"]) <= D.MAP_HALF_M and abs(u["z"]) <= D.MAP_HALF_M for u in mine + theirs)
            sizes.add((len(mine), len(theirs)))
            lords.add(any(u["general"] for u in mine))
        assert len(sizes) > 6 and lords == {True, False}

    def test_counter_broad_has_the_counter_situation(self):
        n_pairs, have = set(), []
        for desc, ours in D.battles(counter.DRILL, range(32), broad=1.0, embed=0.0):
            st = scenario.build([desc])
            sit = counter.transfer(st)[0][0]
            mine = st.u["side"][0] == ours
            have.append(bool((sit & mine).any()))       # a unit of ours has its counter and a better target near
            n_pairs.add(sum(1 for u in desc["sides"][ours]["units"] if not u["general"]))
        assert n_pairs == {2, 4}
        assert np.mean(have) >= 0.75                    # at the start (the rest: the lines a little over 200 m apart)

    def test_hold_fire_broad_engages_the_lord(self):
        kinds = set()
        for desc, ours in D.battles(hold_fire.DRILL, range(32), broad=1.0, embed=0.0):
            enemy = desc["sides"][3 - ours]["units"]
            assert any(u["general"] for u in enemy)
            kinds |= {u["key"] for u in desc["sides"][ours]["units"]}
        assert hold_fire.MILITIA in kinds


def actor():
    cfg = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)
    torch.manual_seed(0)
    return policy.Actor(cfg).eval()


class TestEvaluation:
    def test_play_reports_the_transfer_of_both_sides(self):
        res = evaluate.play(actor(), opponents=("ai_like", "nearest"), generated=8, limit_s=20.0, together=True)
        tr = res["transfer"]
        assert set(tr) == set(D.READY)
        for name, x in tr.items():
            assert set(x["by_opponent"]) == {"ai_like", "nearest"}
            assert x["ai_like"] == x["by_opponent"]["ai_like"]["script"]
            for d in (x["network"], x["ai_like"]):
                assert d["unit_s"] >= 0 and (d["share"] is None or 0 <= d["share"] <= 1)
                assert d["share"] is None or d["share"] + d["mistake"] <= 1 + 1e-6
        lines = test5.drill_block([test5.metrics(res)], ["now"])
        assert any("TRANSFER" in line for line in lines)
        assert T.text(tr)

    def test_tracker_sums_the_situation_seconds(self):
        desc = D.army([D.unit(NR, 0, 0, 0, 30)], [D.unit(SWORD, 0, 40, 180, 30)], 1, SKV, EMP)
        st = scenario.build([desc, desc])
        u = st.u
        u["vz"][:, st.N // 2] = -3.0
        u["vz"][1, 0] = -4.0                                                # battle 1: running back
        from tools.nn.sim.params import load
        params = load()
        mine = u["side"] == 1
        tr = T.Tracker(st, params, mine, names=["kiting"])
        for _ in range(4):
            tr.update(st, torch.ones(2, dtype=torch.bool))
        s = tr.summary(np.array([True, True]))["kiting"]
        assert s["share"] == pytest.approx(0.5) and s["unit_s"] == pytest.approx(4 * params.dt)
        assert tr.summary(np.array([False, True]))["kiting"]["share"] == pytest.approx(1.0)
        assert tr.summary(np.array([True, True]), "other")["kiting"]["share"] is None
        assert math.isclose(tr.summary(np.array([True, False]))["kiting"]["battles"], 1.0)
