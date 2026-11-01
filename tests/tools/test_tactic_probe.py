"""The tactician's first drill probe (tools/nn/tactic_probe.py): its lanes and measures."""
import pytest

from tools.nn import charge_probe as cp
from tools.nn import tactic_probe as tp


def spec(plan, kind):
    return next(l for l in cp.run_config(plan, 1)[0]["lanes"] if l["kind"] == kind)


def test_the_tacpass_legs_go_clean_through_back_and_clean_on_one_line():
    for kind in ("centre", "side", "back"):
        sp = spec("tacpass", kind)
        assert [s["dz"] for s in sp["steps"]] == [80.0, 160.0, 80.0, 0.0]
        assert [s["run"] for s in sp["steps"]] == [True, True, False, False]
        assert all(s["dx"] == 0.0 for s in sp["steps"])
        block_z = sp["t_at"]["z"] + sp["extras"][0]["dz"]
        # the block stands in the middle of the second leg, the first and the last stay clear of it
        assert block_z == pytest.approx(sp["a_at"]["z"] + 120.0)
        assert sp["men_all_tg"] and sp["t_at"]["z"] - block_z > 170      # the enemy out of the bows' reach


def test_the_tacpass_side_lane_keeps_the_centres_apart_but_overlaps_the_blocks():
    sp = spec("tacpass", "side")
    e = sp["extras"][0]
    gap_centres = abs(e["dx"])
    assert gap_centres > 15.0                            # beyond contact.move_crowd's 15 m between the centres
    assert (sp["a_width"] + e["width"]) / 2 - gap_centres == pytest.approx(15.0)


def test_the_tacfire_friends_stand_between_the_shooters_and_the_enemy_at_the_given_gaps():
    for kind, gap, friend in tp.FIRE:
        sp = spec("tacfire", kind)
        assert sp["gap_m"] == gap and sp["mode"] == "shoot" and sp["target_mode"] == "hold"
        if friend is None:
            assert not sp.get("extras")
            continue
        e = sp["extras"][0]
        z = sp["t_at"]["z"] + e["dz"]
        f_d = cp.depth(cp.UNITS["swords"][0], cp.UNITS["swords"][1], tp.FRIEND_W)
        assert z - f_d / 2 == pytest.approx(sp["a_at"]["z"] + sp["a_depth"] / 2 + friend, abs=0.1)
        assert e["width"] > sp["a_width"]                    # every man's line passes over them


def test_the_tacwalk_slaves_walk_into_the_friends_and_stop_short_without_them():
    config = cp.run_config("tacwalk", 1)[0]
    assert config["mem_proj"] and config["mem_extras"]
    assert spec("tacwalk", "w20")["push_m"] < -tp.WALK_GAP
    assert -tp.WALK_GAP < spec("tacwalk", "w0")["push_m"] < 0


def test_fire_measure_counts_to_the_lanes_end_only():
    rows = [(t, {"ammo": 1800 - 90 * (t >= 5) - 90 * (t >= 15) - 90 * (t >= 70), "men": 90}, {"hp": 9000 - 100 * t, "men": 180}, None)
            for t in range(0, 80)]
    out = tp.fire_measure({"spec": {"kind": "x", "max_s": 60}, "samples": rows})
    assert out["shots"] == 180 and out["hp_enemy"] == 6000 and out["first_s"] == 5


def test_nominal_starts_the_twin_at_the_point_less_half_a_depth_without_men():
    sp = spec("tacfire", "f20")
    row = {"x": 0.0, "z": 0.0, "b": 0.0, "men": 90, "hp": 1.0}
    ln = {"spec": sp, "samples": [(0.0, dict(row), dict(row), None)], "extras": [(0.0, [dict(row)])]}
    out = tp.nominal([ln])[0]
    assert out["samples"][0][1]["z"] == pytest.approx(sp["a_at"]["z"] - sp["a_depth"] / 2)
    assert out["samples"][0][2]["z"] == pytest.approx(sp["t_at"]["z"] + sp["t_depth"] / 2)
    assert out["extras"][0][1][0]["z"] < sp["t_at"]["z"] + sp["extras"][0]["dz"]
