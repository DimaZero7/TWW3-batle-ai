"""The morale probe's plans, battle configs and label reading (tools/nn/morale_probe.py). The simulator twin needs
torch (the container)."""
import math

import pytest

from tools.nn import morale_probe as mp


@pytest.mark.parametrize("plan", mp.PLANS)
def test_every_plan_builds_lanes_far_apart_with_a_tested_side(plan):
    for i in range(1, len(mp.battles(plan)) + 1):
        config, model_s, arena = mp.run_config(plan, i)
        lanes = config["lanes"]
        xs = sorted(l["x"] for l in lanes)
        assert all(b - a >= 330 for a, b in zip(xs, xs[1:]))          # no enemy of another lane within 146 m
        names = {u["name"] for l in lanes for u in l["units"]}
        for l in lanes:
            tested = [u for u in l["units"] if not u["fearless"]]
            assert tested and all(u["name"] not in config["fearless"] for u in tested)
            for s in l["steps"]:
                assert s["on"] in ("go", "contact", "rout", "rally", "step")
                assert s["do"] == "end" or s["unit"] in {u["role"] for u in l["units"]}
        assert model_s > max(l["max_s"] for l in lanes)
        assert {p["name"] for p in config["park"]}.isdisjoint(names)


def test_labels_lose_their_colour_markup_and_spans_are_found():
    assert mp.clean("[[col:green]]Натиск[[/col]]") == "Натиск"
    assert mp.runs_of([None, "a", "a", None, "a"], "a") == [(1, 3), (4, 5)]


def test_the_secure_steps_close_in_and_a_friend_stands_at_the_side():
    assert [d for _, d in mp.SECURE_STEPS] == sorted((d for _, d in mp.SECURE_STEPS), reverse=True)
    lane = mp.secure_lane("side")
    u, e, f = lane["units"]
    assert math.isclose(e["z"] - u["z"], 300) and f["x"] == 32 and not f["fearless"] and e["fearless"]
