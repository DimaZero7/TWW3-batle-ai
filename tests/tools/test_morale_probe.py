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
        # no enemy of another lane within 146 m
        assert all(math.hypot(a["x"] - b["x"], a["z"] - b["z"]) >= 330 for i, a in enumerate(lanes) for b in lanes[i + 1:])
        names = {u["name"] for l in lanes for u in l["units"]}
        for l in lanes:
            tested = [u for u in l["units"] if not u["fearless"]]
            assert tested and all(u["name"] not in config["fearless"] for u in tested)
            for s in l["steps"]:
                assert s["on"] in ("go", "contact", "rout", "rally", "waver", "step")
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


def test_the_aura_lanes_ring_the_lord_and_cast_then_move():
    lane = mp.aura_lane("htl")
    lord, *ring = lane["units"]
    assert lord["short"] == "general" and [x["ring_m"] for x in ring] == [30, 33, 36, 39, 42, 45]
    for x in ring:
        assert math.isclose(math.hypot(x["x"], x["z"]), x["ring_m"], abs_tol=0.1)
        # facing the lord: the bearing from the unit to the centre; side-on: a quarter turn from the radius
        assert x["b"] == round((x["ring_deg"] + (90 if x["side_on"] else 180)) % 360)
    cast, out, inn, end = lane["steps"]
    assert cast["do"] == "ability" and cast["key"] == mp.SYG and cast["delay_s"] == mp.AURA_CAST_S
    assert out["unit"] == "U1" and math.isclose(math.hypot(out["x"], out["z"]), 70, abs_tol=0.2)
    assert inn["unit"] == "U6" and math.isclose(math.hypot(inn["x"], inn["z"]), 20, abs_tol=0.2)
    assert mp.aura_lane("rally")["steps"][0]["key"] == mp.RALLY


def test_effect_lists_and_their_spans_are_read_from_the_samples():
    lane = {"rows": {"F": [{"fx": ""}, {"fx": mp.PENITENT}, {"fx": mp.PENITENT + ",x"}, {}]}}
    on = mp.has_fx(lane, "F", mp.PENITENT)
    assert on.tolist() == [False, True, True, False]
    assert mp.spans_of([0.0, 0.5, 1.0, 1.5], on, 0.5) == [(0.0, 1.0)]


def test_the_timer_plan_brings_a_held_unit_down_and_ends_after_its_rout():
    config, model_s, _ = mp.run_config("timer", 1)
    assert config["tick_ms"] == mp.TIMER_TICK_MS and model_s > mp.TIMER_MAX_S
    kinds = {l["kind"]: l for l in config["lanes"]}
    assert set(kinds) == set(mp.TIMER_KINDS)
    for kind, l in kinds.items():
        assert all(x["fearless"] != (x["role"] == "U") for x in l["units"])
        assert l["steps"][-1] == {"on": "rout", "do": "end", "of": "U", "delay_s": float(mp.TIMER_AFTER_ROUT_S)}
    assert [s["on"] for s in kinds["leave"]["steps"] if s["do"] == "move"] == ["waver"]
    assert sum(s["do"] == "move" for s in mp.timer_lane("leave_cr")["steps"]) == 2
    assert [s["share"] for s in kinds["kill"]["steps"] if s["do"] == "kill"] == [x for _, x in mp.TIMER_KILL]
    assert [s["delay_s"] for s in kinds["kill_fast"]["steps"] if s["do"] == "kill"] == [t for t, _ in mp.TIMER_KILL_FAST]
    # battle 2: the same kinds, each on another place
    second = {l["kind"]: (l["x"], l["z"]) for l in mp.run_config("timer", 2)[0]["lanes"]}
    assert all(second[k] != (l["x"], l["z"]) for k, l in kinds.items())


def _timer_lane(mp_, w, r, hp=None, variant="normal"):
    n = len(mp_)
    rows = [{"mp": a, "w": b, "r": c, "hp": (hp or [1.0] * n)[i], "ms": 5 if b else 3}
            for i, (a, b, c) in enumerate(zip(mp_, w, r))]
    return {"run": "x", "variant": variant, "spec": {"name": "L1", "kind": "front"}, "t": [0.25 * i for i in range(n)],
            "rows": {"U": rows}, "events": {"contact": {"U": 0.0}}}


def test_timer_measure_reads_the_delays_and_the_spells_at_zero():
    # wavering at 0.5 s, <= 0 at 1.0 s, back above 0 at 1.5 s, <= 0 again at 2.0 s, rout at 3.0 s
    mp_ = [0.3, 0.2, 0.1, 0.05, 0.0, -0.01, 0.02, 0.01, -0.02, -0.03, -0.05, -0.06, -0.07]
    w = [False, False, True, True, True, True, True, True, True, True, True, True, False]
    r = [False] * 12 + [True]
    row = mp.timer_measure(_timer_lane(mp_, w, r))
    assert row["waver_s"] == 0.5 and row["zero_s"] == 1.0 and row["rout_s"] == 3.0
    assert row["waver_to_rout_s"] == 2.5 and row["zero_to_rout_s"] == 2.0
    assert row["zero_spells_no_rout_s"] == [0.5]
    assert row["zero_run_before_rout_s"] == 1.0 and row["mp_before_rout"] == -0.06 and row["ms_before_rout"] == 5
    assert row["recovered"] is False


def test_timer_summary_splits_the_rule_variants():
    routed = mp.timer_measure(_timer_lane([0.1, 0.0, -0.1], [True, True, False], [False, False, True], variant="a=1"))
    back = mp.timer_measure(_timer_lane([0.1] + [0.2] * 12, [True] + [False] * 12, [False] * 13, variant="a=1"))
    table = mp.timer_summary([routed, back])
    assert set(table) == {"a=1"}
    block = table["a=1"]
    assert block["lanes"] == 2 and block["routed"] == 1 and block["recovered"] == 1 and block["wavered_no_rout"] == 1
    assert block["waver_to_rout_s"]["median"] == 0.5


def test_a_run_s_variant_is_its_build_s_kv_label():
    assert mp.variant({}) == "normal" and mp.variant({"kv_label": "waver_base_timeout=250"}) == "waver_base_timeout=250"
