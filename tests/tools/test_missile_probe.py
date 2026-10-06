"""The missile probe's plans, battle files and measures (tools/nn/missile_probe.py). The simulator twin needs torch
(run in the container)."""
import numpy as np

from tools.nn import missile_probe as mp


def test_plans_have_few_battles_and_lanes_on_both_sides():
    assert len(mp.battles("dist")) == 4
    for plan in mp.PLANS:
        for i in range(1, len(mp.battles(plan)) + 1):
            config, model_s, arena, _ = mp.run_config(plan, i)
            assert 2 <= len(config["lanes"]) <= 5 and model_s < 400
            names = {f"{side}_{u['slot']}" for side in ("own", "enemy") for u in arena["sides"][side]["units"]}
            for lane in config["lanes"]:
                assert lane["shooter"] in names and lane["target"] in names
                assert lane["shooter"].split("_")[0] != lane["target"].split("_")[0]   # two sides
            used = {n for l in config["lanes"] for n in (l["shooter"], l["target"])}
            assert {p["name"] for p in config["park"]} == {"own_lord", "enemy_lord"} - used
            xs = sorted(l["x"] for l in config["lanes"])
            assert all(b - a >= 200 for a, b in zip(xs, xs[1:]))


def test_every_shooter_at_every_distance_once():
    seen = {}
    for b in mp.battles("dist"):
        for l in b:
            seen.setdefault(l["shooter"], []).append(l["d"])
    assert {k: sorted(v) for k, v in seen.items()} == {k: sorted(map(float, v)) for k, v in mp.DIST.items()}
    rng = {"archers": 130, "militia": 90, "slingers": 120, "nr": 140}      # the database's ranges
    assert all(l["d"] < rng[l["shooter"]] for b in mp.battles("dist") for l in b)


def test_battle_file_sets_the_ranks(tmp_path):
    xml = mp.write_scenario("rank", 1, tmp_path / "x.xml").read_text(encoding="utf-8")
    assert xml.count('<unit_experience level="9"/>') == 2 and xml.count("<unit ") == 2 + 8


def test_volleys_group_drops_within_a_second_and_a_half():
    t = np.arange(0, 30, 0.5)
    ammo = np.full(len(t), 1800.0)
    for start in (3.0, 13.0, 23.0):
        ammo[t >= start] -= 60
        ammo[t >= start + 0.5] -= 30
    vs = mp.volleys(t, ammo, np.full(len(t), 90.0))
    assert [(a, b) for a, b in vs] == [(3.0, 90.0), (13.0, 90.0), (23.0, 90.0)]


def test_measure_a_lane():
    spec = dict(mp.lane("archers", "slave", 100), name="L1", x=0.0, z=-80, s_b=0.0, s_key=mp.UNITS["archers"][0],
                t_key=mp.UNITS["slave"][0])
    samples = []
    for k in range(80):
        t = k * 0.5
        fired = 90 * (1 + int((t - 3) // 10)) if t >= 3 else 0
        lost = 0.5 * 16 * fired
        samples.append((t, {"x": 0, "z": -80, "b": 0, "men": 90, "ammo": 1800 - fired},
                        {"x": 0, "z": 20, "b": 180, "men": 180 - lost / 50, "hp": 9000 - lost}, 100))
    m = mp.measure({"run": "x", "spec": spec, "samples": samples, "end": None}, per_hit=16.0)
    assert m["first_s"] == 3.0 and abs(m["first_d"] - 100) < 1e-6 and m["first_share"] == 1.0
    assert m["interval_s"] == 10.0 and abs(m["hits_per_shot"] - 0.5) < 1e-9
