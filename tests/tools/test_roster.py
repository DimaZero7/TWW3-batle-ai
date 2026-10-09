"""tools.roster: scenario from the capture list, roster files from a capture log."""
import json
import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("matplotlib", reason="matplotlib: on the host, not in the training container")

from tools import roster  # noqa: E402


def capture(tmp_path, units):
    path = tmp_path / "capture.json"
    path.write_text(json.dumps({"faction": "wh_main_emp_empire", "units": units, "slots": [[0, 0], [100, 0]],
                                "widths": [40, 10], "baseline_width": 30, "shape_timeout_s": 40}))
    return roster.load_capture(path)


def test_scenario_lists_units_with_slots_and_general(tmp_path):
    spec = capture(tmp_path, [{"key": "lord_key", "men": 1, "general": True}, {"key": "inf_key", "men": 120}])
    tree = ET.fromstring(roster.scenario_xml(spec).encode("utf-8"))
    units = tree.find("alliance/army").findall("unit")
    assert [u.get("script_name") for u in units] == ["roster_1", "roster_2"]
    assert units[0].find("general") is not None and units[1].find("general") is None
    assert units[1].find("position").get("x") == "100" and units[1].get("num_soldiers") == "120"
    config, model_s = roster.run_config(spec, 20, 2000)
    assert model_s == 2 * 42 and config["units"][1]["slot"] == {"x": 100, "z": 0}


def test_too_many_units_or_duplicates_are_rejected(tmp_path):
    with pytest.raises(AssertionError):
        capture(tmp_path, [{"key": "a", "men": 1}, {"key": "b", "men": 1}, {"key": "c", "men": 1}])
    with pytest.raises(AssertionError):
        capture(tmp_path, [{"key": "a", "men": 1}, {"key": "a", "men": 1}])


def test_update_keeps_hand_filled_part(tmp_path):
    out = tmp_path / "roster"
    out.mkdir()
    (out / "inf_key.json").write_text(json.dumps({"ours": {"fire": "arc", "roles": ["x"], "notes": "x"}}))
    soldiers = [0, 0, 100, 0, 0, -20, 100, -20]  # 10 m front, 2 m deep, bearing 0
    rows = [
        {"event": "ready", "wall_iso": "2026-09-27T00:00:00Z", "game_version": "v", "build": "b", "batch": "r"},
        {"event": "unit_card", "key": "inf_key", "profile": {"initial_number_of_men": 4}, "details": {"Mass": 60},
         "stats": {"status": "ok", "list": [{"key": "stat_armour", "Value": 30, "DisplayedValue": 30, "ValueBase": 25}]}},
        {"event": "shape_result", "key": "inf_key", "width": 10, "reason": "arrived", "last_moving_ms": 4000,
         "motion": {"x": 5, "z": -1, "bearing": 0}, "soldiers_dm": soldiers},
    ]
    log = tmp_path / "events.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows))
    roster.update(log, out)
    entry = json.loads((out / "inf_key.json").read_text(encoding="utf-8"))
    assert entry["ours"] == {"fire": "arc", "roles": ["x"], "notes": "x"}
    assert entry["card"]["stats"]["stat_armour"] == {"value": 30, "displayed": 30, "base": 25}
    assert entry["formation"]["widths"] == [{"ordered_m": 10, "front_m": 10.0, "depth_m": 2.0, "reform_s": 5.0,
                                             "ended": "arrived", "men": 4}]
    roster.check([entry])
    entry["ours"]["fire"] = "sideways"
    with pytest.raises(AssertionError):
        roster.check([entry])
