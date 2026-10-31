"""tools/nn/mem_probe.py (the projectiles from the game's memory against the script's ammo) and tools/mem_sites.py (the
sites file outside Git -> the probe's config), on made-up data: no game, no real addresses."""
import json

import pytest

from tools import mem_sites
from tools.nn import mem_probe


def test_split_is_exact_for_a_float32_lua():
    hi, lo = mem_sites.split("0x2fedcb8")
    assert hi * 2 ** 20 + lo == 0x2fedcb8 and hi < 2 ** 24 and lo < 2 ** 20


def test_lua_config_takes_only_the_named_sites_and_offsets(tmp_path):
    sites = {name: {"rva": hex(0x1000 * (i + 1)), "sig": "", "expect": ""} for i, name in enumerate(mem_sites.SITES)}
    sites["proj_update"] = {"rva": "0x123456", "sig": "void(pointer)", "expect": "48 89", "pattern": "48 89 ??"}
    raw = {"sites": dict(sites, other={"rva": "0x1"}),
           "layout": {name: hex(8 * i) for i, name in enumerate(mem_sites.LAYOUT)}}
    path = tmp_path / "sites.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    cfg = mem_sites.lua_config(path)
    assert set(cfg["sites"]) == set(mem_sites.SITES)
    assert cfg["sites"]["proj_update"]["hi"] == 1 and cfg["sites"]["proj_update"]["lo"] == 0x23456
    assert cfg["layout"]["p_shooter"] == 8 and set(cfg["layout"]) == set(mem_sites.LAYOUT)
    with pytest.raises(FileNotFoundError):
        mem_sites.lua_config(tmp_path / "none.json")


def test_dependency_is_pinned():
    d = mem_sites.dependency()
    assert d["pack"] == "memreader_plus.pack" and len(d["sha256"]) == 64


def test_bins_match_takes_one_bin_of_slack():
    assert mem_probe.bins_match([0, 5, 0], [0, 0, 5]) == 5
    assert mem_probe.bins_match([5, 0, 0], [0, 0, 5]) == 0
    assert mem_probe.bins_match([3, 3], [6, 0]) == 6


def _run(tmp_path, spawn_lane2=0):
    run = tmp_path / "20261101-000000"
    run.mkdir()
    lanes = [{"name": "L1", "x": 0, "z": -80}, {"name": "L2", "x": 240, "z": -80}]
    (run / "manifest.json").write_text(json.dumps({"config": {"lanes": lanes}}), encoding="utf-8")
    ev = [{"event": "mem_attach", "status": {"ok": True}}]
    ammo = 100
    for k in range(10):
        ms = 4000 + 500 * k
        fired = 10 if k in (3, 6) else 0
        ammo -= fired
        ev.append({"event": "probe_sample", "model_ms": ms, "lanes": [
            {"lane": "L1", "s": {"x": 0, "z": -80, "ammo": ammo}, "tg": {"x": 0, "z": -30, "hp": 1000}},
            {"lane": "L2", "s": {"x": 240, "z": -80, "ammo": 50}, "tg": {"x": 240, "z": -30, "hp": 1000}}]})
        c = 5 * (k + 1)                                    # 10 simulation ticks a second
        spawn = []
        if fired:
            spawn = [[100 * k + i, c - 2, 0, 0.5 * i, 1.5, -79, 1, 9, 40, 0, 1, 0, 0, -80] for i in range(fired)]
        if k == 6 and spawn_lane2:
            spawn += [[900 + i, c - 2, 0, 240, 1.5, -79, 1, 9, 40, 0, 1, 240, 0, -80] for i in range(spawn_lane2)]
        state = [[100 * 3 + i, c, 5, 0.5 * i, 0, -31, 1.2] for i in range(10)] if k == 5 else []
        ev.append({"event": "mem_proj", "model_ms": ms, "calls": c, "spawn": spawn, "state": state, "gone": [],
                   "fly": [], "errors": [], "records": [], "hook": {"calls": c, "other_thread_calls": 0}})
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n", encoding="utf-8")
    return run


def test_memory_and_script_agree_on_a_lane(tmp_path):
    r = mem_probe.compare(mem_probe.load(_run(tmp_path)))
    assert r["attach_ok"] and r["spawned"] == 20
    assert r["lanes"]["L1"]["memory"] == 20 and r["lanes"]["L1"]["script"] == 20
    assert r["lanes"]["L1"]["share"] == 1.0 and r["matched_share"] == 1.0
    assert r["lanes"]["L1"]["within_15m"] == 1.0                      # came down at the target (z -31 vs -30)
    assert r["spawn_to_shooter_m"]["unattributed"] == 0


def test_projectiles_the_script_did_not_count_lower_the_share(tmp_path):
    r = mem_probe.compare(mem_probe.load(_run(tmp_path, spawn_lane2=5)))
    assert r["lanes"]["L2"]["memory"] == 5 and r["lanes"]["L2"]["script"] == 0 and r["lanes"]["L2"]["share"] == 0
    assert r["matched_share"] == pytest.approx(20 / 25)
