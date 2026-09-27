"""Smoke tests: run entries against a fake battle manager (tests/entries/fake_battle.lua).

They check the wiring between apps, not game behaviour: the real engine is
only exercised by tools/launcher/launch.ps1.
"""
import json
from pathlib import Path

import pytest

from tests.lua_runtime import new_runtime

FAKE = (Path(__file__).parent / "fake_battle.lua").read_text(encoding="utf-8")


@pytest.fixture
def lua(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # entries write their files to the working directory
    runtime = new_runtime()
    runtime.globals().fake = runtime.execute(FAKE)
    return runtime


def events(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class TestDuel:
    def test_duel_charges_and_records_result(self, lua, tmp_path):
        lua.execute("""
            local a = fake.unit('bai_ranged_a', 'wh3_main_ksl_inf_kossars_0', -50, 0)
            local b = fake.unit('bai_ranged_b', 'wh3_main_ksl_inf_kossars_0', 50, 0)
            bm = fake.manager({{a}, {b}})
            find_uicomponent = function() return nil end
            local duel = require('entries.duel')
            local state = duel.main(bm, {build='test', runs=1, speed=20, timeout_ms=60000, tick_ms=1000})
            bm:pump()             -- deployment -> end phase -> start
            bm:tick(); bm:tick()  -- decisions
            bm.outcome, bm.winner = true, 1
            bm:tick()
            assert(state.finished, 'duel did not finish')
            assert(a.attacked > 0 and b.attacked > 0, 'no attack orders')
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "error" not in kinds, rows[-1]
        assert kinds[:2] == ["loaded", "ready"]
        result = next(r for r in rows if r["event"] == "result")
        assert result["status"] == "completed" and result["winner"] == 1
        assert {r["reason"] for r in rows if r["event"] == "decision"} >= {"initial_charge"}
        # 'loaded' is written before the scenario is recognised by unit names.
        assert all(r["build"] == "test" for r in rows)
        assert all(r["scenario"] == "ranged_melee" for r in rows[1:])


class TestMapCapture:
    def test_grid_and_features(self, lua, tmp_path):
        lua.execute("""
            local a = fake.unit('bai_map_probe_one', 'cav', -50, 0)
            local b = fake.unit('bai_map_probe_two', 'inf', 50, 0)
            bm = fake.manager({{a}, {b}})
            require('entries.map_capture').main(bm, {step = 20, features = true},
                {common = fake.common, battle_vector = fake.vector_type})
            bm:pump()
        """)
        rows = events(tmp_path / "tww3_bai_map_capture_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "probe_error" not in kinds, [r for r in rows if r["event"] == "probe_error"]
        assert kinds[-1] == "probe_done"
        frame = next(r for r in rows if r["event"] == "frame")
        assert round(frame["min_x"]) == -100 and round(frame["max_z"]) == 100
        grid = (tmp_path / "tww3_bai_map_capture_grid.csv").read_text().splitlines()
        assert grid[0] == "ix,iz,x,z,height,clear,ground,inside_radar,reach_side_1,reach_side_2"
        assert len(grid) == 1 + 10 * 10
        assert grid[1].endswith(',1,"grass",1,1,1')
