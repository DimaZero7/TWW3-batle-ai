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


class TestAiVsAi:
    def test_player_side_goes_to_ai_planner_and_result_is_recorded(self, lua, tmp_path):
        lua.execute("""
            local a1 = fake.unit('bai_a_kossars_1', 'kossars', 100, 0)
            local a2 = fake.unit('bai_a_lancers', 'lancers', 150, 0)
            local b1 = fake.unit('bai_b_kossars_1', 'kossars', -100, 0)
            bm = fake.manager({{a1, a2}, {b1}})
            -- side 1 is the player's army, side 2 already belongs to the battle AI
            local armies = {}
            for i = 1, 2 do armies[i] = bm:alliances():item(i):armies():item(1) end
            armies[1].is_player_controlled = function() return true end
            armies[2].is_player_controlled = function() return false end
            -- CA library fake: records what the planner was told
            attacks = 0
            script_ai_planner = {new = function(_, name, sunits)
                assert(#sunits == 2, 'own units expected')
                return {attack_force = function(_, foes) assert(#foes == 1); attacks = attacks + 1 end,
                    release = function() end}
            end}
            bm.get_scriptunit_for_unit = function(_, u) return {unit = u} end
            local state = require('entries.ai_vs_ai').main(bm, {build='t', speed=3, timeout_ms=600000, tick_ms=1000})
            bm:pump()
            for _ = 1, 16 do bm:tick() end
            bm.speed = 1              -- the engine slows down once units flee
            bm:tick()
            assert(bm.speed == 3, 'speed guard did not restore x3')
            bm.outcome, bm.winner = true, 2
            bm:tick()
            assert(state.finished and attacks >= 2, 'planner not used or not re-issued: ' .. attacks)
            bm.speed = 1              -- victory countdown after the result: still restored
            bm:tick()
            assert(bm.speed == 3, 'speed not kept after the result')
            bm:set_phase('Complete')
            bm.speed = 1
            bm:tick()
            assert(bm.speed == 1, 'guard must stop at Complete')
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "error" not in kinds, rows[-1]
        ready = next(r for r in rows if r["event"] == "ready")
        assert ready["side_1_player_controlled"] is True and ready["side_2_player_controlled"] is False
        assigned = {r["side"]: r["controller"] for r in rows if r["event"] == "ai_assigned"}
        assert assigned == {1: "script_ai_planner", 2: "general_battle_ai"}
        result = next(r for r in rows if r["event"] == "result")
        assert result["status"] == "completed" and result["winner"] == 2
        assert "progress" in kinds
        restored = [r for r in rows if r["event"] == "speed_restored"]
        assert len(restored) == 2 and restored[0]["from_speed"] == 1 and restored[0]["to_speed"] == 3


class TestUnitReadout:
    def test_all_stages_run_and_report_is_built(self, lua, tmp_path):
        lua.execute("""
            local sides = {{}, {}}
            for _, n in ipairs({'a_general', 'a_spears', 'a_archers'}) do table.insert(sides[1], fake.unit(n, n, 50, 0)) end
            for _, n in ipairs({'b_general', 'b_spears', 'b_archers'}) do table.insert(sides[2], fake.unit(n, n, -50, 0)) end
            bm = fake.manager(sides)
            local state = require('entries.unit_readout').main(bm, {build='t', speed=3, tick_ms=1000},
                {common = fake.common, battle_vector = fake.vector_type})
            bm:pump()
            for _ = 1, 160 do bm:tick() end
            assert(state.finished, 'readout did not finish')
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = {r["event"] for r in rows}
        assert "error" not in kinds, [r for r in rows if r["event"] == "error"]
        assert {"unit_profile", "unit_state", "enemy_gate", "range", "intel", "sampler_frame",
                "nav_state", "stage", "result"} <= kinds
        assert [r["name"] for r in rows if r["event"] == "stage"] == ["idle", "march", "ranged", "cease_fire", "melee", "halt"]
        # The report must build from the log (the fake cannot satisfy game checks).
        from tools.analysis import unit_readout
        (tmp_path / "events.jsonl").write_text((tmp_path / "tww3_bai_events.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        checks, table, _ = unit_readout.analyse(unit_readout.load(tmp_path))
        assert len(table) == 68 and checks
