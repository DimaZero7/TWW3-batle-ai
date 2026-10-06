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
            local state = require('entries.ai_vs_ai').main(bm, {build='t', speed=3, timeout_ms=600000, tick_ms=1000, deadline_s=60, stall_ms=600000})
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
            for _, n in ipairs({'a_general', 'a_spears', 'a_archers', 'a_forest', 'a_stalkers'}) do table.insert(sides[1], fake.unit(n, n, 50, 0)) end
            for _, n in ipairs({'b_general', 'b_spears', 'b_archers', 'b_forest', 'b_stalkers'}) do table.insert(sides[2], fake.unit(n, n, -50, 0)) end
            bm = fake.manager(sides)
            local state = require('entries.unit_readout').main(bm, {build='t', speed=3, tick_ms=1000, deadline_s=60, stall_ms=600000},
                {common = fake.common, battle_vector = fake.vector_type})
            bm:pump()
            for _ = 1, 220 do bm:tick() end
            assert(state.finished, 'readout did not finish')
            assert(bm.ended, 'battle must end, not run forever')
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = {r["event"] for r in rows}
        assert "error" not in kinds, [r for r in rows if r["event"] == "error"]
        assert {"unit_profile", "enemy_gate", "full_view", "side_view",
                "nav_state", "stage", "result"} <= kinds
        assert [r["name"] for r in rows if r["event"] == "stage"] == ["idle", "march", "ranged", "cease_fire", "melee", "scout_a80", "scout_a40", "scout_a15",
                 "scout_b80", "scout_b40", "scout_b15", "halt"]
        # The report must build from the log (the fake cannot satisfy game checks).
        from tools.analysis import unit_readout
        (tmp_path / "events.jsonl").write_text((tmp_path / "tww3_bai_events.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        checks, table, _ = unit_readout.analyse(unit_readout.load(tmp_path))
        assert len(table) == 68 and checks
        side_checks, _, _ = unit_readout.analyse_side(unit_readout.load(tmp_path), 1)
        leaks = [c for c in side_checks if c["group"] == "leaks"]
        assert leaks and all(c["result"] == "pass" for c in leaks)


class TestDeadline:
    def test_readout_ends_itself_when_the_deadline_expires(self, lua, tmp_path):
        lua.execute("""
            local sides = {{}, {}}
            for _, n in ipairs({'a_general', 'a_spears', 'a_archers', 'a_forest', 'a_stalkers'}) do table.insert(sides[1], fake.unit(n, n, 50, 0)) end
            for _, n in ipairs({'b_general', 'b_spears', 'b_archers', 'b_forest', 'b_stalkers'}) do table.insert(sides[2], fake.unit(n, n, -50, 0)) end
            bm = fake.manager(sides)
            local state = require('entries.unit_readout').main(bm, {build='t', speed=3, tick_ms=1000, deadline_s=5, stall_ms=600000},
                {common = fake.common, battle_vector = fake.vector_type})
            bm:pump()          -- deployment ends, battle starts, deadline is queued
            for _ = 1, 3 do bm:tick() end
            bm:fire_timers()   -- wall-clock deadline expires
            assert(state.finished and bm.ended, 'deadline did not end the battle')
        """)
        result = [r for r in events(tmp_path / "tww3_bai_events.jsonl") if r["event"] == "result"]
        assert [r["status"] for r in result] == ["deadline"]


class TestStall:
    def test_ai_vs_ai_ends_when_nobody_takes_damage(self, lua, tmp_path):
        lua.execute("""
            local a = fake.unit('bai_a_kossars_1', 'kossars', 100, 0)
            local b = fake.unit('bai_b_kossars_1', 'kossars', -100, 0)
            bm = fake.manager({{a}, {b}})
            bm:alliances():item(1):armies():item(1).is_player_controlled = function() return false end
            bm:alliances():item(2):armies():item(1).is_player_controlled = function() return false end
            local state = require('entries.ai_vs_ai').main(bm,
                {build='t', speed=20, timeout_ms=3600000, tick_ms=1000, deadline_s=600, stall_ms=30000})
            bm:pump()
            for _ = 1, 20 do bm:tick() end
            a.men = 110                        -- damage resets the quiet period
            for _ = 1, 25 do bm:tick() end
            assert(not state.finished, 'stalled too early')
            for _ = 1, 10 do bm:tick() end
            assert(state.finished, 'stall not detected')
        """)
        result = [r for r in events(tmp_path / "tww3_bai_events.jsonl") if r["event"] == "result"]
        assert [r["status"] for r in result] == ["stalled"]


class TestMoveProbe:
    PLAN = """{name = 'test', monitor = {arrive_m = 5},
        legs = {
            {name = 'shape_w10', kind = 'shape', run = false, timeout_s = 25,
             start = {x = 0, z = 0, facing = 0, width = 30}, target = {x = 0, z = 0, facing = 0, width = 10}},
            {name = 'through', kind = 'traverse', run = true, timeout_s = 70,
             start = {x = 0, z = -50, facing = 0, width = 30}, target = {x = 0, z = 60, facing = 0, width = 30}},
        }}"""

    def test_legs_run_in_order_and_soldiers_are_logged(self, lua, tmp_path):
        lua.execute(f"""
            local a = fake.unit('probe_spears', 'wh_main_emp_inf_spearmen_0', 0, -50)
            local b = fake.unit('far_general', 'wh_main_emp_cha_general_0', 300, -400)
            bm = fake.manager({{{{a}}, {{b}}}})
            local state = require('entries.move_probe').main(bm, {{build = 'test', speed = 20, tick_ms = 1000,
                deadline_s = 100, stall_ms = 900000, settle_ms = 2000, plan = {self.PLAN}}},
                {{common = fake.common, battle_vector = fake.vector_type}})
            bm:pump()
            for _ = 1, 20 do bm:tick() end
            assert(state.finished, 'probe did not finish')
            assert(bm.ended, 'battle was not ended')
            assert(a:position():get_z() == 60)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "error" not in kinds, [r for r in rows if r["event"] == "error"]
        ends = [r for r in rows if r["event"] == "leg_end"]
        assert [(e["leg"], e["reason"]) for e in ends] == [("shape_w10", "arrived"), ("through", "arrived")]
        sample = next(r for r in rows if r["event"] == "move_sample")
        assert sample["soldiers_dm"] == [13, -25, 13, -25] and sample["motion"]["ordered_width"] == 30
        result = next(r for r in rows if r["event"] == "result")
        assert result["status"] == "completed" and result["legs_done"] == 2

    def test_deadline_ends_the_probe(self, lua, tmp_path):
        lua.execute(f"""
            local a = fake.unit('probe_spears', 'inf', 0, -50)
            local b = fake.unit('far_general', 'lord', 300, -400)
            bm = fake.manager({{{{a}}, {{b}}}})
            local state = require('entries.move_probe').main(bm, {{build = 'test', speed = 20, tick_ms = 1000,
                deadline_s = 100, stall_ms = 900000, settle_ms = 2000, plan = {self.PLAN}}},
                {{common = fake.common, battle_vector = fake.vector_type}})
            bm:pump()
            bm:tick()
            bm:fire_timers()
            assert(state.finished and bm.ended)
        """)
        result = next(r for r in events(tmp_path / "tww3_bai_events.jsonl") if r["event"] == "result")
        assert result["status"] == "deadline" and result["legs_done"] == 0


class TestManualRecord:
    def run(self, lua, body):
        lua.execute("""
            units = {}
            for i = 1, 6 do units[i] = fake.unit('spears_' .. i, 'wh_main_emp_inf_spearmen_0', -84 + i * 10, -160) end
            general = fake.unit('far_general', 'lord', 300, -400)
            bm = fake.manager({units, {general}})
            state = require('entries.manual_record').main(bm, {build = 'test', tick_ms = 1000,
                deadline_s = 3600, stall_ms = 1800000}, {common = fake.common, battle_vector = fake.vector_type})
            bm:pump()
        """ + body)

    def test_records_player_orders_without_controlling_them(self, lua, tmp_path):
        self.run(lua, """
            assert(bm.phase == 'Deployment', 'deployment must stay with the player')
            bm:set_phase('Deployed')
            bm:tick()
            local target = fake.vector_type.new(); target:set_x(-84); target:set_z(-40)
            units[1].ordered_position = function() return target end
            for _ = 1, 5 do bm:tick() end
            assert(not units[1]:is_script_controlled() and general:is_script_controlled())
            bm:set_phase('Complete')
            assert(state.finished and not bm.ended)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "error" not in kinds, [r for r in rows if r["event"] == "error"]
        order = next(r for r in rows if r["event"] == "order_seen")
        assert order["unit"] == "spears_1" and order["z"] == -40
        assert next(r for r in rows if r["event"] == "order_end")["reason"] == "stopped"
        sample = next(r for r in rows if r["event"] == "own_sample")
        assert len(sample["units"]) == 6 and sample["units"][0]["soldiers_dm"]
        assert rows[-1]["event"] == "result" and rows[-1]["status"] == "battle_ended"

    def test_deadline_ends_the_battle(self, lua, tmp_path):
        self.run(lua, """
            bm:set_phase('Deployed')
            bm:tick()
            bm:fire_timers()
            assert(state.finished and bm.ended)
        """)
        assert events(tmp_path / "tww3_bai_events.jsonl")[-1]["status"] == "deadline"


class TestRosterCapture:
    def test_cards_and_shapes_are_logged(self, lua, tmp_path):
        lua.execute("""
            local lord = fake.unit('roster_1', 'wh_main_emp_cha_general_0', 0, -100)
            function lord:initial_number_of_men() return 1 end
            local spears = fake.unit('roster_2', 'wh_main_emp_inf_spearmen_0', 100, -200)
            local general = fake.unit('far_general', 'lord', 300, -400)
            bm = fake.manager({{lord, spears}, {general}})
            local state = require('entries.roster_capture').main(bm, {build = 'test', speed = 20, tick_ms = 1000,
                deadline_s = 100, stall_ms = 900000, settle_ms = 2000, widths = {40, 10}, baseline_width = 30,
                shape_timeout_s = 40, units = {
                    {script_name = 'roster_1', key = 'wh_main_emp_cha_general_0', slot = {x = 0, z = -100}},
                    {script_name = 'roster_2', key = 'wh_main_emp_inf_spearmen_0', slot = {x = 100, z = -200}}}},
                {common = fake.common, battle_vector = fake.vector_type})
            bm:pump()
            for _ = 1, 20 do bm:tick() end
            assert(state.finished and bm.ended)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows], [r for r in rows if r["event"] == "error"]
        cards = [r for r in rows if r["event"] == "unit_card"]
        assert [c["key"] for c in cards] == ["wh_main_emp_cha_general_0", "wh_main_emp_inf_spearmen_0"]
        assert cards[1]["stats"]["list"][0] == {"key": "stat_armour", "Value": 30, "DisplayedValue": "unknown:nil",
                                                "ValueBase": "unknown:nil"}
        assert cards[1]["details"]["Mass"] == 60 and cards[1]["profile"]["slow_speed"] == 4
        shapes = [(r["key"], r["width"]) for r in rows if r["event"] == "shape_result"]
        assert shapes == [("wh_main_emp_inf_spearmen_0", 40), ("wh_main_emp_inf_spearmen_0", 10)]
        assert rows[-1]["event"] == "result" and rows[-1]["status"] == "completed"


class TestEnemyLayout:
    def test_game_ai_side_is_watched_not_commanded(self, lua, tmp_path):
        lua.execute("""
            local own = fake.unit('own_1', 'wh_main_emp_inf_spearmen_0', 0, -250)
            local enemy = fake.unit('enemy_1', 'wh_main_emp_inf_spearmen_0', 0, 200)
            bm = fake.manager({{own}, {enemy}})
            local state = require('entries.enemy_layout').main(bm, {build = 'test', speed = 20, tick_ms = 1000,
                deadline_s = 100, stall_ms = 900000, hold_s = 5, layout = 'test'},
                {common = fake.common, battle_vector = fake.vector_type})
            bm:pump()
            for _ = 1, 10 do bm:tick() end
            assert(state.finished and bm.ended)
            assert(own:is_script_controlled() and not enemy:is_script_controlled())
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows]
        assert [r["stage"] for r in rows if r["event"] == "enemy_snapshot"] == ["deployment", "deployed", "end"]
        sample = next(r for r in rows if r["event"] == "enemy_sample")
        assert sample["units"][0]["seen"] is True and sample["layout"] == "test"
        assert rows[-1]["status"] == "completed"


class TestNnArena:
    # Two units a side; the config as tools/nn/scenario.run_config gives it (script names, slots).
    SETUP = """
        own = {fake.unit('own_lord', 'lord', -235, 0), fake.unit('own_spear_1', 'spears', -175, 0)}
        enemy = {fake.unit('enemy_lord', 'lord', 235, 0), fake.unit('enemy_spear_1', 'spears', 175, 0)}
        bm = fake.manager({own, enemy})
        local function slots(side)
            return {{script_name = side .. '_lord', slot = 'lord', key = 'lord'},
                {script_name = side .. '_spear_1', slot = 'spear_1', key = 'spears'}}
        end
        CONFIG = {build = 'test', speed = 20, tick_ms = 1000, deadline_s = 600, stall_ms = 600000,
            timeout_ms = 600000, defend_radius_m = 150, units = {own = slots('own'), enemy = slots('enemy')}}
        GLOBALS = {common = fake.common, battle_vector = fake.vector_type}
    """

    def test_attack_records_every_unit_and_the_result(self, lua, tmp_path):
        # No CA library loaded: the planner falls back to the engine's AI unit planner.
        lua.execute(self.SETUP + """
            CONFIG.own_ai = 'attack'
            local state = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
            bm:pump()
            for _ = 1, 16 do bm:tick() end         -- the attack is re-issued after 15 s
            own[2].routing = true; bm:tick()
            own[2].routing = false; bm:tick()       -- rallied: back into the planner
            for _ = 1, 8 do own[2].men = own[2].men - 1; bm:tick() end  -- idle under fire
            bm:pump()                               -- the last order is repeated
            bm.outcome, bm.winner = true, 2
            bm:tick()
            assert(state.finished, 'the battle did not finish')
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "error" not in kinds, [r for r in rows if r["event"] == "error"]
        assert kinds[:2] == ["loaded", "ready"]
        assert next(r for r in rows if r["event"] == "own_ai")["mode"] == "engine_planner"
        samples = [r for r in rows if r["event"] == "nn_sample"]
        assert len(samples) >= 20
        assert {(u["n"], u["side"]) for u in samples[-1]["units"]} == {
            ("own_lord", 1), ("own_spear_1", 1), ("enemy_lord", 2), ("enemy_spear_1", 2)}
        assert samples[-1]["units"][1]["men"] < 120 and samples[-1]["units"][1]["x"] == -175
        assert [r["units"] for r in rows if r["event"] == "rejoined"] == [["own_spear_1"]]
        assert any(r["event"] == "idle_kick" and r["unit"] == "own_spear_1" for r in rows)
        log = list(lua.eval("bm.planner_log").values())
        assert log[:2] == ["add own_lord", "add own_spear_1"]
        assert log.count("attack enemy_spear_1") >= 3   # the start, 15 s later, after the rally
        assert kinds[-2:] == ["nn_final", "result"]
        result = rows[-1]
        assert result["status"] == "completed" and result["winner"] == 2
        assert result["rejoined"] == 1 and result["idle_kicks"] >= 1
        assert result["side_1_men"] < result["side_2_men"] == 240
        assert all(r["policy"] == "nn_arena_attack" and r["scenario"] == "nn_arena" for r in rows)

    def test_defend_holds_its_place_until_the_timeout(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            CONFIG.own_ai, CONFIG.timeout_ms = 'defend', 20000
            local state = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
            bm:pump()
            for _ = 1, 25 do bm:tick() end
            assert(state.finished and bm.ended, 'the timeout did not end the battle')
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows], [r for r in rows if r["event"] == "error"]
        log = list(lua.eval("bm.planner_log").values())
        assert "defend 150" in log and not any(e.startswith("attack") for e in log)
        assert rows[-1]["event"] == "result" and rows[-1]["status"] == "timeout" and rows[-1]["winner"] == 0

    def test_hold_gives_no_orders_and_each_side_has_its_own_army(self, lua, tmp_path):
        # A named arena: one Skaven unit of ours holds, two Empire units of the game's AI.
        lua.execute("""
            own = {fake.unit('own_slave', 'slaves', -125, 0)}
            enemy = {fake.unit('enemy_archer', 'archers', 125, 0), fake.unit('enemy_lord', 'lord', 185, 0)}
            bm = fake.manager({own, enemy})
            CONFIG = {build = 'test', speed = 20, tick_ms = 1000, deadline_s = 600, stall_ms = 600000,
                timeout_ms = 600000, defend_radius_m = 150, own_ai = 'hold', units = {
                    own = {{script_name = 'own_slave', slot = 'slave', key = 'slaves'}},
                    enemy = {{script_name = 'enemy_archer', slot = 'archer', key = 'archers'},
                             {script_name = 'enemy_lord', slot = 'lord', key = 'lord'}}}}
            local state = require('entries.nn_arena').main(bm, CONFIG,
                {common = fake.common, battle_vector = fake.vector_type})
            bm:pump()
            for _ = 1, 20 do own[1].men = own[1].men - 1; bm:tick() end   -- shot at, stands
            bm.outcome, bm.winner = true, 2
            bm:tick()
            assert(state.finished and state.planner == nil)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "error" not in kinds, [r for r in rows if r["event"] == "error"]
        assert next(r for r in rows if r["event"] == "own_ai")["mode"] == "hold"
        assert "idle_kick" not in kinds and "rejoined" not in kinds
        assert list(lua.eval("bm.planner_log").values()) == []
        sample = [r for r in rows if r["event"] == "nn_sample"][-1]
        assert [(u["n"], u["side"]) for u in sample["units"]] == [
            ("own_slave", 1), ("enemy_archer", 2), ("enemy_lord", 2)]
        assert rows[-1]["status"] == "completed" and rows[-1]["idle_kicks"] == 0
        assert all(r["policy"] == "nn_arena_hold" for r in rows)

    def test_samples_record_the_balance_of_power_and_the_morale_fields(self, lua, tmp_path):
        # The values for the army collapse and morale rules; a field that fails is left out.
        lua.execute(self.SETUP + """
            CONFIG.own_ai, CONFIG.timeout_ms = 'hold', 5000
            bm.get_player_alliance_num = function() return 1 end
            fake.root['BattleRoot.BalanceOfPowerPercent'] = 0.61234
            fake.root['BattleRoot.PlayerAllianceContext.Id'] = 0
            own[1].strategic_value = function() return 812.46 end
            own[2].strategic_value = function() error('no such method') end
            fake.cco['uid_own_lord'] = {PercentCasualtiesRecently = 0.125, PercentHpLostRecently = 0.03,
                MoraleGreatestEffect = 'Losses'}
            fake.cco['uid_own_spear_1'] = {MoraleGreatestEffect = ''}
            local state = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
            bm:pump()
            for _ = 1, 8 do bm:tick() end
            assert(state.finished)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows], [r for r in rows if r["event"] == "error"]
        ready = next(r for r in rows if r["event"] == "ready")
        assert ready["player_side"] == 1 and ready["player_alliance_cco"] == 0
        sample = [r for r in rows if r["event"] == "nn_sample"][-1]
        assert sample["bop"] == 0.6123 and sample["bop_side"] == 1
        lord, spear = sample["units"][0], sample["units"][1]
        assert (lord["sv"], lord["pcr"], lord["phr"], lord["mge"]) == (812.5, 0.125, 0.03, "Losses")
        assert not {"sv", "pcr", "phr", "mge"} & set(spear)

    def test_net_writes_the_state_and_gives_the_companions_orders(self, lua, tmp_path):
        # The test plays the companion (tools/nn/companion/exchange.py writes its answer).
        from tools.nn.companion import exchange
        lua.execute(self.SETUP + """
            CONFIG.own_ai, CONFIG.enemy_role, CONFIG.decide_ms, CONFIG.poll_ms = 'net', 'attack', 1000, 100
            CONFIG.factions = {own = 'wh_main_emp_empire', enemy = 'wh2_main_skv_skaven'}
            STATE = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
            bm:pump()
        """)
        doc = exchange.read_state(tmp_path / exchange.STATE)
        assert doc["move"] == 1 and doc["attacker"] == 2 and not doc["done"]
        assert [(u["n"], u["side"], u["key"], u["v"]) for u in doc["units"]] == [
            ("own_lord", 1, "lord", True), ("own_spear_1", 1, "spears", True),
            ("enemy_lord", 2, "lord", True), ("enemy_spear_1", 2, "spears", True)]
        orders = [{"unit": "own_lord", "kind": "move", "x": -200.0, "z": 30.0, "run": True},
                  {"unit": "own_spear_1", "kind": "attack", "target": "enemy_spear_1", "run": False}]
        exchange.write_atomic(tmp_path / exchange.ORDERS, exchange.orders_text(doc["batch"], 1, orders, 3.5))
        lua.execute("""
            bm:tick(100)                  -- the answer is read and given
            bm:tick(100)                  -- the same answer again: nothing new
            for _ = 1, 9 do bm:tick(100) end   -- the next decision: move 2 is written
            for _ = 1, 10 do bm:tick(100) end  -- no answer: a miss at move 3
        """)
        log = list(lua.eval("bm.orders").values())
        assert log == ["goto own_lord -200 30 true", "attack enemy_spear_1"]
        doc = exchange.read_state(tmp_path / exchange.STATE)
        assert doc["move"] == 3
        exchange.write_atomic(tmp_path / exchange.ORDERS, exchange.orders_text(
            doc["batch"], 3, [{"unit": "own_lord", "kind": "move", "x": -202.0, "z": 31.0, "run": True},
                              {"unit": "own_spear_1", "kind": "hold"}]))
        lua.execute("bm:tick(100)")
        exchange.write_atomic(tmp_path / exchange.ORDERS, exchange.orders_text(      # the answer to move 4 (taken once it is written): keep all
            doc["batch"], 4, [{"unit": "own_lord", "kind": "keep"}, {"unit": "own_spear_1", "kind": "keep"}]))
        lua.execute("""
            bm:tick(100)
            bm.outcome, bm.winner = true, 2
            for _ = 1, 10 do bm:tick(100) end
            assert(STATE.finished, 'the battle did not finish')
        """)
        log = list(lua.eval("bm.orders").values())
        assert log[2:] == ["halt"]          # the lord's point moved < 5 m: the same order, not given again
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "error" not in kinds, [r for r in rows if r["event"] == "error"]
        assert next(r for r in rows if r["event"] == "own_ai")["mode"] == "net"
        given = [r for r in rows if r["event"] == "nn_orders"]
        assert [g["move"] for g in given] == [1, 3, 4]
        assert given[2]["keeps"] == 2 and not given[2]["orders"]   # keep: nothing given
        assert [(o["u"], o["k"], o["status"]) for o in given[0]["orders"]] == [
            ("own_lord", "move", "given"), ("own_spear_1", "attack", "given")]
        assert given[0]["think_ms"] == 3.5 and given[0]["wait_model_ms"] == 100 and given[0]["lag"] == 0
        assert given[1]["kept"] == 1
        assert [r["move"] for r in rows if r["event"] == "nn_miss"] == [2]
        result = rows[-1]
        assert result["event"] == "result" and result["nn_moves"] == 4 and result["nn_answered"] == 3
        assert result["nn_missed"] == 1 and result["nn_orders_given"] == 3 and result["nn_keeps"] == 2
        final = exchange.read_state(tmp_path / exchange.STATE)
        assert final["done"] is True and final["move"] == 4   # the decision just before the end
        assert all(r["policy"] == "nn_arena_net" for r in rows)

    def test_net_runs_shooters_frees_idle_ones_and_gives_orders_again_after_a_rally(self, lua, tmp_path):
        # The gate's in-game findings (01.10.2026): a ranged attack must pass the run flag; a unit that
        # routed and rallied gets its order again; a shooter that cannot hit its target (the target in
        # melee) is released to fire at will and takes the target again once it is out of melee.
        from tools.nn.companion import exchange
        lua.execute(self.SETUP + """
            archer = fake.unit('own_archer_1', 'archers', -200, 40)
            archer.ammo, archer.range = 1800, 150
            table.insert(own, archer)
            table.insert(CONFIG.units.own, {script_name = 'own_archer_1', slot = 'archer_1', key = 'archers'})
            fake.cco['uid_own_archer_1'] = {IsFiringMissiles = true}
            CONFIG.own_ai, CONFIG.enemy_role, CONFIG.decide_ms, CONFIG.poll_ms = 'net', 'attack', 1000, 100
            CONFIG.factions = {own = 'wh_main_emp_empire', enemy = 'wh2_main_skv_skaven'}
            STATE = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
            bm:pump()
        """)
        orders = [{"unit": "own_spear_1", "kind": "attack", "target": "enemy_lord", "run": True},
                  {"unit": "own_archer_1", "kind": "attack", "target": "enemy_spear_1", "run": True}]
        answered = [0]

        def play(seconds):   # the companion answers every move with the same orders
            for _ in range(seconds * 10):
                doc = exchange.read_state(tmp_path / exchange.STATE)
                if doc["move"] > answered[0]:
                    answered[0] = doc["move"]
                    exchange.write_atomic(tmp_path / exchange.ORDERS,
                                          exchange.orders_text(doc["batch"], doc["move"], orders, 1.0))
                lua.execute("bm:tick(100)")

        def log():
            return list(lua.eval("bm.orders").values())
        play(3)
        assert log() == ["attack enemy_lord", "attack enemy_spear_1"]
        args = lua.eval("archer.attack_args")
        assert (args.target, args.primary, args.run) == ("enemy_spear_1", True, True)   # runs as told
        assert lua.eval("archer.free_fire") is True
        lua.execute("own[2].routing = true")
        play(3)
        lua.execute("own[2].routing = false")        # rallied: the same order is given again
        play(2)
        assert log() == ["attack enemy_lord", "attack enemy_spear_1", "attack enemy_lord"]
        lua.execute("fake.cco['uid_own_archer_1'].IsFiringMissiles = false; enemy[2].melee = true")
        play(5)                                      # 4 decisions idle: released to fire at will
        assert log()[3:] == ["halt"] and lua.eval("archer.free_fire") is True
        lua.execute("fake.cco['uid_own_archer_1'].IsFiringMissiles = true")
        play(12)                                     # the target still in melee: stays free
        assert log()[3:] == ["halt"]
        lua.execute("enemy[2].melee = false")
        play(2)                                      # out of melee: the ordered target again
        assert log()[3:] == ["halt", "attack enemy_spear_1"]
        # A new target in melee within range: not an explicit target (it would stand idle), fire at will;
        # the archer shoots standing already, so it is not halted for it (a halt restarts its aim).
        lua.execute("enemy[1].pos.x, enemy[1].pos.z, enemy[1].melee = -120, 40, true")
        play(1)                                      # the bridge sees it at the next decision
        orders[1]["target"] = "enemy_lord"
        play(2)
        assert log()[3:] == ["halt", "attack enemy_spear_1"]
        lua.execute("""
            bm.outcome, bm.winner = true, 2
            for _ = 1, 10 do bm:tick(100) end
            assert(STATE.finished, 'the battle did not finish')
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows], [r for r in rows if r["event"] == "error"]
        assert [(r["u"], r["k"], r["tg"], r["status"]) for r in rows if r["event"] == "nn_rally"] == [
            ("own_spear_1", "attack", "enemy_lord", "given")]
        assert sum(r["skipped"] for r in rows if r["event"] == "nn_orders") >= 2
        assert [(r["u"], r["action"]) for r in rows if r["event"] == "nn_duty"] == [
            ("own_archer_1", "release"), ("own_archer_1", "resume"), ("own_archer_1", "free_kept")]
        result = rows[-1]
        assert (result["nn_regiven"], result["nn_released"], result["nn_resumed"]) == (1, 1, 1)
        assert result["nn_aims_kept"] == 1

    def test_human_gives_no_orders_and_records_everything_it_can(self, lua, tmp_path):
        # A human plays our side: no planner, no bridge; the observer's fields and change events.
        lua.execute(self.SETUP + """
            CONFIG.own_ai, CONFIG.timeout_ms, CONFIG.soldiers_every = 'human', 8000, 5
            own[1].abilities = {ability_x = true}
            fake.cco['uid_own_spear_1'] = {IsUnderMissileAttack = true, IsTakingDamage = false,
                DamageInflictedRecently = 12.345, ['ActiveEffectList.Size'] = 1,
                ['ActiveEffectList.At(0).PhaseRecordContext.Key'] = 'phase_fatigue'}
            local state = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
            bm:pump()
            bm:tick(); bm:tick()
            own[1].abilities = {ability_x = false}   -- used: ready turns false
            for _ = 1, 8 do bm:tick() end
            assert(state.finished and state.planner == nil and state.net == nil)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "error" not in kinds, [r for r in rows if r["event"] == "error"]
        assert next(r for r in rows if r["event"] == "own_ai")["mode"] == "human"
        assert list(lua.eval("bm.planner_log").values()) == []
        assert all(r["policy"] == "nn_arena_human" for r in rows)
        spear = next(u for u in [r for r in rows if r["event"] == "nn_sample"][-1]["units"] if u["n"] == "own_spear_1")
        assert (spear["ob"], spear["ow"], spear["idle"], spear["v"]) == (0, 30, True, True)
        assert (spear["uma"], spear["td"], spear["dir"]) == (True, False, 12.345)
        effects = [r for r in rows if r["event"] == "nn_effects" and r["u"] == "own_spear_1"]
        assert [(r["fx"], r["side"]) for r in effects] == [(["phase_fatigue"], 1)]
        ready = [(r["ready"], r["side"]) for r in rows if r["event"] == "nn_ability_ready" and r["u"] == "own_lord"]
        assert ready == [(True, 1), (False, 1)]
        soldiers = [r for r in rows if r["event"] == "nn_soldiers"]
        assert len(soldiers) == 2 and len(soldiers[0]["units"]) == 4   # ticks 1 and 6; both sides
        assert soldiers[0]["units"][0]["xz_dm"] == [13, -25, 13, -25]

    def test_lord_duel_the_enemy_lord_takes_one_attack_and_again_only_when_lost(self, lua, tmp_path):
        # tools/nn/lord_duel.py: our lord under the network, theirs under one scripted attack order.
        lua.execute("""
            own = {fake.unit('own_lord', 'lord', -50, 0)}
            enemy = {fake.unit('enemy_lord', 'lord', 50, 0)}
            bm = fake.manager({own, enemy})
            local function lord(side) return {{script_name = side .. '_lord', slot = 'lord', key = 'lord'}} end
            CONFIG = {build = 'test', speed = 20, tick_ms = 1000, deadline_s = 600, stall_ms = 600000,
                timeout_ms = 600000, defend_radius_m = 150, units = {own = lord('own'), enemy = lord('enemy')},
                own_ai = 'net', enemy_ai = 'scripted', enemy_role = 'defend', decide_ms = 1000, poll_ms = 100,
                factions = {own = 'wh_main_emp_empire', enemy = 'wh_main_emp_empire'}}
            STATE = require('entries.nn_arena').main(bm, CONFIG, {common = fake.common, battle_vector = fake.vector_type})
            bm:pump()
            assert(enemy[1].controlled and enemy[1].attack_args.target == 'own_lord')
            enemy[1].target = own[1]
            for _ = 1, 5 do bm:tick() end            -- has its target: nothing new
            enemy[1].target = nil                     -- the engine dropped it: again after 3 s
            for _ = 1, 5 do bm:tick() end
            bm.outcome, bm.winner = true, 1
            bm:tick()
            assert(STATE.finished)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows], [r for r in rows if r["event"] == "error"]
        given = [(r["u"], r["tg"], r["why"]) for r in rows if r["event"] == "scripted_order"]
        assert given == [("enemy_lord", "own_lord", "start"), ("enemy_lord", "own_lord", "lost")]
        assert next(r for r in rows if r["event"] == "own_ai")["mode"] == "net"
        assert rows[-1]["event"] == "result" and rows[-1]["scripted_orders"] == 2

    def test_lord_duel_escort_the_script_sends_lord_on_lord_and_infantry_on_infantry(self, lua, tmp_path):
        # scripted_targets 'like': the enemy lord is nearer our spearmen, yet attacks our lord.
        lua.execute(self.SETUP + """
            CONFIG.own_ai, CONFIG.enemy_ai, CONFIG.scripted_targets = 'scripted', 'scripted', 'like'
            enemy[1].pos = fake.vector_type.new(); enemy[1].pos.x = -170
            local state = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
            bm:pump()
            assert(enemy[1].attack_args.target == 'own_lord', enemy[1].attack_args.target)
            assert(enemy[2].attack_args.target == 'own_spear_1' and own[1].attack_args.target == 'enemy_lord')
            assert(own[2].attack_args.target == 'enemy_spear_1')
            enemy[2].men = 0                       -- no enemy infantry left: our spearmen take the lord
            own[2].target = nil
            for _ = 1, 5 do bm:tick() end
            assert(own[2].attack_args.target == 'enemy_lord', own[2].attack_args.target)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows], [r for r in rows if r["event"] == "error"]

    def test_lord_duel_control_both_sides_scripted(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            CONFIG.own_ai, CONFIG.enemy_ai = 'scripted', 'scripted'
            local state = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
            bm:pump()
            assert(state.planner == nil and state.net == nil)
            assert(own[1].attack_args.target == 'enemy_spear_1' and own[2].attack_args.target == 'enemy_spear_1')
            assert(enemy[2].attack_args.target == 'own_spear_1')
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows], [r for r in rows if r["event"] == "error"]
        assert [r["side"] for r in rows if r["event"] == "scripted_side"] == [1, 2]
        assert list(lua.eval("bm.planner_log").values()) == []

    def test_lord_ai_side_2_stays_the_game_ai_and_both_sides_cards_and_abilities_are_recorded(self, lua, tmp_path):
        # tools/nn/lord_ai.py: our side scripted, side 2 the game's AI; observe + cards on both sides.
        lua.execute(self.SETUP + """
            CONFIG.own_ai, CONFIG.observe, CONFIG.cards = 'scripted', true, true
            enemy[1].abilities = {rage = true}
            fake.cco['uid_enemy_lord'] = {CharacterRank = 1, HasCharacterRank = true, ExperienceLevel = 0,
                ['ActiveEffectList.Size'] = 1, ['ActiveEffectList.At(0).PhaseRecordContext.Key'] = 'rage'}
            local state = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
            bm:pump()
            assert(not enemy[1].controlled and own[1].controlled)
            bm:tick(); bm:tick()
            fake.cco['uid_enemy_lord']['UnitDetailsContext.StatList.At(0).Value'] = 45   -- a buff on the card
            enemy[1].abilities = {rage = false}                                          -- used
            for _ = 1, 3 do bm:tick() end
            bm.outcome, bm.winner = true, 2
            bm:tick()
            assert(state.finished and state.observer ~= nil)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows], [r for r in rows if r["event"] == "error"]
        assert [r["side"] for r in rows if r["event"] == "scripted_side"] == [1]
        cards = [r for r in rows if r["event"] == "nn_card"]
        assert sorted(r["u"] for r in cards if r["t"] == 0) == ["enemy_lord", "enemy_spear_1", "own_lord", "own_spear_1"]
        lord = [r for r in cards if r["u"] == "enemy_lord"]
        assert len(lord) == 2 and (lord[0]["rank"], lord[0]["has_rank"], lord[0]["xp"]) == (1, True, 0)
        assert [s["v"] for s in lord[0]["stats"]] == [30, 30] and [s["k"] for s in lord[0]["stats"]] == [
            "stat_armour", "stat_morale"]
        assert [s["v"] for s in lord[1]["stats"]] == [45, 30] and lord[1]["t"] > 0
        ready = [(r["ready"], r["side"]) for r in rows if r["event"] == "nn_ability_ready" and r["u"] == "enemy_lord"]
        assert ready == [(True, 2), (False, 2)]
        assert [r["fx"] for r in rows if r["event"] == "nn_effects" and r["u"] == "enemy_lord"] == [["rage"]]
        assert not [r for r in rows if r["event"] == "nn_soldiers"]

    def test_an_unknown_own_ai_is_an_error(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            CONFIG.own_ai = 'dance'
            local state = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
            assert(state.finished and not state.active)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert any(r["event"] == "error" and "own_ai" in r["message"] for r in rows)

    def test_net_commands_a_generated_army_of_twenty_units(self, lua, tmp_path):
        # A large gate battle (tools/nn/gate.py): 10+ Empire units v 20 Skaven, lords of both factions (the
        # Empire's count follows config/nn/pools.json: 18 before the 02.10.2026 second wave, 12 after),
        # several unit types. Our network attacks: the state says so, and every one of our units takes its order.
        from tools.nn import scenario as nn_scenario
        from tools.nn.armies import generate
        from tools.nn.companion import exchange
        arena = generate.battle(1_000_900_008)
        places = nn_scenario.placements(arena)
        cfg = nn_scenario.run_config(arena)
        assert len(places["own"]) >= 10 and len(places["enemy"]) == 20
        # several unit types a side (the count follows config/nn/pools.json: not fixed here)
        assert len({u["key"] for u in places["own"]}) >= 3 and len({u["key"] for u in places["enemy"]}) >= 3

        def units(side):
            return ", ".join(f"fake.unit('{u['script_name']}', '{u['key']}', {u['x']}, {u['z']})" for u in places[side])

        def specs(side):
            return ", ".join(f"{{script_name = '{u['script_name']}', slot = '{u['slot']}', key = '{u['key']}'}}"
                             for u in cfg["units"][side])
        lua.execute(f"""
            own = {{{units('own')}}}
            enemy = {{{units('enemy')}}}
            bm = fake.manager({{own, enemy}})
            CONFIG = {{build = 'test', speed = 20, tick_ms = 1000, deadline_s = 600, stall_ms = 600000,
                timeout_ms = 600000, defend_radius_m = 150, own_ai = 'net', enemy_role = 'defend',
                decide_ms = 1000, poll_ms = 100, units = {{own = {{{specs('own')}}}, enemy = {{{specs('enemy')}}}}},
                factions = {{own = '{cfg['factions']['own']}', enemy = '{cfg['factions']['enemy']}'}}}}
            STATE = require('entries.nn_arena').main(bm, CONFIG, {{common = fake.common, battle_vector = fake.vector_type}})
            bm:pump()
        """)
        doc = exchange.read_state(tmp_path / exchange.STATE)
        assert doc["attacker"] == 1 and doc["factions"] == cfg["factions"]
        assert len(doc["units"]) == len(places["own"]) + len(places["enemy"])
        assert [(u["n"], u["key"]) for u in doc["units"]] == [(u["script_name"], u["key"])
                                                              for s in ("own", "enemy") for u in places[s]]
        b = exchange.battle(doc)
        n_own, enemies = len(b.own), [n for n, s in zip(b.names, b.side) if s == 2]
        kinds = [("attack", i) if i % 3 == 0 else ("move", i) if i % 3 == 1 else ("hold", i) for i in range(n_own)]
        orders = []
        for (k, i), name in zip(kinds, b.own):
            o = {"unit": name, "kind": k}
            if k == "attack":
                o.update(target=enemies[i % len(enemies)], run=True)
            elif k == "move":
                o.update(x=-100.0 + i, z=10.0 * i, run=False)
            orders.append(o)
        exchange.write_atomic(tmp_path / exchange.ORDERS, exchange.orders_text(doc["batch"], 1, orders, 2.0))
        lua.execute("""
            bm:tick(100)
            bm.outcome, bm.winner = true, 1
            for _ = 1, 10 do bm:tick(100) end
            assert(STATE.finished, 'the battle did not finish')
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows], [r for r in rows if r["event"] == "error"]
        (given,) = [r for r in rows if r["event"] == "nn_orders"]
        assert len(given["orders"]) == n_own == len(places["own"]) and all(o["status"] == "given" for o in given["orders"])
        assert len(list(lua.eval("bm.orders").values())) == n_own
        result = rows[-1]
        assert result["event"] == "result" and result["winner"] == 1 and result["nn_orders_given"] == n_own


class TestLordSwarm:
    SETUP = """
        own = {fake.unit('own_lord', 'lord', -300, 0), fake.unit('own_spear_1', 'spears', 0, 0),
               fake.unit('own_spear_2', 'spears', 0, 0)}
        enemy = {fake.unit('enemy_lord', 'lord', 300, 0), fake.unit('enemy_spear_1', 'spears', 0, 0),
                 fake.unit('enemy_spear_2', 'spears', 0, 0)}
        own[1].melee, enemy[1].melee = true, true
        bm = fake.manager({own, enemy})
        local trial = function(name, places)
            local a = {}
            for _, s in ipairs(places) do a[#a + 1] = {side = s, kind = 'spear'} end
            return {name = name, attackers = a}
        end
        CONFIG = {build = 'test', speed = 20, tick_ms = 200, soldier_ms = 1000, deadline_s = 100,
            start_m = 20, settle_ms = 3000, fight_s = 2, max_s = 10, min_hp = 0.3, radii = {2, 4}, near_m = 6,
            lanes = {{name = 'general', lord = 'own_lord', x = -300, z = 0, bearing = 0,
                      spears = {'enemy_spear_1', 'enemy_spear_2'}, ap = {}, park = {x = -700, z = 600, bearing = 0}},
                     {name = 'warlord', lord = 'enemy_lord', x = 300, z = 0, bearing = 0,
                      spears = {'own_spear_1', 'own_spear_2'}, ap = {}, park = {x = 250, z = -600, bearing = 180}}},
            trials = {trial('s1', {'front'}), trial('s2', {'front', 'back'})}, widths = {}, depths = {}}
        GLOBALS = {common = fake.common, battle_vector = fake.vector_type}
    """

    def test_trials_run_in_turn_and_the_attackers_are_recorded(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            STATE = require('entries.lord_swarm').main(bm, CONFIG, GLOBALS)
            bm:pump()
            for _ = 1, 100 do bm:tick(200); bm:pump() end
            assert(STATE.finished and bm.ended)
            assert(own[2].controlled and enemy[1].controlled)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "error" not in kinds, [r for r in rows if r["event"] == "error"]
        trials = [r for r in rows if r["event"] == "swarm_trial"]
        assert [t["name"] for t in trials] == ["s1", "s2"]
        # Spear units rotate: the second trial starts with the unit after the first trial's.
        assert [a["name"] for a in trials[1]["lanes"][0]["attackers"]] == ["enemy_spear_2", "enemy_spear_1"]
        ends = [r for r in rows if r["event"] == "swarm_lane_end"]
        assert len(ends) == 4 and all(e["why"] == "fight_s" for e in ends)
        sample = next(r for r in rows if r["event"] == "swarm_sample")
        assert {x["lane"] for x in sample["lanes"]} == {"general", "warlord"}
        men = next(r for r in rows if r["event"] == "swarm_men")
        assert men["lanes"][0]["att"][0]["c"] == [0, 0]   # the fake soldiers stand at (1.25, -2.5)
        assert rows[-1]["event"] == "result" and rows[-1]["trials"] == 3

    def test_soldiers_are_counted_by_distance(self, lua):
        counts, near = lua.eval("""(function()
            local c, n = require('entries.lord_swarm').count_near({1, 0, 0, 2.5, 5, 5}, 0, 0, {1.5, 3}, 4)
            return c, n end)()""")
        assert list(counts.values()) == [1, 2] and list(near.values()) == [10, 0, 0, 25]

    def test_a_rival_lord_attacks_in_one_lane_only(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            CONFIG.lanes[1].rival, CONFIG.lanes[2].rival = 'enemy_lord', 'own_lord'
            CONFIG.trials = {{name = 'lord_s1', lanes = {'warlord'},
                attackers = {{side = 'front', kind = 'lord'}, {side = 'back', kind = 'spear'}}}}
            STATE = require('entries.lord_swarm').main(bm, CONFIG, GLOBALS)
            bm:pump()
            for _ = 1, 60 do bm:tick(200); bm:pump() end
            assert(STATE.finished)
            assert(own[1].attack_args.target == 'enemy_lord' and own[2].attack_args.target == 'enemy_lord')
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows]
        trial = next(r for r in rows if r["event"] == "swarm_trial")
        assert trial["lanes"] == [{"lane": "warlord", "attackers": [
            {"kind": "lord", "name": "own_lord", "side": "front"}, {"kind": "spear", "name": "own_spear_1", "side": "back"}]}]
        assert [r["lane"] for r in rows if r["event"] == "swarm_lane_end"] == ["warlord"]
        assert all(x["lane"] == "warlord" for r in rows if r["event"] == "swarm_sample" for x in r["lanes"])


class TestLordFall:
    # The treated army (side 1): lord, two fight units, one idle; the fearless side 2 the same minus one idle.
    SETUP = """
        own = {fake.unit('own_lord', 'lord', -60, 0), fake.unit('own_fight_1', 'spears', -40, 18),
               fake.unit('own_fight_2', 'spears', -40, -18), fake.unit('own_idle_1', 'spears', -40, -130)}
        enemy = {fake.unit('enemy_lord', 'lord', 60, 0), fake.unit('enemy_fight_1', 'spears', 40, 18),
                 fake.unit('enemy_fight_2', 'spears', 40, -18)}
        bm = fake.manager({own, enemy})
        CONFIG = {build = 'test', speed = 20, tick_ms = 500, deadline_s = 100, treated_side = 1,
            treatment = 'kill', lords = {own = 'own_lord', enemy = 'enemy_lord'},
            fight = {{'own_fight_1', 'enemy_fight_1'}, {'own_fight_2', 'enemy_fight_2'}}, idle = {'own_idle_1'},
            treat_after_ms = 20000, treat_latest_ms = 60000, observe_ms = 10000,
            factions = {own = 'emp', enemy = 'skv'}}
        GLOBALS = {common = fake.common, battle_vector = fake.vector_type}
        fake.cco['uid_own_idle_1'] = {MoralePercent = 0.9, MoraleGreatestEffect = 'General died'}
    """

    def test_due_after_contact_or_at_the_latest(self, lua):
        due = lua.eval("require('entries.lord_fall').due")
        assert due(25000, 0, 5000, 20000, 60000) and not due(24000, 0, 5000, 20000, 60000)
        assert not due(59000, 0, None, 20000, 60000) and due(60000, 0, None, 20000, 60000)

    def test_kill_after_contact_records_everyone_and_ends(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            STATE = require('entries.lord_fall').main(bm, CONFIG, GLOBALS)
            own[1].reduce_hitpoints_unary = function(self, p) self.men = 0 end
            bm:pump()
            for _ = 1, 10 do bm:tick(500) end      -- 5 s: no contact yet
            own[2].melee = true                     -- contact at 5.5 s
            for _ = 1, 80 do bm:tick(500) end       -- the kill at 25.5 s, the end 10 s later
            assert(STATE.finished and bm.ended, 'the probe did not end')
            assert(enemy[1].fearless and enemy[2].fearless and not own[2].fearless)
            assert(own[2].attack_args.target == 'enemy_fight_1' and enemy[3].attack_args.target == 'own_fight_2')
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "error" not in kinds, [r for r in rows if r["event"] == "error"]
        (contact,) = [r for r in rows if r["event"] == "contact"]
        (fall,) = [r for r in rows if r["event"] == "lord_fall"]
        assert contact["t"] == 5500 and fall["t"] == 25500
        assert fall["method"] == "reduce_hitpoints_unary" and fall["lord"] == "own_lord" and fall["before"]["men"] == 120
        assert "lord_fall_fallback" not in kinds
        sample = [r for r in rows if r["event"] == "fall_sample"][-1]
        assert sample["treated"] is True
        roles = {u["n"]: (u["side"], u["role"]) for u in sample["units"]}
        assert roles["own_idle_1"] == (1, "idle") and roles["enemy_fight_2"] == (2, "fight") and roles["own_lord"] == (1, "lord")
        idle = next(u for u in sample["units"] if u["n"] == "own_idle_1")
        assert (idle["mp"], idle["mge"]) == (0.9, "General died")
        assert next(u for u in sample["units"] if u["n"] == "own_lord")["men"] == 0
        result = rows[-1]
        assert result["event"] == "result" and result["lord_dead"] == "damage" and result["treated_ms"] == 25500
        assert kinds[-2] == "fall_final"

    def test_kill_falls_back_to_uc_kill_and_rout_routs(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            STATE = require('entries.lord_fall').main(bm, CONFIG, GLOBALS)
            bm:pump()
            for _ = 1, 140 do bm:tick(500) end      -- no contact: the kill at 60 s, no damage method
            assert(STATE.finished)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        fall = next(r for r in rows if r["event"] == "lord_fall")
        assert fall["t"] == 60000 and fall["error"]
        assert next(r for r in rows if r["event"] == "lord_fall_fallback")["method"] == "uc_kill"
        assert "kill own_lord" in list(lua.eval("bm.orders").values())
        assert rows[-1]["lord_dead"] == "uc_kill"

    def test_rout_and_control(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            CONFIG.treatment = 'rout'
            STATE = require('entries.lord_fall').main(bm, CONFIG, GLOBALS)
            bm:pump()
            for _ = 1, 140 do bm:tick(500) end
            assert(STATE.finished and own[1].routing)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert next(r for r in rows if r["event"] == "lord_fall")["method"] == "morale_behavior_rout"
        assert rows[-1]["event"] == "result" and rows[-1].get("lord_dead") is None

    def test_a_fight_unit_out_of_melee_gets_its_attack_again(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            CONFIG.treatment = 'none'
            STATE = require('entries.lord_fall').main(bm, CONFIG, GLOBALS)
            bm:pump()
            own[2].melee, enemy[2].melee = true, true
            own[3].melee, enemy[3].melee = true, true
            for _ = 1, 4 do bm:tick(500) end
            own[2].melee = false                    -- dropped out of the fight for 5 s
            for _ = 1, 12 do bm:tick(500) end
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert [(r["u"], r["tg"]) for r in rows if r["event"] == "fall_reissue"] == [("own_fight_1", "enemy_fight_1")]


class TestChargeProbe:
    """The melee probe (entries.charge_probe; lanes from tools/nn/charge_probe.py)."""
    SETUP = """
        own = {fake.unit('own_lord', 'lord', -60, 0), fake.unit('own_swords_1', 'swords', 0, 0),
               fake.unit('own_swords_2', 'swords', 0, 0)}
        enemy = {fake.unit('enemy_lord', 'lord', 60, 0), fake.unit('enemy_clanrat_1', 'clanrat', 0, 0),
                 fake.unit('enemy_clanrat_2', 'clanrat', 0, 0)}
        own[1].abilities = {syg = true}
        bm = fake.manager({own, enemy})
        CONFIG = {build = 'test', speed = 20, tick_ms = 500, men_ms = 1000, men_near_m = 60, men_after_s = 30,
            deadline_s = 100, settle_ms = 4000,
            lanes = {{name = 'L1', attacker = 'own_swords_1', target = 'enemy_clanrat_1', x = -120, z = 0, gap_m = 80,
                      a_depth = 9, t_depth = 12, a_width = 30, t_width = 30, mode = 'recharge', target_mode = 'stand',
                      answer = true, fight_s = 20, max_s = 60, move_beyond_m = 5, recharge_after_s = 2, back_m = 40,
                      recharge_max_s = 3},
                     {name = 'L2', attacker = 'enemy_clanrat_2', target = 'own_swords_2', x = 120, z = 0, gap_m = 40,
                      a_depth = 9, t_depth = 12, a_width = 30, t_width = 30, mode = 'attack_walk',
                      target_mode = 'stand', answer = true, fight_s = 5, max_s = 60, move_beyond_m = 5,
                      lord = {name = 'own_lord', ability = 'syg', dz = 8}}},
            park = {{name = 'enemy_lord', x = 700, z = -400, bearing = 0}}}
        GLOBALS = {common = fake.common, battle_vector = fake.vector_type}
    """

    def test_orders_contacts_recharge_and_ability(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            STATE = require('entries.charge_probe').main(bm, CONFIG, GLOBALS)
            bm:pump()
            for _ = 1, 10 do bm:tick(500); bm:pump() end
            assert(own[2].attack_args.run == true, 'recharge starts with an attack at a run')
            assert(enemy[3].attack_args.run == false, 'attack_walk walks')
            own[2].melee, enemy[3].melee = true, true
            for _ = 1, 200 do bm:tick(500); bm:pump() end
            assert(STATE.finished and bm.ended)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "error" not in kinds, [r for r in rows if r["event"] == "error"]
        contacts = [(r["lane"], r["n"]) for r in rows if r["event"] == "probe_contact"]
        assert ("L1", 1) in contacts and ("L2", 1) in contacts and ("L1", 2) in contacts
        assert [r["phase"] for r in rows if r["event"] == "probe_phase"] == ["out", "back"]
        (ab,) = [r for r in rows if r["event"] == "probe_ability"]
        assert ab["lane"] == "L2" and ab["status"] == "used"
        sample = next(r for r in rows if r["event"] == "probe_sample")
        assert {x["lane"] for x in sample["lanes"]} == {"L1", "L2"} and "l" in sample["lanes"][1]
        assert any(r["event"] == "probe_men" for r in rows)
        ends = {r["lane"]: r["why"] for r in rows if r["event"] == "probe_lane_end"}
        assert ends == {"L1": "fight_s", "L2": "fight_s"}
        assert rows[-1]["event"] == "result"

    def test_layout_and_recharge_step(self, lua):
        L = lua.eval("""require('entries.charge_probe').layout({x = 10, z = 0, gap_m = 80, a_depth = 10,
            t_depth = 12, target_mode = 'rear'})""")
        assert (L.ax, L.az, L.ab, L.tx, L.tz, L.tb) == (10, 85, 180, 10, -6, 180)
        step = lua.eval("require('entries.charge_probe').recharge_step")
        lane = lua.table_from({"recharge_after_s": 10, "back_m": 40, "recharge_max_s": 25})
        assert step("in", 9000, 0, 0, 0, True, lane) == "in"
        assert step("in", 10000, 0, 0, 0, True, lane) == "out"
        assert step("out", 12000, 0, 10000, 36, False, lane) == "back"
        assert step("out", 12000, 0, 10000, 36, True, lane) == "out"
        assert step("out", 35000, 0, 10000, 3, True, lane) == "back"

class TestMissileProbe:
    """The missile probe (entries.missile_probe; lanes from tools/nn/missile_probe.py)."""
    SETUP = """
        own = {fake.unit('own_lord', 'lord', -60, 0), fake.unit('own_archers_1', 'archers', 0, 0)}
        enemy = {fake.unit('enemy_lord', 'lord', 60, 0), fake.unit('enemy_slave_1', 'slave', 0, 0)}
        bm = fake.manager({own, enemy})
        CONFIG = {build = 'test', speed = 20, tick_ms = 500, deadline_s = 100, settle_ms = 4000,
            lanes = {{name = 'L1', shooter = 'own_archers_1', target = 'enemy_slave_1', x = 0, z = -80, s_b = 0,
                      d = 150, t_angle = 0, t_rot = 0, s_width = 30, t_width = 30, mode = 'fire',
                      target_mode = 'step', start_d = 150, step_m = 2, step_s = 2, max_s = 30, idle_s = 10,
                      settle_s = 2}},
            park = {{name = 'own_lord', x = -700, z = -400, bearing = 0}, {name = 'enemy_lord', x = 700, z = -400,
                     bearing = 0}}}
        GLOBALS = {common = fake.common, battle_vector = fake.vector_type}
    """

    def test_the_target_steps_in_and_the_lane_ends(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            STATE = require('entries.missile_probe').main(bm, CONFIG, GLOBALS)
            bm:pump()
            for _ = 1, 100 do bm:tick(500); bm:pump() end
            assert(STATE.finished and bm.ended)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        kinds = [r["event"] for r in rows]
        assert "error" not in kinds, [r for r in rows if r["event"] == "error"]
        ds = [x["d"] for r in rows if r["event"] == "probe_sample" for x in r["lanes"]]
        assert ds[0] == 150 and min(ds) < 150                        # stepped closer
        ends = {r["lane"]: r["why"] for r in rows if r["event"] == "probe_lane_end"}
        assert set(ends) == {"L1"} and rows[-1]["event"] == "result"

    def test_layout_and_frame(self, lua):
        L = lua.eval("""require('entries.missile_probe').layout({x = 10, z = 0, s_b = 0, t_angle = 90, t_rot = 180},
            100)""")
        assert (round(L.sx), round(L.sz), L.sb, round(L.tx), round(L.tz), round(L.tb)) == (10, 0, 0, 110, 0, 90)
        fx, fz = lua.eval("require('entries.missile_probe').frame")(0, 0, 90, 10, 5)
        assert (round(fx, 6), round(fz, 6)) == (10, -5)
