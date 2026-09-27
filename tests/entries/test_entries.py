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
            local state = duel.main(bm, {build='test', runs=1, speed=20, timeout_ms=60000, tick_ms=1000, deadline_s=60, stall_ms=600000})
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


class TestFormationProbe:
    # Our spearmen and archers against two enemy spearmen: "wall and arc" fits
    # (the enemy has more infantry, we have the only archers, we attack).
    SETUP = """
        local spears = fake.unit('own_1', 'wh_main_emp_inf_spearmen_0', 0, -200)
        local archers = fake.unit('own_2', 'wh2_dlc13_emp_inf_archers_0', 20, -200)
        local e1 = fake.unit('enemy_1', 'wh_main_emp_inf_spearmen_0', 0, 100)
        local e2 = fake.unit('enemy_2', 'wh_main_emp_inf_spearmen_0', 30, 100)
        bm = fake.manager({{spears, archers}, {e1, e2}})
        local function spear(id)
            return {id = id, class = 'inf_mel', men = 120, commanding = false, range_m = 0, health = 8280,
                armour = 30, melee_attack = 20, melee_defence = 34, shapes = {{ordered_m = 40, front_m = 39, depth_m = 8}}}
        end
        CONFIG = {build = 'test', speed = 20, tick_ms = 1000, deadline_s = 100, stall_ms = 900000, align_timeout_s = 30,
            settle_ms = 3000, stage_timeout_s = 20, hold_s = 10, role = 'attack',
            own = {anchor = {x = 0, z = -150}, units = {spear('own_1'),
                {id = 'own_2', class = 'inf_mis', men = 90, commanding = false, fire = 'arc', range_m = 130,
                 health = 6210, missile_damage = 19, shapes = {{ordered_m = 20, front_m = 19, depth_m = 18}}}}},
            enemy = {anchor = {x = 0, z = 150}, units = {spear('enemy_1'), spear('enemy_2')}, placements = {
                {script_name = 'enemy_1', role = 'wall', x = 0, z = 150, bearing = 180, width = 30},
                {script_name = 'enemy_2', role = 'wall', x = 30, z = 150, bearing = 180, width = 30}}}}
        -- Two soldiers per unit around its position, so each side has its own place.
        local by_id = {uid_own_1 = spears, uid_own_2 = archers, uid_enemy_1 = e1, uid_enemy_2 = e2}
        local common = {game_version = fake.common.game_version, get_context_value = function(key, id, field)
            local u = by_id[id]
            if u and field == 'ManList.Size' then return 2 end
            local i = u and field and field:match('^ManList%.At%((%d+)%)%.Position$')
            if i then return u.pos.x + tonumber(i) * 2, 0, u.pos.z end
            return fake.common.get_context_value(key, id, field)
        end}
        GLOBALS = {common = common, battle_vector = fake.vector_type}
    """

    def test_plan_is_applied_and_stages_run(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            CONFIG.turn_test = true
            local state = require('entries.formation_probe').main(bm, CONFIG, GLOBALS)
            bm:pump()
            for _ = 1, 140 do bm:tick() end
            assert(state.finished and bm.ended)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows], [r for r in rows if r["event"] == "error"]
        plan = next(r for r in rows if r["event"] == "plan")
        assert plan["facing_source"] == "visible_enemy" and plan["plan"]["status"] == "ok"
        assert plan["strategy"] == "wall_and_arc" and plan["plan"]["layout"] == "line_and_blocks"
        stages = [r["stage"] for r in rows if r["event"] == "stage_snapshot"]
        assert len([r for r in rows if r["event"] == "hold_sample"]) >= 10
        assert stages == ["placed", "align", "mask", "hold", "turn_right", "back_from_right", "turn_left", "back_from_left",
                          "turn_right_in_place", "back_right_in_place", "turn_left_in_place", "back_left_in_place"]
        assert any(r["event"] == "turn_sample" for r in rows)
        assert rows[-1]["event"] == "result" and rows[-1]["status"] == "completed"

    def test_default_run_only_places_and_holds_without_orders(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            CONFIG.hold_s = 5
            local state = require('entries.formation_probe').main(bm, CONFIG, GLOBALS)
            bm:pump()
            for _ = 1, 60 do bm:tick() end
            assert(state.finished)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert [r["stage"] for r in rows if r["event"] == "stage_snapshot"] == ["placed", "align", "mask", "hold"]
        mask = next(r for r in rows if r["event"] == "mask")
        assert mask["summary"]["known"] == mask["summary"]["cells"] and mask["summary"]["cells"] > 0
        assert not any(r["event"] == "turn_sample" for r in rows)
        assert rows[-1]["orders_after_placed"] == 0 and rows[-1]["status"] == "completed"

    def test_no_fitting_strategy_is_an_error_not_a_guess(self, lua, tmp_path):
        lua.execute(self.SETUP + """
            CONFIG.role = 'defend'
            local state = require('entries.formation_probe').main(bm, CONFIG, GLOBALS)
            bm:pump()
            bm:tick()
            assert(state.finished and not state.active)
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        plan = next(r for r in rows if r["event"] == "plan")
        assert plan["status"] == "no_strategy" and plan["strategy"] == "none"
        assert any(r["event"] == "error" and "No strategy" in r["message"] for r in rows)


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

    def test_battle_picture_from_our_side_view(self, lua, tmp_path):
        lua.execute("""
            local own = fake.unit('own_1', 'wh_main_emp_inf_spearmen_0', 0, -250)
            local seen = fake.unit('enemy_1', 'wh_main_emp_inf_spearmen_0', 0, 200)
            local hidden = fake.unit('enemy_2', 'wh_main_emp_inf_spearmen_0', 600, 200)
            function hidden:is_visible_to_alliance() return false end
            local by_id = {uid_own_1 = own, uid_enemy_1 = seen, uid_enemy_2 = hidden}
            -- Two soldiers per unit, around the unit's position.
            local common = {game_version = fake.common.game_version, get_context_value = function(key, id, field)
                local u = by_id[id]
                if u and field == 'ManList.Size' then return 2 end
                local i = u and field and field:match('^ManList%.At%((%d+)%)%.Position$')
                if i then return u.pos.x + tonumber(i) * 2, 0, u.pos.z end
                return fake.common.get_context_value(key, id, field)
            end}
            bm = fake.manager({{own}, {seen, hidden}})
            local spec = {class = 'inf_mel', men = 120, range_m = 0, health = 8280, melee_attack = 20, melee_defence = 34}
            local state = require('entries.enemy_layout').main(bm, {build = 'test', speed = 20, tick_ms = 1000,
                deadline_s = 100, stall_ms = 900000, hold_s = 5, layout = 'test', picture_every = 1,
                roster = {wh_main_emp_inf_spearmen_0 = spec}}, {common = common, battle_vector = fake.vector_type})
            bm:pump()
            for _ = 1, 3 do bm:tick() end
        """)
        rows = events(tmp_path / "tww3_bai_events.jsonl")
        assert "error" not in [r["event"] for r in rows], [r for r in rows if r["event"] == "error"]
        pic = next(r for r in rows if r["event"] == "battlefield")
        # The hidden unit is not in the picture; its strength still counts in the enemy total.
        assert pic["seen"] == 1 and pic["enemy"]["groups"][0]["ids"] == ["enemy_1"]
        assert pic["enemy"]["seen_share"] == 0.5
        assert pic["field"]["status"] == "ok" and abs(pic["field"]["centres_m"] - 450) < 1
