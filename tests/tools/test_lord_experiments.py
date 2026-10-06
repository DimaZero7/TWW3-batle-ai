"""tools.nn.lord_fall, tools.nn.lord_duel and tools.nn.lord_ai: the battles' layout, the builds, the plans and the tables."""
import json
import math

import pytest

from tools import build
from tools import config as project
from tools.nn import lord_ai, lord_duel, lord_fall
from tools.nn import scenario as nn_scenario


def places(arena):
    return {u["script_name"]: u for side in nn_scenario.SIDES for u in nn_scenario.placements(arena)[side]}


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["z"] - b["z"])


class TestLordFallLayout:
    @pytest.mark.parametrize("faction", sorted(lord_fall.ARMIES))
    def test_fight_units_face_their_pair_and_idle_units_stand_out_of_aura_and_melee(self, faction):
        p = places(lord_fall.arena(faction))
        cfg, _ = lord_fall.run_config(faction, "kill")
        for own, enemy in cfg["fight"]:
            assert p[own]["z"] == p[enemy]["z"] and dist(p[own], p[enemy]) == lord_fall.GAP_M
            # the melee is met halfway: the lord's aura is full there
            meet = {"x": 0.0, "z": p[own]["z"]}
            assert dist(p["own_lord"], meet) < lord_fall.AURA_M[0]
        enemy_fight = [p[e] for _, e in cfg["fight"]] + [{"x": 0.0, "z": p[e]["z"]} for _, e in cfg["fight"]]
        for name in ("own_idle_1", "own_idle_2"):
            assert dist(p[name], p["own_lord"]) > lord_fall.AURA_M[1]
            assert min(dist(p[name], e) for e in enemy_fight) > 110
            assert min(dist(p[name], p[e]) for e in ("enemy_idle_1", "enemy_idle_2")) > 200
        # every unit inside its deployment zone (centre 40 m ahead of the front, half the size back)
        half = lord_fall.DEPLOYMENT_M / 2
        assert all(abs(u["z"]) <= half for u in p.values())

    def test_the_armies_come_from_the_database_keys_and_the_vampires_have_skeleton_spearmen(self):
        a = lord_fall.arena("vmp")
        own, enemy = a["sides"]["own"], a["sides"]["enemy"]
        assert own["faction"] == "wh_main_vmp_vampire_counts" and enemy["faction"] == "wh_main_emp_empire"
        assert [u["key"] for u in own["units"]].count("wh_main_vmp_inf_skeleton_warriors_1") == 4
        assert sum(bool(u.get("general")) for u in own["units"]) == 1

    def test_the_plan_runs_in_rounds(self):
        p = lord_fall.plan(kill=2, rout=2, none=1)
        assert len(p) == 15 and p[:3] == [("emp", "kill"), ("emp", "rout"), ("emp", "none")]
        assert p[9:12] == [("emp", "kill"), ("emp", "rout"), ("skv", "kill")]
        assert sorted(set(p)) == sorted((f, t) for f in lord_fall.ARMIES for t in lord_fall.TREATMENTS)


class TestBuilds:
    def test_lord_fall_build_writes_its_battle_and_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        monkeypatch.setattr(lord_fall, "ROOT", tmp_path / "lord-fall")
        assert build.main(["lord-fall", "--faction", "skv", "--treatment", "rout"]) == 0
        manifest = json.loads((tmp_path / "lord-fall" / "manifest.json").read_text(encoding="utf-8"))
        cfg = manifest["config"]
        assert (cfg["faction"], cfg["treatment"], cfg["treated_side"], cfg["tick_ms"]) == ("skv", "rout", 1, 500)
        assert cfg["fight"] == [["own_fight_1", "enemy_fight_1"], ["own_fight_2", "enemy_fight_2"]]
        assert cfg["stall_ms"] >= 270_000 and cfg["leadership"] == 45
        xml = (tmp_path / "lord-fall" / "lord_fall_skv.xml").read_text(encoding="utf-8")
        assert "wh2_main_skv_inf_clanrat_spearmen_0" in xml and "wh_main_emp_inf_spearmen_0" in xml

    @pytest.mark.parametrize("own_ai,role", [("net", "attack"), ("net", "defend"), ("scripted", "attack")])
    def test_lord_duel_build(self, tmp_path, monkeypatch, own_ai, role):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        monkeypatch.setattr(lord_duel, "ROOT", tmp_path / "lord-duel")
        assert build.main(["lord-duel", "--duel", "skv", "--own-ai", own_ai, "--own-role", role]) == 0
        cfg = json.loads((tmp_path / "lord-duel" / "manifest.json").read_text(encoding="utf-8"))["config"]
        assert (cfg["own_ai"], cfg["enemy_ai"], cfg["own_role"], cfg["duel"]) == (own_ai, "scripted", role, "skv")
        assert cfg["enemy_role"] == ("defend" if role == "attack" else "attack")
        assert [u["script_name"] for u in cfg["units"]["own"]] == ["own_lord"] == [
            u["script_name"].replace("enemy", "own") for u in cfg["units"]["enemy"]]
        assert cfg["timeout_ms"] == 900_000 and ("decide_ms" in cfg) == (own_ai == "net")
        xml = (tmp_path / "lord-duel" / f"lord_duel_skv_{role}.xml").read_text(encoding="utf-8")
        assert xml.count("wh2_main_skv_cha_warlord_0") == 2
        assert f"<timeout_winning_alliance_index>{1 if role == 'attack' else 0}<" in xml

    def test_lord_duel_refuses_the_planner(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        with pytest.raises(SystemExit):
            build.main(["lord-duel", "--own-ai", "attack"])
        with pytest.raises(SystemExit):
            build.main(["nn-arena", "--own-ai", "scripted"])

    def test_the_duel_plan(self):
        p = lord_duel.plan(net=4, control=2)
        assert len(p) == 12
        assert [b for b in p if b[0] == "emp" and b[1] == "net"] == [("emp", "net", r, "solo") for r in
                                                                    ("attack", "defend", "attack", "defend")]
        assert sum(b[1] == "scripted" for b in p) == 4
        assert {b[3] for b in lord_duel.plan(variant="escort")} == {"escort"}

    @pytest.mark.parametrize("kind", sorted(lord_duel.LORDS))
    def test_the_escort_arena_is_mirrored_with_two_infantry_units_beside_the_lord(self, kind):
        a = lord_duel.arena(kind, "escort")
        own, enemy = a["sides"]["own"]["units"], a["sides"]["enemy"]["units"]
        assert own == enemy and [u["slot"] for u in own] == ["lord", "inf_1", "inf_2"]
        assert {u["key"] for u in own[1:]} == {lord_duel.ESCORT[kind][0]}
        p = places(a)
        assert p["own_inf_1"]["z"] == p["enemy_inf_2"]["z"] and p["own_inf_2"]["z"] == p["enemy_inf_1"]["z"]   # face to face
        assert dist(p["own_lord"], p["enemy_lord"]) == lord_duel.GAP_M

    def test_escort_build_tells_the_script_lord_on_lord(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        monkeypatch.setattr(lord_duel, "ROOT", tmp_path / "lord-duel")
        assert build.main(["lord-duel", "--duel", "emp", "--duel-variant", "escort"]) == 0
        cfg = json.loads((tmp_path / "lord-duel" / "manifest.json").read_text(encoding="utf-8"))["config"]
        assert (cfg["variant"], cfg["scripted_targets"]) == ("escort", "like")
        assert [u["script_name"] for u in cfg["units"]["own"]] == ["own_lord", "own_inf_1", "own_inf_2"]
        assert (tmp_path / "lord-duel" / "lord_duel_emp_escort_attack.xml").exists()


def write_run(folder, config, rows):
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    (folder / "events.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows),
                                         encoding="utf-8")
    return folder


def fall_run(folder, treatment, drop):
    """Idle unit: MoralePercent 1.0 until the moment at 25 s, then 1.0 - drop; a fight unit falls 0.01 a second."""
    cfg, _ = lord_fall.run_config("skv", treatment)
    rows = []
    for k in range(0, 171):
        t = 500 * k
        treated = t > 25000
        idle_mp = 1.0 - (drop if treated else 0.0)
        units = [{"n": "own_lord", "side": 1, "role": "lord", "men": 0 if treated and treatment == "kill" else 1, "mp": 1.0},
                 {"n": "own_idle_1", "side": 1, "role": "idle", "mp": round(idle_mp, 3), "hp": 1.0,
                  "mge": "General died recently" if treated else None},
                 {"n": "own_fight_1", "side": 1, "role": "fight", "mp": round(1.0 - t / 100000, 3), "hp": 1.0 - t / 200000,
                  "r": treated and t > 40000},
                 {"n": "enemy_fight_1", "side": 2, "role": "fight", "mp": 1.0}]
        rows.append({"event": "fall_sample", "t": t, "treated": treated, "units": units})
        if t == 25000:
            rows.append({"event": "lord_fall", "t": t, "lord": "own_lord", "method": "x", "treatment": treatment})
    rows.append({"event": "result", "status": "completed"})
    return write_run(folder, cfg, rows)


class TestLordFallTable:
    def test_points_routs_and_the_control_subtracted(self, tmp_path):
        kill = lord_fall.load_run(fall_run(tmp_path / "a", "kill", 0.4))
        ctrl = lord_fall.load_run(fall_run(tmp_path / "b", "none", 0.0))
        units = lord_fall.measure(kill) + lord_fall.measure(ctrl)
        idle = next(u for u in units if u["treatment"] == "kill" and u["role"] == "idle")
        assert idle["d1"] == idle["d60"] == pytest.approx(-0.4 * 45)
        assert idle["mge_after"] == [("General died recently", 40)] and not idle["routed60"]
        fight = next(u for u in units if u["treatment"] == "kill" and u["role"] == "fight")
        assert fight["d10"] == pytest.approx(-0.1 * 45) and not fight["routed10"] and fight["routed60"]
        assert lord_fall.lord_state(kill) == "dead" and lord_fall.lord_state(ctrl) == "standing"
        table = {(r["treatment"], r["role"]): r for r in lord_fall.summary(units)}
        assert table[("kill", "idle")]["net10"] == pytest.approx(-18.0)
        assert table[("kill", "fight")]["net10"] == pytest.approx(0.0)   # the melee drift is the control's
        assert table[("none", "fight")]["net10"] is None

    def test_report_writes_its_json(self, tmp_path, capsys):
        fall_run(tmp_path / "runs" / "a", "kill", 0.2)
        table = lord_fall.report(lord_fall.runs(tmp_path / "runs"), out=tmp_path / "analysis.json")
        assert len(table) == 2 and json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))["units"]
        assert "skv" in capsys.readouterr().out


class TestLordDuelTable:
    def test_trade_orders_abilities_and_walking(self, tmp_path):
        cfg = {"duel": "emp", "own_ai": "net", "own_role": "attack"}
        rows = []
        for k in range(0, 61):
            own = {"n": "own_lord", "hp": 1.0 - 0.005 * max(0, k - 10), "x": -50 + min(k, 10) * 5.0, "z": 0.0, "m": k > 10}
            enemy = {"n": "enemy_lord", "hp": 1.0 - 0.01 * max(0, k - 10), "x": 50.0, "z": 0.0, "m": k > 10}
            rows.append({"event": "nn_sample" if k < 60 else "nn_final", "t": 1000 * k, "units": [own, enemy]})
        rows.append({"event": "nn_orders", "orders": [{"u": "own_lord", "k": "attack", "status": "given"},
                                                      {"u": "own_lord", "k": "move", "status": "given"}]})
        rows.append({"event": "nn_ability", "u": "own_lord", "key": "stand", "status": "used"})
        rows.append({"event": "nn_ability", "u": "own_lord", "key": "stand", "status": "not_ready"})
        rows.append({"event": "scripted_order", "u": "enemy_lord", "why": "start"})
        rows.append({"event": "scripted_order", "u": "enemy_lord", "why": "lost"})
        rows.append({"event": "result", "status": "completed", "winner": 1})
        run = lord_duel.load_run(write_run(tmp_path / "r", cfg, rows))
        m = lord_duel.measure(run)
        assert m["mode"] == "net-attack" and m["winner"] == 1 and m["contact_s"] == 11
        assert m["own_lost"] == pytest.approx(0.25) and m["enemy_lost"] == pytest.approx(0.5)
        assert m["trade"] == pytest.approx(0.25)
        assert m["own_walked_m"] == 50 and m["enemy_walked_m"] == 0
        assert m["order_kinds"] == {"attack": 1, "move": 1} and m["orders_per_min"] == 2.0
        assert m["abilities"] == {"stand": 1} and m["reissues"] == 1
        (row,) = lord_duel.summary([m])
        assert (row["own_wins"], row["enemy_wins"], row["trade"]) == (1, 0, 0.25)
        assert m["variant"] == "solo" and m["side_trade"] is None

    def test_escort_targeting_and_side_trade(self, tmp_path):
        # 60 s: our lord's engine target enemy_lord, enemy_inf_1, (none), enemy_inf_1, enemy_lord; in melee
        # from 10 s; the enemy lord stays on ours. The network's attack targets: lord, inf_1, lord (2 switches).
        cfg = {"duel": "skv", "variant": "escort", "own_ai": "net", "own_role": "defend"}
        rows = []
        for k in range(0, 61):
            tg = ("enemy_lord" if k < 20 else "enemy_inf_1" if k < 30 else "" if k < 35
                  else "enemy_inf_1" if k < 40 else "enemy_lord")
            units = [{"n": "own_lord", "hp": 1.0 - 0.002 * k, "t": tg, "m": k >= 10, "x": 0.0, "z": 0.0},
                     {"n": "own_inf_1", "hp": 1.0 - 0.004 * k, "x": 0.0, "z": 0.0},
                     {"n": "enemy_lord", "hp": 1.0 - 0.004 * k, "t": "own_lord", "m": k >= 10, "x": 0.0, "z": 0.0},
                     {"n": "enemy_inf_1", "hp": 1.0 - 0.006 * k, "x": 0.0, "z": 0.0}]
            rows.append({"event": "nn_sample", "t": 1000 * k, "units": units})
        rows.append({"event": "nn_orders", "orders": [
            {"u": "own_lord", "k": "attack", "tg": "enemy_lord", "status": "given"},
            {"u": "own_inf_1", "k": "attack", "tg": "enemy_inf_1", "status": "given"}]})
        rows.append({"event": "nn_orders", "orders": [{"u": "own_lord", "k": "attack", "tg": "enemy_inf_1", "status": "given"}]})
        rows.append({"event": "nn_orders", "orders": [{"u": "own_lord", "k": "attack", "tg": "enemy_lord", "status": "given"}]})
        rows.append({"event": "result", "status": "completed", "winner": 1})
        m = lord_duel.measure(lord_duel.load_run(write_run(tmp_path / "r", cfg, rows)))
        assert m["own_target_switches_per_min"] == 2 and m["enemy_target_switches_per_min"] == 0
        assert m["order_switches_per_min"] == 2 and m["orders_given"] == 3
        assert m["own_melee_s"] == 51 and m["own_on_lord"] == pytest.approx(31 / 51, abs=1e-3)
        assert m["enemy_on_lord"] == 1.0
        assert m["side_trade"] == pytest.approx((0.24 + 0.36) / 2 - (0.12 + 0.24) / 2)
        (row,) = lord_duel.summary([m])
        assert (row["variant"], row["own_on_lord"], row["order_switches"]) == ("escort", 0.608, 2.0)


class TestLordAi:
    @pytest.mark.parametrize("enemy,role", [("game", "attack"), ("game", "defend"), ("scripted", "attack")])
    def test_build_side_2_the_game_ai_or_the_script_and_everything_recorded(self, tmp_path, monkeypatch, enemy, role):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        monkeypatch.setattr(lord_ai, "ROOT", tmp_path / "lord-ai")
        assert build.main(["lord-ai", "--duel", "skv", "--own-role", role, "--duel-enemy", enemy]) == 0
        cfg = json.loads((tmp_path / "lord-ai" / "manifest.json").read_text(encoding="utf-8"))["config"]
        assert (cfg["own_ai"], cfg["duel_enemy"], cfg["own_role"], cfg["duel"]) == ("scripted", enemy, role, "skv")
        assert cfg.get("enemy_ai") == ("scripted" if enemy == "scripted" else None)   # absent: the game's AI
        assert cfg["observe"] is True and cfg["cards"] is True and "decide_ms" not in cfg
        assert cfg["timeout_ms"] == 900_000
        xml = (tmp_path / "lord-ai" / f"lord_duel_skv_{role}.xml").read_text(encoding="utf-8")
        assert xml.count("wh2_main_skv_cha_warlord_0") == 2
        assert lord_ai.mode_of(cfg) == ("ai" if enemy == "game" else "control")

    def test_build_refuses_a_network_or_a_planner(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        monkeypatch.setattr(lord_ai, "ROOT", tmp_path / "lord-ai")
        with pytest.raises(SystemExit):
            build.main(["lord-ai", "--own-ai", "net"])

    def test_plan(self):
        p = lord_ai.plan(ai=4, control=2)
        assert len(p) == 12 and sum(b[1] == "game" for b in p) == 8
        assert [b[2] for b in p if b[:2] == ("emp", "game")] == ["attack", "defend", "attack", "defend"]
        assert {b[2] for b in p if b[1] == "scripted"} == {"attack"}

    def test_old_duels_count_only_as_scripted_solo_controls(self):
        assert lord_ai.mode_of({"own_ai": "scripted", "enemy_ai": "scripted"}) == "control"
        assert lord_ai.mode_of({"own_ai": "net", "enemy_ai": "scripted"}) is None
        assert lord_ai.mode_of({"own_ai": "scripted", "variant": "escort"}) is None


def ai_run(folder, enemy, rate_on_own, rate_on_enemy, ability_at=None):
    """Lords in melee from 10 s to 70 s, the health falling linearly; the enemy lord's card and (ability_at)
    his ability used then: ready true -> false, the phase in his effects for 10 s, his damage card up."""
    cfg = {"duel": "emp", "own_ai": "scripted", "duel_enemy": enemy, "own_role": "attack"}
    if enemy == "scripted":
        cfg["enemy_ai"] = "scripted"
    rows = []
    for k in range(0, 81):
        m = 10 <= k < 70
        d = max(0, min(k, 70) - 10)
        own = {"n": "own_lord", "hp": 1.0 - rate_on_own * d / 100, "m": m, "bf": 20 <= k < 30}
        en = {"n": "enemy_lord", "hp": 1.0 - rate_on_enemy * d / 100, "m": m and k != 40}
        rows.append({"event": "nn_sample" if k < 80 else "nn_final", "t": 1000 * k, "units": [own, en]})
    card = lambda dmg: [{"k": "stat_melee_attack", "v": 55, "b": 55}, {"k": "stat_weapon_damage", "v": dmg, "b": 430}]
    rows += [{"event": "nn_card", "t": 0, "u": "own_lord", "side": 1, "stats": card(430), "rank": 1, "xp": 0},
             {"event": "nn_card", "t": 0, "u": "enemy_lord", "side": 2, "stats": card(430), "rank": 1, "xp": 0},
             {"event": "nn_ability_ready", "t": 0, "u": "enemy_lord", "side": 2, "key": "x_abilities_rage", "ready": True},
             {"event": "nn_effects", "t": 0, "u": "enemy_lord", "side": 2, "fx": []}]
    if ability_at is not None:
        rows += [{"event": "nn_ability_ready", "t": ability_at * 1000, "u": "enemy_lord", "side": 2,
                  "key": "x_abilities_rage", "ready": False},
                 {"event": "nn_effects", "t": ability_at * 1000, "u": "enemy_lord", "side": 2, "fx": ["x_abilities_rage"]},
                 {"event": "nn_card", "t": ability_at * 1000, "u": "enemy_lord", "side": 2, "stats": card(537)},
                 {"event": "nn_effects", "t": (ability_at + 10) * 1000, "u": "enemy_lord", "side": 2, "fx": []}]
    rows.sort(key=lambda r: r["t"])
    rows.append({"event": "result", "status": "completed", "winner": 2})
    return write_run(folder, cfg, rows)


class TestLordAiTable:
    def test_rates_abilities_cards_and_behaviour(self, tmp_path):
        m = lord_ai.measure(lord_ai.load_run(ai_run(tmp_path / "a", "game", 1.0, 0.5, ability_at=15)))
        # the step 40 -> 41 has the enemy out of melee: 59 duel seconds
        assert m["mode"] == "ai" and m["contact_s"] == 10 and m["duel_s"] == 59
        assert m["on_own"] == pytest.approx(1.0) and m["on_enemy"] == pytest.approx(0.5)
        assert m["fresh_s"] == 30 and m["fresh_on_own"] == pytest.approx(1.0)
        assert m["enemy_reengaged"] == 1 and m["own_rear_share"] == pytest.approx(10 / 59, abs=1e-3)
        assert m["enemy_abilities"] == [("x_abilities_rage", 15.0, 5.0)] and m["own_abilities"] == []
        assert m["enemy_effects"] == ["x_abilities_rage"] and m["ability_on_s"] == 10
        assert m["enemy_card"]["stats"]["stat_weapon_damage"] == (430, 430) and m["enemy_card"]["rank"] == 1
        assert m["enemy_card_changes"] == [(15.0, "stat_weapon_damage", 537)] and m["own_card_changes"] == []

    def test_an_ai_ability_shows_by_its_phase_alone_and_passives_are_no_uses(self):
        # The game's AI side: can_perform_special_ability stays true; the phase in the effects is the use.
        windows = [(0.0, ["x_lord_passive_hold"]), (13.0, ["x_lord_passive_hold", "x_abilities_seek"]),
                   (38.0, ["x_lord_passive_hold"]), (98.0, ["x_abilities_seek", "x_unit_passive_single"])]
        ready = [{"u": "enemy_lord", "t": 0, "key": "x_abilities_seek", "ready": True}]
        assert lord_ai.ability_uses(ready, windows, "enemy_lord") == [(13.0, "x_abilities_seek"), (98.0, "x_abilities_seek")]
        # ours: a ready true -> false without the phase in the effects still counts
        ready += [{"u": "own_lord", "t": 0, "key": "x_abilities_stand", "ready": True},
                  {"u": "own_lord", "t": 20000, "key": "x_abilities_stand", "ready": False}]
        assert lord_ai.ability_uses(ready, [], "own_lord") == [(20.0, "x_abilities_stand")]

    def test_the_ratio_ai_over_control(self, tmp_path):
        ai = [lord_ai.measure(lord_ai.load_run(ai_run(tmp_path / f"a{k}", "game", 1.2, 0.6))) for k in range(3)]
        ctrl = [lord_ai.measure(lord_ai.load_run(ai_run(tmp_path / f"c{k}", "scripted", 0.8, 0.6))) for k in range(2)]
        (row,) = lord_ai.summary(ai + ctrl)
        assert (row["n_ai"], row["n_control"]) == (3, 2)
        assert row["ratio_on_own"] == pytest.approx(1.5) and row["ratio_lo"] == pytest.approx(1.5)
        assert row["ratio_on_enemy"] == pytest.approx(1.0)
        assert row["ai_enemy_wins"] == 3

    def test_report_writes_its_json(self, tmp_path, capsys):
        ai_run(tmp_path / "runs" / "a", "game", 1.0, 0.5, ability_at=15)
        rows, table = lord_ai.report(lord_ai.runs(tmp_path / "runs"), out=tmp_path / "analysis.json")
        assert len(rows) == 1 and table[0]["ai_abilities"]["x_abilities_rage"]["uses"] == 1
        assert json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))["battles"]
        out = capsys.readouterr().out
        assert "weapon_damage=430/430" in out and "(15.0, 'stat_weapon_damage', 537)" in out
