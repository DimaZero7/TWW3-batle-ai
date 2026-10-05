"""tools.nn.lord_fall and tools.nn.lord_duel: the battles' layout, the builds, the plans and the tables."""
import json
import math

import pytest

from tools import build
from tools import config as project
from tools.nn import lord_duel, lord_fall
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
        assert [b for b in p if b[0] == "emp" and b[1] == "net"] == [("emp", "net", r) for r in
                                                                    ("attack", "defend", "attack", "defend")]
        assert sum(b[1] == "scripted" for b in p) == 4


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
