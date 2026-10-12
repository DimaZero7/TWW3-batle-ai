"""tools.pack and tools.build: pack format, bundling and every build target."""
import json

import pytest
pytest.importorskip("lupa", reason="the Lua runtime (lupa): on the host, not in the training container")

from lupa.lua51 import LuaRuntime  # noqa: E402

from tools import build  # noqa: E402
from tools import config as project  # noqa: E402
from tools.pack import bundle, pfh5  # noqa: E402


class TestPfh5:
    def test_roundtrip_is_deterministic(self):
        files = {"script\\b.lua": b"b", "script\\a.lua": b"a"}
        first = pfh5.pack_files(files, ["dep.pack"])
        assert first == pfh5.pack_files(dict(reversed(list(files.items()))), ["dep.pack"])
        assert pfh5.read_pack(first) == (files, ("dep.pack",))

    @pytest.mark.parametrize("name", ["../x.lua", "/abs.lua", "ы.lua"])
    def test_bad_paths(self, name):
        with pytest.raises(ValueError):
            pfh5.pack_files({name: b""})

    def test_bad_dependency(self):
        with pytest.raises(ValueError):
            pfh5.pack_files({}, ["bad/dep.pack"])


class TestBundle:
    def test_collects_dependencies_first(self):
        modules = [name for name, _ in bundle.collect(project.SRC, "entries.ai_vs_ai")]
        assert modules[-1] == "entries.ai_vs_ai"
        assert modules.index("apps.core.value") < modules.index("apps.core.json")

    def test_bundle_runs_entry_with_loader(self):
        script, _ = bundle.bundle(project.SRC, "apps.core.json", {"k": 1})
        lua = LuaRuntime()
        # Replace the entry call: exercise the loader without a battle.
        script = script.replace("if bm then", "if false then")
        lua.execute(script + "\nresult = __require('apps.core.json').encode({a = 1})")
        assert lua.globals().result == '{"a":1}'


class TestTargets:
    def test_the_kept_targets(self):
        assert sorted(build.TARGETS) == ["ai-vs-ai", "archer-range", "charge-probe", "enemy-layout", "human", "lord-ai", "lord-duel", "lord-fall",
                                         "lord-swarm",
                                         "manual", "map-capture", "missile-probe", "morale-probe",
                                         "move-probe", "nn-arena", "roster-capture", "unit-readout"]

    @pytest.mark.parametrize("target", sorted(build.TARGETS))
    def test_every_target_builds_and_compiles(self, target, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        config = {"step": 5, "features": False} if target == "map-capture" else {
            "speed": 20, "timeout_ms": 60000, "tick_ms": 1000, "scenario": "x"}
        manifest = build.build(target, config)
        assert manifest["syntax_checked"] is True
        entries, _ = pfh5.read_pack((tmp_path / target / manifest["pack"]).read_bytes())
        assert any(name.endswith(".xml") for name in entries)
        assert manifest["config"]["build"] == manifest["build"]


class TestRequiredMods:
    def test_every_build_depends_on_true_sight(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        manifest = build.build("ai-vs-ai", {"speed": 3, "timeout_ms": 60000, "tick_ms": 1000,
                                            "scenario": "ai_vs_ai"})
        _, dependencies = pfh5.read_pack((tmp_path / "ai-vs-ai" / manifest["pack"]).read_bytes())
        assert dependencies == ("true_sight.pack",)
        assert manifest["mod_profile"] == "true-sight-v1"
        assert manifest["dependencies"][0]["sha256"] == project.required_mods()[0]["sha256"]


class TestKvOverride:
    def test_a_probe_pack_carries_its_rule_table_and_names_it(self, tmp_path, monkeypatch):
        from tools.nn import gamedb
        monkeypatch.setattr(project, "BUILD", tmp_path)
        assert build.main(["morale-probe", "--morale-plan", "timer", "--speed", "10",
                           "--kv", "waver_base_timeout=x10"]) == 0
        manifest = json.loads((tmp_path / "morale-probe" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["kv_override"] == {"morale": {"waver_base_timeout": 250.0}}
        assert manifest["config"]["kv_label"] == "waver_base_timeout=250" and manifest["config"]["tick_ms"] == 250
        entries, _ = pfh5.read_pack((tmp_path / "morale-probe" / manifest["pack"]).read_bytes())
        table = entries["db\\_kv_morale_tables\\!tww3_bai_probe_kv"]
        assert gamedb.kv_table(table) == {"waver_base_timeout": 250.0}

    def test_an_ordinary_build_has_no_rule_table(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        assert build.main(["morale-probe", "--morale-plan", "timer"]) == 0
        manifest = json.loads((tmp_path / "morale-probe" / "manifest.json").read_text(encoding="utf-8"))
        assert "kv_override" not in manifest and "kv" not in manifest["config"]
        entries, _ = pfh5.read_pack((tmp_path / "morale-probe" / manifest["pack"]).read_bytes())
        assert not any(name.startswith("db\\") for name in entries)

    def test_kv_only_for_probes_and_known_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        with pytest.raises(SystemExit):
            build.main(["nn-arena", "--kv", "waver_base_timeout=0"])
        with pytest.raises(SystemExit):
            build.main(["morale-probe", "--kv", "waver_base_timout=0"])
        with pytest.raises(AssertionError):
            build.build("ai-vs-ai", {"speed": 3, "timeout_ms": 60000, "tick_ms": 1000, "scenario": "ai_vs_ai"},
                        kv={"morale": {"waver_base_timeout": 0.0}})


class TestMovePlan:
    def test_hamlet_plan_loads_and_stall_rule_covers_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        assert build.main(["move-probe", "--plan", "hamlet"]) == 0
        config = json.loads((tmp_path / "move-probe" / "manifest.json").read_text(encoding="utf-8"))["config"]
        model_s = build.move_plan_model_s(config["plan"])
        assert config["stall_ms"] >= (model_s + 120) * 1000
        assert config["deadline_s"] == build.deadline_seconds(model_s, 20)
        assert {leg["kind"] for leg in config["plan"]["legs"]} == {"shape", "traverse"}

    def test_incomplete_leg_is_rejected(self, tmp_path, monkeypatch):
        (tmp_path / "move-plans").mkdir()
        (tmp_path / "move-plans" / "bad.json").write_text(json.dumps({"legs": [
            {"name": "x", "kind": "shape", "run": False, "timeout_s": 5,
             "start": {"x": 0, "z": 0, "facing": 0}, "target": {"x": 0, "z": 0, "facing": 0, "width": 5}}]}))
        monkeypatch.setattr(project, "CONFIG_DIR", tmp_path)
        with pytest.raises(KeyError):
            build.load_move_plan("bad")


class TestRecorders:
    @pytest.mark.parametrize("own_ai,enemy_role", [("attack", "defend"), ("defend", "attack")])
    def test_nn_arena_records_the_planner_against_the_game_s_ai(self, own_ai, enemy_role, tmp_path, monkeypatch):
        from tools.nn import scenario as nn_scenario
        monkeypatch.setattr(project, "BUILD", tmp_path)
        monkeypatch.setattr(nn_scenario, "SCENARIO", tmp_path / "nn_arena.xml")
        written = []
        monkeypatch.setattr(build, "build", lambda target, config, scenario=None: written.append(config) or {})
        assert build.main(["nn-arena", "--own-ai", own_ai]) == 0
        config = written[0]
        assert config["own_ai"] == own_ai and config["enemy_role"] == enemy_role
        assert {u["slot"] for u in config["units"]["own"]} == {u["slot"] for u in config["units"]["enemy"]}
        assert "decide_ms" not in config
        # The side that wins on timeout defends.
        winner = "1" if enemy_role == "defend" else "0"
        assert f"<timeout_winning_alliance_index>{winner}</timeout_winning_alliance_index>" in \
            (tmp_path / "nn_arena.xml").read_text(encoding="utf-8")

    def test_nn_arena_net_gives_our_side_to_the_network(self, tmp_path, monkeypatch):
        from tools.nn import scenario as nn_scenario
        monkeypatch.setattr(nn_scenario, "SCENARIO", tmp_path / "nn_arena.xml")
        written = []
        monkeypatch.setattr(build, "build", lambda target, config, scenario=None: written.append(config) or {})
        assert build.main(["nn-arena", "--own-ai", "net", "--speed", "1", "--decide-ms", "500"]) == 0
        config = written[0]
        assert config["own_ai"] == "net" and config["enemy_role"] == "attack" and config["speed"] == 1
        assert config["decide_ms"] == 500 and config["poll_ms"] == build.NET_POLL_MS
        assert config["factions"] == {"own": "wh_main_emp_empire", "enemy": "wh_main_emp_empire"}
        assert all(u["key"] for u in config["units"]["own"] + config["units"]["enemy"])
        # The game's AI attacks: our side wins on timeout.
        assert "<timeout_winning_alliance_index>0</timeout_winning_alliance_index>" in             (tmp_path / "nn_arena.xml").read_text(encoding="utf-8")
        with pytest.raises(SystemExit):
            build.main(["nn-arena", "--own-ai", "net", "--decide-ms", "100"])

    def test_nn_arena_skirmish_off_only_for_the_bridge(self, tmp_path, monkeypatch):
        """--skirmish off puts skirmish 'off' into the config (both bridges turn the mode off); the default leaves
        the config as before; the planner modes have no bridge."""
        from tools.nn import scenario as nn_scenario
        monkeypatch.setattr(nn_scenario, "SCENARIO", tmp_path / "nn_arena.xml")
        written = []
        monkeypatch.setattr(build, "build", lambda target, config, scenario=None: written.append(config) or {})
        assert build.main(["nn-arena", "--own-ai", "net", "--enemy-ai", "ai_like", "--skirmish", "off"]) == 0
        assert written[-1]["skirmish"] == "off" and written[-1]["enemy_script"] == "ai_like"
        assert build.main(["nn-arena", "--own-ai", "net"]) == 0
        assert "skirmish" not in written[-1]
        with pytest.raises(SystemExit):
            build.main(["nn-arena", "--own-ai", "attack", "--skirmish", "off"])

    def test_enemy_layout_records_the_game_s_ai_only(self, tmp_path, monkeypatch):
        from tools import enemy_layout
        monkeypatch.setattr(enemy_layout, "SCENARIO", tmp_path / "enemy_layout.xml")
        written = []
        monkeypatch.setattr(build, "build", lambda target, config, scenario=None: written.append(config) or {})
        assert build.main(["enemy-layout", "--layout", "balanced_5"]) == 0
        config = written[0]
        assert config["layout"] == "balanced_5" and config["enemy_mode"] == "defend"
        assert "picture_every" not in config and "roster" not in config
