"""tools.pack and tools.build: pack format, bundling and every build target."""
import json

import pytest
from lupa.lua51 import LuaRuntime

from tools import build
from tools import config as project
from tools.pack import bundle, pfh5


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
        modules = [name for name, _ in bundle.collect(project.SRC, "entries.duel")]
        assert modules[-1] == "entries.duel"
        assert modules.index("apps.core.value") < modules.index("apps.core.json")

    def test_bundle_runs_entry_with_loader(self):
        script, _ = bundle.bundle(project.SRC, "apps.core.json", {"k": 1})
        lua = LuaRuntime()
        # Replace the entry call: exercise the loader without a battle.
        script = script.replace("if bm then", "if false then")
        lua.execute(script + "\nresult = __require('apps.core.json').encode({a = 1})")
        assert lua.globals().result == '{"a":1}'


class TestTargets:
    @pytest.mark.parametrize("target", sorted(build.TARGETS))
    def test_every_target_builds_and_compiles(self, target, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        config = {"step": 5, "features": False} if target == "map-capture" else {
            "runs": 1, "speed": 20, "timeout_ms": 60000, "tick_ms": 1000, "scenario": "x"}
        manifest = build.build(target, config)
        assert manifest["syntax_checked"] is True
        entries, _ = pfh5.read_pack((tmp_path / target / manifest["pack"]).read_bytes())
        assert any(name.endswith(".xml") for name in entries)
        assert manifest["config"]["build"] == manifest["build"]


class TestRequiredMods:
    def test_every_build_depends_on_true_sight(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        manifest = build.build("ai-vs-ai", {"runs": 1, "speed": 3, "timeout_ms": 60000,
                                            "tick_ms": 1000, "scenario": "ai_vs_ai"})
        _, dependencies = pfh5.read_pack((tmp_path / "ai-vs-ai" / manifest["pack"]).read_bytes())
        assert dependencies == ("true_sight.pack",)
        assert manifest["mod_profile"] == "true-sight-v1"
        assert manifest["dependencies"][0]["sha256"] == project.required_mods()[0]["sha256"]


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
