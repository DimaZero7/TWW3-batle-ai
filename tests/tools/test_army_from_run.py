"""tools/nn/army_from_run.py and tools.build nn-arena --army-from-run: an old arena run's armies and deployment
read back from its manifest. A run built from a seed is played again from its manifest: the same units, places,
factions, battle file. Levels 1-2: no game."""
import json

import pytest

from tools import build
from tools import config as project
from tools.nn import army_from_run
from tools.nn import scenario as nn_scenario

SEED = 1_000_901_824
SAME = ("units", "factions", "gap_m", "defend_radius_m", "own_ai", "own_role", "enemy_role", "deadline_s")


def build_config(tmp_path, monkeypatch, *args):
    monkeypatch.setattr(project, "BUILD", tmp_path / "build")
    monkeypatch.setattr(nn_scenario, "SCENARIO", tmp_path / "nn_arena.xml")
    written = []
    monkeypatch.setattr(build, "build", lambda target, config, scenario=None: written.append((config, scenario)) or {})
    assert build.main(["nn-arena", "--timeout", "3600", *args]) == 0
    (config, scenario), = written
    return config, open(scenario, encoding="utf-8").read()


def old_run(tmp_path, monkeypatch, *args):
    """A run folder whose manifest is the build of a generated battle."""
    config, xml = build_config(tmp_path, monkeypatch, "--army-seed", str(SEED), *args)
    run = tmp_path / ("run_swap" if args else "run")
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"target": "nn-arena", "config": config}), encoding="utf-8")
    return run, config, xml


def test_side_frame_inverts_the_placement():
    for side in nn_scenario.SIDES:
        arena = {"gap_m": 350, "sides": {s: {"faction": "f", "units": [{"slot": "a", "key": "k", "men": 1,
                                                                       "forward": -52.5, "lateral": 36.0, "width": 30.0}]}
                                         for s in nn_scenario.SIDES}}
        u = nn_scenario.placements(arena)[side][0]
        assert army_from_run.side_frame(side, u["x"], u["z"], 350) == (-52.5, 36.0)


@pytest.mark.parametrize("swap", [False, True])
def test_a_run_is_played_again_from_its_manifest(tmp_path, monkeypatch, swap):
    run, old, old_xml = old_run(tmp_path, monkeypatch)
    want, want_xml = (build_config(tmp_path, monkeypatch, "--army-seed", str(SEED), "--army-swap") if swap
                      else (old, old_xml))
    config, xml = build_config(tmp_path, monkeypatch, "--army-from-run", str(run), *(["--army-swap"] if swap else []))
    assert {k: config[k] for k in SAME} == {k: want[k] for k in SAME}
    assert config["army"] == dict(want["army"], from_run="run")
    assert config["arena"] == "random_%d_runrun%s" % (SEED, "_swap" if swap else "")
    assert xml == want_xml


def test_the_swapped_run_swapped_back_is_the_first(tmp_path, monkeypatch):
    run, old, _ = old_run(tmp_path, monkeypatch, "--army-swap")
    config, _ = build_config(tmp_path, monkeypatch, "--army-from-run", str(run), "--army-swap")
    first, _ = build_config(tmp_path, monkeypatch, "--army-seed", str(SEED))
    assert config["units"] == first["units"] and config["factions"] == first["factions"]
    assert config["army"]["swap"] is False


def test_changed_pools_and_bad_options_are_refused(tmp_path, monkeypatch):
    run, old, _ = old_run(tmp_path, monkeypatch)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["config"]["army"]["men"]["own"] += 1
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        army_from_run.load(run)
    for bad in (["--army-from-run", str(run), "--army-seed", str(SEED)],
                ["--army-from-run", str(tmp_path / "missing")],
                ["--army-from-run", str(run), "--arena", "pair_spear_v_slave"]):
        with pytest.raises(SystemExit):
            build.main(["nn-arena", *bad])
