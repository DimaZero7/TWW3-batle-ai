"""game: the smoke test without the game (host python + lupa; no torch). The chain's battle and a probe build their packs
from this tree (into a temporary build folder: the real build/ is not touched), the guard of the game (ops/guard.*)
refuses on the marks and lets its holder pass, the launcher asks the guard before it touches the game folder, the
analysis reads a kept chain battle (data/, when it is there). Nothing starts the game.

    python -m pytest -q game/tests
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

import game

M1_RUN = game.DATA / "human" / "runs" / "20261025-192124"
SWAP = "wh3_dlc25_emp_inf_nuln_ironsides=wh3_dlc25_emp_inf_hochland_long_rifles"
CHAIN = ["nn-arena", "--own-ai", "net", "--deploy", "strategist", "--map", "ksl_plains_01", "--speed", "3",
         "--timeout", "1500"]


@pytest.fixture
def build_dir(tmp_path, monkeypatch, capsys):
    """bridge.build into tmp_path: the bridge's and the game's build folder, and the probes' scenario folders."""
    pytest.importorskip("lupa")      # the pack's Lua syntax check (bridge/build.py check_syntax; the host's venv)
    from bridge import build, config
    from game.probe import charge_probe, missile_probe, morale_probe
    monkeypatch.setattr(config, "BUILD", tmp_path)
    monkeypatch.setattr(game, "BUILD", tmp_path)
    for m, name in ((charge_probe, "charge-probe"), (missile_probe, "missile-probe"), (morale_probe, "morale-probe")):
        monkeypatch.setattr(m, "ROOT", tmp_path / name)

    def run(argv):
        assert build.main(argv) == 0
        capsys.readouterr()
        return json.loads((tmp_path / argv[0] / "manifest.json").read_text(encoding="utf-8"))
    return run


def test_chain_battle_pack_m3(build_dir):
    """M3 (the generator's seed 1000702112, v3): the chain's battle as game/battle/chain.sh builds it."""
    m = build_dir(CHAIN + ["--army-seed", "1000702112", "--armies", "v3"])
    c = m["config"]
    assert m["syntax_checked"] and m["pack_sha256"]
    assert (c["own_ai"], c["deploy"], c["speed"], c["timeout_ms"]) == ("net", "strategist", 3, 1500000)
    assert c["map_name"] == "ksl_plains_01" and c["army"]["seed"] == 1000702112
    assert len(c["units"]["own"]) == 20 and len(c["units"]["enemy"]) == 20
    from bridge import config
    assert m["dependencies"] == config.required_mods()             # True Sight, pinned: every battle loads it


@pytest.mark.skipif(not (M1_RUN / "manifest.json").exists(), reason="data/human/runs (out of git) is not here")
def test_chain_battle_pack_m1_from_run(build_dir):
    """M1: the person's run of 25.10 again, the Nuln Ironsides -> the Hochland long rifles (game.arena.army_from_run)."""
    m = build_dir(CHAIN + ["--army-from-run", str(M1_RUN), "--army-replace", SWAP])
    a = m["config"]["army"]
    assert a["from_run"] == "20261025-192124" and a["replaced"] == dict([SWAP.split("=")])
    keys = {u["key"] for u in m["config"]["units"]["own"]}
    assert "wh3_dlc25_emp_inf_hochland_long_rifles" in keys and "wh3_dlc25_emp_inf_nuln_ironsides" not in keys


def test_probe_pack(build_dir):
    """A stage 0 probe (the hill plan) as game/probe/run.sh builds it (without --mem: no memreader needed here)."""
    from game.probe import charge_probe
    assert "hill" in charge_probe.PLANS
    m = build_dir(["charge-probe", "--probe-plan", "hill", "--probe-battle", "1", "--deadline", "900"])
    c = m["config"]
    assert m["syntax_checked"] and c["plan"] == "hill" and c["lanes"]
    assert (game.BUILD / "charge-probe").is_dir()


def test_guard_py(tmp_path, monkeypatch):
    from ops import guard
    monkeypatch.delenv(guard.ENV, raising=False)
    assert guard.busy(state=tmp_path) == ""
    (tmp_path / "HUMAN_PLAYING").write_text("")
    assert "user plays" in guard.busy(state=tmp_path)
    assert guard.busy("human", state=tmp_path) == ""            # the user's own battle (live labelling)
    (tmp_path / "HUMAN_PLAYING").unlink()
    with guard.hold("smoke", state=tmp_path) as tok:
        assert guard.owner(tmp_path / "probe_game.lock") == tok == os.environ[guard.ENV]
        assert guard.busy(state=tmp_path) == ""                   # the holder's own launches pass
        with guard.hold("inner", state=tmp_path) as inner:      # a probe's run() under a runner: rides on it
            assert inner == tok
        monkeypatch.setenv(guard.ENV, "someone else")
        assert "held" in guard.busy(state=tmp_path)
        with pytest.raises(guard.Busy):
            with guard.hold("second", state=tmp_path):
                pass
        monkeypatch.setenv(guard.ENV, tok)
    assert not (tmp_path / "probe_game.lock").exists()


def _ps(script):
    return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)


@pytest.mark.skipif(not shutil.which("powershell"), reason="Windows PowerShell")
def test_guard_ps1(tmp_path):
    """The launcher's guard (ops/guard.ps1) on a temporary state folder: refuses on HUMAN_PLAYING (not the human
    target) and on a lock it does not hold; Enter / Exit take and free the lock."""
    g = game.ROOT / "ops" / "guard.ps1"
    s = str(tmp_path)
    (tmp_path / "HUMAN_PLAYING").write_text("")
    r = _ps(f". '{g}'; $env:BAI_LOCK_HELD=''; try {{ Assert-GameFree -Target nn-arena -State '{s}'; 'passed' }} "
            f"catch {{ 'refused' }}; Assert-GameFree -Target human -State '{s}'; 'human ok'")
    assert r.stdout.split() == ["refused", "human", "ok"], r.stdout + r.stderr
    (tmp_path / "HUMAN_PLAYING").unlink()
    r = _ps(f". '{g}'; $env:BAI_LOCK_HELD=''; $t = Enter-GameLock -Label smoke -State '{s}'; "
            f"Assert-GameFree -State '{s}'; 'holder ok'; $env:BAI_LOCK_HELD='other'; "
            f"try {{ Assert-GameFree -State '{s}'; 'passed' }} catch {{ 'refused' }}; $env:BAI_LOCK_HELD=$t; "
            f"Exit-GameLock $t -State '{s}'; Test-Path (Join-Path '{s}' 'probe_game.lock')")
    assert r.stdout.split() == ["holder", "ok", "refused", "False"], r.stdout + r.stderr


def _bash():
    """Git Bash on Windows (not WSL's System32 bash), bash elsewhere; None when there is none."""
    git = shutil.which("git")
    for c in ([os.path.join(os.path.dirname(os.path.dirname(git)), "bin", "bash.exe")] if git else []) + [shutil.which("bash")]:
        if c and os.path.exists(c) and "system32" not in c.lower():
            return c
    return None


@pytest.mark.skipif(_bash() is None, reason="bash")
def test_guard_sh(tmp_path):
    g = (game.ROOT / "ops" / "guard.sh").as_posix()
    env = dict(os.environ, GUARD_STATE=tmp_path.as_posix(), BAI_LOCK_HELD="")
    script = (f". '{g}'; guard_take smoke && echo took; BAI_LOCK_HELD=x guard_busy >/dev/null && echo held-for-others; "
              f"guard_busy >/dev/null || echo free-for-me; guard_release; [ -e \"$GUARD_STATE/probe_game.lock\" ] || echo freed; "
              f"touch \"$GUARD_STATE/HUMAN_PLAYING\"; guard_take smoke2 2>/dev/null || echo refused")
    r = subprocess.run([_bash(), "-c", script], capture_output=True, text=True, env=env, timeout=60)
    assert r.stdout.split() == ["took", "held-for-others", "free-for-me", "freed", "refused"], r.stdout + r.stderr


def test_launcher_checks_the_marks():
    """bridge/launcher/launch.ps1 refuses on state/HUMAN_PLAYING (not the human target) and on a state/probe_game.lock
    it does not hold (BAI_LOCK_HELD: the guard's own name) before it touches the game folder or starts the game - so no
    way to the game (probes, chain, labelling) passes the marks by."""
    from ops import guard
    text = (game.ROOT / "bridge" / "launcher" / "launch.ps1").read_text(encoding="utf-8-sig")
    marks = [text.index(k) for k in ("'HUMAN_PLAYING'", "'probe_game.lock'", "$env:" + guard.ENV)]
    first_touch = min(text.index(k) for k in ("Remove-InstalledFiles -Marker", "Copy-Item -LiteralPath $sourcePack",
                                              "Start-Process"))
    assert max(marks) < first_touch


@pytest.mark.skipif(not (game.DATA / "nn-arena" / "runs" / "20261108-092727").is_dir(), reason="data/ is not here")
def test_chain_battle_table():
    """The after6 M1 battle 1 (08.11): the chain lost, 361 s, 72 % of the unit-seconds inside their areas."""
    from game.battle import table
    row = table.row("20261108-092727")
    assert (row["winner"], row["t"], row["inside"], row["strat_err"]) == (2, 361, 0.72, 0)


@pytest.mark.skipif(not (game.DATA / "nn-arena" / "runs" / "20261107-220829").is_dir(), reason="data/ is not here")
def test_vision(tmp_path):
    """game.vision on the base M1 chain battle (07.11 22:08): the strip and the text, within ~5 k tokens; the
    long rifles 2 that never fired routed first at 108 s; the operative's 7 groups."""
    from game import vision
    from game.vision import record
    res = vision.render(record.load("20261107-220829"), tmp_path)
    text = (tmp_path / "20261107-220829.txt").read_text(encoding="utf-8")
    assert (tmp_path / "20261107-220829.png").stat().st_size > 0
    assert 1000 < res["tokens"]["image"] + res["tokens"]["text"] < 5000
    row = next(line for line in text.splitlines() if line.startswith("long_rifles_2 S | +55,-78 f0"))
    assert "| 0/0/0 | 97 | 108 |" in row and "our routs: 108 long_rifles_2" in text
    assert text.count("\n g") == 7


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
