"""tools/launcher/preferences.ps1: the run's values written into the game's preferences text, nothing else."""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(POWERSHELL is None, reason="PowerShell not available")

PREFS = (
    "gfx_aa 0; # gfx_aa <int>, Set antialiasing, 0-no, 1 = FXAA, 2 = TAA, 3 = TAA High #\r\n"
    "gfx_ssao false; # gfx_ssao <bool>, Enable Screen Space Ambient Occlusion buffer #\r\n"
    "gfx_shadow_quality 0; # gfx_shadow_quality <int>, Set shadow quality. 0 - off, 4 - extreme #\r\n"
    "autoresolve_difficulty 1; # autoresolve_difficulty <int>, 0 is easy #\r\n"
    "battle_difficulty 3; # battle_difficulty <int>, 0 is easy, 1 is normal #\r\n"
)


def run(tmp_path, body):
    (tmp_path / "prefs.txt").write_bytes(PREFS.encode("latin-1"))
    script = tmp_path / "case.ps1"
    script.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f". '{REPO / 'tools' / 'launcher' / 'preferences.ps1'}'\n"
        f"$text = [IO.File]::ReadAllText('{tmp_path / 'prefs.txt'}', [Text.Encoding]::GetEncoding(28591))\n" + body,
        encoding="utf-8-sig",
    )
    return subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                          capture_output=True, text=True, timeout=60)


def test_only_the_named_values_change(tmp_path):
    done = run(tmp_path, "$out = Set-PreferenceValues -Text $text -Values ([ordered]@{battle_difficulty = 1; gfx_ssao = 'true'})\n"
                         f"[IO.File]::WriteAllText('{tmp_path / 'out.txt'}', $out, [Text.Encoding]::GetEncoding(28591))\n")
    assert done.returncode == 0, done.stderr
    out = (tmp_path / "out.txt").read_bytes().decode("latin-1")
    assert out == PREFS.replace("battle_difficulty 3;", "battle_difficulty 1;").replace("gfx_ssao false;", "gfx_ssao true;")


def test_the_ultra_preset_sets_every_quality_to_its_ultra_value(tmp_path):
    done = run(tmp_path, "$GraphicsPresets.ultra.Keys | ForEach-Object { '{0}={1}' -f $_, $GraphicsPresets.ultra[$_] }\n")
    assert done.returncode == 0, done.stderr
    ultra = dict(line.split("=") for line in done.stdout.split())
    assert ultra["gfx_shadow_quality"] == "3" and ultra["gfx_texture_quality"] == "3" and ultra["gfx_aa"] == "2"
    assert ultra["gfx_ssao"] == "true" and "battle_difficulty" not in ultra and "x_res" not in ultra


def test_a_missing_key_stops_the_run(tmp_path):
    done = run(tmp_path, "Set-PreferenceValues -Text $text -Values @{gfx_tree_quality = 3}\n")
    assert done.returncode != 0 and "gfx_tree_quality not found" in done.stderr + done.stdout
