"""tools/launcher/installed.ps1: what the launcher puts into the game folder is removed by its marker, only if unchanged
(a True Sight copy left in data/ by a crashed run broke the user's multiplayer)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(POWERSHELL is None, reason="PowerShell not available")


def run(tmp_path, body):
    script = tmp_path / "case.ps1"
    script.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f". '{REPO / 'tools' / 'launcher' / 'installed.ps1'}'\n"
        f"$marker = '{tmp_path / 'build' / 'marker.json'}'\n" + body,
        encoding="utf-8-sig",
    )
    done = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()


def test_marker_files_removed_changed_kept_others_untouched(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "true_sight.pack").write_bytes(b"mod")
    (data / "ours.pack").write_bytes(b"ours")
    (data / "user.pack").write_bytes(b"user")
    mods = tmp_path / "mods.txt"
    out = run(tmp_path, f"""
$text = 'mod "true_sight.pack";' + [Environment]::NewLine
Save-InstalledMarker -Marker $marker -Entries @(
  @{{path = '{data / 'true_sight.pack'}'; sha256 = (Get-FileSha256 '{data / 'true_sight.pack'}')}},
  @{{path = '{data / 'ours.pack'}'; sha256 = (Get-FileSha256 '{data / 'ours.pack'}')}},
  @{{path = '{mods}'; sha256 = (Get-TextSha256 $text)}})
[IO.File]::WriteAllText('{mods}', $text, [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText('{data / 'ours.pack'}', 'changed')
$g = Remove-InstalledFiles -Marker $marker
Write-Output ("removed " + $g.removed.Count + " kept " + $g.kept.Count)
""")
    assert out == ["removed 2 kept 1"]
    assert not (data / "true_sight.pack").exists() and not mods.exists()
    assert (data / "ours.pack").exists() and (data / "user.pack").exists()
    left = json.loads((tmp_path / "build" / "marker.json").read_text(encoding="utf-8"))["files"]
    left = left if isinstance(left, list) else [left]
    assert [Path(e["path"]).name for e in left] == ["ours.pack"]


def test_marker_gone_when_all_removed_and_no_marker_is_a_no_op(tmp_path):
    (tmp_path / "x.pack").write_bytes(b"x")
    out = run(tmp_path, f"""
Save-InstalledMarker -Marker $marker -Entries @(@{{path = '{tmp_path / 'x.pack'}'; sha256 = (Get-FileSha256 '{tmp_path / 'x.pack'}')}})
$g = Remove-InstalledFiles -Marker $marker
Write-Output ("removed " + $g.removed.Count + " marker " + (Test-Path -LiteralPath $marker))
$g = Remove-InstalledFiles -Marker $marker
Write-Output ("again " + $g.removed.Count)
""")
    assert out == ["removed 1 marker False", "again 0"]
    assert not (tmp_path / "x.pack").exists()
