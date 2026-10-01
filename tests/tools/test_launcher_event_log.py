"""tools/launcher/event_log.ps1: the game-folder event log is removed only once fully copied."""
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
        f". '{REPO / 'tools' / 'telemetry' / 'read_jsonl.ps1'}'\n"
        f". '{REPO / 'tools' / 'launcher' / 'event_log.ps1'}'\n"
        f"$log = '{tmp_path / 'game.jsonl'}'\n"
        f"$runLog = '{tmp_path / 'run.jsonl'}'\n" + body,
        encoding="utf-8-sig",
    )
    done = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()


COPY = """
$reader = New-JsonlReader -Offset 0
$lines = @(Read-JsonlLines -State $reader -Path $log)
[IO.File]::AppendAllLines($runLog, [string[]]$lines, [Text.UTF8Encoding]::new($false))
$copied = [long]$lines.Count
"""


def test_removed_after_full_copy(tmp_path):
    (tmp_path / "game.jsonl").write_bytes(b'{"event":"start"}\n{"event":"result"}\n')
    out = run(tmp_path, COPY + "Remove-CopiedEventLog -Path $log -State $reader -RunLog $runLog -CopiedLines $copied\n")
    assert out == ["removed"]
    assert not (tmp_path / "game.jsonl").exists()
    assert (tmp_path / "run.jsonl").read_text(encoding="utf-8").split() == ['{"event":"start"}', '{"event":"result"}']


def test_kept_when_written_after_the_copy(tmp_path):
    (tmp_path / "game.jsonl").write_bytes(b'{"event":"start"}\n')
    out = run(tmp_path, COPY + "[IO.File]::AppendAllText($log, \"{`\"event`\":`\"late`\"}`n\")\n"
              "Remove-CopiedEventLog -Path $log -State $reader -RunLog $runLog -CopiedLines $copied\n")
    assert out[0].startswith("kept:")
    assert (tmp_path / "game.jsonl").exists()


def test_kept_with_a_partial_line_or_short_copy(tmp_path):
    (tmp_path / "game.jsonl").write_bytes(b'{"event":"start"}\n{"event":"res')
    out = run(tmp_path, COPY + "Remove-CopiedEventLog -Path $log -State $reader -RunLog $runLog -CopiedLines $copied\n"
              "$reader.Pending = ''\n"
              "Remove-CopiedEventLog -Path $log -State $reader -RunLog $runLog -CopiedLines ($copied + 1)\n")
    assert out[0] == "kept: a partial line is not copied"
    assert out[1].startswith("kept: 2 lines copied, 1 in the run copy")
    assert (tmp_path / "game.jsonl").exists()


def test_kept_when_not_read_from_the_start(tmp_path):
    (tmp_path / "game.jsonl").write_bytes(b'{"event":"old"}\n{"event":"new"}\n')
    out = run(tmp_path, "$reader = New-JsonlReader -Offset 16\n"
              "$lines = @(Read-JsonlLines -State $reader -Path $log)\n"
              "[IO.File]::AppendAllLines($runLog, [string[]]$lines, [Text.UTF8Encoding]::new($false))\n"
              "Remove-CopiedEventLog -Path $log -State $reader -RunLog $runLog -CopiedLines $lines.Count\n")
    assert out == ["kept: not read from the start"]


def test_stale_log_moved_and_empty_one_removed(tmp_path):
    (tmp_path / "game.jsonl").write_bytes(b'{"event":"old"}\n')
    out = run(tmp_path, f"Move-StaleEventLog -Path $log -Destination '{tmp_path / 'before.jsonl'}'\n"
              "Remove-CopiedEventLog -Path $log -State (New-JsonlReader) -RunLog $runLog -CopiedLines 0\n"
              "[IO.File]::WriteAllText($log, '')\n"
              f"Move-StaleEventLog -Path $log -Destination '{tmp_path / 'again.jsonl'}'\n")
    assert out == ["16", "absent", "0"]
    assert (tmp_path / "before.jsonl").read_bytes() == b'{"event":"old"}\n'
    assert not (tmp_path / "game.jsonl").exists() and not (tmp_path / "again.jsonl").exists()
