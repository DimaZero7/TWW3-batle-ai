"""tools/launcher/game_process.ps1: the game's priority and fast (P) cores, chosen from parameters, env and config."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(POWERSHELL is None or os.name != "nt", reason="Windows PowerShell not available")


def run(tmp_path, body, env=None):
    script = tmp_path / "case.ps1"
    script.write_text("$ErrorActionPreference = 'Stop'\n"
                      f". '{REPO / 'tools' / 'launcher' / 'game_process.ps1'}'\n" + body, encoding="utf-8-sig")
    e = {k: v for k, v in os.environ.items() if k not in ("BAI_GAME_PRIORITY", "BAI_FAST_CORES")}
    e.update(env or {})
    done = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                          capture_output=True, text=True, timeout=120, env=e)
    assert done.returncode == 0, done.stderr
    return done.stdout.split()


def cores(rows):
    return "$c = @(" + ", ".join(f"[pscustomobject]@{{efficiency = {e}; mask = [long]{m}}}" for e, m in rows) + ")\n"


def test_hybrid_cpu_keeps_the_p_cores_with_their_second_threads(tmp_path):
    # i9-13900HK: 6 P cores with two threads each (logical 0-11), 8 E cores (12-19)
    rows = [(1, 3 << (2 * i)) for i in range(6)] + [(0, 1 << (12 + i)) for i in range(8)]
    out = run(tmp_path, cores(rows) + "'{0:X}' -f (Get-FastCoreMask -Cores $c)\n")
    assert out == ["FFF"]


def test_a_cpu_with_one_core_class_is_left_alone(tmp_path):
    out = run(tmp_path, cores([(0, 1 << i) for i in range(8)]) + "$null -eq (Get-FastCoreMask -Cores $c)\n")
    assert out == ["True"]


def test_this_machine_s_cores_cover_disjoint_logical_processors(tmp_path):
    out = run(tmp_path, "$c = @(Get-CpuCores); $all = [long]0; $overlap = $false\n"
                        "foreach ($x in $c) { if ($all -band $x.mask) { $overlap = $true }; $all = $all -bor $x.mask }\n"
                        "$c.Count; $overlap; [Environment]::ProcessorCount -eq @($c | ForEach-Object { $_.logical }).Count\n")
    assert int(out[0]) >= 1 and out[1] == "False" and out[2] == "True"


def test_defaults_are_high_priority_on_fast_cores(tmp_path):
    out = run(tmp_path, "$o = Resolve-GameProcessOptions -Settings ([pscustomobject]@{game_dir = 'x'}); $o.priority; $o.fast_cores\n")
    assert out == ["High", "True"]


def test_parameter_beats_environment_beats_config(tmp_path):
    cfg = "$s = [pscustomobject]@{game_priority = 'AboveNormal'; game_fast_cores = $false}\n"
    out = run(tmp_path, cfg + "$o = Resolve-GameProcessOptions -Settings $s; $o.priority; $o.fast_cores\n")
    assert out == ["AboveNormal", "False"]
    out = run(tmp_path, cfg + "$o = Resolve-GameProcessOptions -Settings $s; $o.priority; $o.fast_cores\n",
              env={"BAI_GAME_PRIORITY": "Normal", "BAI_FAST_CORES": "on"})
    assert out == ["Normal", "True"]
    out = run(tmp_path, cfg + "$o = Resolve-GameProcessOptions -Settings $s -Priority High -FastCores off; $o.priority; $o.fast_cores\n",
              env={"BAI_GAME_PRIORITY": "Normal"})
    assert out == ["High", "False"]


def test_a_wrong_priority_stops_the_run(tmp_path):
    script = tmp_path / "bad.ps1"
    script.write_text(f". '{REPO / 'tools' / 'launcher' / 'game_process.ps1'}'\n"
                      "Resolve-GameProcessOptions -Settings $null -Priority RealTime\n", encoding="utf-8-sig")
    done = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                          capture_output=True, text=True, timeout=60)
    assert "must be Normal, AboveNormal or High" in done.stderr + done.stdout


def test_set_game_process_changes_only_the_given_process(tmp_path):
    # a child process of our own: High priority and the fast-core mask (or 'not hybrid'), then it is stopped
    out = run(tmp_path, "$p = Start-Process -FilePath powershell -ArgumentList '-NoProfile','-Command','Start-Sleep 20' "
                        "-WindowStyle Hidden -PassThru\n"
                        "try { $i = Set-GameProcess -Process $p -Priority High -FastCores $true; $p.Refresh()\n"
                        "$p.PriorityClass; $i.priority; $m = Get-FastCoreMask -Cores @(Get-CpuCores)\n"
                        "if ($null -eq $m) { $i.affinity -eq 'not hybrid' } else { [long]$p.ProcessorAffinity -eq $m }\n"
                        "} finally { Stop-Process -Id $p.Id }\n"
                        "[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass\n")
    assert out == ["High", "High", "True", "Normal"]


def test_a_priority_the_game_put_back_is_restored_once(tmp_path):
    # the game sets itself back to Normal while loading; the launcher's poll puts High back
    out = run(tmp_path, "$p = Start-Process -FilePath powershell -ArgumentList '-NoProfile','-Command','Start-Sleep 20' "
                        "-WindowStyle Hidden -PassThru\n"
                        "try { $p.PriorityClass = 'Normal'\n"
                        "Restore-GamePriority -Process $p -Priority High; $p.Refresh(); $p.PriorityClass\n"
                        "Restore-GamePriority -Process $p -Priority High\n"
                        "} finally { Stop-Process -Id $p.Id }\n")
    assert out == ["True", "High", "False"]
