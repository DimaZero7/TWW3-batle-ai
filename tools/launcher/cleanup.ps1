# Removes what a launcher run put into the game folder and left there (a crash, a killed run):
# the files of build/launcher-installed.json whose hash is unchanged (installed.ps1). Every launch
# does this first by itself; run it by hand before playing the game normally (multiplayer) when a
# run was interrupted.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/cleanup.ps1
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $PSScriptRoot 'installed.ps1')
if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 is running; close it first.' }
$gone = Remove-InstalledFiles -Marker (Join-Path $repo 'build\launcher-installed.json')
if ($gone.removed.Count -eq 0 -and $gone.kept.Count -eq 0) { Write-Output 'Nothing of ours in the game folder.' }
foreach ($p in $gone.removed) { Write-Output ("removed: " + $p) }
foreach ($p in $gone.kept) { Write-Output ("changed since, kept: " + $p) }
