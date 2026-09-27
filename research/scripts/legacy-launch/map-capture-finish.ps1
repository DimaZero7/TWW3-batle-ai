param([Parameter(Mandatory=$true)][string]$Run)
$ErrorActionPreference='Stop'
$finishRepo=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$finishRoot=(Resolve-Path -LiteralPath (Join-Path $finishRepo 'build\map-capture')).Path
$finishRun=(Resolve-Path -LiteralPath $Run).Path
if (-not $finishRun.StartsWith($finishRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Run must be inside build/map-capture.' }
$finishLaunch=Get-Content -LiteralPath (Join-Path $finishRun 'launch.json') -Raw | ConvertFrom-Json
$finishStatus=Get-Content -LiteralPath (Join-Path $finishRun 'status.json') -Raw | ConvertFrom-Json
if ($finishStatus.status -ne 'probe_done') { throw 'Incomplete run; inspect and preserve it before cleanup.' }
$finishGame='C:\Program Files (x86)\Steam\steamapps\common\Total War WARHAMMER III'
$finishPack=Join-Path $finishGame 'data\tww3_bai_map_capture.pack'
$finishMods=Join-Path $finishGame 'tww3_bai_map_capture_mods.txt'
if ((Get-FileHash -LiteralPath $finishPack).Hash.ToLowerInvariant() -ne $finishLaunch.pack_sha256) { throw 'Installed pack changed; not removing it.' }
if ((Get-Content -LiteralPath $finishMods -Raw) -ne 'mod "tww3_bai_map_capture.pack";') { throw 'Modlist changed; not removing it.' }
$finishOutputs=@('tww3_bai_map_capture_events.jsonl','tww3_bai_map_capture_ready.xml','tww3_bai_map_capture_grid.csv')
foreach ($finishName in $finishOutputs) {
    if ((Get-FileHash -LiteralPath (Join-Path $finishGame $finishName)).Hash -ne (Get-FileHash -LiteralPath (Join-Path $finishRun $finishName)).Hash) { throw ('Saved output differs: '+$finishName) }
}
$finishProcess=Get-Process -Id $finishLaunch.pid -ErrorAction SilentlyContinue
if ($finishProcess) {
    if ($finishProcess.Path -ne (Join-Path $finishGame 'Warhammer3.exe') -or [Math]::Abs(($finishProcess.StartTime.ToUniversalTime()-[datetime]$finishLaunch.started_utc).TotalSeconds) -gt 15) { throw 'Process identity differs; not stopping it.' }
    Stop-Process -Id $finishProcess.Id
}
for ($finishAttempt=0;$finishAttempt -lt 40;$finishAttempt++) {
    $finishRemaining=Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue
    if (-not $finishRemaining) { break }
    if ($finishRemaining | Where-Object { $_.Id -ne $finishLaunch.pid }) { throw 'Another WH3 process is running; files left intact.' }
    Start-Sleep -Milliseconds 500
}
if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 still running; files left intact.' }
foreach ($finishName in $finishOutputs) { Remove-Item -LiteralPath (Join-Path $finishGame $finishName) }
Remove-Item -LiteralPath $finishPack
Remove-Item -LiteralPath $finishMods
@{closed_pid=$finishLaunch.pid;closed_utc=(Get-Date).ToUniversalTime().ToString('o');outputs_verified_and_preserved=$true;private_files_removed=$true} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $finishRun 'cleanup.json') -Encoding utf8
Write-Output ('Test closed; game-side files removed; results preserved at '+$finishRun)
