param([int]$TimeoutSeconds=1200)
$ErrorActionPreference='Stop'
throw 'Legacy vanilla entry point retired. Use tools/competition/run.py; True Sight is mandatory. Migrate this diagnostic with dependency verification before reuse.'
$scanTools=$PSScriptRoot
$scanRoot=Join-Path (Resolve-Path -LiteralPath (Join-Path $scanTools '..\..')).Path 'build\map-capture'
$scanRepo=(Resolve-Path -LiteralPath (Join-Path $scanTools '..\..')).Path
$scanGame='C:\Program Files (x86)\Steam\steamapps\common\Total War WARHAMMER III'
$scanPrefix='tww3_bai_map_capture_'
if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 already running; not interrupting.' }
$scanPack=Join-Path $scanGame 'data\tww3_bai_map_capture.pack'
$scanMods=Join-Path $scanGame ($scanPrefix+'mods.txt')
$scanLog=Join-Path $scanGame ($scanPrefix+'events.jsonl')
if ((Test-Path -LiteralPath $scanPack) -or (Get-ChildItem -LiteralPath $scanGame -Filter ($scanPrefix+'*'))) { throw 'Existing experiment files; preserve before reusing.' }
$scanRun=Join-Path $scanRoot ('run-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $scanRun | Out-Null
foreach ($scanName in @('capture.lua','reader.lua','map_probe.xml','manifest.json')) { Copy-Item -LiteralPath (Join-Path $scanRoot $scanName) -Destination $scanRun }
$scanManifest=Get-Content -LiteralPath (Join-Path $scanRoot 'manifest.json') -Raw | ConvertFrom-Json
Copy-Item -LiteralPath (Join-Path $scanRoot 'tww3_bai_map_capture.pack') -Destination $scanPack
if ((Get-FileHash -LiteralPath $scanPack).Hash.ToLowerInvariant() -ne $scanManifest.sha256) { throw 'Hash mismatch' }
[IO.File]::WriteAllText($scanMods,'mod "tww3_bai_map_capture.pack";',[Text.UTF8Encoding]::new($false))
& (Join-Path $scanRepo 'tools\set-test-graphics.ps1')
$scanStart=Get-Date
$scanProcess=Start-Process -FilePath (Join-Path $scanGame 'Warhammer3.exe') -WorkingDirectory $scanGame -ArgumentList 'game_startup_mode battle script/battle/tww3_bai_map_capture/map_probe.xml; tww3_bai_map_capture_mods.txt;' -WindowStyle Hidden -PassThru
try {
@{pid=$scanProcess.Id;started_utc=$scanStart.ToUniversalTime().ToString('o');pack_sha256=$scanManifest.sha256;static_scan=$true;step=$scanManifest.step} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $scanRun 'launch.json') -Encoding utf8
Write-Output ('Launched map capture PID '+$scanProcess.Id+'; run '+$scanRun)
$scanSeen=[System.Collections.Generic.HashSet[string]]::new()
while (((Get-Date)-$scanStart).TotalSeconds -lt $TimeoutSeconds) {
    if ($scanProcess.HasExited) { throw 'Process exited; inspect crash reports.' }
    $scanCrash=Get-ChildItem -LiteralPath "$env:APPDATA\The Creative Assembly\Warhammer3\crash_report" -Filter '*.stack.txt' -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt $scanStart } | Select-Object -First 1
    if ($scanCrash) { Copy-Item -LiteralPath $scanCrash.FullName -Destination $scanRun; throw 'New crash report found.' }
    if (Test-Path -LiteralPath $scanLog) {
        $scanLines=Get-Content -LiteralPath $scanLog
        foreach ($scanLine in $scanLines) {
            if ($scanLine -match '"event":"(frame|grid_begin|grid_done|probe_error)"' -and $scanSeen.Add($scanLine)) { Write-Output $scanLine }
        }
        if ($scanLines -match '"event":"probe_error"') { Copy-Item -LiteralPath $scanLog -Destination $scanRun; throw 'Lua diagnostic error; see preserved log.' }
        if ($scanLines -match '"event":"probe_done"') {
            Get-ChildItem -LiteralPath $scanGame -Filter ($scanPrefix+'*') | Where-Object { $_.Extension -in @('.csv','.xml','.jsonl') } | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $scanRun }
            $scanProcess.Refresh()
            @{status='probe_done';pid=$scanProcess.Id;game_left_running=$true;game_paused=$true;elapsed_including_load_s=((Get-Date)-$scanStart).TotalSeconds;peak_working_set_bytes=$scanProcess.PeakWorkingSet64} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $scanRun 'status.json') -Encoding utf8
            & (Join-Path $scanTools 'finish.ps1') -Run $scanRun
            $scanFinalStatus=Get-Content -LiteralPath (Join-Path $scanRun 'status.json') -Raw | ConvertFrom-Json
            $scanFinalStatus.game_left_running=$false
            $scanFinalStatus.game_paused=$false
            $scanFinalStatus | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $scanRun 'status.json') -Encoding utf8
            Write-Output 'Map capture complete; game closed automatically; results preserved.'
            exit 0
        }
    }
    Start-Sleep -Seconds 1
}
throw 'Experiment timeout; partial files preserved for diagnosis.'
} finally {
    # Errors also close only our own process; unfinished output is preserved.
    $scanRemaining=Get-Process -Id $scanProcess.Id -ErrorAction SilentlyContinue
    if ($scanRemaining -and $scanRemaining.Path -eq (Join-Path $scanGame 'Warhammer3.exe') -and [Math]::Abs(($scanRemaining.StartTime.ToUniversalTime()-$scanStart.ToUniversalTime()).TotalSeconds) -lt 15) {
        Stop-Process -Id $scanRemaining.Id
        for ($scanWait=0;$scanWait -lt 40;$scanWait++) {
            if (-not (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue | Where-Object { $_.Id -eq $scanProcess.Id })) { break }
            Start-Sleep -Milliseconds 500
        }
    }
}
