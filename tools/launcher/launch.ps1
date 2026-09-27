# Installs a built pack, starts WH3 on its scenario, waits for the result and
# cleans up. Build first: python -m tools.build <target>
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target duel
#
# Safety rules (kept from the research launchers):
#  * refuses to start while any Warhammer3 process runs;
#  * uses its own mod list file; the user's used_mods.txt is never touched;
#  * stops only the process it started (same PID, path and start time);
#  * removes only its own pack and mod list, and only if unchanged;
#  * results are copied to build/<target>/runs/<time>/ before cleanup.
param(
    [Parameter(Mandatory = $true)][ValidateSet('duel', 'arena', 'ai-vs-ai', 'unit-readout', 'move-probe', 'manual', 'roster-capture', 'formation-probe', 'map-capture')][string]$Target,
    [int]$TimeoutSeconds = 0,
    [switch]$KeepGameOpen
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $repo 'tools\telemetry\read_jsonl.ps1')

# Settings: config/default.json overridden by config/local.json.
$settings = Get-Content -LiteralPath (Join-Path $repo 'config\default.json') -Raw | ConvertFrom-Json
$localConfig = Join-Path $repo 'config\local.json'
if (Test-Path -LiteralPath $localConfig) {
    $local = Get-Content -LiteralPath $localConfig -Raw | ConvertFrom-Json
    foreach ($property in $local.PSObject.Properties) {
        $settings | Add-Member -NotePropertyName $property.Name -NotePropertyValue $property.Value -Force
    }
}
$game = (Resolve-Path -LiteralPath $settings.game_dir).Path
$exe = Join-Path $game 'Warhammer3.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Warhammer3.exe not found in $game (set game_dir in config/local.json)" }

$buildDir = Join-Path $repo ('build\' + $Target)
$manifest = Get-Content -LiteralPath (Join-Path $buildDir 'manifest.json') -Raw | ConvertFrom-Json
$sourcePack = Join-Path $buildDir $manifest.pack
$installedPack = Join-Path $game ('data\' + $manifest.pack)
$modList = Join-Path $game ('tww3_bai_' + ($Target -replace '-', '_') + '_mods.txt')

# Where the entry writes its events and when the run is complete.
if ($Target -eq 'map-capture') {
    $eventLog = Join-Path $game 'tww3_bai_map_capture_events.jsonl'
    $doneEvents = @('probe_done')
    $failEvents = @('probe_error')
    $outputs = @('tww3_bai_map_capture_events.jsonl', 'tww3_bai_map_capture_ready.xml', 'tww3_bai_map_capture_grid.csv')
} else {
    $eventLog = Join-Path $game 'tww3_bai_events.jsonl'
    $failEvents = @('error')
    if ($Target -eq 'arena') { $doneEvents = @('arena_complete') } else { $doneEvents = @('result') }
    $outputs = @()
}
$expectedDone = 1
if ($Target -eq 'duel') { $expectedDone = [int]$manifest.config.runs }
# Default wait: 4 minutes for loading plus the in-game deadline of every battle.
if ($TimeoutSeconds -le 0) {
    if ($manifest.config.deadline_s) { $TimeoutSeconds = 240 + [int]$manifest.config.deadline_s * $expectedDone }
    else { $TimeoutSeconds = 1200 }
}
Write-Output ("Launcher timeout: {0} s" -f $TimeoutSeconds)

if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 is already running; not interrupting it.' }
if ((Test-Path -LiteralPath $installedPack) -or (Test-Path -LiteralPath $modList)) {
    throw "Files of a previous run remain ($installedPack or $modList). Inspect them and remove manually."
}
foreach ($name in $outputs) {
    if (Test-Path -LiteralPath (Join-Path $game $name)) { throw "Previous output $name remains in the game folder; preserve it first." }
}

$run = Join-Path $buildDir ('runs\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $run | Out-Null
Copy-Item -LiteralPath (Join-Path $buildDir 'manifest.json') -Destination $run

# Required mods come from config/mod-dependencies.json and must be baked into
# the build: every battle runs with them, there is no vanilla fallback.
$pinned = Get-Content -LiteralPath (Join-Path $repo 'config\mod-dependencies.json') -Raw | ConvertFrom-Json
foreach ($mod in $pinned.mods) {
    $inBuild = @($manifest.dependencies | Where-Object { $_.pack -eq $mod.pack_name -and $_.sha256 -eq $mod.sha256 })
    if ($inBuild.Count -eq 0) { throw "Build lacks required mod $($mod.pack_name) ($($pinned.profile)). Rebuild with python -m tools.build $Target." }
}

# Dependencies (from the build manifest): the Workshop pack is copied into data/
# after a SHA-256 check and listed before our pack. A matching file already in
# data/ is kept; a different one is never overwritten. Only our copies are removed.
# All checks run before anything is written to the game folder.
$lines = @()
$toCopy = @()
foreach ($dependency in $manifest.dependencies) {
    $workshopPack = Join-Path (Join-Path $settings.workshop_dir $dependency.workshop_id) $dependency.pack
    if (-not (Test-Path -LiteralPath $workshopPack)) { throw "Dependency not downloaded: $workshopPack" }
    if ((Get-FileHash -LiteralPath $workshopPack).Hash.ToLowerInvariant() -ne $dependency.sha256) { throw "Dependency hash differs: $workshopPack" }
    $dataPack = Join-Path $game ('data\' + $dependency.pack)
    if (Test-Path -LiteralPath $dataPack) {
        if ((Get-FileHash -LiteralPath $dataPack).Hash.ToLowerInvariant() -ne $dependency.sha256) { throw "A different $($dependency.pack) is already in data/; not overwriting it." }
    } else {
        $toCopy += , @($workshopPack, $dataPack)
    }
    $lines += 'mod "' + $dependency.pack + '";'
}
$lines += 'mod "' + $manifest.pack + '";'

# Install dependencies, our pack and the private mod list.
$copiedDependencies = @()
foreach ($pair in $toCopy) {
    Copy-Item -LiteralPath $pair[0] -Destination $pair[1]
    $copiedDependencies += $pair[1]
}
Copy-Item -LiteralPath $sourcePack -Destination $installedPack
if ((Get-FileHash -LiteralPath $installedPack).Hash.ToLowerInvariant() -ne $manifest.pack_sha256) { throw 'Installed pack hash mismatch' }
$modText = ($lines -join [Environment]::NewLine) + [Environment]::NewLine
[IO.File]::WriteAllText($modList, $modText, [Text.UTF8Encoding]::new($false))

$offset = 0
if (Test-Path -LiteralPath $eventLog) { $offset = (Get-Item -LiteralPath $eventLog).Length }
$reader = New-JsonlReader -Offset $offset
$started = Get-Date
$arguments = 'game_startup_mode battle ' + $manifest.scenario + '; ' + (Split-Path -Leaf $modList) + ';'
$process = Start-Process -FilePath $exe -WorkingDirectory $game -ArgumentList $arguments -PassThru
@{pid = $process.Id; started_utc = $started.ToUniversalTime().ToString('o'); target = $Target; build = $manifest.build;
  pack_sha256 = $manifest.pack_sha256; log_offset = $offset; arguments = $arguments} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'launch.json') -Encoding utf8
Write-Output ("Started WH3 PID {0}; run folder {1}. Process start is not success: waiting for events." -f $process.Id, $run)

$status = 'timeout'
$done = 0
$runLog = Join-Path $run 'events.jsonl'
try {
    while (((Get-Date) - $started).TotalSeconds -lt $TimeoutSeconds) {
        if ($process.HasExited) { $status = 'process_exited'; break }
        $crash = Get-ChildItem -LiteralPath "$env:APPDATA\The Creative Assembly\Warhammer3\crash_report" -Filter '*.stack.txt' -ErrorAction SilentlyContinue |
            Where-Object { $_.LastWriteTime -gt $started } | Select-Object -First 1
        if ($crash) { Copy-Item -LiteralPath $crash.FullName -Destination $run; $status = 'crash_report'; break }
        # Fast path: runs can write tens of thousands of large rows. Lines are
        # appended in one call per poll and only key events are parsed.
        $lines = @(Read-JsonlLines -State $reader -Path $eventLog)
        if ($lines.Count -gt 0) {
            [IO.File]::AppendAllLines($runLog, [string[]]$lines, [Text.UTF8Encoding]::new($false))
            foreach ($line in $lines) {
                if ($line -notmatch '"event":"(ready|start|result|error|frame|grid_done|probe_done|probe_error|arena_complete|skipped|speed_restored)"') { continue }
                $event = $Matches[1]
                Write-Output $line
                if ($event -in $failEvents -or $event -eq 'skipped') { $status = 'lua_error' }
                if ($event -in $doneEvents) {
                    $done++
                    # A duel result other than 'completed' (timeout, deadline) ends the series.
                    if ($Target -eq 'duel' -and $line -notmatch '"status":"completed"') { $done = $expectedDone }
                }
            }
        }
        if ($status -eq 'lua_error') { break }
        if ($done -ge $expectedDone) { $status = 'completed'; break }
        Start-Sleep -Seconds 1
    }
    foreach ($name in $outputs) {
        $path = Join-Path $game $name
        if (Test-Path -LiteralPath $path) { Copy-Item -LiteralPath $path -Destination $run }
    }
} finally {
    $cleanup = @{status = $status; completed_events = $done; game_left_running = $false; private_files_removed = $false}
    $current = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
    $ours = $current -and $current.Path -eq $exe -and [Math]::Abs(($current.StartTime - $started).TotalSeconds) -lt 15
    if ($ours -and $KeepGameOpen) {
        $cleanup.game_left_running = $true
    } elseif ($ours) {
        Stop-Process -Id $current.Id
        for ($i = 0; $i -lt 40 -and (Get-Process -Id $current.Id -ErrorAction SilentlyContinue); $i++) { Start-Sleep -Milliseconds 500 }
    }
    if (-not $cleanup.game_left_running -and -not (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue)) {
        $packSame = (Test-Path -LiteralPath $installedPack) -and (Get-FileHash -LiteralPath $installedPack).Hash.ToLowerInvariant() -eq $manifest.pack_sha256
        $modsSame = (Test-Path -LiteralPath $modList) -and (Get-Content -LiteralPath $modList -Raw) -eq $modText
        if ($packSame -and $modsSame) {
            Remove-Item -LiteralPath $installedPack
            Remove-Item -LiteralPath $modList
            foreach ($copy in $copiedDependencies) { Remove-Item -LiteralPath $copy -ErrorAction SilentlyContinue }
            foreach ($name in $outputs) {
                $gamePath = Join-Path $game $name
                $savedPath = Join-Path $run $name
                if ((Test-Path -LiteralPath $gamePath) -and (Test-Path -LiteralPath $savedPath) -and
                    (Get-FileHash -LiteralPath $gamePath).Hash -eq (Get-FileHash -LiteralPath $savedPath).Hash) {
                    Remove-Item -LiteralPath $gamePath
                }
            }
            $cleanup.private_files_removed = $true
        }
    }
    $cleanup | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'status.json') -Encoding utf8
}
Write-Output ("Run finished: {0}. Results: {1}" -f $status, $run)
if ($status -ne 'completed') { exit 1 }
