param(
    [string]$GamePath = 'C:\Program Files (x86)\Steam\steamapps\common\Total War WARHAMMER III',
    [switch]$Watch,
    [string]$Python = 'python',
    [string]$NextPolicy,
    [int]$TimeoutSeconds = 600,
    [switch]$CloseWhenDone
)
$ErrorActionPreference='Stop'
throw 'Legacy vanilla entry point retired. Use tools/competition/run.py; True Sight is mandatory. Migrate this diagnostic with dependency verification before reuse.'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskGame = (Resolve-Path -LiteralPath $GamePath).Path
if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 is already running. Close it before a direct scenario launch.' }
if (-not (Test-Path -LiteralPath (Join-Path $taskGame 'data\tww3_bai_duel.pack'))) { throw 'Install the pack first.' }
$taskManifest = Get-Content -LiteralPath (Join-Path $taskRoot 'build\manifest.json') -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath (Join-Path $taskGame 'data\tww3_bai_duel.pack')).Hash.ToLowerInvariant() -ne $taskManifest.pack_sha256) { throw 'Installed pack differs from build. Install first.' }
$taskScenario = $taskManifest.config.scenario
if ($taskScenario -notin @('ranged_melee', 'triple_melee')) { throw 'Unsupported manifest scenario.' }
if ($taskScenario -eq 'triple_melee' -and $NextPolicy) { throw 'NextPolicy is for sequential battles, not parallel pairs.' }
if ($NextPolicy) { $NextPolicy = (Resolve-Path -LiteralPath $NextPolicy).Path }
if ($Watch) { Get-Command $Python -ErrorAction Stop | Out-Null }
$taskLog = Join-Path $taskGame 'tww3_bai_events.jsonl'
$taskOffset = if (Test-Path -LiteralPath $taskLog) { (Get-Item -LiteralPath $taskLog).Length } else { 0 }
$taskOutput = Join-Path $taskRoot ('reports\' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
$taskPending = Join-Path $taskGame 'tww3_bai_pending.txt'
if (Test-Path -LiteralPath $taskPending) { Copy-Item -LiteralPath $taskPending -Destination (Join-Path $taskOutput 'previous-pending.txt') }
[System.IO.File]::WriteAllText($taskPending, '', [System.Text.UTF8Encoding]::new($false))
# A private mod list keeps the user's campaign mod selection intact.
[System.IO.File]::WriteAllText((Join-Path $taskGame 'tww3_bai_mods.txt'), 'mod "tww3_bai_duel.pack";' + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
& (Join-Path $PSScriptRoot 'set-test-graphics.ps1')
# Syntax found in the installed executable's own command help, not a Linux benchmark flag.
$taskProcess = Start-Process -FilePath (Join-Path $taskGame 'Warhammer3.exe') -WorkingDirectory $taskGame -ArgumentList ('game_startup_mode battle script/battle/tww3_bai/' + $taskScenario + '.xml; tww3_bai_mods.txt;') -WindowStyle Hidden -PassThru
Write-Output ('Started WH3 process, PID ' + $taskProcess.Id + '. Scenario success requires fresh telemetry; process creation alone is not success.')
$taskLaunch = @{ pid = $taskProcess.Id; started_utc = (Get-Date).ToUniversalTime().ToString('o'); build = $taskManifest.config.build; pack_sha256 = $taskManifest.pack_sha256; log_offset = $taskOffset; requested_speed = $taskManifest.config.speed; scenario = $taskScenario }
$taskLaunch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOutput 'launch.json') -Encoding utf8
if ($Watch) {
    $taskArgs = @((Join-Path $PSScriptRoot 'watch-batch.py'), '--game', $taskGame, '--build', $taskManifest.config.build, '--runs', $taskManifest.config.runs, '--output', $taskOutput, '--offset', $taskOffset, '--timeout', $TimeoutSeconds)
    $taskArgs += @('--pid', $taskProcess.Id)
    if ($taskScenario -eq 'triple_melee') { $taskArgs += '--single-map' }
    if ($NextPolicy) { $taskArgs += @('--next-policy', $NextPolicy) }
    try {
        & $Python @taskArgs
        if ($LASTEXITCODE -ne 0) { throw ('Batch did not complete. See ' + $taskOutput) }
    } finally {
        if ($CloseWhenDone -and -not $taskProcess.HasExited) {
            # This handle belongs only to the dedicated process started above.
            $taskProcess.Kill()
            $taskProcess.WaitForExit(10000) | Out-Null
        }
    }
}
Write-Output ('Evidence directory: ' + $taskOutput)
