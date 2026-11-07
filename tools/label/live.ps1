# Разметка вживую (docs/ru/training/label.md "Разметка вживую", tools/label/live.py): a battle the person plays as usual
# (his own orders, his speed and pause) on the user's 25.10 roster generator (seed 1000702579 swapped, armies v3: 20 v 20,
# Empire v Skaven, the game's AI on Normal), the enemy's cards in the bottom strip, the groups recorded in 15 s windows.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/label/live.ps1
#   ... -Seed 1000702579 -Timeout 1800 -Graphics low
param(
    [int]$Seed = 1000702579,
    [int]$TimeoutModelSeconds = 1800,
    [ValidateSet('', 'ultra', 'low', 'lowigpu', 'human2k')][string]$Graphics = '',
    [string]$Container = 'label-live'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$settings = Get-Content -LiteralPath (Join-Path $repo 'config\default.json') -Raw | ConvertFrom-Json
$localConfig = Join-Path $repo 'config\local.json'
if (Test-Path -LiteralPath $localConfig) {
    $local = Get-Content -LiteralPath $localConfig -Raw | ConvertFrom-Json
    foreach ($property in $local.PSObject.Properties) {
        $settings | Add-Member -NotePropertyName $property.Name -NotePropertyValue $property.Value -Force
    }
}
$game = (Resolve-Path -LiteralPath $settings.game_dir).Path
$lock = Join-Path $repo 'build\steps\probe_game.lock'
$exchange = @('tww3_bai_nn_state.json', 'tww3_bai_nn_state.json.tmp', 'tww3_bai_nn_overlay.txt', 'tww3_bai_nn_overlay.txt.tmp',
    'tww3_bai_live_cmd.txt', 'tww3_bai_nn_label_click.txt', 'tww3_bai_keys.txt')
function Clear-Exchange {
    if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { return }
    foreach ($n in $exchange) { $p = Join-Path $game $n; if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p } }
}

if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 is already running; not interrupting it.' }
if (Test-Path -LiteralPath $lock) { throw ("The game is held: build/steps/probe_game.lock says '{0}'" -f (Get-Content -LiteralPath $lock -Raw).Trim()) }
if (docker ps -a --filter "name=^$Container$" --format '{{.Names}}') { throw "Container $Container exists: docker stop $Container" }
& $python -m tools.build human --army-seed $Seed --army-swap --armies v3 --live-label --free-speed --timeout $TimeoutModelSeconds | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
Set-Content -LiteralPath $lock -Value ("label live, начало {0}" -f (Get-Date -Format 'HH:mm')) -Encoding utf8
Clear-Exchange
$runArgs = @('run', '-d', '--rm', '--init', '--name', $Container, '--cpus', '2', '-e', 'OMP_NUM_THREADS=2',
    '-v', "${repo}:/repo", '-v', "${game}:/game", '-w', '/repo', '-e', 'PYTHONPATH=/repo', '-e', 'PYTHONUNBUFFERED=1',
    'snake-ai-trainer', 'python', '-m', 'tools.label.live', '--game', '/game', '--idle-exit', '1800')
$keys = $null
try {
    # Ctrl / Shift for the enemy's pick (tools/label/keys.ps1: reads the two keys only, ends with the game)
    $keys = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -PassThru -ArgumentList @('-NoProfile',
        '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f (Join-Path $repo 'tools\label\keys.ps1')), '-Game', ('"{0}"' -f $game))
    & docker @runArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'The recorder did not start' }
    $graphicsArgs = @{}
    if ($Graphics) { $graphicsArgs['Graphics'] = $Graphics }
    & (Join-Path $repo 'tools\launcher\launch.ps1') -Target human -LingerSeconds 5 @graphicsArgs
    Start-Sleep -Seconds 3
    docker logs $Container 2>&1 | Select-Object -Last 3
} finally {
    $ErrorActionPreference = 'Continue'
    docker stop -t 10 $Container 2>$null | Out-Null
    if ($keys -and -not $keys.HasExited) { Stop-Process -Id $keys.Id -ErrorAction SilentlyContinue }
    Clear-Exchange
    if (Test-Path -LiteralPath $lock) { Remove-Item -LiteralPath $lock }
    # the recorder's folder: build/label/<date>_<time>_live (not matchups/ or the other folders there)
    $last = Get-ChildItem -LiteralPath (Join-Path $repo 'build\label') -Directory -Filter '*_live*' -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}_\d{6}_live' } | Sort-Object Name | Select-Object -Last 1
    if ($last) { Write-Output ("Запись: {0}" -f $last.FullName) }
}
