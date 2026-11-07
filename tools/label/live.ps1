# Разметка вживую (docs/ru/training/label.md "Разметка вживую", tools/label/live.py): a battle the person plays as usual
# (his own orders, his speed and pause): the exact roster of his battle of 25.10 19:21 (build/human/runs/20261025-192124:
# Empire v Skaven, 20 v 20, the game's AI on Normal; tools/nn/army_from_run.py) on the captured Kislev plains
# (build/maps/ksl_plains_01 and its deployment zones; tools/nn/arena_map.py), 'ultra' full screen on the RTX, his mod of
# the 3D models at any distance (Workshop 3812267837, !!!wh3_cinematic_lod_remover.pack) for this battle only. The
# game's deployment, the enemy's cards in the bottom strip, the groups recorded in 15 s windows.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/label/live.ps1
#   ... -Run build/human/runs/20261025-192124 -Map ksl_plains_01 -Graphics ultrafull -ExtraMods 3812267837
#   ... -Run '' -Seed 1000702579   (the generator's armies instead of a run's)
#   ... -Match 1   (a prepared matchup of tools/label/live_matches.json: the generator's seed, the sides, the map)
param(
    [int]$Match = 0,
    [string]$Run = 'build/human/runs/20261025-192124',
    [int]$Seed = 1000702579,
    [string]$Map = 'ksl_plains_01',
    [int]$TimeoutModelSeconds = 1800,
    [ValidateSet('', 'ultra', 'ultrafull', 'low', 'lowigpu', 'human2k')][string]$Graphics = 'ultrafull',
    [string[]]$ExtraMods = @('3812267837'),
    # a key of the run's roster replaced on the same places (user 07.11: the Nuln Ironsides are no snipers - the
    # Hochland long rifles instead)
    [string[]]$Swap = @('wh3_dlc25_emp_inf_nuln_ironsides=wh3_dlc25_emp_inf_hochland_long_rifles'),
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
$buildArgs = @('-m', 'tools.build', 'human', '--live-label', '--free-speed', '--timeout', $TimeoutModelSeconds)
if ($Match -gt 0) {
    # a prepared matchup (tools/label/live_matches.json): the v3 generator's seed, its sides as they are or swapped, the map
    $table = Get-Content -LiteralPath (Join-Path $repo 'tools\label\live_matches.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $m = $table.("$Match")
    if (-not $m) { throw "No matchup $Match in tools/label/live_matches.json" }
    Write-Output ("Матчап {0}: {1}, карта {2}" -f $Match, $m.name, $m.map)
    $Map = $m.map
    $buildArgs += @('--army-seed', $m.seed, '--armies', 'v3')
    if ($m.swap) { $buildArgs += '--army-swap' }
} elseif ($Run) {
    $buildArgs += @('--army-from-run', (Join-Path $repo $Run))
    foreach ($x in $Swap) { if ($x) { $buildArgs += @('--army-replace', $x) } }
} else { $buildArgs += @('--army-seed', $Seed, '--army-swap', '--armies', 'v3') }
if ($Map) { $buildArgs += @('--map', $Map) }
foreach ($m in $ExtraMods) { if ($m) { $buildArgs += @('--extra-mod', $m) } }
& $python @buildArgs | Out-Null
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
    # (the recorder ends with the battle and docker removes its container: no logs to show then - 07.11)
    if (docker ps -a --filter "name=^$Container$" --format '{{.Names}}') {
        try { docker logs $Container 2>&1 | Select-Object -Last 3 } catch { }
    }
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
