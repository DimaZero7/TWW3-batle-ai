# Разметка в игре (docs/ru/training/label.md "Разметка в игре"): the labelling server with the game folder mounted,
# the page, the game launched when the page asks (its «Запустить игру»: build/label/game/request.json), the battle,
# the cleanup. The person is the strategist (the deployment) and the operative (every 15 s: the game pauses and waits).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/label/game.ps1
#   ... -Graphics low -NoBrowser -StopAfterSeconds 20     (the scripted check: tools/label/game_check.py)
#   ... -Battles 2                                        (two battles one after another, the server stays)
param(
    [int]$Port = 8790,
    [string]$Container = 'label',
    [int]$Cpus = 4,
    [int]$TimeoutModelSeconds = 600,
    [ValidateSet('', 'ultra', 'low', 'lowigpu', 'human2k')][string]$Graphics = '',
    [switch]$NoBrowser,
    [int]$WaitRequestSeconds = 3600,
    [int]$StopAfterSeconds = -1,
    [int]$Battles = 1
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
$exchangeFiles = @('tww3_bai_nn_state.json', 'tww3_bai_nn_state.json.tmp', 'tww3_bai_nn_orders.txt', 'tww3_bai_nn_orders.txt.tmp',
    'tww3_bai_nn_deploy.json', 'tww3_bai_nn_deploy.json.tmp', 'tww3_bai_nn_plan.txt', 'tww3_bai_nn_plan.txt.tmp')
$gameRoot = Join-Path $repo 'build\label\game'
$request = Join-Path $gameRoot 'request.json'
$over = Join-Path $gameRoot 'over.json'
$lock = Join-Path $repo 'build\steps\probe_game.lock'

function Clear-Exchange {
    if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { return }
    foreach ($name in $exchangeFiles) {
        $path = Join-Path $game $name
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
    }
}

if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 is already running; not interrupting it.' }
if (Test-Path -LiteralPath $lock) { throw ("The game is held: build/steps/probe_game.lock says '{0}'" -f (Get-Content -LiteralPath $lock -Raw).Trim()) }
docker image inspect snake-ai-trainer --format '{{.Id}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker image snake-ai-trainer not found (is Docker Desktop running?)' }
if (docker ps -a --filter "name=^$Container$" --format '{{.Names}}') { throw "Container $Container exists (a labelling server already?). Stop it: docker stop $Container" }

New-Item -ItemType Directory -Force -Path $gameRoot | Out-Null
foreach ($f in @($request, $over)) { if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f } }
Set-Content -LiteralPath $lock -Value ("label game, начало {0}" -f (Get-Date -Format 'HH:mm')) -Encoding utf8
Clear-Exchange
$n = [Math]::Max(1, $Cpus)
$runArgs = @('run', '-d', '--rm', '--init', '--name', $Container, '--cpus', "$Cpus", '-e', "OMP_NUM_THREADS=$n",
    '-e', "MKL_NUM_THREADS=$n", '-p', "127.0.0.1:${Port}:8790", '-v', "${repo}:/repo", '-v', "${game}:/game", '-w', '/repo',
    '-e', 'PYTHONPATH=/repo', '-e', 'PYTHONUNBUFFERED=1', 'snake-ai-trainer', 'python', '-m', 'tools.label.server',
    '--port', '8790', '--game', '/game')
try {
    & docker @runArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'The labelling server did not start' }
    $up = $false
    for ($i = 0; $i -lt 90 -and -not $up; $i++) {
        try { Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/config" -TimeoutSec 2 | Out-Null; $up = $true } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $up) { throw 'The labelling server does not answer' }
    Write-Output ("Разметка в игре: http://localhost:{0} -> «Игра» -> «Запустить игру»" -f $Port)
    if (-not $NoBrowser) { Start-Process "http://localhost:$Port" }
    for ($k = 1; $k -le $Battles; $k++) {
        $t0 = Get-Date
        while (-not (Test-Path -LiteralPath $request)) {
            if (((Get-Date) - $t0).TotalSeconds -gt $WaitRequestSeconds) { throw 'No request from the page' }
            Start-Sleep -Milliseconds 500
        }
        Start-Sleep -Milliseconds 200
        $req = Get-Content -LiteralPath $request -Raw | ConvertFrom-Json
        Remove-Item -LiteralPath $request
        $role = if ($req.task -eq 'destroy') { 'attack' } else { 'defend' }
        Write-Output ("Battle {0}: task {1}, speed x{2}" -f $k, $req.task, $req.speed)
        & $python -m tools.build nn-arena --own-ai net --deploy strategist --label-game --speed $req.speed --timeout $TimeoutModelSeconds --own-role $role --label-own 'Человек (разметка)' | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
        Clear-Exchange
        $runs = Join-Path $repo 'build\nn-arena\runs'
        $before = Get-ChildItem -LiteralPath $runs -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
        $graphicsArgs = @{}
        if ($Graphics) { $graphicsArgs['Graphics'] = $Graphics }
        & (Join-Path $repo 'tools\launcher\launch.ps1') -Target nn-arena -LingerSeconds 5 @graphicsArgs
        $code = $LASTEXITCODE
        $run = Get-ChildItem -LiteralPath $runs -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
        if ($run -and $before -and $run.Name -eq $before.Name) { $run = $null }
        $runRel = if ($run) { 'build/nn-arena/runs/' + $run.Name } else { $null }
        @{ code = $code; run = $runRel; at = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') } | ConvertTo-Json | Set-Content -LiteralPath $over -Encoding utf8
        Write-Output ("Battle {0} over (code {1}), run {2}" -f $k, $code, $runRel)
        Clear-Exchange
    }
    if ($StopAfterSeconds -ge 0) { Start-Sleep -Seconds $StopAfterSeconds }
    else { Read-Host 'Бой окончен. Enter — закрыть сервер разметки' | Out-Null }
} finally {
    $ErrorActionPreference = 'Continue'
    docker logs $Container *> (Join-Path $gameRoot 'server.log')
    docker stop -t 5 $Container 2>$null | Out-Null
    Clear-Exchange
    if (Test-Path -LiteralPath $lock) { Remove-Item -LiteralPath $lock }
}
