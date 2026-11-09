# The chain's battle in the game (bridge/docs/launch/watch.md): builds the arena with our side under the network (or
# takes a build made before: -NoBuild), starts the companion in the training container (snake-ai-trainer, the game
# folder mounted at /game), runs the battle through launch.ps1 (fair Normal difficulty, the user's preferences
# restored, the locks state/HUMAN_PLAYING and state/probe_game.lock, cleanup) and stops the companion when the run ends.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File bridge/launcher/watch.ps1 -NoBuild -Strategist default -StrategistTask hold -Chain
#       (a build with: python -m bridge.build nn-arena --own-ai net --deploy strategist ... : the strategist places our
#        side, then the chain operative -> tacticians -> unit network commands it, companion --chain)
#   ... -ChainOp formula|net|<ckpt> -ChainTactic formula|net|<ckpt> -ChainGreedy   (the operative: formula by default;
#                                          no network of 40 groups yet)
#   ... -NoBuild -Strategist formula      (only the deployment answered: no -Chain)
#   ... -Speed 20 -LingerSeconds 0        (a quick check; without -NoBuild: the default arena, our side under the chain)
#   ... -Skirmish game                    (the game's skirmish mode as the game sets it; default off for our units, as
#                                          the simulator's network units: build --skirmish; with -NoBuild the build's)
#   ... -Graphics low                     (a graphics preset for the run, launch.ps1 -Graphics; default: BAI_GRAPHICS)
#   ... -Container agent-deploy           (the companion container's name; default tww3-bai-companion)
param(
    [ValidateSet(1, 3, 10, 20)][int]$Speed = 1,
    [int]$DecideMs = 1000,
    [int]$TimeoutModelSeconds = 600,
    [ValidateSet('', 'game', 'off')][string]$Skirmish = '',
    [int]$LingerSeconds = 30,
    [ValidateSet('', 'ultra', 'ultrafull', 'low')][string]$Graphics = '',
    [string]$Strategist = '',
    [ValidateSet('hold', 'destroy')][string]$StrategistTask = 'hold',
    [switch]$Chain,
    [string]$ChainOp = 'formula',
    [string]$ChainTactic = 'formula',
    [switch]$ChainGreedy,
    [string]$Container = 'tww3-bai-companion',
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'
$Target = 'nn-arena'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = @((Join-Path $repo '.venv\Scripts\python.exe'), (Join-Path $repo '..\TWW3-batle-ai\.venv\Scripts\python.exe')) |
    Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $python) { $python = 'python' }
$container = $Container
$exchangeFiles = @('tww3_bai_nn_state.json', 'tww3_bai_nn_state.json.tmp', 'tww3_bai_nn_orders.txt', 'tww3_bai_nn_orders.txt.tmp',
    'tww3_bai_nn_deploy.json', 'tww3_bai_nn_deploy.json.tmp', 'tww3_bai_nn_plan.txt', 'tww3_bai_nn_plan.txt.tmp')

$settings = Get-Content -LiteralPath (Join-Path $repo 'bridge\config\default.json') -Raw | ConvertFrom-Json
$localConfig = Join-Path $repo 'bridge\config\local.json'
if (Test-Path -LiteralPath $localConfig) {
    $local = Get-Content -LiteralPath $localConfig -Raw | ConvertFrom-Json
    foreach ($property in $local.PSObject.Properties) {
        $settings | Add-Member -NotePropertyName $property.Name -NotePropertyValue $property.Value -Force
    }
}
$game = (Resolve-Path -LiteralPath $settings.game_dir).Path

if (-not ($Chain -or $Strategist)) { throw 'Nobody to answer: give -Chain and / or -Strategist' }
if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 is already running; not interrupting it.' }
docker image inspect snake-ai-trainer --format '{{.Id}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker image snake-ai-trainer not found (is Docker Desktop running?)' }
$running = docker ps -a --filter "name=^$container$" --format '{{.Names}}'
if ($running) { throw "Container $container exists (a previous companion?). Remove it: docker rm -f $container" }

if (-not $NoBuild) {
    $buildArgs = @()
    if ($Skirmish) { $buildArgs += @('--skirmish', $Skirmish) }
    if ($Strategist) { $buildArgs += @('--deploy', 'strategist') }
    & $python -m bridge.build nn-arena --own-ai net --speed $Speed --decide-ms $DecideMs --timeout $TimeoutModelSeconds @buildArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
}
$manifest = Get-Content -LiteralPath (Join-Path $repo ('build\' + $Target + '\manifest.json')) -Raw | ConvertFrom-Json
if ($manifest.config.own_ai -ne 'net') { throw "build/$Target is not a net build: python -m bridge.build nn-arena --own-ai net ..." }
if ($Strategist -and $manifest.config.deploy -ne 'strategist') { throw "build/$Target has no strategist deployment: python -m bridge.build nn-arena --deploy strategist ..." }
if ($Strategist -and $Strategist -notin @('formula', 'default') -and -not (Test-Path -LiteralPath (Join-Path $repo $Strategist))) { throw "No strategist checkpoint $Strategist (a path inside the repository)" }
Write-Output ("Build {0}: arena {1}, speed x{2}, a decision every {3} ms" -f $manifest.build, $manifest.config.arena, $manifest.config.speed, $manifest.config.decide_ms)

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logRel = "build/$Target/companion/$stamp.jsonl"
$dockerArgs = "run --rm --init --name $container -v `"${repo}:/repo`" -v `"${game}:/game`" -w /repo -e PYTHONPATH=/repo " +
    "-e PYTHONUNBUFFERED=1 snake-ai-trainer python -m bridge.companion --game /game --log /repo/$logRel"
if ($Chain) {
    $dockerArgs += ' --chain --chain-op ' + ($ChainOp -replace '\\', '/') + ' --chain-tactic ' + ($ChainTactic -replace '\\', '/')
    if ($ChainGreedy) { $dockerArgs += ' --chain-greedy' }
    Write-Output ("Our side: the chain (operative {0}, tacticians {1}) in the companion" -f $ChainOp, $ChainTactic)
}
if ($Strategist) {
    $who = if ($Strategist -in @('formula', 'default')) { $Strategist } else { '/repo/' + ($Strategist -replace '\\', '/') }
    $dockerArgs += ' --strategist ' + $who + ' --strategist-task ' + $StrategistTask
    Write-Output ("The deployment: the strategist ({0}, task {1}) in the companion" -f $Strategist, $StrategistTask)
}
$runs = Join-Path $repo ('build\' + $Target + '\runs')
$before = Get-ChildItem -LiteralPath $runs -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
$companion = Start-Process -FilePath docker -ArgumentList $dockerArgs -NoNewWindow -PassThru
$code = 1
$run = $null
try {
    for ($i = 0; $i -lt 30 -and -not (docker ps --filter "name=^$container$" --format '{{.Names}}'); $i++) { Start-Sleep -Seconds 1 }
    if ($companion.HasExited) { throw 'The companion did not start' }
    $graphicsArgs = @{}
    if ($Graphics) { $graphicsArgs['Graphics'] = $Graphics }
    & (Join-Path $PSScriptRoot 'launch.ps1') -Target $Target -LingerSeconds $LingerSeconds @graphicsArgs
    $code = $LASTEXITCODE
} finally {
    $ErrorActionPreference = 'Continue'
    docker stop -t 5 $container 2>$null | Out-Null
    if (-not $companion.WaitForExit(15000)) { $companion.Kill() }
    if (-not (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue)) {
        foreach ($name in $exchangeFiles) {
            $path = Join-Path $game $name
            if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
        }
    }
    $run = Get-ChildItem -LiteralPath $runs -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
    if ($run -and $before -and $run.Name -eq $before.Name) { $run = $null }
    $log = Join-Path $repo ($logRel -replace '/', '\')
    if ($run -and (Test-Path -LiteralPath $log)) { Copy-Item -LiteralPath $log -Destination (Join-Path $run.FullName 'companion.jsonl') }
}
if ($run) { Write-Output ("Companion stopped. Run: {0}" -f $run.FullName) } else { Write-Output 'Companion stopped.' }
exit $code
