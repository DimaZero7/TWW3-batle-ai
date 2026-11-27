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
#   ... -Chain                            (lock-step, the default with -Chain: the build's own - bridge.build turns it
#                                          on for --own-ai net; with -NoBuild a lock-step build must be a strict one -
#                                          bridge/launcher/lockstep.ps1 Test-LockstepBuild - and needs -Chain, or the run
#                                          is refused; an older build without it runs free. -StepLock: kept, and with
#                                          -NoBuild it refuses a build without lock-step. The game starts only once the
#                                          companion wrote its readiness mark for this run)
#   ... -Chain -NoStepLock                (lock-step off: the build with --no-lockstep; with -NoBuild a lock-step build
#                                          is refused. Without -Chain the build is always without lock-step: nobody
#                                          would answer the moves)
#   ... -PyThreads 2                      (the companion's torch threads, written to the run's companion.json)
#   ... -Chain -LordScript T              (the lord line's series T: our lord by bridge/companion/lord_script.py;
#                                          default '' - off, whatever BAI_LORD_SCRIPT the caller's shell has)
#   ... -Chain -NoDraw                    (no frames drawn, always x40 - -Speed with it is refused -, never lingers: the
#                                          build with --no-draw --speed 40; only under lock-step, never an exam -
#                                          bridge/config/memreader.json; with -NoBuild the build must be a no-draw one,
#                                          and a no-draw build without -NoDraw is refused)
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
    [switch]$NoBuild,
    # (declared last and named so that every older call form binds as before: -L, -T, the positional container)
    [switch]$StepLock,
    [int]$PyThreads = 2,
    [switch]$NoDraw,
    [switch]$NoStepLock,
    # the lord line's lane instrument (bridge/companion/lord_script.py): 'T' only by this parameter - an inherited
    # BAI_LORD_SCRIPT of the caller's shell never reaches the companion
    [ValidateSet('', 'T')][string]$LordScript = ''
)
$ErrorActionPreference = 'Stop'
$Target = 'nn-arena'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = @((Join-Path $repo '.venv\Scripts\python.exe'), (Join-Path $repo '..\TWW3-batle-ai\.venv\Scripts\python.exe')) |
    Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $python) { $python = 'python' }
$container = $Container
$exchangeFiles = @('tww3_bai_nn_state.json', 'tww3_bai_nn_state.json.tmp', 'tww3_bai_nn_orders.txt', 'tww3_bai_nn_orders.txt.tmp',
    'tww3_bai_nn_deploy.json', 'tww3_bai_nn_deploy.json.tmp', 'tww3_bai_nn_plan.txt', 'tww3_bai_nn_plan.txt.tmp',
    'tww3_bai_nn_ready.json', 'tww3_bai_nn_ready.json.tmp')
. (Join-Path $PSScriptRoot 'lockstep.ps1')
$ReadySeconds = 180    # the companion loads its networks before it answers: the game waits for its mark this long

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
if ($StepLock -and -not $Chain) { throw '-StepLock waits for the chain''s answers: give -Chain' }
if ($LordScript -and -not $Chain) { throw '-LordScript drives our lord inside the chain: give -Chain' }
if ($StepLock -and $NoStepLock) { throw '-StepLock (the default with -Chain) and -NoStepLock: give one' }
# lock-step: on whenever the chain answers the moves (bridge.build's default for --own-ai net), unless -NoStepLock
$lock = $Chain -and -not $NoStepLock
if ($NoDraw -and -not $lock) { throw '-NoDraw runs the model at x40 undrawn: only under lock-step (-Chain, no -NoStepLock)' }
if ($NoDraw -and $PSBoundParameters.ContainsKey('Speed')) { throw '-NoDraw is always x40: no -Speed with it' }
# a no-draw battle is never left open after its end (launch.ps1 forces it too)
if ($NoDraw) { $LingerSeconds = 0 }
# (the param block is not advanced: an unknown name lands in $args silently - refuse the lockstep2 names, never lose the mode)
if (@($args | Where-Object { "$_" -match '^-(Lockstep|Threads)' }).Count) { throw "-Lockstep / -Threads are -StepLock / -PyThreads: $args" }
if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 is already running; not interrupting it.' }
# the user plays / another run holds the game (ops/guard.ps1): refused before the build and the exchange files
. (Join-Path $repo 'ops\guard.ps1')
Assert-GameFree -Target $Target
docker image inspect snake-ai-trainer --format '{{.Id}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker image snake-ai-trainer not found (is Docker Desktop running?)' }
$running = docker ps -a --filter "name=^$container$" --format '{{.Names}}'
if ($running) { throw "Container $container exists (a previous companion?). Remove it: docker rm -f $container" }

if (-not $NoBuild) {
    $buildArgs = @()
    if ($Skirmish) { $buildArgs += @('--skirmish', $Skirmish) }
    if ($Strategist) { $buildArgs += @('--deploy', 'strategist') }
    if (-not $lock) { $buildArgs += @('--no-lockstep') }
    if ($NoDraw) { $buildArgs += @('--no-draw') }
    $buildSpeed = if ($NoDraw) { 40 } else { $Speed }
    & $python -m bridge.build nn-arena --own-ai net --speed $buildSpeed --decide-ms $DecideMs --timeout $TimeoutModelSeconds @buildArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
}
$manifest = Get-Content -LiteralPath (Join-Path $repo ('build\' + $Target + '\manifest.json')) -Raw | ConvertFrom-Json
if ($manifest.config.own_ai -ne 'net') { throw "build/$Target is not a net build: python -m bridge.build nn-arena --own-ai net ..." }
if ($manifest.config.lockstep -or $StepLock) {
    $why = Test-LockstepBuild $manifest.config
    if ($why) { throw "build/$Target is not a strict lock-step build ($why): python -m bridge.build nn-arena ... (lock-step is its default)" }
}
if ($manifest.config.lockstep -and -not $lock) { throw "build/$Target is a lock-step build: it needs -Chain without -NoStepLock (or rebuild with --no-lockstep)" }
if ($NoDraw -and -not $manifest.no_draw) { throw "build/$Target is not a no-draw build: python -m bridge.build nn-arena --no-draw ..." }
if (-not $NoDraw -and $manifest.no_draw) { throw "build/$Target is a no-draw build: give -NoDraw (or rebuild without --no-draw)" }
if ($Strategist -and $manifest.config.deploy -ne 'strategist') { throw "build/$Target has no strategist deployment: python -m bridge.build nn-arena --deploy strategist ..." }
if ($Strategist -and $Strategist -notin @('formula', 'default') -and -not (Test-Path -LiteralPath (Join-Path $repo $Strategist))) { throw "No strategist checkpoint $Strategist (a path inside the repository)" }
Write-Output ("Build {0}: arena {1}, speed x{2}, a decision every {3} ms, lock-step {4}" -f $manifest.build, $manifest.config.arena, $manifest.config.speed, $manifest.config.decide_ms, $(if ($manifest.config.lockstep) { 'on' } else { 'off' }))

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logRel = "build/$Target/companion/$stamp.jsonl"
$dockerArgs = "run --rm --init --name $container -v `"${repo}:/repo`" -v `"${game}:/game`" -w /repo -e PYTHONPATH=/repo " +
    "-e PYTHONUNBUFFERED=1 -e BAI_TICK_MS=$env:BAI_TICK_MS -e BAI_OP_SCRIPT=$env:BAI_OP_SCRIPT -e BAI_LORD_SCRIPT=$LordScript -e BAI_READY_TOKEN=$stamp snake-ai-trainer python -m bridge.companion --game /game --log /repo/$logRel --threads $PyThreads"
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
$ready = Join-Path $game 'tww3_bai_nn_ready.json'
Assert-GameFree -Target $Target       # once more after the build, right before the first file of the game folder
if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 is already running; not interrupting it.' }
if (-not (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue)) {    # (a dead battle's exchange files: gone)
    foreach ($name in $exchangeFiles) { $path = Join-Path $game $name; if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path } }
}
if (Test-Path -LiteralPath $ready) { Remove-Item -LiteralPath $ready }
$companion = Start-Process -FilePath docker -ArgumentList $dockerArgs -NoNewWindow -PassThru
$code = 1
$run = $null
try {
    for ($i = 0; $i -lt 30 -and -not (docker ps --filter "name=^$container$" --format '{{.Names}}'); $i++) { Start-Sleep -Seconds 1 }
    if ($companion.HasExited) { throw 'The companion did not start' }
    if (-not (Wait-CompanionReady $ready $stamp $ReadySeconds $companion)) { throw ("The companion wrote no readiness mark for this run within {0} s: the game is not started" -f $ReadySeconds) }
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
    if ($run) {
        @{threads = $PyThreads; lockstep = [bool]$manifest.config.lockstep; container = $container; chain = [bool]$Chain} |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run.FullName 'companion.json') -Encoding utf8
    }
}
if ($run) { Write-Output ("Companion stopped. Run: {0}" -f $run.FullName) } else { Write-Output 'Companion stopped.' }
exit $code
