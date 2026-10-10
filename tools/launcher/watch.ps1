# Watch the network command our side in a real battle (docs/en/launch/watch.md):
# builds the arena with our side under the network, starts the companion in the
# training container (snake-ai-trainer, the game folder mounted at /game), runs
# the battle through launch.ps1 (fair Normal difficulty, the user's preferences
# restored, cleanup) and stops the companion when the run ends.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/watch.ps1
#   ... -Speed 20 -LingerSeconds 0      (a quick check)
#   ... -Checkpoint build/nn-train/random.pt -Greedy
#   ... -Target lord-duel -NoBuild        (another build of entries.nn_arena under the network: tools/nn/lord_duel.py)
#   ... -EnemyAi ai_like                 (the enemy side under the simulator's script in the companion, not the
#                                         game's AI: build --enemy-ai; with -NoBuild the build's own setting is used)
#   ... -Skirmish off                    (the game's skirmish mode off for every unit the bridges command: build
#                                         --skirmish; with -NoBuild the build's own setting is used)
param(
    [ValidateSet('nn-arena', 'lord-duel')][string]$Target = 'nn-arena',
    [ValidateSet(1, 3, 10, 20)][int]$Speed = 1,
    [string]$Arena = 'arena',
    [int]$DecideMs = 1000,
    [int]$TimeoutModelSeconds = 600,
    [string]$Checkpoint = '',
    [ValidateSet('', 'game', 'ai_like', 'nearest', 'hold_shoot', 'hold')][string]$EnemyAi = '',
    [ValidateSet('', 'game', 'off')][string]$Skirmish = '',
    [switch]$Greedy,
    [int]$LingerSeconds = 30,
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$container = 'tww3-bai-companion'
$exchangeFiles = @('tww3_bai_nn_state.json', 'tww3_bai_nn_state.json.tmp', 'tww3_bai_nn_orders.txt', 'tww3_bai_nn_orders.txt.tmp',
    'tww3_bai_nn_state_enemy.json', 'tww3_bai_nn_state_enemy.json.tmp', 'tww3_bai_nn_orders_enemy.txt', 'tww3_bai_nn_orders_enemy.txt.tmp')

$settings = Get-Content -LiteralPath (Join-Path $repo 'config\default.json') -Raw | ConvertFrom-Json
$localConfig = Join-Path $repo 'config\local.json'
if (Test-Path -LiteralPath $localConfig) {
    $local = Get-Content -LiteralPath $localConfig -Raw | ConvertFrom-Json
    foreach ($property in $local.PSObject.Properties) {
        $settings | Add-Member -NotePropertyName $property.Name -NotePropertyValue $property.Value -Force
    }
}
$game = (Resolve-Path -LiteralPath $settings.game_dir).Path

if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 is already running; not interrupting it.' }
docker image inspect snake-ai-trainer --format '{{.Id}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker image snake-ai-trainer not found (is Docker Desktop running?)' }
$running = docker ps -a --filter "name=^$container$" --format '{{.Names}}'
if ($running) { throw "Container $container exists (a previous companion?). Remove it: docker rm -f $container" }

if (-not $NoBuild) {
    if ($Target -ne 'nn-arena') { throw "Build $Target yourself (python -m tools.build $Target ...) and pass -NoBuild" }
    $enemyArgs = @()
    if ($EnemyAi) { $enemyArgs = @('--enemy-ai', $EnemyAi) }
    if ($Skirmish) { $enemyArgs += @('--skirmish', $Skirmish) }
    & $python -m tools.build nn-arena --own-ai net --speed $Speed --arena $Arena --decide-ms $DecideMs --timeout $TimeoutModelSeconds @enemyArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
}
$manifest = Get-Content -LiteralPath (Join-Path $repo ('build\' + $Target + '\manifest.json')) -Raw | ConvertFrom-Json
if ($manifest.config.own_ai -ne 'net') { throw "build/$Target is not a net build; run without -NoBuild" }
$enemyScript = ''
if ($manifest.config.enemy_ai -eq 'companion') { $enemyScript = [string]$manifest.config.enemy_script }
if ($EnemyAi -and ($EnemyAi -ne 'game') -ne [bool]$enemyScript) { throw "build/$Target enemy ($($manifest.config.enemy_ai) $enemyScript) is not -EnemyAi $EnemyAi; build it again" }
if ($EnemyAi -and $EnemyAi -ne 'game' -and $enemyScript -ne $EnemyAi) { throw "build/$Target enemy script is $enemyScript, not $EnemyAi; build it again" }
if ($enemyScript) { Write-Output ("The enemy side: the simulator's script {0} in the companion" -f $enemyScript) }
Write-Output ("Build {0}: arena {1}, speed x{2}, a decision every {3} ms" -f $manifest.build, $manifest.config.arena, $manifest.config.speed, $manifest.config.decide_ms)

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logRel = "build/$Target/companion/$stamp.jsonl"
$dockerArgs = "run --rm --init --name $container -v `"${repo}:/repo`" -v `"${game}:/game`" -w /repo -e PYTHONPATH=/repo " +
    "-e PYTHONUNBUFFERED=1 snake-ai-trainer python -m tools.nn.companion --game /game --log /repo/$logRel"
if ($Checkpoint) { $dockerArgs += ' --checkpoint /repo/' + ($Checkpoint -replace '\\', '/') }
if ($Greedy) { $dockerArgs += ' --greedy' }
if ($enemyScript) { $dockerArgs += ' --enemy-script ' + $enemyScript }
$runs = Join-Path $repo ('build\' + $Target + '\runs')
$before = Get-ChildItem -LiteralPath $runs -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
$companion = Start-Process -FilePath docker -ArgumentList $dockerArgs -NoNewWindow -PassThru
$code = 1
$run = $null
try {
    for ($i = 0; $i -lt 30 -and -not (docker ps --filter "name=^$container$" --format '{{.Names}}'); $i++) { Start-Sleep -Seconds 1 }
    if ($companion.HasExited) { throw 'The companion did not start' }
    & (Join-Path $PSScriptRoot 'launch.ps1') -Target $Target -LingerSeconds $LingerSeconds
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
    $enemyLog = $log -replace '\.jsonl$', '_enemy.jsonl'
    if ($run -and (Test-Path -LiteralPath $enemyLog)) { Copy-Item -LiteralPath $enemyLog -Destination (Join-Path $run.FullName 'companion_enemy.jsonl') }
}
if ($run) { Write-Output ("Companion stopped. Run: {0}" -f $run.FullName) } else { Write-Output 'Companion stopped.' }
exit $code
