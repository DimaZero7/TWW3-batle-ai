# Watch the network command our side in a real battle (docs/en/launch/watch.md):
# builds the arena with our side under the network, starts the companion in the
# training container (snake-ai-trainer, the game folder mounted at /game), runs
# the battle through launch.ps1 (fair Normal difficulty, the user's preferences
# restored, cleanup) and stops the companion when the run ends.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/watch.ps1
#   ... -Speed 20 -LingerSeconds 0      (a quick check)
#   ... -Checkpoint build/nn-train/random.pt -Greedy
param(
    [ValidateSet(1, 3, 10, 20)][int]$Speed = 1,
    [string]$Arena = 'arena',
    [int]$DecideMs = 1000,
    [int]$TimeoutModelSeconds = 600,
    [string]$Checkpoint = '',
    [switch]$Greedy,
    [int]$LingerSeconds = 30,
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$container = 'tww3-bai-companion'
$exchangeFiles = @('tww3_bai_nn_state.json', 'tww3_bai_nn_state.json.tmp', 'tww3_bai_nn_orders.txt', 'tww3_bai_nn_orders.txt.tmp')

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
    & $python -m tools.build nn-arena --own-ai net --speed $Speed --arena $Arena --decide-ms $DecideMs --timeout $TimeoutModelSeconds | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
}
$manifest = Get-Content -LiteralPath (Join-Path $repo 'build\nn-arena\manifest.json') -Raw | ConvertFrom-Json
if ($manifest.config.own_ai -ne 'net') { throw 'build/nn-arena is not a net build; run without -NoBuild' }
Write-Output ("Build {0}: arena {1}, speed x{2}, a decision every {3} ms" -f $manifest.build, $manifest.config.arena, $manifest.config.speed, $manifest.config.decide_ms)

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logRel = "build/nn-arena/companion/$stamp.jsonl"
$dockerArgs = "run --rm --init --name $container -v `"${repo}:/repo`" -v `"${game}:/game`" -w /repo -e PYTHONPATH=/repo " +
    "-e PYTHONUNBUFFERED=1 snake-ai-trainer python -m tools.nn.companion --game /game --log /repo/$logRel"
if ($Checkpoint) { $dockerArgs += ' --checkpoint /repo/' + ($Checkpoint -replace '\\', '/') }
if ($Greedy) { $dockerArgs += ' --greedy' }
$runs = Join-Path $repo 'build\nn-arena\runs'
$before = Get-ChildItem -LiteralPath $runs -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
$companion = Start-Process -FilePath docker -ArgumentList $dockerArgs -NoNewWindow -PassThru
$code = 1
$run = $null
try {
    for ($i = 0; $i -lt 30 -and -not (docker ps --filter "name=^$container$" --format '{{.Names}}'); $i++) { Start-Sleep -Seconds 1 }
    if ($companion.HasExited) { throw 'The companion did not start' }
    & (Join-Path $PSScriptRoot 'launch.ps1') -Target nn-arena -LingerSeconds $LingerSeconds
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
