# The in-game gate (docs/en/launch/gate.md): N battles in the real game, our side commanded by a
# trained network (a checkpoint), the other by the game's AI at fair Normal difficulty, on armies
# from the random army generator (EVAL seeds, never used in training), our network attacking and
# defending in turn, in swapped pairs (the same seed twice, our network on either army). One battle per game launch (watch.ps1 -> launch.ps1: difficulty Normal, the
# user's preferences restored byte for byte, cleanup). Then build/nn-gate/<time>/summary.json and
# a table. Gate: 4 battles, at least 3 wins.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/gate.ps1 -Checkpoint build/nn-train/latest.pt
#   ... -Battles 4 -Speed 20        (the defaults)
#   ... -Battles 1 -Offset 3        (one battle: the plan's 4th, the largest armies)
param(
    [string]$Checkpoint = 'build/nn-train/latest.pt',
    [int]$Battles = 4,
    [int]$Offset = 0,
    [ValidateSet(1, 3, 10, 20)][int]$Speed = 20,
    [int]$DecideMs = 1000,
    [int]$TimeoutModelSeconds = 0,
    [switch]$Greedy,
    [int]$Retries = 1
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'

# The checkpoint as a path inside the repository: the companion's container sees the repo at /repo.
$full = [IO.Path]::GetFullPath((Join-Path $repo $Checkpoint))
if ([IO.Path]::IsPathRooted($Checkpoint)) { $full = [IO.Path]::GetFullPath($Checkpoint) }
if (-not (Test-Path -LiteralPath $full)) { throw "Checkpoint not found: $full" }
if (-not $full.StartsWith($repo + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "The checkpoint must be inside the repository: $full" }
$checkpointRel = $full.Substring($repo.Length + 1) -replace '\\', '/'
if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 is already running; not interrupting it.' }

$planArgs = @('-m', 'tools.nn.gate', 'plan', '--battles', $Battles, '--offset', $Offset, '--timeout', $TimeoutModelSeconds)
$plan = (& $python @planArgs) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Gate plan failed' }

$gate = Join-Path $repo ('build\nn-gate\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $gate | Out-Null
$runs = Join-Path $repo 'build\nn-arena\runs'
$done = @()
$failure = $null
Write-Output ("Gate: {0} battles, checkpoint {1}, x{2}, battle limit {3} s, need {4} wins; folder {5}" -f $Battles, $checkpointRel, $Speed, $plan.timeout_s, $plan.min_wins, $gate)

function Save-Battles {
    $doc = [ordered]@{checkpoint = $checkpointRel; greedy = [bool]$Greedy; speed = $Speed; timeout_s = $plan.timeout_s;
        planned = $Battles; offset = $Offset; min_wins = $plan.min_wins; battles = @($done)}
    [IO.File]::WriteAllText((Join-Path $gate 'battles.json'), ($doc | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
}

try {
    foreach ($b in $plan.battles) {
        $attempt = 0
        while ($true) {
            $attempt++
            Write-Output ("--- battle {0} (pair {8}{9}): seed {1}, our network {2}s, {3} v {4} units ({5} v {6}), attempt {7}" -f $b.battle, $b.seed, $b.role, $b.own_units, $b.enemy_units, $b.factions.own, $b.factions.enemy, $attempt, $b.pair, $(if ($b.swap) { ', armies swapped' } else { '' }))
            $swapArgs = @()
            if ($b.swap) { $swapArgs = @('--army-swap') }
            & $python -m tools.build nn-arena --own-ai net --army-seed $b.seed @swapArgs --own-role $b.role --speed $Speed `
                --timeout $plan.timeout_s --deadline $plan.deadline_s --decide-ms $DecideMs | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "Build failed for seed $($b.seed)" }
            $before = @(Get-ChildItem -LiteralPath $runs -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
            $watchArgs = @{NoBuild = $true; LingerSeconds = 0; Checkpoint = $checkpointRel}
            if ($Greedy) { $watchArgs.Greedy = $true }
            & (Join-Path $PSScriptRoot 'watch.ps1') @watchArgs
            $code = $LASTEXITCODE
            $new = @(Get-ChildItem -LiteralPath $runs -Directory -ErrorAction SilentlyContinue | Where-Object { $before -notcontains $_.Name } | Sort-Object Name)
            $run = $null
            if ($new.Count -gt 0) { $run = $new[-1].FullName }
            $hasResult = $run -and (Test-Path -LiteralPath (Join-Path $run 'events.jsonl')) -and
                (Select-String -LiteralPath (Join-Path $run 'events.jsonl') -Pattern '"event":"result"' -SimpleMatch -Quiet)
            if ($hasResult -or $attempt -gt $Retries) {
                $done += [ordered]@{battle = $b.battle; pair = $b.pair; swap = [bool]$b.swap; seed = $b.seed; role = $b.role; run = $run; attempts = $attempt; launcher_exit = $code}
                Save-Battles
                break
            }
            Write-Output ("Battle {0}: no result (launcher exit {1}); trying again" -f $b.battle, $code)
        }
    }
} catch {
    $failure = $_
} finally {
    Save-Battles
}
& $python -m tools.nn.gate summary $gate
$code = $LASTEXITCODE
if ($failure) { Write-Output ("Gate stopped: {0}" -f $failure); exit 2 }
exit $code
