# Lock-step checks of the launcher and watch.ps1 (bridge/docs/apps/bridge.md, "Шаг в шаг"); dot-sourced, no side effects.

# '' when the build's config.lockstep is a strict lock-step (bridge.build --lockstep: timeout_ms, first_timeout_ms, spin_ms,
# strikes, strict = true), else why not (a pack built before lockstep3 has no strict and runs on free after timeouts).
function Test-LockstepBuild($config) {
    $ls = $config.lockstep
    if (-not $ls) { return 'no lockstep in the manifest' }
    if (-not ($ls.strict -is [bool]) -or -not $ls.strict) { return 'lockstep without strict = true' }
    foreach ($key in @('timeout_ms', 'first_timeout_ms', 'spin_ms', 'strikes')) {
        if (-not $ls.$key) { return "lockstep without $key" }
    }
    return ''
}

# '' when the battle's result line has a normal end (status completed, or timeout = the model time limit) and ls_ok
# true (a JSON boolean), else why the battle is void (deadline = the wall clock, stalled, incomplete, no result).
function Get-LockstepVoid([string]$line) {
    if (-not $line) { return 'no result' }
    try { $r = $line | ConvertFrom-Json } catch { return 'the result is not JSON' }
    if ($r.status -notin @('completed', 'timeout')) { return ('result status ' + $r.status) }
    if (-not ($r.ls_ok -is [bool]) -or -not $r.ls_ok) { return ('ls_ok ' + $r.ls_ok) }
    return ''
}

# $true once the companion's readiness mark (bridge/companion/loop.py, written after its networks are loaded) with this
# run's token is in the file, $false after $seconds or when the companion's process exits.
function Wait-CompanionReady([string]$path, [string]$token, [int]$seconds, $process) {
    $end = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $end) {
        if ($process -and $process.HasExited) { return $false }
        if (Test-Path -LiteralPath $path) {
            try { if (((Get-Content -LiteralPath $path -Raw) | ConvertFrom-Json).token -eq $token) { return $true } } catch { }
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}
