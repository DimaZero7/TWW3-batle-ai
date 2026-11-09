# Is the game free for us? The marks of state/ (not in git); the same rule as ops/guard.py and ops/guard.sh.
#   state/HUMAN_PLAYING     the user plays: no game of ours starts, only the user's own battle (-Target human)
#   state/probe_game.lock   one runner holds the game; its text is the holder's token. A process whose
#                           $env:BAI_LOCK_HELD equals that text is the holder (its own launches pass).
# bridge/launcher/launch.ps1 checks the same rule itself before it touches the game folder.
# A runner (game/label/live.ps1):  $token = Enter-GameLock 'label live'; try { ... } finally { Exit-GameLock $token }
$GuardState = Join-Path (Split-Path -Parent $PSScriptRoot) 'state'

function Get-GameLockOwner([string]$State = $GuardState) {
    $lock = Join-Path $State 'probe_game.lock'
    if (-not (Test-Path -LiteralPath $lock)) { return $null }
    return ([string](Get-Content -LiteralPath $lock -Raw -Encoding UTF8)).Trim()
}

function Get-GameBusy([string]$Target = '', [string]$State = $GuardState) {
    if ((Test-Path -LiteralPath (Join-Path $State 'HUMAN_PLAYING')) -and $Target -ne 'human') {
        return 'the user plays (state/HUMAN_PLAYING)'
    }
    $owner = Get-GameLockOwner $State
    if ($null -ne $owner -and $owner -ne ([string]$env:BAI_LOCK_HELD).Trim()) {
        return ("the game is held (state/probe_game.lock: '{0}')" -f $owner)
    }
    return ''
}

function Assert-GameFree([string]$Target = '', [string]$State = $GuardState) {
    $why = Get-GameBusy -Target $Target -State $State
    if ($why) { throw "Not starting the game: $why" }
}

function Enter-GameLock([string]$Label, [string]$Target = '', [string]$State = $GuardState) {
    Assert-GameFree -Target $Target -State $State
    $owner = Get-GameLockOwner $State
    if ($null -ne $owner) { return $null }          # the caller's own lock: ride on it, free nothing at the end
    New-Item -ItemType Directory -Force -Path $State | Out-Null
    # an ASCII token: the launcher reads the lock without an encoding (Windows PowerShell: the ANSI page)
    $Label = $Label -replace '[^ -~]', '?'
    $token = '{0} {1} pid {2}' -f $Label, (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'), $PID
    [IO.File]::WriteAllText((Join-Path $State 'probe_game.lock'), $token + "`n", [Text.UTF8Encoding]::new($false))
    $env:BAI_LOCK_HELD = $token
    return $token
}

function Exit-GameLock([string]$Token, [string]$State = $GuardState) {
    if (-not $Token) { return }
    $lock = Join-Path $State 'probe_game.lock'
    if ((Get-GameLockOwner $State) -eq $Token) { Remove-Item -LiteralPath $lock }
    if ($env:BAI_LOCK_HELD -eq $Token) { Remove-Item Env:\BAI_LOCK_HELD }
}
