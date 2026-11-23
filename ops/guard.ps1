# Is the game free for us? The marks of state/ (not in git); the same rule as ops/guard.py and ops/guard.sh.
#   state/HUMAN_PLAYING     the user plays: no game of ours starts, only the user's own battle (-Target human)
#   state/probe_game.lock   one runner holds the game; its text is the holder's token. A process whose
#                           $env:BAI_LOCK_HELD equals that text is the holder (its own launches pass). It holds every
#                           launcher slot too (its token is in probe_game.1.lock .. probe_game.3.lock).
#   state/probe_game.N.lock one runner holds launcher slot N = 1..3 (bridge/docs/launch/slots.md; $env:BAI_SLOT_LOCK_HELD_N).
#   Enter-SlotLock 2 'lane a' / Exit-SlotLock $held                     (bridge/launcher/launch.ps1 -Slot 2)
# The protocol (ops/guard.py's text): exclusive create is the only way in; the whole game takes 1, 2, 3, whole in this
# order; a dead holder (token '... pid <pid> id <8 hex>' with no such process, or an empty file older than 60 s) is
# removed only under '<lock>.break'.
# bridge/launcher/launch.ps1 checks the same rule itself before it touches the game folder.
# A runner (game/label/live.ps1):  $token = Enter-GameLock 'label live'; try { ... } finally { Exit-GameLock $token }
$GuardState = Join-Path (Split-Path -Parent $PSScriptRoot) 'state'

function Read-GuardLock([string]$Path) {
    try { $fs = [IO.File]::Open($Path, 'Open', 'Read', [IO.FileShare]'ReadWrite, Delete') }
    catch [IO.FileNotFoundException] { return $null }
    catch [IO.DirectoryNotFoundException] { return $null }
    catch { return '<unreadable>' }
    try { $ms = New-Object IO.MemoryStream; $fs.CopyTo($ms); $bytes = $ms.ToArray() } finally { $fs.Dispose() }
    return ([Text.UTF8Encoding]::new($false).GetString($bytes)).TrimStart([char]0xFEFF).Trim()
}

function Get-GameLockOwner([string]$State = $GuardState, [int]$Slot = 0) {
    $lock = Join-Path $State $(if ($Slot -gt 0) { 'probe_game.' + $Slot + '.lock' } else { 'probe_game.lock' })
    return Read-GuardLock $lock
}

# The one rule for a dead holder (ops/guard.py dead()).
function Test-GuardLockDead([string]$Path, $Token) {
    if ($null -eq $Token -or $Token -ceq '<unreadable>') { return $false }
    if ($Token -eq '') {
        try { return ((Get-Date) - [IO.File]::GetLastWriteTime($Path)).TotalSeconds -gt 60 } catch { return $false }
    }
    if ($Token -cnotmatch ' pid (\d+) id [0-9a-f]{8}$') { return $false }
    return -not (Get-Process -Id ([int]$Matches[1]) -ErrorAction SilentlyContinue)
}

function Remove-GuardPath([string]$Path) {
    for ($i = 0; $i -lt 40; $i++) {
        try { [IO.File]::Delete($Path); if (-not [IO.File]::Exists($Path)) { return $true } } catch { }
        Start-Sleep -Milliseconds 50
    }
    return $false
}

function Remove-DeadGuardLock([string]$Path) {
    $tok = Read-GuardLock $Path
    if ($null -eq $tok) { return $true }
    if (-not (Test-GuardLockDead $Path $tok)) { return $false }
    $brk = $Path + '.break'
    try { (([IO.File]::Open($brk, 'CreateNew', 'Write', 'None'))).Dispose() } catch { return $false }
    try { if ((Read-GuardLock $Path) -ceq $tok) { [void](Remove-GuardPath $Path) } } finally { [void](Remove-GuardPath $brk) }
    return ($null -eq (Read-GuardLock $Path))
}

function New-GuardLock([string]$Path, [string]$Token) {
    foreach ($attempt in 0, 1) {
        try { $f = [IO.File]::Open($Path, 'CreateNew', 'Write', 'Read') }
        catch { if ($attempt -eq 0 -and (Remove-DeadGuardLock $Path)) { continue }; return $false }
        try { $b = [Text.Encoding]::ASCII.GetBytes($Token + "`n"); $f.Write($b, 0, $b.Length); $f.Flush() }
        catch { $f.Dispose(); [void](Remove-GuardPath $Path); return $false }
        $f.Dispose()
        return $true
    }
    return $false
}

function Remove-OwnGuardLock([string]$Path, [string]$Token) {
    if ($Token -and (Read-GuardLock $Path) -ceq $Token) { [void](Remove-GuardPath $Path) }
}

function New-GuardToken([string]$Label) {
    # an ASCII token: the launcher reads the lock without an encoding (Windows PowerShell: the ANSI page)
    $Label = ($Label -replace '[^ -~]', '?').Trim()
    if (-not $Label) { $Label = 'lock' }
    return '{0} {1} pid {2} id {3}' -f $Label, (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'), $PID, [Guid]::NewGuid().ToString('N').Substring(0, 8)
}

function Get-GameBusy([string]$Target = '', [string]$State = $GuardState) {
    if ((Test-Path -LiteralPath (Join-Path $State 'HUMAN_PLAYING')) -and $Target -ne 'human') {
        return 'the user plays (state/HUMAN_PLAYING)'
    }
    $owner = Get-GameLockOwner $State
    if ($null -ne $owner -and (Test-GuardLockDead (Join-Path $State 'probe_game.lock') $owner)) { $owner = $null }   # a dead holder's lock is no one's
    if ($null -ne $owner -and -not ($owner -and $owner -ceq ([string]$env:BAI_LOCK_HELD).Trim())) {
        return ("the game is held (state/probe_game.lock: '{0}')" -f $owner)
    }
    if ($null -eq $owner) {
        foreach ($n in 1..3) {
            $s = Get-GameLockOwner $State $n
            if ($null -ne $s -and -not (Test-GuardLockDead (Join-Path $State ('probe_game.' + $n + '.lock')) $s)) {   # a dead holder's slot lock is no one's
                return ("launcher slots hold the game (state/probe_game.{0}.lock: '{1}')" -f $n, $s)
            }
        }
    }
    return ''
}

function Get-SlotBusy([int]$Slot, [string]$State = $GuardState) {
    if (Test-Path -LiteralPath (Join-Path $State 'HUMAN_PLAYING')) { return 'the user plays (state/HUMAN_PLAYING)' }
    $owner = Get-GameLockOwner $State
    if ($null -ne $owner -and (Test-GuardLockDead (Join-Path $State 'probe_game.lock') $owner)) { $owner = $null }   # a dead holder's lock is no one's
    if ($null -ne $owner) {
        if ($owner -and $owner -ceq ([string]$env:BAI_LOCK_HELD).Trim()) { return '' }      # the whole-game holder holds every slot
        return ("the game is held (state/probe_game.lock: '{0}')" -f $owner)
    }
    $s = Get-GameLockOwner $State $Slot
    $mine = ([string][Environment]::GetEnvironmentVariable('BAI_SLOT_LOCK_HELD_' + $Slot)).Trim()
    if ($null -ne $s -and -not ($s -and $s -ceq $mine) -and -not (Test-GuardLockDead (Join-Path $State ('probe_game.' + $Slot + '.lock')) $s)) { return ("slot {0} is held (state/probe_game.{0}.lock: '{1}')" -f $Slot, $s) }
    return ''
}

# Get-GameBusy / Get-SlotBusy for a taker: a dead holder's lock counts as free (the taker breaks it).
function Get-GuardBusyLive([string]$State, [string]$Target = '', [int]$Slot = 0) {
    if ((Test-Path -LiteralPath (Join-Path $State 'HUMAN_PLAYING')) -and ($Slot -gt 0 -or $Target -ne 'human')) { return 'the user plays (state/HUMAN_PLAYING)' }
    $whole = Join-Path $State 'probe_game.lock'
    $owner = Read-GuardLock $whole
    if ($null -ne $owner) {
        if ($owner -and $owner -ceq ([string]$env:BAI_LOCK_HELD).Trim()) { return '' }
        if (-not (Test-GuardLockDead $whole $owner)) { return ("the game is held (state/probe_game.lock: '{0}')" -f $owner) }
    }
    foreach ($n in $(if ($Slot -gt 0) { @($Slot) } else { 1..3 })) {
        $path = Join-Path $State ('probe_game.' + $n + '.lock')
        $s = Read-GuardLock $path
        if ($null -eq $s) { continue }
        if ($Slot -gt 0 -and $s -and $s -ceq ([string][Environment]::GetEnvironmentVariable('BAI_SLOT_LOCK_HELD_' + $n)).Trim()) { return '' }
        if (Test-GuardLockDead $path $s) { continue }
        if ($Slot -gt 0) { return ("slot {0} is held (state/probe_game.{0}.lock: '{1}')" -f $n, $s) }
        return ("launcher slots hold the game (state/probe_game.{0}.lock: '{1}')" -f $n, $s)
    }
    return ''
}

function Assert-GameFree([string]$Target = '', [string]$State = $GuardState) {
    $why = Get-GameBusy -Target $Target -State $State
    if ($why) { throw "Not starting the game: $why" }
}

function Enter-GameLock([string]$Label, [string]$Target = '', [string]$State = $GuardState) {
    $owner = Get-GameLockOwner $State
    $human = (Test-Path -LiteralPath (Join-Path $State 'HUMAN_PLAYING')) -and $Target -ne 'human'
    if ($owner -and $owner -ceq ([string]$env:BAI_LOCK_HELD).Trim() -and -not $human) { return $null }   # the caller's own lock: ride on it, free nothing at the end
    $why = Get-GuardBusyLive $State $Target
    if ($why) { throw "Not starting the game: $why" }
    New-Item -ItemType Directory -Force -Path $State | Out-Null
    $token = New-GuardToken $Label
    $taken = @()
    foreach ($name in 'probe_game.1.lock', 'probe_game.2.lock', 'probe_game.3.lock', 'probe_game.lock') {
        $path = Join-Path $State $name
        if (-not (New-GuardLock $path $token)) {
            $held = Read-GuardLock $path
            [array]::Reverse($taken); foreach ($t in $taken) { Remove-OwnGuardLock $t $token }
            if ($name -eq 'probe_game.lock') { throw ("Not starting the game: the game is held (state/probe_game.lock: '{0}')" -f $held) }
            throw ("Not starting the game: launcher slots hold the game (state/{0}: '{1}')" -f $name, $held)
        }
        $taken += $path
    }
    if ((Test-Path -LiteralPath (Join-Path $State 'HUMAN_PLAYING')) -and $Target -ne 'human') {
        [array]::Reverse($taken); foreach ($t in $taken) { Remove-OwnGuardLock $t $token }
        throw 'Not starting the game: the user plays (state/HUMAN_PLAYING)'
    }
    $env:BAI_LOCK_HELD = $token
    return $token
}

function Exit-GameLock([string]$Token, [string]$State = $GuardState) {
    if (-not $Token) { return }
    foreach ($name in 'probe_game.lock', 'probe_game.3.lock', 'probe_game.2.lock', 'probe_game.1.lock') {
        Remove-OwnGuardLock (Join-Path $State $name) $Token
    }
    if ($env:BAI_LOCK_HELD -ceq $Token) { Remove-Item Env:\BAI_LOCK_HELD }
}

function Enter-SlotLock([int]$Slot, [string]$Label, [string]$State = $GuardState) {
    if ($Slot -lt 1 -or $Slot -gt 3) { throw 'slot must be 1, 2 or 3' }
    $why = Get-GuardBusyLive $State '' $Slot
    if ($why) { throw "Not starting the game: $why" }
    $w = Get-GameLockOwner $State; $s = Get-GameLockOwner $State $Slot
    if (($w -and $w -ceq ([string]$env:BAI_LOCK_HELD).Trim()) -or ($s -and $s -ceq ([string][Environment]::GetEnvironmentVariable('BAI_SLOT_LOCK_HELD_' + $Slot)).Trim())) { return $null }   # the caller's: ride on it
    New-Item -ItemType Directory -Force -Path $State | Out-Null
    $token = New-GuardToken $Label
    $path = Join-Path $State ('probe_game.' + $Slot + '.lock')
    $whole = Join-Path $State 'probe_game.lock'
    if (-not (New-GuardLock $path $token)) { throw ("Not starting the game: slot {0} is held (state/probe_game.{0}.lock: '{1}')" -f $Slot, (Read-GuardLock $path)) }
    if ($null -ne (Read-GuardLock $whole) -and -not (Remove-DeadGuardLock $whole)) {        # an older runner's whole-game lock
        Remove-OwnGuardLock $path $token; throw ("Not starting the game: the game is held (state/probe_game.lock: '{0}')" -f (Read-GuardLock $whole))
    }
    if (Test-Path -LiteralPath (Join-Path $State 'HUMAN_PLAYING')) {
        Remove-OwnGuardLock $path $token; throw 'Not starting the game: the user plays (state/HUMAN_PLAYING)'
    }
    [Environment]::SetEnvironmentVariable('BAI_SLOT_LOCK_HELD_' + $Slot, $token)
    return @{slot = $Slot; token = $token}
}

function Exit-SlotLock($Held, [string]$State = $GuardState) {
    if (-not $Held) { return }
    Remove-OwnGuardLock (Join-Path $State ('probe_game.' + $Held.slot + '.lock')) $Held.token
    if ([Environment]::GetEnvironmentVariable('BAI_SLOT_LOCK_HELD_' + $Held.slot) -ceq $Held.token) {
        [Environment]::SetEnvironmentVariable('BAI_SLOT_LOCK_HELD_' + $Held.slot, $null)
    }
}
