# Launcher SLOTS (bridge/docs/launch/slots.md): up to three game processes at once for scripted work (probes, lanes,
# AI against AI), each in slot N = 1..3 with its own files; slot 0 is the ordinary single launch.
#   Get-SlotPlan ...             every name and path a slot uses (launch.ps1 -ShowPlan prints it)
#   Enter-LauncherMutex / Exit-LauncherMutex
#                                Global\tww3_bai_launcher: every launcher holds it while it checks, reserves, installs and
#                                starts, and while it cleans up (the single launch: its whole run); a dead holder's
#                                mutex is released by Windows
#   Add-SlotMember / Initialize-SlotShared / Set-SlotGame / Exit-SlotSession / Clear-DeadSlotSession / Get-ForeignGames
#                                the "session of slots", build/slot-session.json (written whole and atomically): the
#                                user's preferences (ONE backup build/preferences.slots.backup.txt, its SHA-256 in the
#                                session BEFORE the preferences change), the Workshop packs copied into data/ (shared
#                                marker) and the members: one per launch (run id, launcher PID + start, its game's PID +
#                                start + path)
#   Start-OnDesktop              CreateProcess on a private desktop of the user's window station; the process is kept
#                                by its handle (no PID reuse)
# Usage: . bridge/launcher/slots.ps1 (after installed.ps1 and preferences.ps1)

$SlotIds = @(1, 2, 3)
$SlotMutexName = 'Global\tww3_bai_launcher'
$script:SlotMutex = $null
$script:SlotMutexHeld = 0

# The slot's name of a game-folder file: tww3_bai_<x>.<ext> -> tww3_bai_<x>_s<N>.<ext> (bridge/pack/slot.py, the same
# rule as the Lua shim in a slot's pack); slot 0 - unchanged.
function Get-SlotFileName([string]$Name, [int]$Slot) {
    if ($Slot -eq 0 -or -not $Name.StartsWith('tww3_bai_') -or $Name -match '[/\\:]') { return $Name }
    $i = $Name.IndexOf('.')
    if ($i -lt 0) { return $Name + '_s' + $Slot }
    return $Name.Substring(0, $i) + '_s' + $Slot + $Name.Substring($i)
}

# Every name a launch in slot $Slot uses (the game folder, our build/ and state/ files, the desktop, the switch).
function Get-SlotPlan([string]$Repo, [string]$Game, [string]$Target, [string]$Pack, [int]$Slot) {
    $t = $Target -replace '-', '_'
    $build = Join-Path $Repo 'build'
    $packName = if ($Slot -eq 0) { $Pack } else { [IO.Path]::GetFileNameWithoutExtension($Pack) + '_s' + $Slot + '.pack' }
    $eventName = if ($Target -eq 'map-capture') { 'tww3_bai_map_capture_events.jsonl' } else { 'tww3_bai_events.jsonl' }
    $outputs = if ($Target -eq 'map-capture') { @('tww3_bai_map_capture_events.jsonl', 'tww3_bai_map_capture_ready.xml', 'tww3_bai_map_capture_grid.csv') } else { @() }
    # the exchange and other files our Lua writes in the game folder (bridge/lua: grep "'tww3_bai_"): for the record
    $luaFiles = @('tww3_bai_events.jsonl', 'tww3_bai_sequence.txt', 'tww3_bai_nn_state.json', 'tww3_bai_nn_state.json.tmp',
        'tww3_bai_nn_orders.txt', 'tww3_bai_nn_orders.txt.tmp', 'tww3_bai_nn_deploy.json', 'tww3_bai_nn_deploy.json.tmp',
        'tww3_bai_nn_plan.txt', 'tww3_bai_nn_plan.txt.tmp', 'tww3_bai_nn_overlay.txt', 'tww3_bai_nn_label_click.txt',
        'tww3_bai_live_cmd.txt', 'tww3_bai_keys.txt', 'tww3_bai_map_capture_events.jsonl', 'tww3_bai_map_capture_ready.xml',
        'tww3_bai_map_capture_grid.csv')
    $sfx = if ($Slot -eq 0) { '' } else { '.s' + $Slot }
    $slotBuild = if ($Slot -eq 0) { '' } else { Join-Path $build ('slots\s' + $Slot + '\' + $Target) }
    [ordered]@{
        slot = $Slot
        game = $Game
        pack = Join-Path $Game ('data\' + $packName)
        mod_list = Join-Path $Game (Get-SlotFileName ('tww3_bai_' + $t + '_mods.txt') $Slot)
        event_log = Join-Path $Game (Get-SlotFileName $eventName $Slot)
        outputs = @($outputs | ForEach-Object { Get-SlotFileName $_ $Slot })
        lua_files = @($luaFiles | ForEach-Object { Join-Path $Game (Get-SlotFileName $_ $Slot) })
        marker = Join-Path $build ('launcher-installed' + $sfx + '.json')
        slot_build = $slotBuild
        # a slot's runs live beside its build, never among build/<target>/runs (watch.ps1 takes the newest folder there)
        runs = if ($Slot -eq 0) { '' } else { Join-Path $slotBuild 'runs' }
        run_suffix = if ($Slot -eq 0) { '' } else { '-s' + $Slot }
        desktop = if ($Slot -eq 0) { '' } else { 'tww3_bai_s' + $Slot }
        lock = Join-Path $Repo $(if ($Slot -eq 0) { 'state\probe_game.lock' } else { 'state\probe_game.' + $Slot + '.lock' })
        switch = if ($Slot -eq 0) { '' } else { '-multirun' }
        prefs_backup = Join-Path $build $(if ($Slot -eq 0) { 'preferences.script.backup.txt' } else { 'preferences.slots.backup.txt' })
        slots_backup = Join-Path $build 'preferences.slots.backup.txt'
        # shared by the session of slots (one each)
        backup_tmp = Join-Path $build 'preferences.slots.backup.tmp'
        session = Join-Path $build 'slot-session.json'
        shared_marker = Join-Path $build 'launcher-installed.shared.json'
        # every slot's marker: a slot's own pack can stay open in ANOTHER slot's game (the game opens every pack in
        # data/ - game check 23.11), so the last slot out sweeps them all
        slot_markers = @($SlotIds | ForEach-Object { Join-Path $build ('launcher-installed.s' + $_ + '.json') })
        # what a crashed single launch left (a slot that comes first puts it right, as the next single launch would)
        single_marker = Join-Path $build 'launcher-installed.json'
        single_backup = Join-Path $build 'preferences.script.backup.txt'
    }
}

# --- the launcher mutex ----------------------------------------------------------------------------------------------
function Enter-LauncherMutex([int]$Seconds) {
    if (-not $script:SlotMutex) {
        try { $script:SlotMutex = New-Object System.Threading.Mutex($false, $SlotMutexName) }
        catch { throw ("Cannot open the launcher mutex {0} ({1}): another user or an elevated process made it; not starting." -f $SlotMutexName, $_.Exception.InnerException.Message) }
    }
    $got = $false
    try { $got = $script:SlotMutex.WaitOne([TimeSpan]::FromSeconds($Seconds)) }
    catch [System.Threading.AbandonedMutexException] { $got = $true }     # a dead holder: ours now (the state is journaled)
    if ($got) { $script:SlotMutexHeld++ }
    return $got
}

function Exit-LauncherMutex {
    while ($script:SlotMutexHeld -gt 0) { $script:SlotMutex.ReleaseMutex(); $script:SlotMutexHeld-- }
}

# --- small durable files -----------------------------------------------------------------------------------------------
function Get-BytesSha256([byte[]]$Bytes) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($Bytes)) -replace '-', '').ToLowerInvariant() } finally { $sha.Dispose() }
}

function Write-DurableBytes([string]$Path, [byte[]]$Bytes) {
    $f = [IO.File]::Open($Path, 'Create', 'Write', 'None')
    try { $f.Write($Bytes, 0, $Bytes.Length); $f.Flush($true) } finally { $f.Dispose() }
}

# The session file is replaced whole (temp file, flushed, then Replace / Move): a cut write leaves the old version.
function Write-SlotSession([string]$Path, $Session) {
    $dir = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    $tmp = $Path + '.tmp'
    Write-DurableBytes $tmp ([Text.UTF8Encoding]::new($false).GetBytes(($Session | ConvertTo-Json -Depth 6)))
    if ([IO.File]::Exists($Path)) { [IO.File]::Replace($tmp, $Path, [NullString]::Value) } else { [IO.File]::Move($tmp, $Path) }
}

# $null when there is no session; a file that does not parse or has another format stops everything (by hand).
function Read-SlotSession([string]$Path) {
    if (-not [IO.File]::Exists($Path)) { return $null }
    try { $s = [IO.File]::ReadAllText($Path, [Text.Encoding]::UTF8) | ConvertFrom-Json } catch { $s = $null }
    if (-not $s -or [int]$s.v -ne 2) { throw "The session of slots $Path cannot be read: nothing touched; inspect it (and build/preferences.slots.backup.txt) by hand." }
    return [pscustomobject]@{v = 2; phase = [string]$s.phase; graphics = [string]$s.graphics; prefs = [string]$s.prefs;
        prefs_sha256 = [string]$s.prefs_sha256; normal_sha256 = [string]$s.normal_sha256; user_difficulty = $s.user_difficulty; deps = @($s.deps | Where-Object { $_ });
        members = @($s.members | Where-Object { $_ })}
}

function New-SlotSession([string]$Graphics, [string]$Prefs) {
    return [pscustomobject]@{v = 2; phase = 'reserved'; graphics = $Graphics; prefs = $Prefs; prefs_sha256 = ''; normal_sha256 = ''; user_difficulty = $null; deps = @(); members = @()}
}

# --- processes ---------------------------------------------------------------------------------------------------------
function Test-MemberLive($m) {
    foreach ($try in 1, 2) {
        $p = Get-Process -Id ([int]$m.pid) -ErrorAction SilentlyContinue
        if (-not $p) { return $false }
        try { return ($p.StartTime.ToUniversalTime().Ticks -eq [int64]$m.start) } catch { Start-Sleep -Milliseconds 200 }
    }
    return $false      # its start time cannot be read twice: gone meanwhile, or another user's process (never our launcher)
}

function Test-MemberGame($m, $p) {
    try { return ([int]$m.game_pid -eq $p.Id -and [int64]$m.game_start -eq $p.StartTime.ToUniversalTime().Ticks -and [string]$m.game_path -eq $p.Path) }
    catch { return $false }
}

# Warhammer3 processes that are not the game of a live member (PID, start time and path); empty: every running game is ours.
function Get-ForeignGames($Session) {
    $live = @(if ($Session) { $Session.members | Where-Object { Test-MemberLive $_ } })
    return @(Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue | Where-Object { $p = $_; -not ($live | Where-Object { Test-MemberGame $_ $p }) })
}

# --- our files in the game folder ----------------------------------------------------------------------------------------
# Like Remove-InstalledFiles (installed.ps1) for the slot markers, but the marker is read as UTF-8 and only a file of
# ours in the game folder is removed: <game>\data\*.pack or <game>\tww3_bai_*, not a link; anything else stays (kept).
# A copy in progress, <game>\data\*.pack.tww3_bai_part (Copy-SlotFile), is ours by its name: removed whatever its bytes.
function Remove-SlotInstalledFiles([string]$Marker, [string]$Game) {
    $out = @{removed = @(); kept = @()}
    if (-not [IO.File]::Exists($Marker)) { return $out }
    try { $files = @(([IO.File]::ReadAllText($Marker, [Text.Encoding]::UTF8) | ConvertFrom-Json).files) } catch { $out.kept += ($Marker + ': cannot be read'); return $out }
    $data = Join-Path $Game 'data'
    $left = @()
    foreach ($e in $files) {
        $p = [string]$e.path
        $ok = $false
        try {
            $full = [IO.Path]::GetFullPath($p); $dir = Split-Path -Parent $full; $leaf = Split-Path -Leaf $full
            $ok = ($full -eq $p) -and (($dir -eq $data -and ($leaf -like '*.pack' -or $leaf -like '*.pack.tww3_bai_part')) -or ($dir -eq $Game -and $leaf -like 'tww3_bai_*'))
        } catch { $ok = $false }
        if (-not $ok) { $out.kept += ($p + ' (not a file of ours in the game folder)'); $left += $e; continue }
        if (-not (Test-Path -LiteralPath $p)) { $out.removed += $p; continue }
        try {
            if ((Get-Item -LiteralPath $p -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'a link' }
            if ($p -notlike '*.tww3_bai_part' -and (Get-FileSha256 $p) -ne [string]$e.sha256) { throw 'changed since' }
            Remove-Item -LiteralPath $p
            $out.removed += $p
        } catch { $out.kept += ($p + ': ' + $_.Exception.Message); $left += $e }
    }
    if ($left.Count -eq 0) { Remove-Item -LiteralPath $Marker }
    else { [IO.File]::WriteAllText($Marker, (@{files = @($left)} | ConvertTo-Json -Depth 4), [Text.UTF8Encoding]::new($false)) }
    return $out
}

# The marker is written (flushed) before the first copy; a pack's entry brings its copy-in-progress name along.
function Save-SlotMarker([string]$Marker, [object[]]$Entries) {
    if ([IO.File]::Exists($Marker)) { throw "$Marker is still there: an earlier run's files were not cleaned" }
    $files = @(foreach ($e in $Entries) {
        [pscustomobject]@{path = $e.path; sha256 = $e.sha256}
        if ($e.path -like '*.pack') { [pscustomobject]@{path = $e.path + '.tww3_bai_part'; sha256 = 'copy in progress'} }
    })
    Write-DurableBytes $Marker ([Text.UTF8Encoding]::new($false).GetBytes((@{files = $files} | ConvertTo-Json -Depth 4)))
}

# A pack into data/: copied under <name>.tww3_bai_part, checked, then renamed - a cut copy never looks like the pack.
function Copy-SlotFile([string]$Source, [string]$Dest, [string]$Sha256) {
    $part = $Dest + '.tww3_bai_part'
    if ([IO.File]::Exists($part)) { Remove-Item -LiteralPath $part }
    Copy-Item -LiteralPath $Source -Destination $part
    if ((Get-FileSha256 $part) -ne $Sha256) { Remove-Item -LiteralPath $part; throw "the copy of $Source differs from the pinned file ($Sha256)" }
    [IO.File]::Move($part, $Dest)
}

# --- the single launch's preferences backup ------------------------------------------------------------------------------
# build/preferences.script.backup.txt as the session of slots keeps its own: the user's bytes into a temp file (flushed,
# read back, compared), their SHA-256 into the journal build/preferences.script.backup.json, then the rename; the
# journal gets the SHA-256 of our Normal before it is written. The backup goes back only when its hash is the journal's
# (a backup without a journal - left by a launcher before this scheme - is taken as it is) and the user's file now is
# the backup itself, our Normal, or still carries our values (the game rewrites other lines of the file); anything
# else is the user's own change: both files kept, a refusal that says so.
function Get-SingleJournal([string]$Backup) { return [IO.Path]::ChangeExtension($Backup, '.json') }

function Write-SingleJournal([string]$Backup, [string]$BackupSha, [string]$NormalSha) {
    $j = Get-SingleJournal $Backup
    Write-DurableBytes ($j + '.tmp') ([Text.UTF8Encoding]::new($false).GetBytes((@{v = 1; backup_sha256 = $BackupSha; normal_sha256 = $NormalSha} | ConvertTo-Json)))
    if ([IO.File]::Exists($j)) { [IO.File]::Replace($j + '.tmp', $j, [NullString]::Value) } else { [IO.File]::Move($j + '.tmp', $j) }
}

# The user's bytes into the backup (returns them); nothing is changed when any step fails.
function Backup-SinglePreferences([string]$Prefs, [string]$Backup) {
    $bytes = [IO.File]::ReadAllBytes($Prefs)
    $h = Get-BytesSha256 $bytes
    $tmp = $Backup + '.tmp'
    Write-DurableBytes $tmp $bytes
    if ((Get-BytesSha256 ([IO.File]::ReadAllBytes($tmp))) -ne $h) { Remove-Item -LiteralPath $tmp; throw "The preferences backup $tmp did not read back as written: nothing changed, not starting." }
    Write-SingleJournal $Backup $h ''
    [IO.File]::Move($tmp, $Backup)
    return $bytes
}

# Our Normal into the user's file, its hash into the journal first.
function Set-SingleNormal([string]$Prefs, [string]$Backup, [byte[]]$Normal) {
    $j = [IO.File]::ReadAllText((Get-SingleJournal $Backup), [Text.Encoding]::UTF8) | ConvertFrom-Json
    Write-SingleJournal $Backup ([string]$j.backup_sha256) (Get-BytesSha256 $Normal)
    Write-DurableBytes $Prefs $Normal
}

# Returns @{restored; conflict} ('' or the refusal); $Values: the values our run wrote (Normal + its graphics).
function Restore-SinglePreferences([string]$Prefs, [string]$Backup, $Values) {
    $r = @{restored = $false; conflict = ''}
    $j = Get-SingleJournal $Backup
    if (-not [IO.File]::Exists($Backup)) {
        # a temp backup alone: the user's file was never changed
        foreach ($t in ($Backup + '.tmp'), $j, ($j + '.tmp')) { if ([IO.File]::Exists($t)) { Remove-Item -LiteralPath $t } }
        return $r
    }
    $bytes = [IO.File]::ReadAllBytes($Backup)
    $h = Get-BytesSha256 $bytes
    $normalSha = ''
    if ([IO.File]::Exists($j)) {
        $jo = $null
        try { $jo = [IO.File]::ReadAllText($j, [Text.Encoding]::UTF8) | ConvertFrom-Json } catch { }
        if (-not $jo -or [string]$jo.backup_sha256 -ne $h) {
            $r.conflict = ("conflict: {0} is not the backup its journal {1} wrote down; both kept, put it right by hand" -f $Backup, $j)
            return $r
        }
        $normalSha = [string]$jo.normal_sha256
    }
    if ([IO.File]::Exists($Prefs)) {
        $now = [IO.File]::ReadAllBytes($Prefs)
        $hn = Get-BytesSha256 $now
        if ($hn -ne $h -and -not ($normalSha -and $hn -eq $normalSha) -and -not (Test-PreferenceValues ([Text.Encoding]::GetEncoding(28591).GetString($now)) $Values)) {
            $r.conflict = ("conflict: {0} changed since our run wrote Normal (not our values any more - the user's own change?); both it and {1} kept, put it right by hand" -f $Prefs, $Backup)
            return $r
        }
    }
    Write-DurableBytes $Prefs $bytes
    if ((Get-BytesSha256 ([IO.File]::ReadAllBytes($Prefs))) -ne $h) { $r.conflict = "preferences not restored: $Prefs differs from the backup $Backup (both kept)"; return $r }
    Remove-Item -LiteralPath $Backup
    foreach ($t in $j, ($j + '.tmp')) { if ([IO.File]::Exists($t)) { Remove-Item -LiteralPath $t } }
    $r.restored = $true
    return $r
}

# --- the user's preferences --------------------------------------------------------------------------------------------
# The backup back into the user's file: only when its SHA-256 is the one the session wrote down before the change; the
# file written and flushed, read back and compared; the backup removed only then. A temp backup alone means the
# preferences were never changed (it is removed). Returns $true when restored, $false when there was nothing to do.
function Restore-SlotPreferences($Plan, [string]$Prefs, $Session) {
    if ([IO.File]::Exists($Plan.backup_tmp)) { Remove-Item -LiteralPath $Plan.backup_tmp }
    if (-not [IO.File]::Exists($Plan.slots_backup)) { return $false }
    $bytes = [IO.File]::ReadAllBytes($Plan.slots_backup)
    $h = Get-BytesSha256 $bytes
    if (-not $Session -or $Session.prefs_sha256 -ne $h) { throw ("conflict: {0} is not the backup the session wrote down; both files kept, put it right by hand" -f $Plan.slots_backup) }
    if ($Session.prefs -and $Session.prefs -ne $Prefs) { throw ("conflict: the session's preferences file is {0}, not {1}; both kept" -f $Session.prefs, $Prefs) }
    if ($Session.normal_sha256 -and [IO.File]::Exists($Prefs) -and $Session.phase -notin @('reserved', 'backup', 'normal_writing')) {
        $now = [IO.File]::ReadAllBytes($Prefs)
        $hn = Get-BytesSha256 $now
        if ($hn -ne $h -and $hn -ne $Session.normal_sha256 -and -not (Test-PreferenceValues ([Text.Encoding]::GetEncoding(28591).GetString($now)) (Get-SlotPrefValues $Session.graphics))) {
            throw ("conflict: {0} changed since our run wrote Normal (not our values any more - the user's own change?); both it and {1} kept, put it right by hand" -f $Prefs, $Plan.slots_backup)
        }
    }
    Write-DurableBytes $Prefs $bytes
    if ((Get-BytesSha256 ([IO.File]::ReadAllBytes($Prefs))) -ne $h) { throw "preferences not restored: $Prefs differs from the backup" }
    Remove-Item -LiteralPath $Plan.slots_backup
    return $true
}

# The values a session writes (Normal + the graphics preset) and whether a preferences text still has all of them (the
# game may rewrite other lines of the file; our values still in place = still our file).
function Get-SlotPrefValues([string]$Graphics) {
    $values = [ordered]@{battle_difficulty = 1}
    if ($Graphics) { foreach ($key in $GraphicsPresets[$Graphics].Keys) { $values[$key] = $GraphicsPresets[$Graphics][$key] } }
    return $values
}
function Test-PreferenceValues([string]$Text, $Values) {
    foreach ($key in $Values.Keys) { if ($Text -notmatch ('(?m)^' + [regex]::Escape($key) + ' ' + [regex]::Escape([string]$Values[$key]) + ';')) { return $false } }
    return $true
}

# What a session left when no member is live and no Warhammer3 runs (the caller checks both, under the mutex): the
# preferences first (restored before anything else can fail), then the shared and every slot's marker; with
# -AlsoSingle also what a crashed single launch left (its marker, its backup - as the next single launch does). The
# session file goes only when no backup is left. Returns @{restored; removed; kept; conflict}.
function Clear-DeadSlotSession($Plan, [string]$Prefs, [switch]$AlsoSingle) {
    $r = @{restored = $false; removed = @(); kept = @(); conflict = ''}
    $s = $null
    try { $s = Read-SlotSession $Plan.session } catch { $r.conflict = $_.Exception.Message; return $r }
    if ($AlsoSingle) {
        # as the next single launch would (Restore-SinglePreferences): not over the user's own later change
        try {
            $one = Restore-SinglePreferences $Prefs $Plan.single_backup @{battle_difficulty = 1}
            if ($one.conflict) { $r.conflict = $one.conflict; return $r }
            if ($one.restored) { $r.restored = $true }
        } catch { $r.conflict = 'the single launch''s backup: ' + $_.Exception.Message; return $r }
    }
    try { if (Restore-SlotPreferences $Plan $Prefs $s) { $r.restored = $true } } catch { $r.conflict = $_.Exception.Message }
    $markers = @($Plan.shared_marker) + @($Plan.slot_markers)
    foreach ($m in $markers) {
        $g = Remove-SlotInstalledFiles $m $Plan.game
        $r.removed += @($g.removed); $r.kept += @($g.kept)
    }
    if ($AlsoSingle) {
        try { $g = Remove-InstalledFiles -Marker $Plan.single_marker; $r.removed += @($g.removed); $r.kept += @($g.kept) }
        catch { $r.kept += ($Plan.single_marker + ': ' + $_.Exception.Message) }
    }
    if (-not $r.conflict -and -not [IO.File]::Exists($Plan.slots_backup) -and [IO.File]::Exists($Plan.session)) { Remove-Item -LiteralPath $Plan.session }
    return $r
}

# --- the session of slots ------------------------------------------------------------------------------------------------
# A launch reserves its slot (under the mutex, before it creates any file): refused when its slot is live, when a
# Warhammer3 runs that is not a live member's game, when it would join with other graphics or other dependencies than
# the live members'. The first live member puts a dead session right first (refused when that leaves a conflict).
# $Deps: @(@{path; sha256}...) - the data/ packs this build needs. Returns @{run; first; user_difficulty}.
function Add-SlotMember($Plan, [int]$Slot, [string]$Prefs, [string]$Graphics, [object[]]$Deps,
                       [int]$LauncherPid = $PID, [int64]$LauncherStart = 0) {
    if ($LauncherStart -eq 0) { $LauncherStart = (Get-Process -Id $LauncherPid).StartTime.ToUniversalTime().Ticks }
    $s = Read-SlotSession $Plan.session
    $live = @(if ($s) { $s.members | Where-Object { Test-MemberLive $_ } })
    $same = @($live | Where-Object { [int]$_.slot -eq $Slot })
    if ($same.Count -gt 0) { throw ("Slot {0} is live (launcher PID {1}): one game per slot." -f $Slot, $same[0].pid) }
    $foreign = @(Get-ForeignGames $s)
    if ($foreign.Count -gt 0) { throw ('WH3 is already running and is not a live slot of ours (PID {0}); not interrupting it.' -f ($foreign.Id -join ', ')) }
    $first = $live.Count -eq 0
    $cleared = @()
    if ($first) {
        $c = Clear-DeadSlotSession $Plan $Prefs -AlsoSingle
        $cleared = @($c.removed)
        if ($c.conflict) { throw ('Not starting: ' + $c.conflict) }
        if ($c.kept.Count -gt 0) { throw ('Files an earlier session put into the game folder have changed since; not touching them: ' + ($c.kept -join ', ')) }
        $s = New-SlotSession $Graphics $Prefs
    } else {
        if ($s.graphics -ne $Graphics) { throw ("The live slots run with graphics '{0}', this launch asks '{1}': refused (one preferences file for all slots)." -f $s.graphics, $Graphics) }
        foreach ($d in $Deps) {
            $known = @($s.deps | Where-Object { $_.path -eq $d.path -and $_.sha256 -eq $d.sha256 })
            if ($known.Count -eq 0) { throw ("The live slots run without {0} ({1}): refused (other dependencies than the live slots')." -f $d.path, $d.sha256) }
            if (-not [IO.File]::Exists($d.path) -or (Get-FileSha256 $d.path) -ne $d.sha256) { throw ("{0} in data/ is not the pinned file: refused." -f $d.path) }
        }
    }
    $run = [Guid]::NewGuid().ToString('N')
    $s.members = @($live) + @([pscustomobject]@{slot = $Slot; run = $run; pid = $LauncherPid; start = $LauncherStart; phase = 'preparing'; game_pid = 0; game_start = 0; game_path = ''})
    Write-SlotSession $Plan.session $s
    return @{run = $run; first = $first; user_difficulty = $s.user_difficulty; cleared = $cleared}
}

# The first member's shared work (under the mutex): the Workshop packs (written down in the session and the shared
# marker before the copy, each copy checked), then the preferences: the user's bytes into a temp backup (flushed and
# read back), their SHA-256 into the session, the temp backup renamed into the backup, and only then Normal + the
# graphics written into the user's file. $Deps: @(@{source; path; sha256}...). Returns @{user_difficulty; warnings}.
function Initialize-SlotShared($Plan, [string]$Run, [string]$Prefs, [string]$Graphics, [object[]]$Deps) {
    $s = Read-SlotSession $Plan.session
    if (-not ($s.members | Where-Object { $_.run -eq $Run })) { throw "this launch ($Run) is not a member of the session" }
    $rec = @(); $copy = @(); $warnings = @()
    foreach ($d in $Deps) {
        if ([IO.File]::Exists($d.path)) {
            if ((Get-FileSha256 $d.path) -ne $d.sha256) { throw "A different $($d.path) is already in data/; not overwriting it." }
            $warnings += ("Warning: {0} is in data/ and not ours - left there after the run (remove it unless it is yours)." -f $d.path)
        } else { $copy += $d }
        $rec += [pscustomobject]@{path = $d.path; sha256 = $d.sha256}
    }
    $s.deps = $rec
    Write-SlotSession $Plan.session $s
    if ($copy.Count -gt 0) { Save-SlotMarker $Plan.shared_marker $copy }
    foreach ($d in $copy) { Copy-SlotFile $d.source $d.path $d.sha256 }
    $bytes = [IO.File]::ReadAllBytes($Prefs)
    $latin = [Text.Encoding]::GetEncoding(28591)
    $text = $latin.GetString($bytes)
    if ($text -notmatch 'battle_difficulty (\d+);') { throw "battle_difficulty not found in $Prefs" }
    $difficulty = [int]$Matches[1]
    $h = Get-BytesSha256 $bytes
    Write-DurableBytes $Plan.backup_tmp $bytes
    if ((Get-BytesSha256 ([IO.File]::ReadAllBytes($Plan.backup_tmp))) -ne $h) { throw "the temp backup of $Prefs does not read back" }
    $s.prefs = $Prefs; $s.prefs_sha256 = $h; $s.user_difficulty = $difficulty; $s.phase = 'backup'
    Write-SlotSession $Plan.session $s
    [IO.File]::Move($Plan.backup_tmp, $Plan.slots_backup)
    $normal = $latin.GetBytes((Set-PreferenceValues -Text $text -Values (Get-SlotPrefValues $Graphics)))
    $s.normal_sha256 = Get-BytesSha256 $normal; $s.phase = 'normal_writing'
    Write-SlotSession $Plan.session $s
    Write-DurableBytes $Prefs $normal
    $s.phase = 'normal'
    Write-SlotSession $Plan.session $s
    return @{user_difficulty = $difficulty; warnings = $warnings}
}

# The game of a member, by its PID, start time and path (under the mutex, right after the start).
function Set-SlotGame($Plan, [string]$Run, $Process) {
    $s = Read-SlotSession $Plan.session
    $m = @($s.members | Where-Object { $_.run -eq $Run })
    if ($m.Count -ne 1) { throw "this launch ($Run) is not a member of the session: its game is not registered" }
    $m[0].game_pid = $Process.Id; $m[0].game_start = $Process.StartTime.ToUniversalTime().Ticks; $m[0].game_path = $Process.Path
    $m[0].phase = 'running'
    Write-SlotSession $Plan.session $s
}

# A launch leaves (under the mutex): its member is dropped, dead members too. The LAST one out restores the user's
# preferences and sweeps the shared and every slot's files - but only when no Warhammer3 runs at all (a crashed
# member's game, an unknown one): then the backup and the session stay (deferred) for the next launch or cleanup.ps1.
# Returns @{last; deferred; preferences_restored; removed; kept; conflict}.
function Exit-SlotSession($Plan, [string]$Run, [string]$Prefs) {
    $out = @{last = $false; deferred = ''; preferences_restored = $false; removed = @(); kept = @(); conflict = ''}
    $s = Read-SlotSession $Plan.session
    if (-not $s) { $out.conflict = 'no session file'; return $out }
    $s.members = @($s.members | Where-Object { $_.run -ne $Run } | Where-Object { Test-MemberLive $_ })
    if ($s.members.Count -gt 0) { Write-SlotSession $Plan.session $s; return $out }
    $out.last = $true
    $games = @(Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue)
    if ($games.Count -gt 0) {
        $s.phase = 'deferred'
        Write-SlotSession $Plan.session $s
        $out.deferred = ('Warhammer3 still runs (PID {0}): the preferences backup is kept for the next launch or bridge/launcher/cleanup.ps1' -f ($games.Id -join ', '))
        return $out
    }
    $s.phase = 'closing'
    Write-SlotSession $Plan.session $s
    $c = Clear-DeadSlotSession $Plan $Prefs
    $out.preferences_restored = $c.restored; $out.removed = $c.removed; $out.kept = $c.kept; $out.conflict = $c.conflict
    return $out
}

# --- a private desktop ---------------------------------------------------------------------------------------------
# Starts $Exe with $Arguments in $Dir on the private desktop WinSta0\$Desktop (created if missing; no rights needed;
# the user's desktop is never switched). Returns @{process; desktop_handle}: the process object holds its own handle
# (taken while CreateProcess's handle was still open), so its PID cannot be reused while the launcher watches it;
# close the desktop handle after the game is gone.
function Start-OnDesktop([string]$Exe, [string]$Arguments, [string]$Dir, [string]$Desktop) {
    if ($Desktop -notmatch '^[A-Za-z0-9_]{1,32}$') { throw "bad desktop name $Desktop" }
    if (-not ('SlotDesk' -as [type])) {
        Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices; using System.Text;
public static class SlotDesk {
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)] public struct SI { public int cb; public string r; public string desk; public string title;
    public int x, y, w, h, cx, cy, fill, flags; public short show, r2; public IntPtr r3, i, o, e; }
  [StructLayout(LayoutKind.Sequential)] public struct PI { public IntPtr hp, ht; public int pid, tid; }
  [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)] public static extern IntPtr CreateDesktopW(string n, IntPtr d, IntPtr m, int f, uint a, IntPtr s);
  [DllImport("user32.dll", SetLastError = true)] public static extern bool CloseDesktop(IntPtr h);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] public static extern bool CreateProcessW(string app, StringBuilder cmd, IntPtr pa, IntPtr ta, bool inh, uint fl, IntPtr env, string dir, ref SI si, out PI pi);
  [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
}
'@
    }
    $h = [SlotDesk]::CreateDesktopW($Desktop, [IntPtr]::Zero, [IntPtr]::Zero, 0, 0x10000000, [IntPtr]::Zero)
    if ($h -eq [IntPtr]::Zero) { throw ("CreateDesktop failed " + [Runtime.InteropServices.Marshal]::GetLastWin32Error()) }
    $si = New-Object SlotDesk+SI; $si.cb = [Runtime.InteropServices.Marshal]::SizeOf($si); $si.desk = 'WinSta0\' + $Desktop
    $pi = New-Object SlotDesk+PI
    $cmd = New-Object System.Text.StringBuilder ('"' + $Exe + '" ' + $Arguments)
    if (-not [SlotDesk]::CreateProcessW($Exe, $cmd, [IntPtr]::Zero, [IntPtr]::Zero, $false, 0, [IntPtr]::Zero, $Dir, [ref]$si, [ref]$pi)) {
        $e = [Runtime.InteropServices.Marshal]::GetLastWin32Error(); [void][SlotDesk]::CloseDesktop($h); throw ("CreateProcess failed " + $e) }
    try {
        $p = [Diagnostics.Process]::GetProcessById($pi.pid)
        $null = $p.Handle; $null = $p.StartTime                   # our own handle, while CreateProcess's is still open
    } finally { [void][SlotDesk]::CloseHandle($pi.ht); [void][SlotDesk]::CloseHandle($pi.hp) }
    return @{process = $p; desktop_handle = $h}
}
