# Removes what a launcher run put into the game folder and left there (a crash, a killed run):
# the files of build/launcher-installed.json whose hash is unchanged (installed.ps1), and what a dead session of
# launcher slots left (slots.ps1: the user's preferences back from build/preferences.slots.backup.txt, then the slot
# and shared markers' files). Every launch does this first by itself; run it by hand before playing the game normally
# (multiplayer) when a run was interrupted.
#   powershell -NoProfile -ExecutionPolicy Bypass -File bridge/launcher/cleanup.ps1
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $PSScriptRoot 'installed.ps1')
. (Join-Path $PSScriptRoot 'slots.ps1')
if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 is running; close it first.' }
$gone = Remove-InstalledFiles -Marker (Join-Path $repo 'build\launcher-installed.json')
$slotsLeft = @((Join-Path $repo 'build\slot-session.json'), (Join-Path $repo 'build\preferences.slots.backup.txt'),
    (Join-Path $repo 'build\preferences.slots.backup.tmp'), (Join-Path $repo 'build\launcher-installed.shared.json')) +
    @($SlotIds | ForEach-Object { Join-Path $repo ('build\launcher-installed.s' + $_ + '.json') }) | Where-Object { [IO.File]::Exists($_) }
$dead = @{restored = $false; removed = @(); kept = @(); conflict = ''}
if (@($slotsLeft).Count -gt 0) {
    $settings = Get-Content -LiteralPath (Join-Path $repo 'bridge\config\default.json') -Raw | ConvertFrom-Json
    $localConfig = Join-Path $repo 'bridge\config\local.json'
    if (Test-Path -LiteralPath $localConfig) {
        foreach ($property in (Get-Content -LiteralPath $localConfig -Raw | ConvertFrom-Json).PSObject.Properties) {
            $settings | Add-Member -NotePropertyName $property.Name -NotePropertyValue $property.Value -Force
        }
    }
    if (-not (Enter-LauncherMutex 60)) { throw "A launcher holds ${SlotMutexName}: run this again when it is done." }
    try {
        $s = Read-SlotSession (Join-Path $repo 'build\slot-session.json')
        if ($s -and @($s.members | Where-Object { Test-MemberLive $_ }).Count -gt 0) { throw 'Launcher slots are running: nothing touched.' }
        if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { throw 'WH3 is running; close it first.' }
        $plan = Get-SlotPlan -Repo $repo -Game (Resolve-Path -LiteralPath $settings.game_dir).Path -Target 'nn-arena' -Pack 'x.pack' -Slot 1
        $dead = Clear-DeadSlotSession $plan (Join-Path $env:APPDATA 'The Creative Assembly\Warhammer3\scripts\preferences.script.txt')
    } finally { Exit-LauncherMutex }
}
if ($gone.removed.Count -eq 0 -and $gone.kept.Count -eq 0 -and $dead.removed.Count -eq 0 -and $dead.kept.Count -eq 0 -and -not $dead.restored -and -not $dead.conflict) { Write-Output 'Nothing of ours in the game folder.' }
foreach ($p in @($gone.removed) + @($dead.removed)) { Write-Output ("removed: " + $p) }
foreach ($p in @($gone.kept) + @($dead.kept)) { Write-Output ("changed since, kept: " + $p) }
if ($dead.restored) { Write-Output 'the user''s preferences restored from a dead session of slots' }
if ($dead.conflict) { Write-Output ('NOT restored: ' + $dead.conflict) }
