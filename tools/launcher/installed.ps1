# What the launcher puts into the game folder (our pack, the Workshop packs it copies into data/
# such as true_sight.pack, its private mod list) is written to a marker file BEFORE the first copy:
# build/launcher-installed.json, a list of {path, sha256}. The launcher's cleanup and every later
# launch remove the files of the marker whose hash is unchanged, so a run that crashed, timed out
# or was killed (its finally never ran) leaves nothing behind for the next normal start of the
# game: a mod pack left in data/ is loaded by the game's own launcher too, and a True Sight in
# data/ broke multiplayer with a friend who had no such mod (user, 18.10.2026).
# A file not in the marker is never touched (a pack the user put into data/ himself).
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/cleanup.ps1   # by hand
# Usage: . tools/launcher/installed.ps1

function Get-FileSha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256 {
    param([string]$Text)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $sha.ComputeHash([Text.UTF8Encoding]::new($false).GetBytes($Text))
        return ([BitConverter]::ToString($bytes) -replace '-', '').ToLowerInvariant()
    } finally { $sha.Dispose() }
}

# Writes the marker: $Entries is a list of @{path = ...; sha256 = ...} (all the files this run will
# put into the game folder). Entries of a marker already there (not cleaned) are kept in it.
function Save-InstalledMarker {
    param([string]$Marker, [object[]]$Entries)
    $all = @()
    if (Test-Path -LiteralPath $Marker) {
        $all += @((Get-Content -LiteralPath $Marker -Raw | ConvertFrom-Json).files)
    }
    foreach ($e in $Entries) {
        if (-not ($all | Where-Object { $_.path -eq $e.path -and $_.sha256 -eq $e.sha256 })) {
            $all += [pscustomobject]@{path = $e.path; sha256 = $e.sha256}
        }
    }
    $dir = Split-Path -Parent $Marker
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    [IO.File]::WriteAllText($Marker, (@{files = @($all)} | ConvertTo-Json -Depth 4), [Text.UTF8Encoding]::new($false))
}

# Removes the marker's files whose hash is unchanged; a file gone already counts as removed, a
# changed one is kept (and stays in the marker). The marker goes when nothing of it is left.
# Returns @{removed = [paths]; kept = [paths]}. Call it only with no game running.
function Remove-InstalledFiles {
    param([string]$Marker)
    $out = @{removed = @(); kept = @()}
    if (-not (Test-Path -LiteralPath $Marker)) { return $out }
    $left = @()
    foreach ($e in @((Get-Content -LiteralPath $Marker -Raw | ConvertFrom-Json).files)) {
        if (-not (Test-Path -LiteralPath $e.path)) { $out.removed += $e.path; continue }
        if ((Get-FileSha256 $e.path) -eq $e.sha256) {
            Remove-Item -LiteralPath $e.path
            $out.removed += $e.path
        } else {
            $out.kept += $e.path
            $left += $e
        }
    }
    if ($left.Count -eq 0) {
        Remove-Item -LiteralPath $Marker
    } else {
        [IO.File]::WriteAllText($Marker, (@{files = @($left)} | ConvertTo-Json -Depth 4), [Text.UTF8Encoding]::new($false))
    }
    return $out
}
