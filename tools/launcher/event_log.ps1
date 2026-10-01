# The game-folder event log (tww3_bai_events.jsonl) that the Lua entries append to.
# The launcher copies a run's lines into build/<target>/runs/<time>/events.jsonl; these
# helpers keep the game-folder file from growing:
#  * before the game starts, lines left from outside a launcher run are moved into the
#    run folder (events.before.jsonl), so the run reads the log from offset 0;
#  * after the run, with no game running, the file is removed only when everything in it
#    is verifiably in the run's copy (whole file read, no partial line, same line count).
# Usage: . tools/launcher/event_log.ps1

function Move-StaleEventLog {
    param([string]$Path, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Path)) { return [long]0 }
    $length = (Get-Item -LiteralPath $Path).Length
    if ($length -eq 0) { Remove-Item -LiteralPath $Path; return [long]0 }
    Move-Item -LiteralPath $Path -Destination $Destination
    return [long]$length
}

function Measure-JsonlLines {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return [long]0 }
    $count = [long]0
    foreach ($line in [IO.File]::ReadLines($Path)) { if ($line.Length -gt 0) { $count++ } }
    return $count
}

# Returns 'removed', 'absent' or 'kept: <reason>'.
function Remove-CopiedEventLog {
    param([string]$Path, [hashtable]$State, [string]$RunLog, [long]$CopiedLines)
    if (-not (Test-Path -LiteralPath $Path)) { return 'absent' }
    $length = (Get-Item -LiteralPath $Path).Length
    if ($State.BytesRead -ne $State.Offset) { return 'kept: not read from the start' }
    if ($length -ne $State.Offset) { return ('kept: {0} bytes in the log, {1} read' -f $length, $State.Offset) }
    if ($State.Pending.Length -gt 0) { return 'kept: a partial line is not copied' }
    $saved = Measure-JsonlLines -Path $RunLog
    if ($saved -ne $CopiedLines) { return ('kept: {0} lines copied, {1} in the run copy' -f $CopiedLines, $saved) }
    Remove-Item -LiteralPath $Path
    return 'removed'
}
