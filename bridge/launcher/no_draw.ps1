# The no-draw build's events in the launcher (bridge.build --no-draw; bridge/docs/apps/memreader.md, section V2).
# New-NoDrawState: the counts; Add-NoDrawEvent: one key event line (v2_on / v2_off / v2_refused / v2_drawn) - a v2_off
# without "restored":true marks the run unrestored at once (the launcher then closes its game); Test-NoDrawEnd: at the
# battle's end (result or error) a v2_on with no restoring v2_off is unrestored too. Only v2_off rows with restored true
# count as off.
function New-NoDrawState { @{on = 0; off = 0; off_failed = 0; refused = 0; drawn = $null; unrestored = ''} }

function Add-NoDrawEvent($State, [string]$Event, [string]$Line) {
    switch ($Event) {
        'v2_on' { $State.on++ }
        'v2_refused' { $State.refused++ }
        'v2_off' {
            if ($Line -match '"restored":true') { $State.off++ }
            else { $State.off_failed++; if (-not $State.unrestored) { $State.unrestored = 'a v2_off row without restored true' } }
        }
        'v2_drawn' { $State.drawn = $Line -match '"drawn":true' }
    }
}

function Test-NoDrawEnd($State, [string]$Status) {
    if (-not $State.unrestored -and $State.on -gt $State.off -and $Status -in @('completed', 'lua_error')) {
        $State.unrestored = 'v2_on without a restoring v2_off at the battle''s end'
    }
    return $State.unrestored
}
