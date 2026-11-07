# The modifier keys for the live labelling (src/apps/battle/live_label.lua M.keys): a game script sees no keys, so this
# helper on the host reads two keys only - Ctrl and Shift (GetAsyncKeyState, no key is ever sent) - every PollMs and
# writes 'ctrl=0|1 shift=0|1 t=<unix s>' to the game folder's tww3_bai_keys.txt on every change and twice a second.
# It ends when the game it saw ends, when the game never comes in WaitSeconds, or after MaxMinutes.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/label/keys.ps1 -Game <game folder>
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [int]$PollMs = 30,
    [int]$WaitSeconds = 300,
    [int]$MaxMinutes = 240
)
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class BaiKeys {
    [DllImport("user32.dll")] public static extern short GetAsyncKeyState(int vKey);
}
'@
$file = Join-Path $Game 'tww3_bai_keys.txt'
$started = Get-Date
$seen = $false
$state = @{ last = ''; written = [DateTime]::MinValue }
function Write-Keys {
    $ctrl = ([BaiKeys]::GetAsyncKeyState(0x11) -band 0x8000) -ne 0
    $shift = ([BaiKeys]::GetAsyncKeyState(0x10) -band 0x8000) -ne 0
    $line = 'ctrl={0} shift={1}' -f [int]$ctrl, [int]$shift
    $now = Get-Date
    if ($line -ne $state.last -or ($now - $state.written).TotalMilliseconds -ge 500) {
        $t = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        try { [System.IO.File]::WriteAllText($file, "$line t=$t") } catch { }
        $state.last = $line
        $state.written = $now
    }
}
try {
    while ($true) {
        $now = Get-Date
        if (($now - $started).TotalMinutes -ge $MaxMinutes) { break }
        # (the game process looked up once a second, the keys every PollMs)
        if (Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue) { $seen = $true }
        elseif ($seen -or ($now - $started).TotalSeconds -ge $WaitSeconds) { break }
        for ($i = 0; $i -lt [Math]::Max(1, [int](1000 / $PollMs)); $i++) {
            Write-Keys
            Start-Sleep -Milliseconds $PollMs
        }
    }
} finally {
    if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -ErrorAction SilentlyContinue }
}
