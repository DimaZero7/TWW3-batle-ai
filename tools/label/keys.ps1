# The modifier keys and the cursor for the live labelling (src/apps/battle/live_label.lua M.keys, M.cursor): a game script
# sees no keys and no mouse, so this helper on the host only reads - Ctrl, Shift (GetAsyncKeyState) and the cursor's place
# in the game window's client area (GetCursorPos, ScreenToClient; the client's size for the scale) - every PollMs, and
# writes 'ctrl=0|1 shift=0|1 x=<px> y=<px> w=<client px> h=<client px> t=<unix s>' to the game folder's tww3_bai_keys.txt
# on every change and twice a second. Nothing is ever sent to any window. It ends when the game it saw ends, when the game
# never comes in WaitSeconds, or after MaxMinutes.
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
using System;
using System.Runtime.InteropServices;
public static class BaiKeys {
    [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X; public int Y; }
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }
    [DllImport("user32.dll")] public static extern short GetAsyncKeyState(int vKey);
    [DllImport("user32.dll")] public static extern bool GetCursorPos(out POINT p);
    [DllImport("user32.dll")] public static extern bool ScreenToClient(IntPtr hWnd, ref POINT p);
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr hWnd, out RECT r);
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
}
'@
[void][BaiKeys]::SetProcessDPIAware()   # (physical pixels: the game draws in them)
$file = Join-Path $Game 'tww3_bai_keys.txt'
$started = Get-Date
$seen = $false
$state = @{ last = ''; written = [DateTime]::MinValue; hwnd = [IntPtr]::Zero }
function Write-Keys {
    $ctrl = ([BaiKeys]::GetAsyncKeyState(0x11) -band 0x8000) -ne 0
    $shift = ([BaiKeys]::GetAsyncKeyState(0x10) -band 0x8000) -ne 0
    $line = 'ctrl={0} shift={1}' -f [int]$ctrl, [int]$shift
    if ($state.hwnd -ne [IntPtr]::Zero) {
        $p = New-Object BaiKeys+POINT
        $r = New-Object BaiKeys+RECT
        if ([BaiKeys]::GetCursorPos([ref]$p) -and [BaiKeys]::ScreenToClient($state.hwnd, [ref]$p) -and
            [BaiKeys]::GetClientRect($state.hwnd, [ref]$r)) {
            $line += ' x={0} y={1} w={2} h={3}' -f $p.X, $p.Y, ($r.Right - $r.Left), ($r.Bottom - $r.Top)
        }
    }
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
        # (the game process and its window looked up once a second, the keys and the cursor every PollMs)
        $game = Get-Process -Name Warhammer3 -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($game) { $seen = $true; $state.hwnd = $game.MainWindowHandle }
        elseif ($seen -or ($now - $started).TotalSeconds -ge $WaitSeconds) { break }
        for ($i = 0; $i -lt [Math]::Max(1, [int](1000 / $PollMs)); $i++) {
            Write-Keys
            Start-Sleep -Milliseconds $PollMs
        }
    }
} finally {
    if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -ErrorAction SilentlyContinue }
}
