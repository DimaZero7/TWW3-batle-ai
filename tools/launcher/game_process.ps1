# The game process's own properties after the launcher starts it: CPU priority and the cores it may run on
# (docs/en/launch/run.md "Game process: priority and fast cores"). A big battle is bound by one game thread
# (the battle model); on a hybrid CPU (Intel P + E cores) the launcher keeps the game on the fast P cores and
# raises its priority over Docker and training. Only this process changes: no power plan, no system setting.
#
#   . tools/launcher/game_process.ps1
#   $cores = Get-CpuCores                      # [{efficiency, mask, logical}] per physical core (group 0)
#   $mask = Get-FastCoreMask -Cores $cores     # P-core logical processors as a bit mask; $null: not hybrid
#   $info = Set-GameProcess -Process $p -Priority High -FastCores $true

if (-not ('Bai.CpuTopology' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
namespace Bai {
    public static class CpuTopology {
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool GetLogicalProcessorInformationEx(int relation, IntPtr buffer, ref uint length);
        // One row per physical core of processor group 0: {EfficiencyClass, logical processor mask}.
        public static long[][] Cores() {
            uint len = 0;
            GetLogicalProcessorInformationEx(0, IntPtr.Zero, ref len);   // RelationProcessorCore
            IntPtr buf = Marshal.AllocHGlobal((int)len);
            try {
                if (!GetLogicalProcessorInformationEx(0, buf, ref len)) throw new System.ComponentModel.Win32Exception();
                var rows = new List<long[]>();
                long off = 0;
                while (off < len) {
                    IntPtr p = new IntPtr(buf.ToInt64() + off);
                    int size = Marshal.ReadInt32(p, 4);
                    byte eff = Marshal.ReadByte(p, 9);
                    ushort groups = (ushort)Marshal.ReadInt16(p, 30);
                    for (int g = 0; g < groups; g++) {
                        long mask = Marshal.ReadInt64(p, 32 + 16 * g);
                        ushort group = (ushort)Marshal.ReadInt16(p, 40 + 16 * g);
                        if (group == 0) rows.Add(new long[] { eff, mask });
                    }
                    off += size;
                }
                return rows.ToArray();
            } finally { Marshal.FreeHGlobal(buf); }
        }
    }
}
'@
}

function Get-CpuCores {
    foreach ($row in [Bai.CpuTopology]::Cores()) {
        $m = [long]$row[1]
        $logical = @(0..63 | Where-Object { ($m -shr $_) -band 1 })
        [pscustomobject]@{efficiency = [int]$row[0]; mask = $m; logical = $logical}
    }
}

# The logical processors of the fastest cores (the highest efficiency class: Intel P cores) as one mask;
# $null when every core has the same class (not hybrid: the game keeps all cores).
function Get-FastCoreMask {
    param([Parameter(Mandatory = $true)][object[]]$Cores)
    $classes = @($Cores | ForEach-Object { $_.efficiency } | Sort-Object -Unique)
    if ($classes.Count -lt 2) { return $null }
    $top = ($classes | Measure-Object -Maximum).Maximum
    $mask = [long]0
    foreach ($c in $Cores) { if ($c.efficiency -eq $top) { $mask = $mask -bor [long]$c.mask } }
    return $mask
}

# The run's choice: a launcher parameter, else the environment (BAI_GAME_PRIORITY, BAI_FAST_CORES), else
# config (game_priority, game_fast_cores), else on (High, fast cores).
function Resolve-GameProcessOptions {
    param($Settings, [string]$Priority = '', [string]$FastCores = '')
    if (-not $Priority) { $Priority = $env:BAI_GAME_PRIORITY }
    if (-not $Priority -and $Settings -and $Settings.PSObject.Properties['game_priority']) { $Priority = [string]$Settings.game_priority }
    if (-not $Priority) { $Priority = 'High' }
    if ($Priority -notin @('Normal', 'AboveNormal', 'High')) { throw "game priority must be Normal, AboveNormal or High, not '$Priority'" }
    if (-not $FastCores) { $FastCores = $env:BAI_FAST_CORES }
    if (-not $FastCores -and $Settings -and $Settings.PSObject.Properties['game_fast_cores']) {
        $FastCores = if ($Settings.game_fast_cores) { 'on' } else { 'off' }
    }
    if (-not $FastCores) { $FastCores = 'on' }
    if ($FastCores -notin @('on', 'off')) { throw "fast cores must be on or off, not '$FastCores'" }
    return [pscustomobject]@{priority = $Priority; fast_cores = ($FastCores -eq 'on')}
}

# The game sets its own priority back to Normal while it loads (16.10: High at the start, Normal a minute
# later; the cores stay): called on every launcher poll, it puts the wanted priority back. $true when it had to.
function Restore-GamePriority {
    param([Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process, [string]$Priority = 'High')
    try {
        $Process.Refresh()
        if ($Process.HasExited -or $Process.PriorityClass -eq [System.Diagnostics.ProcessPriorityClass]$Priority) { return $false }
        $Process.PriorityClass = [System.Diagnostics.ProcessPriorityClass]$Priority
        return $true
    } catch { return $false }
}

# Sets the started game's priority and (on a hybrid CPU) its cores; never throws: what was set, or why not.
function Set-GameProcess {
    param([Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process, [string]$Priority = 'High', [bool]$FastCores = $true)
    $info = [ordered]@{priority = 'unchanged'; affinity = 'unchanged'}
    try {
        $Process.PriorityClass = [System.Diagnostics.ProcessPriorityClass]$Priority
        $info.priority = $Priority
    } catch { $info.priority = 'failed: ' + $_.Exception.Message }
    if ($FastCores) {
        try {
            $mask = Get-FastCoreMask -Cores @(Get-CpuCores)
            if ($null -eq $mask) {
                $info.affinity = 'not hybrid'
            } else {
                $Process.ProcessorAffinity = [IntPtr]$mask
                $info.affinity = '0x{0:X}' -f $mask
            }
        } catch { $info.affinity = 'failed: ' + $_.Exception.Message }
    }
    return [pscustomobject]$info
}
