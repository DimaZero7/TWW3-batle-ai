# Incremental UTF-8 JSONL reader for a file another process appends to.
# Keeps partial lines and split UTF-8 sequences between polls.
# Usage: . tools/telemetry/read_jsonl.ps1
#        $reader = New-JsonlReader -Offset 0
#        Read-JsonlLines -State $reader -Path $log | ForEach-Object { $_ | ConvertFrom-Json }
function New-JsonlReader {
    param([long]$Offset = 0)
    return @{Offset = $Offset; Pending = ''; Decoder = [Text.UTF8Encoding]::new($false, $true).GetDecoder(); BytesRead = [long]0}
}

function Read-JsonlLines {
    param([hashtable]$State, [string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $stream = [IO.FileStream]::new($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    try {
        if ($stream.Length -lt $State.Offset) { throw 'Log was truncated' }
        $stream.Position = $State.Offset
        $bytes = [byte[]]::new(65536)
        $chars = [char[]]::new(65536)
        while (($count = $stream.Read($bytes, 0, $bytes.Length)) -gt 0) {
            $State.Offset += $count
            $State.BytesRead += $count
            $decoded = $State.Decoder.GetChars($bytes, 0, $count, $chars, 0, $false)
            $State.Pending += [string]::new($chars, 0, $decoded)
            while (($newline = $State.Pending.IndexOf("`n")) -ge 0) {
                $line = $State.Pending.Substring(0, $newline).TrimEnd("`r")
                $State.Pending = $State.Pending.Substring($newline + 1)
                if ($line.Length -gt 0) { Write-Output $line }
            }
            if ($State.Pending.Length -gt 4194304) { throw 'Log line exceeds bounded buffer' }
        }
    } finally {
        $stream.Dispose()
    }
}
