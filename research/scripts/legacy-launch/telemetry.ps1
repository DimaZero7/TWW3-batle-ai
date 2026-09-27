# Incremental UTF-8 JSONL reader. Retains partial text/UTF-8 across append polls.
function New-TrialAuditReader {
 return @{Offset=[long]0;Pending='';Decoder=[Text.UTF8Encoding]::new($false,$true).GetDecoder();BytesRead=[long]0}
}
function Read-TrialAudit {
 param([hashtable]$State,[string]$Path)
 if(-not (Test-Path -LiteralPath $Path)){return}
 $auditStream=[IO.FileStream]::new($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
 try {
  if($auditStream.Length -lt $State.Offset){throw 'Audit was truncated'}
  $auditStream.Position=$State.Offset
  $auditBytes=[byte[]]::new(65536);$auditChars=[char[]]::new(65536)
  while(($auditN=$auditStream.Read($auditBytes,0,$auditBytes.Length)) -gt 0){
   $State.Offset+=$auditN;$State.BytesRead+=$auditN
   $auditCount=$State.Decoder.GetChars($auditBytes,0,$auditN,$auditChars,0,$false)
   $State.Pending+=[string]::new($auditChars,0,$auditCount)
   while(($auditNewline=$State.Pending.IndexOf("`n")) -ge 0){
    $auditLine=$State.Pending.Substring(0,$auditNewline).TrimEnd("`r")
    $State.Pending=$State.Pending.Substring($auditNewline+1)
    if($auditLine.Length -gt 0){Write-Output $auditLine}
   }
   if($State.Pending.Length -gt 4194304){throw 'Audit line exceeds bounded buffer'}
  }
 }finally{$auditStream.Dispose()}
}
