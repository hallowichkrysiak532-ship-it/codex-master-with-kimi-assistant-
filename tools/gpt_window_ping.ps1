# Anchor Codex's 5-hour usage window with a minimal prompt so the timer
# starts before working hours. Registered as GPT-WindowPing-HHMM tasks.
# This file must stay UTF-8 with BOM (PowerShell 5.1 reads 你好 correctly).
param([switch]$DryRun)

[Console]::OutputEncoding = [Text.Encoding]::UTF8

$root    = 'E:\Multi-Agent'
$logDir  = Join-Path $root '.agent_runtime\logs'
$logFile = Join-Path $logDir 'gpt_window_ping.log'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

if ($DryRun) {
    "[$stamp] DRYRUN ok" | Out-File -Append -Encoding utf8 $logFile
    exit 0
}

$codex = (Get-Command codex.cmd -ErrorAction SilentlyContinue).Source
if (-not $codex) { $codex = Join-Path $env:APPDATA 'npm\codex.cmd' }
if (-not (Test-Path $codex)) {
    "[$stamp] ERROR codex not found" | Out-File -Append -Encoding utf8 $logFile
    exit 1
}

Push-Location $root
try {
    $output = & $codex exec "你好" 2>&1 | Out-String
    $code = $LASTEXITCODE
} catch {
    $output = $_.Exception.Message
    $code = -1
}
Pop-Location

"[$stamp] exit=$code" | Out-File -Append -Encoding utf8 $logFile
$tail = (($output.Trim() -split "\r?\n" | Select-Object -Last 5) -join "; ")
"  $tail" | Out-File -Append -Encoding utf8 $logFile
exit $code
