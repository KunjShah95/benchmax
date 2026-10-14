# Retry loop for the model download.
#
# WSL is restarting intermittently during this session and each restart kills
# whatever job was in flight. huggingface_hub resumes from .incomplete blobs,
# so re-invoking makes forward progress rather than starting over -- this loop
# simply survives the restarts instead of fighting them.
param(
  [string[]]$Models = @('qwen3-1.7b','qwen3-0.6b','qwen3-4b','llama32-1b','llama32-3b'),
  [int]$MaxAttempts = 25,
  [int]$AttemptSeconds = 420
)

$deadline = (Get-Date).AddSeconds($AttemptSeconds)
for ($i = 1; $i -le $MaxAttempts; $i++) {
  Write-Host "=== download attempt $i/$MaxAttempts ==="
  $proc = Start-Process -FilePath 'wsl.exe' `
      -ArgumentList @('-d','Ubuntu','--','bash','/mnt/c/benchmaxing/scripts/stage4_download.sh','drive',($Models -join ' ')) `
      -NoNewWindow -PassThru

  while (-not $proc.HasExited -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 10
  }

  if (-not $proc.HasExited) {
    Write-Host "  attempt still running at deadline; leaving it to continue"
    break
  }
  Write-Host "  attempt exited rc=$($proc.ExitCode)"

  $size = wsl -d Ubuntu -- bash -lc "du -sm ~/benchmax-env/models | cut -f1"
  Write-Host "  models now: $size MiB"

  # stop when the driver reports zero missing artifacts; it writes this to a
  # scratch file precisely because a WSL restart can truncate the log
  $rem = wsl -d Ubuntu -- bash -lc "cat ~/benchmax-logs/.remaining 2>/dev/null || echo -1"
  Write-Host "  remaining artifacts: $rem"
  if ($rem.Trim() -eq '0') { Write-Host "DOWNLOADS COMPLETE"; break }
  Write-Host "  incomplete; retrying"
}