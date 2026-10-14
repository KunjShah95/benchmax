param(
  [int]$MaxRestarts = 40,
  # Generous: a full 29-cell matrix runs ~2-3h. The slice cap exists only to
  # rescue a genuinely wedged server, so it must be long enough that a healthy
  # run is never interrupted. An earlier 900s cap cut runs that were making
  # steady progress.
  [int]$SliceSeconds = 3600
)
# Supervisor for the benchmark matrix.
#
# The WSL VM in this environment restarts unpredictably (multiple times over the
# session), and each restart kills the in-flight run. `--resume` skips cells that
# already completed, so relaunching is safe and makes forward progress; this
# loop just rides out the restarts instead of losing hours of work to them.
#
# Each slice is deliberately time-boxed: if a slice runs past SliceSeconds we
# assume a stall, terminate it, and resume -- a wedged server should not block
# the whole matrix.

$proj = 'C:\benchmaxing'
$logDir = "$env:USERPROFILE\benchmax-logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

for ($i = 1; $i -le $MaxRestarts; $i++) {
  Write-Host ""
  Write-Host "=== slice $i/$MaxRestarts  $(Get-Date -Format 'HH:mm:ss') ==="

  $proc = Start-Process -FilePath 'wsl.exe' -NoNewWindow -PassThru -ArgumentList @(
    '-d','Ubuntu','--','bash','/mnt/c/benchmaxing/scripts/stage5_run.sh','slice',
    'llamacpp','vllm','vllm_gguf','--repeats','3','--warmup','3'
  ) -RedirectStandardOutput "$logDir\slice-$i.log" -RedirectStandardError "$logDir\slice-$i.err"

  $deadline = (Get-Date).AddSeconds($SliceSeconds)
  $lastCount = -1
  while (-not $proc.HasExited -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 20
    $c = (wsl -d Ubuntu -- bash -lc "grep -c '\[cell\] done' /home/jksha/benchmax-logs/slice.log 2>/dev/null || echo 0").Trim()
    if ($c -ne $lastCount) { Write-Host "  cells completed: $c"; $lastCount = $c }
    if ($c -eq '29') { break }
  }

  if ($c -eq '29') { Write-Host "MATRIX COMPLETE"; break }

  if (-not $proc.HasExited) {
    Write-Host "  slice hit the ${SliceSeconds}s cap; restarting (resume)"
    wsl -d Ubuntu -- bash /mnt/c/benchmaxing/scripts/kill_all.sh | Out-Null
    try { $proc.Kill() } catch {}
  } else {
    Write-Host "  slice exited rc=$($proc.ExitCode) (likely a WSL restart); resuming"
  }

  $done = (wsl -d Ubuntu -- bash -lc "grep -c '\[cell\] done' /home/jksha/benchmax-logs/slice.log 2>/dev/null || echo 0").Trim()
  Write-Host "  cumulative cells done: $done / 29"
  if ($done -eq '29') { Write-Host "MATRIX COMPLETE"; break }
  Start-Sleep -Seconds 5
}

Write-Host ""
Write-Host "supervisor finished"