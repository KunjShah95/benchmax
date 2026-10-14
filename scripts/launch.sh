#!/usr/bin/env bash
# Launch a long-running job DETACHED inside WSL.
#
# Launching long jobs via `wsl.exe` from PowerShell does not work: when the
# PowerShell session ends, the WSL guest can be torn down and the job dies
# mid-compile (this happened twice). Detaching with setsid + nohup inside the
# guest keeps it alive, and we poll a sentinel file instead of holding a pipe
# open.
#
#   ./scripts/launch.sh <script.sh> <sentinel-name> [args...]
set -euo pipefail

SCRIPT="$1"; shift
NAME="$1"; shift

# /tmp did NOT survive WSL restarts during this session; logs live under the
# persistent home directory instead. Jobs still die on VM restart (see note
# below) but their evidence survives, so a restart is diagnosable rather than
# silent.
LOGDIR="$HOME/benchmax-logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${NAME}.log"
DONE="$LOGDIR/${NAME}.done"

rm -f "$DONE"
{
  echo "launched at $(date -Is)"
  echo "script: $SCRIPT"
  echo "args: $*"
  echo "---"
} > "$LOG"

# setsid detaches from the controlling terminal so guest shutdown-on-session-exit
# does not propagate; the inner wrapper writes the sentinel either way so we can
# always tell how far it got.
setsid nohup bash -c "
  bash '$SCRIPT' $* >> '$LOG' 2>&1
  echo \"exit=\$?\" >> '$LOG'
  date -Is > '$DONE'
" > /dev/null 2>&1 < /dev/null &

sleep 1
echo "launched: $NAME"
echo "log:      $LOG"
echo "sentinel: $DONE"
echo
echo "NOTE: a WSL VM restart kills the job; re-run this launcher to resume."
pgrep -fa "$(basename "$SCRIPT")" | head -3 || true