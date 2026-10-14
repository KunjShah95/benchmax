#!/usr/bin/env bash
# Watch a single in-flight blob to distinguish "slow" from "stalled".
M=~/benchmax-env/models
snap() {
  find "$M" -name '*.incomplete' -printf '%s %f\n' 2>/dev/null | sort -rn | head -3
}
echo "=== t0 ==="; snap; T0=$(du -sb "$M" | cut -f1)
sleep 20
echo "=== t+20s ==="; snap; T1=$(du -sb "$M" | cut -f1)
echo "delta: $(( (T1-T0)/1024/1024 )) MiB / 20s"
echo
echo "=== network sockets for the download proc ==="
PID=$(pgrep -f "stage4_download" | head -1)
echo "pid=$PID"
if [ -n "$PID" ]; then
  ls -l /proc/$PID/fd 2>/dev/null | grep -c socket | sed 's/^/  open sockets: /'
fi
ss -tnp 2>/dev/null | grep -c ESTAB | sed 's/^/  established conns: /'
echo
echo "=== dl_all progress ==="
grep -cE '^\s+-> ' ~/benchmax-logs/dl_all.log