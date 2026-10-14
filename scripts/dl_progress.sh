#!/usr/bin/env bash
# Is the download actually progressing? huggingface_hub writes into a blob cache
# first, so the model dir can look static while bytes are arriving.
LOGDIR="$HOME/benchmax-logs"
M=~/benchmax-env/models

echo "=== running download procs ==="
pgrep -fa "stage4_download" | head -4 || echo "  none"

echo
echo "=== sizes (10s apart) ==="
a=$(du -sb "$M" 2>/dev/null | cut -f1)
echo "  t0: $(( a / 1024 / 1024 )) MiB"
sleep 10
b=$(du -sb "$M" 2>/dev/null | cut -f1)
echo "  t1: $(( b / 1024 / 1024 )) MiB"
echo "  delta: $(( (b - a) / 1024 / 1024 )) MiB in 10s"

echo
echo "=== incomplete blobs in flight ==="
find "$M" -name '*.incomplete' -printf '  %s %p\n' 2>/dev/null | sort -rn | head -5

echo
echo "=== newest files written ==="
find "$M" -type f -newermt '-3 minutes' -printf '%T@ %s %p\n' 2>/dev/null | sort -rn | head -5 | cut -d' ' -f2-
echo "  (count in last 3 min: $(find "$M" -type f -newermt '-3 minutes' 2>/dev/null | wc -l))"

echo
echo "=== q17 sentinel ==="
cat "$LOGDIR/download_q17.done" 2>/dev/null || echo "  RUNNING"