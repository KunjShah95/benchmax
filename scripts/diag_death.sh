#!/usr/bin/env bash
echo "=== uptime / memory ==="
uptime
free -m | head -2
echo
echo "=== OOM kills in kernel log ==="
dmesg 2>/dev/null | grep -iE 'out of memory|oom.kill|killed process' | tail -10 || echo "  (dmesg unavailable in container)"
echo
echo "=== any surviving benchmax procs ==="
pgrep -fa 'benchmax|stage4|vllm|llama-server|hf_' | head -10 || echo "  none"
echo
echo "=== dl_all sentinel / log tail ==="
cat ~/benchmax-logs/dl_all.done 2>/dev/null || echo "  no sentinel"
tail -4 ~/benchmax-logs/dl_all.log | cut -c1-160
echo
echo "=== free disk ==="
df -h / | tail -1