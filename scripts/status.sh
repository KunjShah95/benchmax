#!/usr/bin/env bash
LOGDIR="$HOME/benchmax-logs"
for j in "$@"; do
  echo "=== $j ==="
  if [ -f "$LOGDIR/$j.done" ]; then
    echo "  DONE at $(cat "$LOGDIR/$j.done")"
  else
    echo "  RUNNING"
  fi
  grep -E '^\s+-> |!!' "$LOGDIR/$j.log" 2>/dev/null | tail -5
  tail -1 "$LOGDIR/$j.log" 2>/dev/null | sed 's/^/  last: /'
done
echo
echo "total models: $(du -sh ~/benchmax-env/models 2>/dev/null | cut -f1)"
echo "qwen3-1.7b safetensors:"; ls -la ~/benchmax-env/models/Qwen__Qwen3-1.7B/*.safetensors 2>/dev/null | sed 's/^/  /' || echo "  none yet"