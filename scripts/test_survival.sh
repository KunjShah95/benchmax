#!/usr/bin/env bash
# Test whether a detached process survives a WSL session ending.
# Writes a heartbeat every 2s; if the VM dies the heartbeat stops permanently.
LOG="$HOME/benchmax-logs/survivor.txt"
rm -f "$LOG"
setsid nohup bash -c "
  for i in \$(seq 1 300); do
    echo \"\$(date +%H:%M:%S) alive \$i\" >> $LOG
    sleep 2
  done
" > /dev/null 2>&1 < /dev/null &
echo "launched survivor pid=$!"
sleep 3
echo "initial lines: $(wc -l < $LOG)"