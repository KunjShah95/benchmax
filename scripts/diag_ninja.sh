#!/usr/bin/env bash
L=$(ls -t ~/benchmax-logs/vllmfix-*.log 2>/dev/null | head -1)
echo "log: $L"
echo
echo "=== distinct compile errors ==="
grep -oE 'error: #[a-z]+|error: [^,{]{0,120}|fatal error: [^,]{0,120}' "$L" \
  | sort | uniq -c | sort -rn | head -12
echo
echo "=== linker diagnostics (cannot find -lcudart / undefined ref) ==="
grep -oE 'cannot find -l[a-zA-Z0-9_.]+|undefined reference to [^,]{0,90}|/usr/bin/ld: [^,]{0,110}' "$L" \
  | sort | uniq -c | sort -rn | head -12
echo
echo "=== raw lines around 'ld returned' ==="
grep -B8 'ld returned 1 exit status' "$L" | tail -18 | cut -c1-180