#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/benchmaxing
mkdir -p reports/pdf/html
mv -f reports/pdf/*.pdf.html reports/pdf/html/ 2>/dev/null || true

source ~/benchmax-env/venv-bench/bin/activate
python - <<'PY'
from pypdf import PdfReader
import os

print(f"{'file':22} {'size':>10} {'pages':>6} {'imgs':>6}")
for f in ("REPORT.pdf", "DATA_APPENDIX.pdf", "README.pdf"):
    p = f"reports/pdf/{f}"
    r = PdfReader(p)
    imgs = sum(len(pg.images) for pg in r.pages)
    print(f"{f:22} {os.path.getsize(p)/1024:7.0f} KiB {len(r.pages):6} {imgs:6}")
PY
echo
echo "=== reports/pdf/ ==="
ls -la reports/pdf/ | sed 's/^/  /'