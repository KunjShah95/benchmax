#!/usr/bin/env bash
set -euo pipefail
source ~/benchmax-env/venv-bench/bin/activate

echo "=== installing markdown + weasyprint (pure-python, no apt) ==="
pip install --quiet markdown weasyprint

python - <<'PY'
import os
import markdown
import weasyprint
print("markdown   ", markdown.__version__)
print("weasyprint ", weasyprint.__version__)

weasyprint.HTML(string="<h1>smoke</h1><p>hello</p>").write_pdf("/tmp/_t.pdf")
print("pdf bytes:", os.path.getsize("/tmp/_t.pdf"))
os.remove("/tmp/_t.pdf")
print("PDF RENDER OK")
PY