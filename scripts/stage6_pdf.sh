#!/usr/bin/env bash
# Finish the reporting stage: tables + figures, data appendix, and PDFs.
# Run `report` first (it writes reports/tables and reports/figures).
set -uo pipefail

ENV_ROOT="$HOME/benchmax-env"
source "$ENV_ROOT/venv-bench/bin/activate"
cd /mnt/c/benchmaxing
export PYTHONPATH=/mnt/c/benchmaxing

echo "### 1/2  data appendix markdown from generated tables"
python scripts/build_appendix.py

echo
echo "### 2/2  render PDFs (self-contained, figures embedded)"
python scripts/md_to_pdf.py \
  reports/pdf/DATA_APPENDIX.md \
  reports/REPORT.md \
  README.md \
  -o reports/pdf

echo
echo "=== outputs ==="
ls -la reports/pdf/*.pdf | sed 's/^/  /'

echo
echo "NOTE: md_to_pdf.py leaves rendered_html:  reports/pdf/*.pdf.html"
echo "keeping the .html alongside each PDF aids inspection and re-rendering"