#!/usr/bin/env bash
for t in pandoc wkhtmltopdf weasyprint libreoffice; do
  printf '%-14s %s\n' "$t" "$(command -v $t 2>/dev/null || echo MISSING)"
done
echo
source ~/benchmax-env/venv-bench/bin/activate
python - <<'PY'
for mod in ("markdown", "weasyprint", "reportlab", "fpdf"):
    try:
        m = __import__(mod)
        print(f"{mod:12} OK  {getattr(m, '__version__', '')}")
    except Exception:
        print(f"{mod:12} MISSING")
PY