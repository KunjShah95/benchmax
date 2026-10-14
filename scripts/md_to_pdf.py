#!/usr/bin/env python
"""Render a Markdown report to a self-contained PDF.

Design decisions:

* **Base64-embedded images.** A PDF with relative image links only renders on
  the machine that produced it. Embedding makes the artifact portable.
* **Print CSS.** Screen-oriented markdown themes are awful on paper: wide
  tables overflow, code blocks break mid-token, figures spill across pages. The
  stylesheet here handles page breaks, keeps table rows intact, and avoids
  splitting figures.
* **No pandoc / wkhtmltopd / LibreOffice dependency.** None of those are
  installed and installing them means apt or a multi-GB download. `markdown`
  plus weasyprint is pure Python.

Usage:
    python md_to_pdf.py results/../reports/REPORT.md out/REPORT.pdf
"""
from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import re
import sys

import markdown

CSS = """
@page {
    size: A4;
    margin: 18mm 16mm 20mm 16mm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 8pt; color: #888;
    }
}
html { font-size: 10pt; }
body {
    font-family: "DejaVu Sans", "Segoe UI", Arial, sans-serif;
    line-height: 1.45; color: #1a1a1a; margin: 0;
}
h1 { font-size: 19pt; margin: 0 0 6pt 0; color: #0f2740; border-bottom: 2px solid #2e86ab;
     padding-bottom: 5pt; }
h2 { font-size: 14pt; margin: 18pt 0 7pt 0; color: #2e86ab;
     border-bottom: 1px solid #d8e4ec; padding-bottom: 3pt;
     break-after: avoid; }
h3 { font-size: 11.5pt; margin: 13pt 0 5pt 0; color: #24455f; break-after: avoid; }
p { margin: 0 0 7pt 0; text-align: justify; }
ul, ol { margin: 0 0 8pt 0; padding-left: 16pt; }
li { margin-bottom: 3pt; }
hr { border: none; border-top: 1px solid #dde3e8; margin: 14pt 0; }

table {
    border-collapse: collapse; width: 100%;
    margin: 8pt 0 12pt 0; font-size: 8.2pt;
    break-inside: avoid;      /* keep a table's rows together where possible */
}
th, td { border: 1px solid #c9d6e0; padding: 3.5pt 5pt; text-align: left;
         vertical-align: top; }
th { background: #eef4f8; font-weight: 600; color: #14405c; }
tr:nth-child(even) td { background: #fafcfd; }

code {
    font-family: "DejaVu Sans Mono", Consolas, monospace;
    background: #f3f6f8; padding: 0.5pt 2.5pt; border-radius: 2px;
    font-size: 8.4pt;
}
pre {
    background: #f3f6f8; border-left: 3px solid #2e86ab;
    padding: 7pt 9pt; margin: 8pt 0; overflow-wrap: break-word;
    font-size: 7.8pt; line-height: 1.3; break-inside: avoid;
}
pre code { background: none; padding: 0; }

img { max-width: 100%; height: auto; display: block; margin: 9pt auto;
      break-inside: avoid; page-break-inside: avoid; }
blockquote {
    margin: 8pt 0; padding: 5pt 10pt; border-left: 3px solid #d1495b;
    background: #fdf6f7; color: #5a2b33;
}
strong { color: #0f2740; }
"""

# --------------------------------------------------------------------------
# image embedding
# --------------------------------------------------------------------------

_IMG_RE = re.compile(
    r'<img\s+([^>]*?)\ssrc="([^"]+)"([^>]*)>', re.IGNORECASE | re.DOTALL)


def embed_images(html: str, base_dir: str) -> tuple[str, int]:
    """Inline images as data URIs relative to the markdown file."""
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        src = m.group(2)
        # already inlined, or not a filesystem path
        if src.startswith(("data:", "http://", "https://")):
            return m.group(0)
        path = src if os.path.isabs(src) else os.path.join(base_dir, src)
        path = os.path.normpath(path)
        if not os.path.isfile(path):
            return m.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        n += 1
        return f'<img {m.group(1)} src="data:{mime};base64,{b64}"{m.group(3)}>'

    return _IMG_RE.sub(repl, html), n


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------

def md_to_html(src: str) -> tuple[str, str]:
    with open(src, "r", encoding="utf-8") as f:
        text = f.read()
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "attr_list", "md_in_html",
                    "sane_lists"],
        output_format="html5",
    )
    title = os.path.splitext(os.path.basename(src))[0]
    return body, title


def render_pdf(src: str, dst: str) -> dict:
    body, title = md_to_html(src)
    base_dir = os.path.dirname(os.path.abspath(src))
    body, embedded = embed_images(body, base_dir)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head>
<body>{body}</body></html>"""

    html_path = dst + ".html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    import weasyprint
    weasyprint.HTML(string=html, base_url=base_dir).write_pdf(dst)

    return {
        "source": src,
        "pdf": dst,
        "pdf_bytes": os.path.getsize(dst),
        "html_kept": html_path,
        "images_embedded": embedded,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="markdown -> PDF (self-contained)")
    ap.add_argument("sources", nargs="+", help="markdown files")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="output directory (default: alongside each source)")
    a = ap.parse_args()

    ok = True
    for src in a.sources:
        if not os.path.isfile(src):
            print(f"MISSING: {src}")
            ok = False
            continue
        out_dir = os.path.abspath(a.out_dir or os.path.dirname(src) or ".")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(out_dir, f"{stem}.pdf")
        try:
            info = render_pdf(src, dst)
            print(f"OK   {os.path.basename(src)} -> {os.path.basename(dst)} "
                  f"({info['pdf_bytes']/1024:.0f} KiB, "
                  f"{info['images_embedded']} figures embedded)")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {src}: {type(exc).__name__}: {exc}")
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()