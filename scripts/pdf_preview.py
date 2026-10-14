#!/usr/bin/env python
"""Render PDF pages to PNG so layout can be eyeballed."""
import os
import sys

# Install: pip install pypdfium2
try:
    import pypdfium2 as pdfium
except ImportError:
    os.system("pip install --quiet pypdfium2")
    import pypdfium2 as pdfium

src = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pdfpages"
pages = [int(p) for p in sys.argv[3].split(",")] if len(sys.argv) > 3 else None

os.makedirs(out_dir, exist_ok=True)
doc = pdfium.PdfDocument(src)
print(f"{src}: {len(doc)} pages")
for i in range(len(doc)):
    if pages and i not in pages:
        continue
    page = doc[i]
    img = page.render(scale=1.4).to_pil()
    p = os.path.join(out_dir, f"page_{i+1:02d}.png")
    img.save(p)
    print(f"  page {i+1}: {img.width}x{img.height} -> {p}")