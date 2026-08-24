"""PDF → plain text with PyMuPDF."""
from pathlib import Path

import pymupdf

from app.config import RAW_PDF_DIR, TXT_DIR


def extract_pdf(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    text = "\n".join(pages)
    # strip trailing spaces — keeps chunking clean
    lines = [ln.rstrip() for ln in text.splitlines()]
    out, blanks = [], 0
    for ln in lines:
        blanks = blanks + 1 if not ln else 0
        if blanks <= 2:
            out.append(ln)
    return "\n".join(out)


def extract_all(pdf_dir: Path = RAW_PDF_DIR, txt_dir: Path = TXT_DIR) -> int:
    """Extract every PDF that doesn't have a .txt yet. Returns count of new files."""
    txt_dir.mkdir(parents=True, exist_ok=True)
    new = 0
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        target = txt_dir / (pdf.stem + ".txt")
        if target.exists():
            continue
        try:
            target.write_text(extract_pdf(pdf), encoding="utf-8")
            new += 1
        except Exception as e:
            print(f"  extract FAILED for {pdf.name}: {e}")
    return new
