from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    """Extract readable text from a PDF using a pure-Python parser."""
    text_parts: list[str] = []

    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            text_parts.append(f"[Page {page_number}]\n{page_text}")

    text = "\n\n".join(text_parts).strip()
    if not text:
        raise ValueError("No selectable text was found in this PDF. Try a text-based PDF instead of a scanned image.")

    return text, page_count
