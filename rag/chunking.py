import re


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 180) -> list[str]:
    """Create overlapping chunks with stable boundaries for semantic search."""
    clean = normalize_text(text)
    if len(clean) <= chunk_size:
        return [clean]

    chunks: list[str] = []
    start = 0

    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        window = clean[start:end]

        if end < len(clean):
            split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if split_at > chunk_size * 0.55:
                end = start + split_at + 1
                window = clean[start:end]

        chunk = window.strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(clean):
            break
        start = max(0, end - overlap)

    return chunks
