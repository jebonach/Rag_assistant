from __future__ import annotations

from typing import List, Dict, Any
import regex as re

_PAGE_MARK_RE = re.compile(r"\[\s*(\d+)\s*\]\s*$")


def extract_page_number(text: str) -> int | None:
    """
    Extracts trailing page marker [ N ] from the end of text.
    Returns N if found else None.
    """
    t = (text or "").strip()
    m = _PAGE_MARK_RE.search(t)
    if not m:
        return None
    return int(m.group(1))


def validate_page_marker(page: int, text: str) -> None:
    n = extract_page_number(text)
    if n is None:
        raise ValueError(f"Page {page}: missing trailing marker '[ N ]'.")
    if n != page:
        raise ValueError(f"Page {page}: marker mismatch, got [ {n} ] at end.")


def pages_to_page_chunks(
    pages: List[Dict[str, Any]],
    prefix: str = "book",
    strict: bool = True,
) -> List[Dict[str, Any]]:
    """
    Strict chunking: each page becomes exactly one chunk.
    Requires that text ends with marker [ page ] if strict=True.
    """
    out: List[Dict[str, Any]] = []
    for row in pages:
        page = int(row["page"])
        text = str(row.get("text") or "")
        if strict:
            validate_page_marker(page, text)

        out.append(
            {
                "chunk_id": f"{prefix}_p{page}",
                "page": page,
                "start": 0,
                "end": len(text),
                "text": text,
            }
        )
    return out
