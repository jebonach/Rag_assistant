from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.chunking import extract_page_number


def split_for_embeddings(
    page_text: str,
    max_chars: int,
    overlap: int,
    page: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Split a page into overlapping subchunks for embeddings.
    The page id is taken from the trailing marker unless provided explicitly.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be < max_chars")

    if page is None:
        page = extract_page_number(page_text)
    if page is None:
        raise ValueError("Page text must end with a '[ N ]' marker or provide page id")

    text = page_text or ""
    out: List[Dict[str, Any]] = []

    start = 0
    sub_id = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        sub_text = text[start:end]
        out.append(
            {
                "page": page,
                "sub_id": sub_id,
                "text": sub_text,
                "start": start,
                "end": end,
            }
        )
        if end >= n:
            break
        start = max(0, end - overlap)
        sub_id += 1

    return out
