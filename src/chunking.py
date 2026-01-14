from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: int
    start: int
    end: int
    text: str


def chunk_text(
    text: str,
    page: int,
    chunk_size: int,
    chunk_overlap: int,
    prefix: str,
) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    chunks: List[Chunk] = []
    n = len(text)
    start = 0
    idx = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk_txt = text[start:end].strip()
        if chunk_txt:
            chunk_id = f"{prefix}_p{page}_c{idx}"
            chunks.append(Chunk(chunk_id=chunk_id, page=page, start=start, end=end, text=chunk_txt))
            idx += 1
        if end == n:
            break
        start = end - chunk_overlap

    return chunks


def pages_to_chunks(
    pages: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    prefix: str = "book",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in pages:
        page = int(row["page"])
        text = str(row["text"] or "")
        for ch in chunk_text(text, page, chunk_size, chunk_overlap, prefix):
            out.append(
                {
                    "chunk_id": ch.chunk_id,
                    "page": ch.page,
                    "start": ch.start,
                    "end": ch.end,
                    "text": ch.text,
                }
            )
    return out
