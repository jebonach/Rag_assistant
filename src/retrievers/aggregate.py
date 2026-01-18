from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def aggregate_subchunk_hits(
    hits: Iterable[Dict[str, Any]],
    page_text_by_page: Dict[int, str],
    page_chunk_id_by_page: Optional[Dict[int, str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    best_by_page: Dict[int, Dict[str, Any]] = {}

    for h in hits:
        page = int(h["page"])
        score = float(h["score"])
        current = best_by_page.get(page)
        if current is None or score > float(current["score"]):
            best_by_page[page] = dict(h)

    out: List[Dict[str, Any]] = []
    for page, row in best_by_page.items():
        page_text = page_text_by_page.get(page, "")
        row = dict(row)
        row["page"] = page
        row["text"] = page_text
        row["best_snippet"] = row.get("snippet", "")
        if page_chunk_id_by_page:
            row["chunk_id"] = page_chunk_id_by_page.get(page, f"p{page}")
        else:
            row["chunk_id"] = row.get("chunk_id", f"p{page}")
        out.append(row)

    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out[:top_k]
