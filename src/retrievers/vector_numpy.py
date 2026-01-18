from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.embeddings import EmbeddingConfig, embed_texts, l2_normalize
from src.retrievers.aggregate import aggregate_subchunk_hits
from src.subchunking import split_for_embeddings


def _make_snippet(text: str, max_chars: int = 200) -> str:
    snippet = " ".join((text or "").split())
    if len(snippet) > max_chars:
        return snippet[:max_chars].rstrip() + "..."
    return snippet


def _assert_max_chars(texts: List[str], max_chars: int) -> None:
    if not texts:
        return
    longest = max(len(t) for t in texts)
    if longest > max_chars:
        raise ValueError(f"Embedding input exceeds max_chars={max_chars}: got {longest}")


@dataclass
class VectorIndex:
    emb: np.ndarray                 # shape (n, d), normalized float32
    chunks: List[Dict[str, Any]]    # subchunks aligned with emb rows
    page_text_by_page: Dict[int, str]
    page_chunk_id_by_page: Dict[int, str]
    subchunk_params: Dict[str, int]
    snippet_chars: int = 200

    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        k: int,
        subchunk_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if self.emb.size == 0:
            return []
        # cosine similarity since vectors are normalized
        sims = self.emb @ query_vec.reshape(-1)  # (n,)
        k_sub = subchunk_k or k
        idxs = np.argsort(-sims)[:k_sub]
        sub_hits: List[Dict[str, Any]] = []
        for i in idxs.tolist():
            row = dict(self.chunks[i])
            row["score"] = float(sims[i])
            if not row.get("snippet"):
                row["snippet"] = _make_snippet(row.get("text", ""), self.snippet_chars)
            sub_hits.append(row)
        return aggregate_subchunk_hits(
            hits=sub_hits,
            page_text_by_page=self.page_text_by_page,
            page_chunk_id_by_page=self.page_chunk_id_by_page,
            top_k=k,
        )


def build_vector_index(
    chunks: List[Dict[str, Any]],
    cfg: EmbeddingConfig,
    subchunk_max_chars: int,
    subchunk_overlap: int,
    snippet_chars: int = 200,
) -> VectorIndex:
    page_text_by_page: Dict[int, str] = {}
    page_chunk_id_by_page: Dict[int, str] = {}
    subchunks: List[Dict[str, Any]] = []

    for ch in chunks:
        page = int(ch["page"])
        text = str(ch.get("text") or "")
        if page not in page_text_by_page:
            page_text_by_page[page] = text
            page_chunk_id_by_page[page] = str(ch.get("chunk_id", f"p{page}"))
        if not text.strip():
            continue
        for sub in split_for_embeddings(
            text,
            max_chars=subchunk_max_chars,
            overlap=subchunk_overlap,
            page=page,
        ):
            subchunks.append(sub)

    texts = [str(ch["text"]) for ch in subchunks]
    _assert_max_chars(texts, subchunk_max_chars)
    mat = embed_texts(texts, cfg)
    mat = l2_normalize(mat).astype(np.float32)
    return VectorIndex(
        emb=mat,
        chunks=subchunks,
        page_text_by_page=page_text_by_page,
        page_chunk_id_by_page=page_chunk_id_by_page,
        subchunk_params={"max_chars": subchunk_max_chars, "overlap": subchunk_overlap},
        snippet_chars=snippet_chars,
    )


def embed_query(query: str, cfg: EmbeddingConfig) -> np.ndarray:
    mat = embed_texts([query], cfg)
    mat = l2_normalize(mat).astype(np.float32)
    return mat[0]


def save_vector_index(index: VectorIndex, emb_path: Path, meta_path: Path) -> None:
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, index.emb)
    with open(meta_path, "w", encoding="utf-8") as f:
        pages = []
        for page in sorted(index.page_text_by_page):
            pages.append(
                {
                    "page": int(page),
                    "chunk_id": index.page_chunk_id_by_page.get(page, f"p{page}"),
                    "text": index.page_text_by_page.get(page, ""),
                }
            )
        meta = {
            "format": "subchunk_v1",
            "subchunk_params": index.subchunk_params,
            "snippet_chars": index.snippet_chars,
            "pages": pages,
            "subchunks": index.chunks,
        }
        json.dump(meta, f, ensure_ascii=False)


def load_vector_index(emb_path: Path, meta_path: Path) -> VectorIndex:
    emb = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if isinstance(meta, list):
        pages = meta
        subchunks = []
        page_text_by_page: Dict[int, str] = {}
        page_chunk_id_by_page: Dict[int, str] = {}
        for ch in pages:
            page = int(ch.get("page", 0))
            text = str(ch.get("text") or "")
            subchunks.append(
                {
                    "page": page,
                    "sub_id": 0,
                    "text": text,
                    "start": 0,
                    "end": len(text),
                }
            )
            if page not in page_text_by_page:
                page_text_by_page[page] = text
                page_chunk_id_by_page[page] = str(ch.get("chunk_id", f"p{page}"))
        return VectorIndex(
            emb=emb.astype(np.float32),
            chunks=subchunks,
            page_text_by_page=page_text_by_page,
            page_chunk_id_by_page=page_chunk_id_by_page,
            subchunk_params={},
        )

    pages = meta.get("pages", [])
    subchunks = meta.get("subchunks", [])
    subchunk_params = meta.get("subchunk_params", {})
    snippet_chars = int(meta.get("snippet_chars", 200))

    page_text_by_page = {}
    page_chunk_id_by_page = {}
    for row in pages:
        page = int(row.get("page", 0))
        page_text_by_page[page] = str(row.get("text") or "")
        page_chunk_id_by_page[page] = str(row.get("chunk_id", f"p{page}"))

    if not page_text_by_page and subchunks:
        for row in subchunks:
            page = int(row.get("page", 0))
            if page not in page_text_by_page:
                page_text_by_page[page] = ""
            page_chunk_id_by_page.setdefault(page, f"p{page}")

    return VectorIndex(
        emb=emb.astype(np.float32),
        chunks=subchunks,
        page_text_by_page=page_text_by_page,
        page_chunk_id_by_page=page_chunk_id_by_page,
        subchunk_params=subchunk_params,
        snippet_chars=snippet_chars,
    )
