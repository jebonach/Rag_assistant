from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from src.retrievers.bm25 import BM25Index
from src.retrievers.hybrid_rrf import rrf_fuse
from src.retrievers.vector_numpy import VectorIndex, embed_query
from src.embeddings import EmbeddingConfig


Mode = Literal["bm25", "fulltext", "vector", "hybrid"]


def format_hit(hit: Dict[str, Any], score_key: str = "score") -> Dict[str, Any]:
    return {
        "chunk_id": hit["chunk_id"],
        "page": int(hit["page"]),
        "score": float(hit.get(score_key, hit.get("score", 0.0))),
        "text": hit["text"],
    }


def retrieve(
    query: str,
    mode: Mode,
    top_k: int,
    bm25: BM25Index,
    vector: Optional[VectorIndex] = None,
    emb_cfg: Optional[EmbeddingConfig] = None,
    vector_top_k: Optional[int] = None,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    if mode in ("bm25", "fulltext"):
        hits = bm25.search(query, k=top_k)
        return [format_hit(h) for h in hits]

    if mode == "vector":
        if vector is None or emb_cfg is None:
            raise ValueError("Vector retrieval requires vector index and embedding config")
        k = vector_top_k or top_k
        qv = embed_query(query, emb_cfg)
        hits = vector.search(query, qv, k=k)
        return [format_hit(h) for h in hits]

    if mode == "hybrid":
        if vector is None or emb_cfg is None:
            raise ValueError("Hybrid retrieval requires vector index and embedding config")
        k_vec = vector_top_k or top_k
        bm25_hits = bm25.search(query, k=top_k)
        qv = embed_query(query, emb_cfg)
        vec_hits = vector.search(query, qv, k=k_vec)
        fused = rrf_fuse(bm25_hits=bm25_hits, vec_hits=vec_hits, k=top_k, rrf_k=rrf_k)
        # fused uses score_rrf
        return [format_hit(h, score_key="score_rrf") for h in fused]

    raise ValueError(f"Unknown mode: {mode}")


def build_context(hits: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """
    Concatenate retrieved pages into a context block (future: for LLM prompt).
    Keeps page numbers to preserve citation ability.
    """
    parts = []
    total = 0
    for h in hits:
        block = f"---\nPAGE {h['page']} | {h['chunk_id']}\n{h['text']}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)
