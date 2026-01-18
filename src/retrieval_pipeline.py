from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from src.retrievers.bm25 import BM25Index
from src.retrievers.hybrid_rrf import fuse_rrf
from src.retrievers.vector_numpy import VectorIndex, embed_query
from src.retrievers.vector_qdrant import QdrantIndex
from src.embeddings import EmbeddingConfig


Mode = Literal["bm25", "vector", "hybrid"]


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
    vector: Optional[VectorIndex | QdrantIndex] = None,
    emb_cfg: Optional[EmbeddingConfig] = None,
    vector_top_k: Optional[int] = None,
    vector_subchunk_k: Optional[int] = None,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    def _dense_search(k_pages: int) -> List[Dict[str, Any]]:
        if vector is None:
            raise ValueError("Vector retrieval requires vector index")
        if isinstance(vector, QdrantIndex):
            return vector.search(query=query, k=k_pages, top_k_subchunks=vector_subchunk_k)
        if emb_cfg is None:
            raise ValueError("Vector retrieval requires embedding config")
        qv = embed_query(query, emb_cfg)
        return vector.search(query, qv, k=k_pages, subchunk_k=vector_subchunk_k)

    if mode == "bm25":
        hits = bm25.search(query, k=top_k)
        return [format_hit(h) for h in hits]

    if mode == "vector":
        k = vector_top_k or top_k
        hits = _dense_search(k)
        return [format_hit(h) for h in hits]

    if mode == "hybrid":
        if vector is None:
            raise ValueError("Hybrid retrieval requires vector index")
        k_vec = vector_top_k or top_k
        bm25_hits = bm25.search(query, k=top_k)
        vec_hits = _dense_search(k_vec)
        fused = fuse_rrf(page_hits_a=bm25_hits, page_hits_b=vec_hits, k=top_k, rrf_k=rrf_k)
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
        block = f"---\nстр. {h['page']} | {h['chunk_id']}\n{h['text']}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)
