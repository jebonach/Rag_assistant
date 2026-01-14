from __future__ import annotations

from typing import Any, Dict, List, Tuple


def rrf_fuse(
    bm25_hits: List[Dict[str, Any]],
    vec_hits: List[Dict[str, Any]],
    k: int,
    rrf_k: int = 60,
    id_key: str = "chunk_id",
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion:
      score(d) = sum_i 1 / (rrf_k + rank_i(d))
    Ranks are 1-based.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Dict[str, Any]] = {}

    for rank, h in enumerate(bm25_hits, start=1):
        cid = str(h[id_key])
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        docs.setdefault(cid, dict(h))

    for rank, h in enumerate(vec_hits, start=1):
        cid = str(h[id_key])
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        docs.setdefault(cid, dict(h))

    merged = []
    for cid, sc in scores.items():
        row = docs[cid]
        row["score_rrf"] = float(sc)
        merged.append(row)

    merged.sort(key=lambda x: x["score_rrf"], reverse=True)
    return merged[:k]
