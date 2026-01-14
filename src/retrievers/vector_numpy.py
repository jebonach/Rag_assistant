from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.embeddings import EmbeddingConfig, embed_texts, l2_normalize


@dataclass
class VectorIndex:
    emb: np.ndarray                 # shape (n, d), normalized float32
    chunks: List[Dict[str, Any]]    # aligned with emb rows

    def search(self, query: str, query_vec: np.ndarray, k: int) -> List[Dict[str, Any]]:
        if self.emb.size == 0:
            return []
        # cosine similarity since vectors are normalized
        sims = self.emb @ query_vec.reshape(-1)  # (n,)
        idxs = np.argsort(-sims)[:k]
        out: List[Dict[str, Any]] = []
        for i in idxs.tolist():
            row = dict(self.chunks[i])
            row["score"] = float(sims[i])
            out.append(row)
        return out


def build_vector_index(chunks: List[Dict[str, Any]], cfg: EmbeddingConfig) -> VectorIndex:
    texts = [str(ch["text"]) for ch in chunks]
    mat = embed_texts(texts, cfg)
    mat = l2_normalize(mat).astype(np.float32)
    return VectorIndex(emb=mat, chunks=chunks)


def embed_query(query: str, cfg: EmbeddingConfig) -> np.ndarray:
    mat = embed_texts([query], cfg)
    mat = l2_normalize(mat).astype(np.float32)
    return mat[0]


def save_vector_index(index: VectorIndex, emb_path: Path, meta_path: Path) -> None:
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, index.emb)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(index.chunks, f, ensure_ascii=False)


def load_vector_index(emb_path: Path, meta_path: Path) -> VectorIndex:
    emb = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return VectorIndex(emb=emb.astype(np.float32), chunks=chunks)
