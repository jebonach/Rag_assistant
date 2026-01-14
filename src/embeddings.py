from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from openai import OpenAI


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str
    base_url: Optional[str]
    model: str
    batch_size: int = 64


def make_client(cfg: EmbeddingConfig) -> OpenAI:
    if cfg.base_url:
        return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    return OpenAI(api_key=cfg.api_key)


def embed_texts(texts: List[str], cfg: EmbeddingConfig) -> np.ndarray:
    """
    Returns float32 matrix: (n_texts, dim)
    """
    client = make_client(cfg)
    vectors: List[np.ndarray] = []

    bs = max(1, int(cfg.batch_size))
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        resp = client.embeddings.create(model=cfg.model, input=batch)
        # OpenAI returns in the same order as inputs
        batch_vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        vectors.append(np.stack(batch_vecs, axis=0))

    mat = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0), dtype=np.float32)
    return mat


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, eps, None)
