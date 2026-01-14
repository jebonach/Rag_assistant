from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import regex as re
from rank_bm25 import BM25Okapi


_WORD_RE = re.compile(r"\p{L}+\p{M}*|\p{N}+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return _WORD_RE.findall(text)


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: List[Dict[str, Any]]  # each has chunk_id, page, text, etc.

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        q = tokenize(query)
        scores = self.bm25.get_scores(q)  # numpy array
        # top-k indices by score
        idxs = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
        res = []
        for i in idxs:
            ch = dict(self.chunks[i])
            ch["score"] = float(scores[i])
            res.append(ch)
        return res


def build_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Index:
    tokenized_corpus = [tokenize(ch["text"]) for ch in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return BM25Index(bm25=bm25, chunks=chunks)


def save_bm25(index: BM25Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_bm25(path: Path) -> BM25Index:
    with open(path, "rb") as f:
        return pickle.load(f)
