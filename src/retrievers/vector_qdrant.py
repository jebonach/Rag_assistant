from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, PointStruct

from src.embeddings import EmbeddingConfig, embed_texts, l2_normalize
from src.retrievers.aggregate import aggregate_subchunk_hits
from src.subchunking import split_for_embeddings
from src.vectorstores.qdrant_store import QdrantVectorStore, make_collection_name


def _get_text(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("text") or "")
    return str(getattr(row, "text", ""))


def _get_page(row: Any) -> Optional[int]:
    if isinstance(row, dict):
        page = row.get("page")
    else:
        page = getattr(row, "page", None)
    if page is None:
        return None
    return int(page)


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
class QdrantIndex:
    store: QdrantVectorStore
    emb_cfg: EmbeddingConfig
    page_text_by_page: Dict[int, str]
    page_chunk_id_by_page: Dict[int, str]
    top_k_subchunks: int
    subchunk_max_chars: int
    subchunk_overlap: int
    snippet_chars: int = 200

    def search(
        self,
        query: str,
        k: int,
        top_k_subchunks: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        return dense_search(
            store=self.store,
            query=query,
            emb_cfg=self.emb_cfg,
            page_text_by_page=self.page_text_by_page,
            top_k_pages=k,
            top_k_subchunks=top_k_subchunks or self.top_k_subchunks,
            page_chunk_id_by_page=self.page_chunk_id_by_page,
            score_threshold=score_threshold,
            filter=filter,
        )


def save_qdrant_meta(meta: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_qdrant_meta(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_point_id(page: int, sub_id: int) -> int:
    return int(page) * 1_000_000 + int(sub_id)

def build_qdrant_index(
    client: QdrantClient,
    base_collection: str,
    book_id: str,
    pages: Sequence[Any],
    emb_cfg: EmbeddingConfig,
    subchunk_max_chars: int,
    subchunk_overlap: int,
    recreate_collection: bool = False,
    batch_size: Optional[int] = None,
    snippet_chars: int = 200,
    top_k_subchunks: int = 50,
    meta_path: Optional[Path] = None,
) -> tuple[QdrantIndex, Dict[str, Any]]:
    page_text_by_page: Dict[int, str] = {}
    page_chunk_id_by_page: Dict[int, str] = {}
    subchunk_count = 0

    bs = max(1, int(batch_size or emb_cfg.batch_size))
    batch: List[Dict[str, Any]] = []

    store: Optional[QdrantVectorStore] = None
    collection_name: Optional[str] = None
    embedding_dim: Optional[int] = None

    def _embed_upsert_batch(batch_items: Sequence[Dict[str, Any]]) -> None:
        nonlocal store, collection_name, embedding_dim, subchunk_count
        texts = [str(ch["text"]) for ch in batch_items]
        _assert_max_chars(texts, subchunk_max_chars)
        vectors = embed_texts(texts, emb_cfg)
        if vectors.size == 0:
            return
        vectors = l2_normalize(vectors).astype(np.float32)

        if store is None:
            embedding_dim = int(vectors.shape[1])
            collection_name = make_collection_name(
                base=base_collection,
                book_id=book_id,
                model=emb_cfg.model,
                dim=embedding_dim,
            )
            store = QdrantVectorStore(client=client, collection=collection_name)
            store.ensure_collection(dim=embedding_dim, recreate=recreate_collection)

        points: List[PointStruct] = []
        for vec, ch in zip(vectors, batch_items):
            page = int(ch["page"])
            sub_id = int(ch["sub_id"])
            payload = {
                "page": page,
                "sub_id": sub_id,
                "snippet": _make_snippet(ch["text"], max_chars=snippet_chars),
                "start": int(ch["start"]),
                "end": int(ch["end"]),
            }
            point_id = f"{page}:{sub_id}"
            points.append(PointStruct(
            id=make_point_id(page, sub_id),
            vector=vec.tolist(),
            payload={"page": page, "sub_id": sub_id},
            ))

        store.upsert_points(points)
        subchunk_count += len(batch_items)

    for row in pages:
        page = _get_page(row)
        if page is None:
            raise ValueError("Each page row must include a page id")
        text = _get_text(row)
        if page not in page_text_by_page:
            page_text_by_page[page] = text
            if isinstance(row, dict) and "chunk_id" in row:
                page_chunk_id_by_page[page] = str(row.get("chunk_id"))
            else:
                page_chunk_id_by_page[page] = str(getattr(row, "chunk_id", f"p{page}"))
        if not text.strip():
            continue
        for subchunk in split_for_embeddings(
            text,
            max_chars=subchunk_max_chars,
            overlap=subchunk_overlap,
            page=page,
        ):
            batch.append(subchunk)
            if len(batch) >= bs:
                _embed_upsert_batch(batch)
                batch = []

    if batch:
        _embed_upsert_batch(batch)

    if store is None or collection_name is None or embedding_dim is None:
        raise ValueError("No subchunks to index for Qdrant")

    meta = {
        "collection": collection_name,
        "embedding_model": emb_cfg.model,
        "embedding_dim": embedding_dim,
        "subchunk_params": {"max_chars": subchunk_max_chars, "overlap": subchunk_overlap},
        "n_pages": len(page_text_by_page),
        "n_subchunks": subchunk_count,
    }
    if meta_path:
        save_qdrant_meta(meta, meta_path)

    return (
        QdrantIndex(
            store=store,
            emb_cfg=emb_cfg,
            page_text_by_page=page_text_by_page,
            page_chunk_id_by_page=page_chunk_id_by_page,
            top_k_subchunks=top_k_subchunks,
            subchunk_max_chars=subchunk_max_chars,
            subchunk_overlap=subchunk_overlap,
            snippet_chars=snippet_chars,
        ),
        meta,
    )


def build_dense_store(
    store: QdrantVectorStore,
    pages: Sequence[Any],
    emb_cfg: EmbeddingConfig,
    subchunk_max_chars: int,
    subchunk_overlap: int,
    recreate_collection: bool = False,
    batch_size: Optional[int] = None,
    snippet_chars: int = 200,
) -> None:
    bs = max(1, int(batch_size or emb_cfg.batch_size))
    collection_ready = False

    batch: List[Dict[str, Any]] = []
    for row in pages:
        page = _get_page(row)
        text = _get_text(row)
        if not text.strip():
            continue
        for subchunk in split_for_embeddings(
            text,
            subchunk_max_chars,
            subchunk_overlap,
            page=page,
        ):
            batch.append(subchunk)
            if len(batch) >= bs:
                collection_ready = _embed_and_upsert(
                    store=store,
                    batch=batch,
                    emb_cfg=emb_cfg,
                    subchunk_max_chars=subchunk_max_chars,
                    recreate_collection=recreate_collection,
                    snippet_chars=snippet_chars,
                    collection_ready=collection_ready,
                )
                batch = []

    if batch:
        _embed_and_upsert(
            store=store,
            batch=batch,
            emb_cfg=emb_cfg,
            subchunk_max_chars=subchunk_max_chars,
            recreate_collection=recreate_collection,
            snippet_chars=snippet_chars,
            collection_ready=collection_ready,
        )


def _embed_and_upsert(
    store: QdrantVectorStore,
    batch: Sequence[Dict[str, Any]],
    emb_cfg: EmbeddingConfig,
    subchunk_max_chars: int,
    recreate_collection: bool,
    snippet_chars: int,
    collection_ready: bool,
) -> bool:
    texts = [str(ch["text"]) for ch in batch]
    _assert_max_chars(texts, subchunk_max_chars)
    vectors = embed_texts(texts, emb_cfg)
    if vectors.size == 0:
        return collection_ready
    vectors = l2_normalize(vectors).astype(np.float32)

    if not collection_ready:
        store.ensure_collection(
            dim=int(vectors.shape[1]),
            recreate=recreate_collection,
        )
        collection_ready = True

    points: List[PointStruct] = []
    for vec, ch in zip(vectors, batch):
        page = int(ch["page"])
        sub_id = int(ch["sub_id"])
        payload = {
            "page": page,
            "sub_id": sub_id,
            "snippet": _make_snippet(ch["text"], max_chars=snippet_chars),
            "start": int(ch["start"]),
            "end": int(ch["end"]),
        }
        point_id = f"{page}:{sub_id}"
        points.append(PointStruct(id=point_id, vector=vec.tolist(), payload=payload))

    store.upsert_points(points)
    return collection_ready


def dense_search(
    store: QdrantVectorStore,
    query: str,
    emb_cfg: EmbeddingConfig,
    page_text_by_page: Dict[int, str],
    top_k_pages: int,
    top_k_subchunks: int,
    page_chunk_id_by_page: Optional[Dict[int, str]] = None,
    score_threshold: Optional[float] = None,
    filter: Optional[Filter] = None,
) -> List[Dict[str, Any]]:
    qv = embed_texts([query], emb_cfg)
    qv = l2_normalize(qv).astype(np.float32)
    if qv.size == 0:
        return []

    hits = store.search(
        query_vector=qv[0],
        limit=top_k_subchunks,
        score_threshold=score_threshold,
        query_filter=filter,
    )

    subchunk_hits: List[Dict[str, Any]] = []
    for hit in hits:
        payload = hit.payload or {}
        if "page" not in payload:
            continue
        subchunk_hits.append(
            {
                "page": int(payload.get("page")),
                "sub_id": int(payload.get("sub_id", 0)),
                "snippet": payload.get("snippet", ""),
                "start": payload.get("start"),
                "end": payload.get("end"),
                "score": float(hit.score),
                "point_id": str(hit.id),
            }
        )

    return aggregate_subchunk_hits(
        hits=subchunk_hits,
        page_text_by_page=page_text_by_page,
        page_chunk_id_by_page=page_chunk_id_by_page,
        top_k=top_k_pages,
    )
