from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams


def get_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)


_COLL_SAFE_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _sanitize_collection_part(text: str) -> str:
    cleaned = _COLL_SAFE_RE.sub("-", (text or "").strip())
    cleaned = cleaned.strip("-_").lower()
    return cleaned or "default"


def make_collection_name(base: str, book_id: str, model: str, dim: int) -> str:
    parts = [
        _sanitize_collection_part(base),
        _sanitize_collection_part(book_id),
        _sanitize_collection_part(model),
        str(int(dim)),
    ]
    return "_".join([p for p in parts if p])


def _get_vector_dim(info) -> int:
    vectors = info.config.params.vectors
    if hasattr(vectors, "size"):
        return int(vectors.size)
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()))
        return int(first.size)
    raise ValueError("Unsupported Qdrant vectors config format")


def ensure_collection(
    client: QdrantClient,
    name: str,
    dim: int,
    distance: Distance = Distance.COSINE,
    recreate: bool = False,
) -> None:
    if recreate:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=distance),
        )
        return

    if client.collection_exists(name):
        info = client.get_collection(name)
        existing_dim = _get_vector_dim(info)
        if existing_dim != dim:
            raise ValueError(
                f"Collection '{name}' has dim={existing_dim}, expected dim={dim}."
            )
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=distance),
    )


def upsert_points(
    client: QdrantClient,
    collection: str,
    points: Sequence[PointStruct],
) -> None:
    if not points:
        return
    client.upsert(collection_name=collection, points=list(points))


def search(
    client: QdrantClient,
    collection: str,
    query_vector: Sequence[float],
    limit: int,
    score_threshold: Optional[float] = None,
    filter: Optional[Filter] = None,
):
    return client.search(
        collection_name=collection,
        query_vector=list(query_vector),
        limit=limit,
        score_threshold=score_threshold,
        query_filter=filter,
        with_payload=True,
        with_vectors=False,
    )


@dataclass
class QdrantVectorStore:
    client: QdrantClient
    collection: str

    @classmethod
    def from_url(cls, url: str, collection: str) -> "QdrantVectorStore":
        return cls(client=get_client(url), collection=collection)

    def ensure_collection(
        self,
        dim: int,
        distance: Distance = Distance.COSINE,
        recreate: bool = False,
    ) -> None:
        ensure_collection(
            client=self.client,
            name=self.collection,
            dim=dim,
            distance=distance,
            recreate=recreate,
        )

    def upsert_points(self, points: Sequence[PointStruct]) -> None:
        upsert_points(self.client, self.collection, points)

    def search(
        self,
        query_vector: Sequence[float],
        limit: int,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Filter] = None,
    ):
        return search(
            client=self.client,
            collection=self.collection,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            filter=query_filter,
        )
