"""The vector store, and why the flat index is kept beside it.

`DenseIndex` in `retrieval.py` compares a query against every vector. At this corpus size that
is both exact and instant, and it is the baseline that makes the approximate index below
meaningful: the only way to know what HNSW costs in recall is to have the exact answer to
compare it against. A repository that ships only the approximate index can report a recall
figure and cannot tell you what it gave up to get it.

`QdrantStore` is the same interface backed by a real vector database, with payload filtering
— which is the part that matters operationally. A claims adjuster asks about *this* claim, and
searching the whole corpus and discarding other claims afterwards is both slower and wrong:
the top-k is filled with other people's documents before the filter ever runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .documents import Chunk
from .retrieval import Scored

#: Where the collection lives when no URL is given: an on-disk Qdrant with no server. Enough
#: to exercise the real client, the real HNSW index and the real filters in a test.
DEFAULT_PATH = ".qdrant"
COLLECTION = "claims"


@dataclass(frozen=True)
class Filter:
    """What to search within. Every field is optional; all given are required to match."""

    claim_id: str | None = None
    kinds: tuple[str, ...] = ()
    #: Chunks the OCR engine was unsure about. Excluding them raises precision and loses
    #: recall, which is a trade worth being able to make deliberately rather than by accident.
    min_confidence: float | None = None


class QdrantStore:
    """A Qdrant collection of chunks, searchable by vector with a payload filter.

    The client is imported lazily so the package installs and the whole test suite runs
    without it; `pip install claims-intake-rag[serve]` is only needed to actually serve.
    """

    def __init__(self, location: str = DEFAULT_PATH, collection: str = COLLECTION) -> None:
        from qdrant_client import QdrantClient

        # ":memory:" for a test, a path for a local run, a URL for a real deployment. The
        # same three lines cover all of them, which is the only reason the deployment config
        # is not a separate code path that nobody exercises.
        if location.startswith("http"):
            self._client = QdrantClient(url=location)
        elif location == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(path=location)
        self._collection = collection

    def recreate(self, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        # Not `recreate_collection`, which the client deprecated: check, drop, create.
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        points = [
            PointStruct(
                # Qdrant ids are integers or UUIDs, and the chunk id is neither. A stable hash
                # keeps the mapping deterministic across runs; the real id rides in the
                # payload, which is what every consumer reads.
                id=abs(hash(chunk.id)) % (2**63),
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "claim_id": chunk.claim_id,
                    "kind": chunk.kind,
                    "page": chunk.page,
                    "text": chunk.text,
                    "headings": list(chunk.headings),
                    "confidence": chunk.confidence,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(
        self, query_vector: list[float], limit: int = 20, where: Filter | None = None
    ) -> list[Scored]:
        from qdrant_client.models import (
            FieldCondition,
            MatchAny,
            MatchValue,
            Range,
        )
        from qdrant_client.models import Filter as QFilter

        conditions = []
        if where:
            if where.claim_id:
                conditions.append(
                    FieldCondition(key="claim_id", match=MatchValue(value=where.claim_id))
                )
            if where.kinds:
                conditions.append(FieldCondition(key="kind", match=MatchAny(any=list(where.kinds))))
            if where.min_confidence is not None:
                conditions.append(
                    FieldCondition(key="confidence", range=Range(gte=where.min_confidence))
                )

        # `query_points`, not `search`: the client removed `search` outright in 1.19 rather
        # than deprecating it, which is the sort of thing only running the code finds.
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=limit,
            query_filter=QFilter(must=conditions) if conditions else None,
        )
        return [
            Scored(point.payload["chunk_id"], float(point.score))
            for point in response.points
            if point.payload
        ]

    def count(self) -> int:
        return int(self._client.count(collection_name=self._collection).count)

    def close(self) -> None:
        self._client.close()
