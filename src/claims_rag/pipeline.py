"""Assembling the index from what is committed.

One place builds the corpus, so the evaluation, the calibration and the service are measuring
and serving the same thing. The subtle part is that **the index holds re-segmented text**: the
chunks on disk are raw OCR output, and embedding the merged text while searching the segmented
text would compare two different corpora and report the difference as a retrieval result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .documents import Chunk
from .embedding import RecordedEmbedder
from .metrics import Query
from .retrieval import HybridRetriever, RecordedReranker
from .segmentation import WordSegmenter

DEFAULT_FIXTURES = Path("fixtures")


def load_chunks(path: str | Path) -> list[Chunk]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Chunk(
            id=row["id"],
            document_id=row["document_id"],
            claim_id=row["claim_id"],
            kind=row["kind"],
            page=row["page"],
            text=row["text"],
            headings=tuple(row["headings"]),
            confidence=float(row["confidence"]),
        )
        for row in payload["chunks"]
    ]


@dataclass
class Corpus:
    """Everything an evaluation or a request needs, built once."""

    chunks: list[Chunk]
    retriever: HybridRetriever
    embedder: RecordedEmbedder
    segmenter: WordSegmenter
    queries: list[Query]
    claim_of: dict[str, str]
    unanswerable: list[str]

    @property
    def by_id(self) -> dict[str, Chunk]:
        return {c.id: c for c in self.chunks}


def build(fixtures: str | Path = DEFAULT_FIXTURES) -> Corpus:
    root = Path(fixtures)
    raw = load_chunks(root / "chunks.json")
    segmenter = WordSegmenter.from_texts([c.text for c in raw])

    # What the index holds. Headings are re-segmented too, because they are prepended to the
    # embedded text and a merged heading would make every chunk under it harder to find.
    indexed = [
        replace(
            c,
            text=segmenter.resegment(c.text),
            headings=tuple(segmenter.resegment(h) for h in c.headings),
        )
        for c in raw
    ]

    embedder = RecordedEmbedder.load(root / "vectors.json")
    vectors = {c.id: embedder.embed_passages([c.embedding_text])[0] for c in indexed}

    reranker = None
    reranker_path = root / "reranker.json"
    if reranker_path.exists():
        payload = json.loads(reranker_path.read_text(encoding="utf-8"))
        reranker = RecordedReranker(payload["model"], payload["scores"])

    queries: list[Query] = []
    claim_of: dict[str, str] = {}
    unanswerable: list[str] = []
    queries_path = root / "queries.json"
    if queries_path.exists():
        payload = json.loads(queries_path.read_text(encoding="utf-8"))
        for row in payload["queries"]:
            queries.append(
                Query(
                    id=row["id"],
                    text=row["text"],
                    relevant=frozenset(row["relevant"]),
                    tags=tuple(row["tags"]),
                )
            )
            claim_of[row["id"]] = row["claim_id"]
        unanswerable = list(payload.get("unanswerable", []))

    return Corpus(
        chunks=indexed,
        retriever=HybridRetriever(indexed, vectors, reranker),
        embedder=embedder,
        segmenter=segmenter,
        queries=queries,
        claim_of=claim_of,
        unanswerable=unanswerable,
    )
