"""Finding the right chunk: lexical, dense, both, and both plus a reranker.

The pipeline production retrieval converged on is three steps, and each one fixes a failure
the step before it cannot:

1. **Hybrid search** fixes *the answer was never retrieved*. Dense search matches meaning and
   misses identifiers; BM25 matches identifiers and misses paraphrase. A claim corpus is full
   of both — "AV-2026-004417" and "o carro capotou na alça de acesso" — so either alone has a
   recall ceiling that no amount of tuning moves.
2. **Rank fusion** merges the two without pretending their scores are comparable. A cosine
   similarity and a BM25 score live on different scales and are not made commensurable by
   normalising them; RRF ignores the scores entirely and fuses the *ranks*.
3. **Reranking** fixes *the answer was retrieved but buried*. A cross-encoder reads the query
   and the chunk together instead of comparing two vectors computed apart, which is far more
   accurate and far too slow to run over a corpus — so it runs over the shortlist.

BM25 is implemented here rather than imported. It is thirty lines, it has to tokenise
Portuguese the way this corpus needs, and a retrieval repository that cannot show its lexical
scoring is not showing very much.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .documents import Chunk

#: Reciprocal Rank Fusion's only parameter, at its conventional value. It damps the
#: contribution of the very top ranks so one confident list cannot dominate the other.
RRF_K = 60

#: Enough Portuguese function words to stop them dominating the term statistics, and no more.
#: A larger list starts removing words that carry meaning in a claim ("sem", "contra", "após").
_STOPWORDS = frozenset(
    # fmt: off
    [
        "a",
        "o",
        "e",
        "as",
        "os",
        "de",
        "da",
        "do",
        "das",
        "dos",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "um",
        "uma",
        "uns",
        "umas",
        "por",
        "para",
        "com",
        "que",
        "se",
        "ao",
        "aos",
        "à",
        "às",
        "pelo",
        "pela",
        "pelos",
        "pelas",
        "este",
        "esta",
        "estes",
        "estas",
        "esse",
        "essa",
        "isso",
        "aquilo",
        "foi",
        "era",
        "ser",
        "sido",
        "sendo",
        "tem",
        "ter",
        "teve",
        "havia",
        "há",
        "são",
        "está",
        "estão",
    ]
    # fmt: on
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    """Lowercased, accent-free, stopwords dropped.

    Accents are stripped on purpose, and the reason is OCR. A scanner that renders "orçamento"
    as "orcamento" or "veículo" as "veiculo" is not a rare event, it is most pages; a lexical
    index that requires the diacritic to match would miss exactly the documents this system
    exists to read. The cost is that a handful of Portuguese word pairs collapse together,
    which is a far smaller error than the one it prevents.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    folded = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return [t for t in _TOKEN.findall(folded) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Scored:
    """One chunk and what a retriever thought of it."""

    chunk_id: str
    score: float


# --------------------------------------------------------------------------------------
# Lexical
# --------------------------------------------------------------------------------------


class Bm25Index:
    """Okapi BM25 over the chunk texts.

    ``k1`` controls how fast term frequency saturates and ``b`` how much document length is
    penalised; the values are the usual ones and are exposed because a corpus of short table
    rows next to long narrative paragraphs is exactly where they matter.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunk_ids = [c.id for c in chunks]
        self._tokens = [tokenise(c.embedding_text) for c in chunks]
        self._lengths = [len(t) for t in self._tokens]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._frequencies = [Counter(t) for t in self._tokens]

        document_count = len(chunks)
        containing: Counter[str] = Counter()
        for counts in self._frequencies:
            containing.update(counts.keys())
        # The BM25 IDF with the +1 inside the logarithm, which keeps it non-negative for terms
        # present in more than half the corpus instead of letting them score below zero.
        self._idf = {
            term: math.log(1 + (document_count - n + 0.5) / (n + 0.5))
            for term, n in containing.items()
        }

    def search(self, query: str, limit: int = 20) -> list[Scored]:
        terms = tokenise(query)
        if not terms or not self.chunk_ids:
            return []

        scored: list[Scored] = []
        for position, chunk_id in enumerate(self.chunk_ids):
            counts = self._frequencies[position]
            length = self._lengths[position]
            total = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * (length / self._avg_length if self._avg_length else 1.0)
                )
                total += self._idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            if total > 0:
                scored.append(Scored(chunk_id, total))

        scored.sort(key=lambda s: (-s.score, s.chunk_id))
        return scored[:limit]


# --------------------------------------------------------------------------------------
# Dense
# --------------------------------------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vectors of different length: {len(a)} and {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DenseIndex:
    """Cosine similarity over precomputed chunk vectors.

    Flat and exhaustive on purpose. A corpus of this size has no use for an approximate index,
    and an exact baseline is what tells you what an approximate one costs in recall when the
    corpus is big enough to need one. `qdrant.py` is the same interface backed by a real
    vector database for when it is.
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        if not vectors:
            raise ValueError("a dense index needs at least one vector")
        self._vectors = vectors
        sizes = {len(v) for v in vectors.values()}
        if len(sizes) > 1:
            raise ValueError(f"vectors of mixed dimension: {sorted(sizes)}")
        self.dimension = sizes.pop()

    def search(self, query_vector: list[float], limit: int = 20) -> list[Scored]:
        scored = [
            Scored(chunk_id, cosine(query_vector, vector))
            for chunk_id, vector in self._vectors.items()
        ]
        scored.sort(key=lambda s: (-s.score, s.chunk_id))
        return scored[:limit]


# --------------------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------------------


def reciprocal_rank_fusion(rankings: list[list[Scored]], k: int = RRF_K) -> list[Scored]:
    """Fuse ranked lists by position, ignoring their scores.

    This is the step people most often replace with a weighted sum of normalised scores, and
    it is the step where that goes wrong. A cosine similarity is bounded in [-1, 1] and
    clusters tightly; a BM25 score is unbounded and depends on corpus statistics. Min-max
    normalising them makes the numbers the same shape without making them mean the same thing,
    and the weight that results is tuned on one query set and silently wrong on the next.

    RRF throws the scores away. A chunk ranked 1 by one retriever and 40 by the other beats a
    chunk ranked 8 by both, which is the behaviour that makes hybrid search work.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, scored in enumerate(ranking, start=1):
            fused[scored.chunk_id] = fused.get(scored.chunk_id, 0.0) + 1.0 / (k + position)

    out = [Scored(chunk_id, score) for chunk_id, score in fused.items()]
    out.sort(key=lambda s: (-s.score, s.chunk_id))
    return out


# --------------------------------------------------------------------------------------
# Reranking
# --------------------------------------------------------------------------------------


class Reranker(Protocol):
    """Reads the query and the chunk together, and reorders the shortlist."""

    @property
    def name(self) -> str: ...

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Scored]: ...


class RecordedReranker:
    """Replays cross-encoder scores captured earlier.

    What makes the demo, the tests and CI run with no model and no GPU while still exercising
    the real fusion and scoring code. It raises on a pair it has no score for rather than
    returning zero, because a reranker that silently scores an unseen chunk as irrelevant
    would quietly remove it from every result and every metric.
    """

    def __init__(self, name: str, scores: dict[str, dict[str, float]]) -> None:
        self._name = name
        self._scores = scores

    @property
    def name(self) -> str:
        return self._name

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Scored]:
        row = self._scores.get(query)
        if row is None:
            raise KeyError(
                f"no recorded reranker scores for query {query[:60]!r}. A recorded reranker "
                "answers only for what it recorded, and never guesses."
            )
        missing = [c.id for c in chunks if c.id not in row]
        if missing:
            raise KeyError(
                f"recorded reranker {self._name!r} has no score for {len(missing)} chunk(s) of "
                f"this query: {missing[:3]}"
            )
        scored = [Scored(c.id, row[c.id]) for c in chunks]
        scored.sort(key=lambda s: (-s.score, s.chunk_id))
        return scored


# --------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalConfig:
    """One point in the ablation."""

    name: str
    use_dense: bool = True
    use_lexical: bool = True
    rerank: bool = False
    #: How many candidates each retriever contributes before fusion. Larger costs nothing in a
    #: flat index and costs latency in a real one; it is the main recall/latency dial.
    candidates: int = 20


class HybridRetriever:
    """Dense, lexical, fused, optionally reranked."""

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: dict[str, list[float]] | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.chunks = {c.id: c for c in chunks}
        self.lexical = Bm25Index(chunks)
        self.dense = DenseIndex(vectors) if vectors else None
        self.reranker = reranker

    def search(
        self,
        query: str,
        config: RetrievalConfig,
        query_vector: list[float] | None = None,
        limit: int = 10,
    ) -> list[Scored]:
        rankings: list[list[Scored]] = []

        if config.use_lexical:
            rankings.append(self.lexical.search(query, config.candidates))

        if config.use_dense:
            if self.dense is None:
                raise ValueError(f"{config.name} asks for dense search but no vectors were given")
            if query_vector is None:
                raise ValueError(
                    f"{config.name} asks for dense search but no query vector was given"
                )
            rankings.append(self.dense.search(query_vector, config.candidates))

        if not rankings:
            raise ValueError(f"{config.name} disables every retriever, so there is nothing to run")

        # One retriever needs no fusion, and running RRF over a single list would quietly
        # replace its scores with reciprocal ranks — harmless for the order, misleading for
        # anything that reads the score.
        fused = rankings[0] if len(rankings) == 1 else reciprocal_rank_fusion(rankings)

        if config.rerank:
            if self.reranker is None:
                raise ValueError(f"{config.name} asks for reranking but no reranker was given")
            shortlist = [self.chunks[s.chunk_id] for s in fused[: config.candidates]]
            fused = self.reranker.rerank(query, shortlist)

        return fused[:limit]
