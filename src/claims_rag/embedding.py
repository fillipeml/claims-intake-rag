"""Turning text into vectors, and the two reasons this is an interface rather than a call.

**Reproducibility.** The ablation in this repository compares lexical, dense, hybrid and
reranked retrieval on the same queries. If the dense side moved every time a model was
re-downloaded or a library changed its pooling, the comparison would measure the weather. So
the committed vectors are computed once and replayed, and CI exercises the real fusion,
reranking and scoring code against them without a model, a download or a GPU.

**Swappability.** The interesting claims here are about retrieval architecture, not about
which encoder is current this quarter. The encoder is named in the report and can be replaced
without touching anything else.

The shipped vectors come from ``intfloat/multilingual-e5-small``: 118M parameters, 384
dimensions, genuinely multilingual, and small enough that regenerating the corpus is a few
minutes on a CPU. A larger encoder would score better and would make this repository
reproducible only for people who already have the hardware.

One detail that is easy to get wrong and silently costly: **E5 models require a prefix**.
A passage must be embedded as ``passage: ...`` and a question as ``query: ...``. Embed both
sides the same way and retrieval still works well enough to look fine, while giving up a
chunk of the model's trained asymmetry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

#: The prefixes E5 was trained with. Not cosmetic.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def text_key(text: str) -> str:
    """A stable key for a piece of text, so vectors can be stored and replayed by content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


class Embedder(Protocol):
    """Anything that can turn text into vectors."""

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class RecordedEmbedder:
    """Replays vectors computed earlier, keyed by the text they came from.

    Raises on text it has no vector for rather than returning zeros. A zero vector has cosine
    similarity 0 with everything, so an unknown chunk would simply never be retrieved and the
    recall figure would quietly fall without anything reporting an error.
    """

    def __init__(self, name: str, dimension: int, vectors: dict[str, list[float]]) -> None:
        self._name = name
        self._dimension = dimension
        self._vectors = vectors

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension

    @classmethod
    def load(cls, path: str | Path) -> RecordedEmbedder:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=str(payload["model"]),
            dimension=int(payload["dimension"]),
            vectors={str(k): [float(x) for x in v] for k, v in payload["vectors"].items()},
        )

    def _lookup(self, text: str, prefix: str) -> list[float]:
        key = text_key(prefix + text)
        vector = self._vectors.get(key)
        if vector is None:
            raise KeyError(
                f"no recorded vector for {prefix.strip()} {text[:60]!r}. A recorded embedder "
                "answers only for what it recorded; regenerate the vectors after changing a "
                "chunk or adding a query."
            )
        return vector

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._lookup(t, PASSAGE_PREFIX) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._lookup(text, QUERY_PREFIX)


class SentenceTransformerEmbedder:
    """The real encoder. Only needed to regenerate vectors or to serve live documents.

    Imported lazily so that the package, the demo and the whole test suite stay installable
    and runnable without torch.
    """

    def __init__(self, model: str = "intfloat/multilingual-e5-small") -> None:
        from sentence_transformers import SentenceTransformer

        self._name = model
        self._model = SentenceTransformer(model)
        self._dimension = int(self._model.get_sentence_embedding_dimension() or 0)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Normalised, because every consumer here compares with cosine and normalising once at
        # encode time is cheaper and less error-prone than normalising at every comparison.
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in row] for row in vectors]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode([PASSAGE_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([QUERY_PREFIX + text])[0]


def save_vectors(
    path: str | Path, model: str, dimension: int, vectors: dict[str, list[float]]
) -> None:
    """Writes the recording. Rounded to six decimals: the file is committed and read by people,
    and the difference is far below anything that changes a ranking."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            {
                "model": model,
                "dimension": dimension,
                "vectors": {k: [round(x, 6) for x in v] for k, v in vectors.items()},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
