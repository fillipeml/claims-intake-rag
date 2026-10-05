"""Answering, refusing, and measuring how long the first token took.

Two things here are not decoration.

**The refusal.** A retrieval system asked something its corpus cannot answer will happily
return its five least-bad chunks, and a model handed those will write a fluent answer from
them. In a claims file that is the worst available behaviour: the adjuster gets a number with
a citation and the citation does not support it. So the answerer declines when retrieval is
weak — and the threshold it declines at is **measured**, not picked. `claims calibrate` runs
the answerable and unanswerable queries through retrieval, prints the score distributions, and
reports what each candidate threshold would cost in missed answers and buy in refused
nonsense.

**Time to first token.** TTFT is the number a user feels and the one a mean latency hides: a
pipeline that retrieves for 900ms and then streams fast reads as broken, while one that starts
in 200ms and streams slowly reads as working. It is measured here rather than asserted, which
is why answering is a generator and not a function returning a string.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from .documents import Chunk
from .retrieval import Scored, tokenise


@dataclass(frozen=True)
class Answer:
    """What came back, with everything needed to check it."""

    text: str
    citations: tuple[str, ...]
    refused: bool
    reason: str = ""
    #: Milliseconds from the request arriving to the first character being emitted.
    ttft_ms: float = 0.0
    #: Milliseconds to the last character.
    total_ms: float = 0.0


class Answerer(Protocol):
    """Anything that turns a question and its evidence into a stream of text."""

    @property
    def name(self) -> str: ...

    def stream(self, question: str, chunks: list[Chunk]) -> Iterator[str]: ...


class ExtractiveAnswerer:
    """Composes the answer out of the retrieved text, with no model at all.

    Not a placeholder for a model: it is the baseline a model has to beat. On a claim file most
    questions are answered by one line of one document — a total, a date, a deductible — and
    quoting that line with its provenance is both correct and impossible to hallucinate. When a
    generative answerer is wired in, this is what its added value is measured against.
    """

    name = "extractive"

    def stream(self, question: str, chunks: list[Chunk]) -> Iterator[str]:
        if not chunks:
            return
        terms = set(tokenise(question))
        best = chunks[0]

        where = " > ".join(best.headings) if best.headings else best.kind
        yield f"Segundo {best.document_id} (p. {best.page}, {where}):\n\n"
        for line in best.text.splitlines():
            yield line + "\n"

        supporting = [
            c
            for c in chunks[1:4]
            if terms & set(tokenise(c.text)) and c.document_id != best.document_id
        ]
        if supporting:
            yield "\nOutros documentos do mesmo sinistro mencionam o assunto:\n"
            for chunk in supporting:
                yield f"- {chunk.cite()}\n"


@dataclass
class GroundingPolicy:
    """When the system is allowed to answer.

    ``min_score`` is the fused-or-reranked score the best chunk must reach. Its default is the
    value `claims calibrate` selected on the shipped golden set; it is a property of this
    corpus and this encoder and should be recalibrated for another, which is why it is a
    parameter and not a constant buried in a function.

    ``min_overlap`` is a second, cheaper guard: at least one content word of the question must
    appear in the chunk being quoted. It costs nothing and catches the case where a dense
    retriever returns something topically adjacent and lexically unrelated.
    """

    #: Chosen from the table `claims calibrate` prints, not by feel — and measured against the
    #: **dense cosine**, not the reranker score, which is the finding that shaped this class.
    #:
    #: The cross-encoder is the better ranker and the worse gate. Asked to separate the 64
    #: answerable questions from the 8 unanswerable ones, its top score gives the answerable ones a median
    #: of 0.016 against an unanswerable maximum of 0.0066: refusing every unanswerable question
    #: costs half the answerable ones. The dense cosine, which ranks worse, separates cleanly —
    #: answerable in [0.840, 0.916], unanswerable in [0.828, 0.847] — so 0.853 refuses 8 of 8
    #: while keeping 51 of 64.
    #:
    #: Ordering and deciding-whether-to-answer are different jobs, and the signal that is best
    #: at one is not best at the other.
    min_score: float = 0.853

    #: Off by default, after measuring what it costs.
    #:
    #: The idea was cheap insurance: require at least one content word of the question to
    #: appear in the chunk being quoted, to catch a dense hit that is topically adjacent and
    #: lexically unrelated. Measured on the 64 golden questions it refuses 38 answers, and 8
    #: of those were the correct chunk — a third of every correct answer the system produces.
    #:
    #: The reason is the reason dense retrieval exists. Asked "o motorista podia dirigir no
    #: dia do acidente", the right chunk reads `Habilitação: válida`, and the two share no
    #: word at all. A lexical floor under a semantic retriever punishes it for working.
    #:
    #: It does raise top-1 precision from 41% to 69%, so it is kept and configurable for work
    #: where a wrong answer costs more than a missing one. It is not the default.
    min_overlap: int = 0
    provenance: str = (
        "claims calibrate, dense cosine on the shipped corpus: 8/8 refused, 51/64 kept"
    )


@dataclass
class Grounded:
    """The decision, and why."""

    allowed: bool
    reason: str = ""
    best_score: float = 0.0
    overlap: int = 0


def check_grounding(
    question: str,
    hits: list[Scored],
    chunks: dict[str, Chunk],
    policy: GroundingPolicy,
    grounding_score: float | None = None,
) -> Grounded:
    """`grounding_score` is the signal the policy was calibrated against.

    It is separate from `hits[0].score` on purpose. After reranking, the top hit carries a
    cross-encoder score, and that score is a poor gate even though the reranker is the better
    ranker. The caller passes the dense cosine here and lets the reranker do the ordering.
    """
    if not hits:
        return Grounded(False, "retrieval returned nothing for this question")

    best = hits[0]
    chunk = chunks.get(best.chunk_id)
    if chunk is None:
        return Grounded(False, f"the top hit {best.chunk_id} is not in the corpus")

    overlap = len(set(tokenise(question)) & set(tokenise(chunk.embedding_text)))
    score = best.score if grounding_score is None else grounding_score

    if score < policy.min_score:
        return Grounded(
            False,
            f"the best match scored {score:.4f}, below the {policy.min_score:.4f} this "
            f"corpus was calibrated at ({policy.provenance})",
            score,
            overlap,
        )
    if overlap < policy.min_overlap:
        return Grounded(
            False,
            "the best match shares no content word with the question, which is what a "
            "topically-adjacent but unrelated passage looks like",
            score,
            overlap,
        )
    return Grounded(True, "", score, overlap)


@dataclass
class Timing:
    """Where the milliseconds went. Every field is measured, none is estimated."""

    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    chunks_considered: int = 0
    stages: dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"retrieval {self.retrieval_ms:.0f}ms, rerank {self.rerank_ms:.0f}ms, "
            f"first token at {self.ttft_ms:.0f}ms, complete at {self.total_ms:.0f}ms"
        )


def answer(
    question: str,
    hits: list[Scored],
    chunks: dict[str, Chunk],
    answerer: Answerer,
    policy: GroundingPolicy,
    started: float | None = None,
    grounding_score: float | None = None,
) -> tuple[Answer, Timing]:
    """Collects the stream. The streaming path is :func:`stream_answer`; this is for tests,
    the CLI and anything that wants the whole thing before deciding what to do with it."""
    started = started if started is not None else time.perf_counter()
    pieces: list[str] = []
    ttft = 0.0
    refusal: Answer | None = None

    for event in stream_answer(question, hits, chunks, answerer, policy, started, grounding_score):
        if isinstance(event, Answer):
            refusal = event
            break
        if not pieces:
            ttft = (time.perf_counter() - started) * 1000
        pieces.append(event)

    total = (time.perf_counter() - started) * 1000
    if refusal is not None:
        return refusal, Timing(ttft_ms=total, total_ms=total, chunks_considered=len(hits))

    cited = tuple(dict.fromkeys(h.chunk_id for h in hits[:4] if h.chunk_id in chunks))
    return (
        Answer(
            text="".join(pieces),
            citations=cited,
            refused=False,
            ttft_ms=ttft,
            total_ms=total,
        ),
        Timing(ttft_ms=ttft, total_ms=total, chunks_considered=len(hits)),
    )


def stream_answer(
    question: str,
    hits: list[Scored],
    chunks: dict[str, Chunk],
    answerer: Answerer,
    policy: GroundingPolicy,
    started: float | None = None,
    grounding_score: float | None = None,
) -> Iterator[str | Answer]:
    """Yields text pieces, or a single :class:`Answer` when the question is refused.

    The refusal is a value rather than an exception because it is an ordinary outcome, and
    because an HTTP stream that has already begun cannot raise at the client usefully.
    """
    started = started if started is not None else time.perf_counter()
    grounded = check_grounding(question, hits, chunks, policy, grounding_score)
    if not grounded.allowed:
        elapsed = (time.perf_counter() - started) * 1000
        yield Answer(
            text=(
                "Não há base suficiente nos documentos deste sinistro para responder a essa "
                f"pergunta. Motivo: {grounded.reason}."
            ),
            citations=(),
            refused=True,
            reason=grounded.reason,
            ttft_ms=elapsed,
            total_ms=elapsed,
        )
        return

    ordered = [chunks[h.chunk_id] for h in hits if h.chunk_id in chunks]
    yield from answerer.stream(question, ordered)
