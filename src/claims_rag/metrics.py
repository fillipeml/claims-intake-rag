"""What the retrieval actually achieved, measured against a golden set.

Most retrieval work is tuned by reading a handful of results and forming an impression. That
works until the day a change helps the queries you look at and hurts the ones you do not, and
nothing in the system can tell you so.

Four numbers, each answering a different question:

* **Hit@k** — did *any* relevant chunk make the top k? The question that matters when the
  model only needs one good passage to answer.
* **Recall@k** — what *share* of the relevant chunks made the top k? The question that matters
  when an answer has to reconcile a figure in an estimate with a date in a police report.
* **MRR** — how far down was the first relevant chunk? Sensitive to rank in a way recall is not.
* **nDCG@k** — the whole ranking, discounted by position.

Hit@k is a proportion over queries, so it carries a Wilson interval. The other three are means
of per-query values, so they carry a bootstrap interval. Using Wilson on a mean would compute
cleanly and be wrong, which is the only reason the distinction is worth this paragraph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .stats import Interval, PairedComparison, bootstrap_mean, sign_test, wilson


@dataclass(frozen=True)
class Query:
    """One question, and the chunks a person decided actually answer it."""

    id: str
    text: str
    #: Chunk ids judged relevant. More than one is normal and is the point of Recall@k: a
    #: claim question is usually answered by two documents that have to agree.
    relevant: frozenset[str]
    #: Free tags for slicing the report — which document kind the answer lives in, whether the
    #: question is about a figure or a date.
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("every query needs an id")
        if not self.relevant:
            raise ValueError(
                f"query {self.id!r} has no relevant chunk. A query nothing answers measures the "
                "corpus, not the retriever, and silently drags every mean down."
            )


@dataclass(frozen=True)
class Ranking:
    """What one configuration returned for one query, best first."""

    query_id: str
    chunk_ids: tuple[str, ...]


def hit_at_k(ranking: Ranking, relevant: frozenset[str], k: int) -> bool:
    """Did anything relevant reach the top k."""
    return any(c in relevant for c in ranking.chunk_ids[:k])


def recall_at_k(ranking: Ranking, relevant: frozenset[str], k: int) -> float:
    """What share of the relevant chunks reached the top k."""
    if not relevant:
        return 0.0
    found = sum(1 for c in ranking.chunk_ids[:k] if c in relevant)
    return found / len(relevant)


def reciprocal_rank(ranking: Ranking, relevant: frozenset[str]) -> float:
    """1/rank of the first relevant chunk, or 0 if none was retrieved at all."""
    for position, chunk_id in enumerate(ranking.chunk_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranking: Ranking, relevant: frozenset[str], k: int) -> float:
    """Normalised discounted cumulative gain, binary relevance.

    The ideal ranking is every relevant chunk first, which is why the denominator stops at
    ``min(k, len(relevant))`` — with three relevant chunks and k=5, perfect is three hits in
    the first three places, not five.
    """
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(ranking.chunk_ids[:k], start=1)
        if chunk_id in relevant
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(k, len(relevant)) + 1))
    return gain / ideal if ideal else 0.0


@dataclass(frozen=True)
class RetrievalReport:
    """One configuration, scored over one golden set."""

    configuration: str
    queries: int
    k: int
    hit: Interval
    recall: Interval
    mrr: Interval
    ndcg: Interval
    #: Per-query nDCG, kept so two configurations can be compared as the paired data they are.
    per_query_ndcg: dict[str, float] = field(default_factory=dict)

    def row(self) -> str:
        return (
            f"| `{self.configuration}` | {self.hit.as_percent()} | {self.recall.as_score()} "
            f"| {self.mrr.as_score()} | {self.ndcg.as_score()} |"
        )


def evaluate(
    queries: list[Query],
    rankings: list[Ranking],
    *,
    configuration: str,
    k: int = 5,
) -> RetrievalReport:
    """Scores one configuration. Every query must have a ranking, including an empty one.

    A missing ranking is not the same as a ranking that found nothing: dropping the query
    would quietly raise every score by removing the hardest cases from the denominator.
    """
    by_id = {r.query_id: r for r in rankings}
    missing = [q.id for q in queries if q.id not in by_id]
    if missing:
        raise ValueError(
            f"{configuration}: no ranking for {len(missing)} query(ies): {missing[:3]}. "
            "A query that returned nothing needs an empty ranking, not an absent one."
        )

    hits = 0
    recalls: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []
    per_query: dict[str, float] = {}

    for query in queries:
        ranking = by_id[query.id]
        if hit_at_k(ranking, query.relevant, k):
            hits += 1
        recalls.append(recall_at_k(ranking, query.relevant, k))
        rrs.append(reciprocal_rank(ranking, query.relevant))
        score = ndcg_at_k(ranking, query.relevant, k)
        ndcgs.append(score)
        per_query[query.id] = score

    return RetrievalReport(
        configuration=configuration,
        queries=len(queries),
        k=k,
        hit=wilson(hits, len(queries)),
        recall=bootstrap_mean(recalls),
        mrr=bootstrap_mean(rrs),
        ndcg=bootstrap_mean(ndcgs),
        per_query_ndcg=per_query,
    )


def compare(before: RetrievalReport, after: RetrievalReport) -> PairedComparison:
    """Two configurations over the same queries, as paired data."""
    shared = sorted(set(before.per_query_ndcg) & set(after.per_query_ndcg))
    if len(shared) != len(before.per_query_ndcg) or len(shared) != len(after.per_query_ndcg):
        raise ValueError(
            "refusing to compare: the two reports do not cover the same queries, so the "
            "comparison would not be paired and the p-value would be meaningless."
        )
    return sign_test(
        [before.per_query_ndcg[q] for q in shared],
        [after.per_query_ndcg[q] for q in shared],
    )


def render(reports: list[RetrievalReport], ablation_note: str = "") -> str:
    """The ablation table. Every figure carries its interval; none is quoted alone."""
    if not reports:
        return "no configuration was scored"
    k = reports[0].k
    lines = [
        f"# Retrieval, measured at k={k}",
        "",
        f"{reports[0].queries} queries, each with the chunks a person judged to answer it.",
        "",
        "Hit@k is a proportion over queries and carries a Wilson interval. Recall, MRR and nDCG",
        "are means of per-query values and carry a seeded percentile bootstrap. Neither is a",
        "formality: the intervals here overlap, and a table without them would read as though",
        "the ordering were settled.",
        "",
        f"| Configuration | Hit@{k} | Recall@{k} | MRR | nDCG@{k} |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(report.row() for report in reports)
    lines.append("")

    if len(reports) >= 2:
        best = max(reports, key=lambda r: r.ndcg.point)
        lines += [
            f"Best by nDCG@{k}: **`{best.configuration}`**. Whether that ordering is real is a",
            "separate question from whether it is the largest number, which is what the paired",
            "comparison below answers.",
            "",
        ]
    if ablation_note:
        lines += [ablation_note, ""]
    return "\n".join(lines)
