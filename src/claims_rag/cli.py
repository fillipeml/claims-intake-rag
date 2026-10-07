"""The command line.

``claims evaluate`` is the one to run first: it produces the ablation table, which is the
thing this repository exists to produce. Everything runs offline against committed fixtures,
with no model, no download and no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .answering import ExtractiveAnswerer, GroundingPolicy, answer
from .metrics import Ranking, compare, compare_by_question, evaluate, render
from .pipeline import DEFAULT_FIXTURES, Corpus, build
from .retrieval import RetrievalConfig

# The report is Portuguese and the output is Markdown; neither survives a terminal defaulting
# to a single-byte code page, which is what Windows still does.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def out(line: str = "") -> None:
    print(line, flush=True)


#: The ablation. `candidates` matters more than it looks: with the search scoped to one claim
#: there are only about twenty chunks to choose from, so a shortlist of twenty is the whole
#: claim and the reranked rows would measure the reranker alone rather than the pipeline.
CONFIGURATIONS = [
    RetrievalConfig("lexical-only", use_dense=False),
    RetrievalConfig("dense-only", use_lexical=False),
    RetrievalConfig("hybrid-rrf"),
    RetrievalConfig("dense+rerank", use_lexical=False, rerank=True, candidates=8),
    RetrievalConfig("hybrid+rerank", rerank=True, candidates=8),
]

#: Comparisons worth drawing, as (before, after). Each answers a question the table alone
#: cannot: whether a difference between two columns is a difference at all.
COMPARISONS = [
    ("lexical-only", "hybrid-rrf"),
    ("dense-only", "hybrid-rrf"),
    ("dense-only", "dense+rerank"),
    ("hybrid-rrf", "hybrid+rerank"),
    ("dense+rerank", "hybrid+rerank"),
]


def run_configuration(
    corpus: Corpus, config: RetrievalConfig, *, scoped: bool, limit: int = 10
) -> list[Ranking]:
    rankings: list[Ranking] = []
    for query in corpus.queries:
        vector = corpus.embedder.embed_query(query.text) if config.use_dense else None
        hits = corpus.retriever.search(
            query.text,
            config,
            query_vector=vector,
            limit=limit,
            claim_id=corpus.claim_of[query.id] if scoped else None,
        )
        rankings.append(Ranking(query.id, tuple(h.chunk_id for h in hits)))
    return rankings


def cmd_evaluate(args: argparse.Namespace) -> int:
    corpus = build(args.fixtures)
    if not corpus.queries:
        out("no golden set: run scripts/make_queries.py first")
        return 2

    reports = [
        evaluate(
            corpus.queries,
            run_configuration(corpus, config, scoped=not args.whole_corpus),
            configuration=config.name,
            k=args.k,
        )
        for config in CONFIGURATIONS
    ]

    scope = "the whole corpus" if args.whole_corpus else "the claim the question is about"
    out(render(reports, ablation_note=f"Search was scoped to **{scope}**."))

    by_name = {report.configuration: report for report in reports}
    question_of = {q.id: q.text for q in corpus.queries}
    distinct = len(set(question_of.values()))

    out("## Is the difference a difference")
    out()
    out(
        f"Paired, on nDCG@{args.k}, by exact sign test. Reported twice, because the golden set "
        f"has {len(corpus.queries)} rows and only **{distinct} distinct questions** — each asked "
        f"of every claim. The question is the unit that is independent; the row is not, and a "
        f"test over rows counts one question that fails everywhere as eight separate failures."
    )
    out()
    out(f"| Comparison | By question (n={distinct}) | By row (n={len(corpus.queries)}) |")
    out("| --- | --- | --- |")
    for before, after in COMPARISONS:
        if before not in by_name or after not in by_name:
            continue
        clustered = compare_by_question(by_name[before], by_name[after], question_of)
        naive = compare(by_name[before], by_name[after])
        out(
            f"| `{before}` -> `{after}` | {clustered.better}+ {clustered.worse}- "
            f"p={clustered.p_value:.4f} | {naive.better}+ {naive.worse}- "
            f"p={naive.p_value:.4f} |"
        )
    out()
    out(
        f"The left column is the one to read. With {distinct} questions no comparison here has "
        f"the power to detect anything: even a clean four-to-nothing split is p=0.125. This "
        f"measures that the pipeline runs; it does not establish which configuration is better."
    )
    out()
    return 0


def cmd_scope(args: argparse.Namespace) -> int:
    """What filtering by claim is worth, which on this corpus is more than any other choice."""
    corpus = build(args.fixtures)
    if not corpus.queries:
        out("no golden set: run scripts/make_queries.py first")
        return 2

    out("# What scoping the search to one claim is worth")
    out()
    out("A claims question is always about one claim, and eight repair estimates differ only in")
    out("their totals. Searching the whole corpus is not a slower way to get the same answer —")
    out("it is being asked a question the corpus cannot answer.")
    out()
    out(f"| Configuration | Hit@{args.k}, whole corpus | Hit@{args.k}, scoped |")
    out("| --- | --- | --- |")
    for config in CONFIGURATIONS[:3]:
        wide = evaluate(
            corpus.queries,
            run_configuration(corpus, config, scoped=False),
            configuration=config.name,
            k=args.k,
        )
        narrow = evaluate(
            corpus.queries,
            run_configuration(corpus, config, scoped=True),
            configuration=config.name,
            k=args.k,
        )
        out(f"| `{config.name}` | {wide.hit.as_percent()} | {narrow.hit.as_percent()} |")
    out()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Choose the score below which the system should decline to answer.

    Not by feel: by running the questions this corpus can answer and the questions it cannot
    through the same retrieval, and reporting what each candidate threshold would cost in
    missed answers and buy in refused nonsense.
    """
    corpus = build(args.fixtures)
    if not corpus.queries or not corpus.unanswerable:
        out("calibration needs both answerable and unanswerable queries")
        return 2

    config = next(c for c in CONFIGURATIONS if c.name == args.configuration)

    answerable_scores: list[float] = []
    for query in corpus.queries:
        vector = corpus.embedder.embed_query(query.text) if config.use_dense else None
        hits = corpus.retriever.search(
            query.text, config, query_vector=vector, limit=1, claim_id=corpus.claim_of[query.id]
        )
        answerable_scores.append(hits[0].score if hits else 0.0)

    unanswerable_scores: list[float] = []
    some_claim = next(iter(corpus.claim_of.values()))
    for question in corpus.unanswerable:
        vector = corpus.embedder.embed_query(question) if config.use_dense else None
        hits = corpus.retriever.search(
            question, config, query_vector=vector, limit=1, claim_id=some_claim
        )
        unanswerable_scores.append(hits[0].score if hits else 0.0)

    out(f"# Where to refuse, measured on `{config.name}`")
    out()
    out(f"{len(answerable_scores)} answerable and {len(unanswerable_scores)} unanswerable")
    out("questions, scored by the top hit each one retrieves.")
    out()
    out(
        f"answerable:   min {min(answerable_scores):.4f}  "
        f"median {sorted(answerable_scores)[len(answerable_scores) // 2]:.4f}  "
        f"max {max(answerable_scores):.4f}"
    )
    out(
        f"unanswerable: min {min(unanswerable_scores):.4f}  "
        f"median {sorted(unanswerable_scores)[len(unanswerable_scores) // 2]:.4f}  "
        f"max {max(unanswerable_scores):.4f}"
    )
    out()
    out("| Threshold | Answerable kept | Unanswerable refused |")
    out("| --- | --- | --- |")

    candidates = sorted({round(s, 3) for s in answerable_scores + unanswerable_scores})
    step = max(1, len(candidates) // 12)
    for threshold in candidates[::step]:
        kept = sum(1 for s in answerable_scores if s >= threshold)
        refused = sum(1 for s in unanswerable_scores if s < threshold)
        out(
            f"| {threshold:.3f} | {kept}/{len(answerable_scores)} "
            f"({kept / len(answerable_scores):.0%}) | {refused}/{len(unanswerable_scores)} "
            f"({refused / len(unanswerable_scores):.0%}) |"
        )
    out()
    out("A threshold is a position on this table, not a fact about the world. Pick it where")
    out("the cost of a missed answer and the cost of a confident wrong one balance for the")
    out("work being done, and record which row was chosen.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    corpus = build(args.fixtures)
    config = next(c for c in CONFIGURATIONS if c.name == args.configuration)

    started = time.perf_counter()
    vector = None
    if config.use_dense:
        try:
            vector = corpus.embedder.embed_query(args.question)
        except KeyError:
            # Offline, the embedder only answers for questions whose vectors are committed.
            # Falling back to the lexical half keeps an arbitrary question answerable without
            # pretending a vector exists — and says so, because a silent downgrade would make
            # the timings and the scores below mean something other than what they claim.
            out(
                f"[no committed vector for this question: falling back to `lexical-only`. "
                f"Install the [models] extra to embed it for real, or ask one of the "
                f"{len(corpus.queries)} questions in the golden set.]"
            )
            out()
            config = next(c for c in CONFIGURATIONS if c.name == "lexical-only")
    hits = corpus.retriever.search(
        args.question, config, query_vector=vector, limit=5, claim_id=args.claim
    )
    retrieval_ms = (time.perf_counter() - started) * 1000

    # The gate reads the dense cosine, not whatever the top hit happens to carry. After
    # reranking that would be a cross-encoder score, and `claims calibrate` measured that to
    # be the worse signal for this decision by a wide margin.
    grounding_score = None
    dense = next(c for c in CONFIGURATIONS if c.name == "dense-only")
    if vector is not None:
        dense_hits = corpus.retriever.search(
            args.question, dense, query_vector=vector, limit=1, claim_id=args.claim
        )
        grounding_score = dense_hits[0].score if dense_hits else 0.0

    policy = GroundingPolicy()
    if args.min_score is not None:
        policy = GroundingPolicy(min_score=args.min_score, provenance=args.provenance)
    result, timing = answer(
        args.question,
        hits,
        corpus.by_id,
        ExtractiveAnswerer(),
        policy,
        started,
        grounding_score,
    )

    out(result.text)
    out()
    if result.refused:
        out(f"[refused after {timing.total_ms:.0f}ms]")
    else:
        out(
            f"[retrieval {retrieval_ms:.0f}ms, first token at {result.ttft_ms:.0f}ms, "
            f"complete at {result.total_ms:.0f}ms]"
        )
        for chunk_id in result.citations:
            out(f"  {corpus.by_id[chunk_id].cite()}")
    return 0


def cmd_ocr_report(args: argparse.Namespace) -> int:
    path = Path(args.fixtures) / "ocr-measurements.json"
    if not path.exists():
        out("no measurements: run scripts/make_fixtures.py first")
        return 2
    pages = json.loads(path.read_text(encoding="utf-8"))["pages"]

    out("# What the OCR engine made of the corpus")
    out()
    out(f"{len(pages)} pages, rendered at 300 dpi, degraded, and read back.")
    out()
    out("| | CER | WER | accent-folded CER |")
    out("| --- | --- | --- | --- |")
    out(
        f"| all pages | {sum(p['cer'] for p in pages) / len(pages):.2%} "
        f"| {sum(p['wer'] for p in pages) / len(pages):.2%} "
        f"| {sum(p['cer_accent_folded'] for p in pages) / len(pages):.2%} |"
    )
    out()
    by_kind: dict[str, list[dict]] = {}
    for page in pages:
        by_kind.setdefault(page["kind"], []).append(page)
    out("| Document kind | pages | CER | layout correct |")
    out("| --- | --- | --- | --- |")
    for kind, rows in sorted(by_kind.items()):
        correct = sum(r["layout_correct"] for r in rows)
        total = sum(r["layout_total"] for r in rows)
        out(
            f"| `{kind}` | {len(rows)} | {sum(r['cer'] for r in rows) / len(rows):.2%} "
            f"| {correct}/{total} ({correct / total:.0%}) |"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claims",
        description="Retrieval over scanned Portuguese claim documents, measured.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
        return p

    p = common(sub.add_parser("evaluate", help="the ablation table"))
    p.add_argument("-k", type=int, default=5)
    p.add_argument(
        "--whole-corpus",
        action="store_true",
        help="do not scope the search to the claim, which is the wrong experiment and is "
        "included because the difference is the largest effect here",
    )
    p.set_defaults(func=cmd_evaluate)

    p = common(sub.add_parser("scope", help="what filtering by claim is worth"))
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_scope)

    p = common(sub.add_parser("calibrate", help="where to refuse, from measured scores"))
    p.add_argument("--configuration", default="hybrid+rerank")
    p.set_defaults(func=cmd_calibrate)

    p = common(sub.add_parser("ask", help="one question, end to end, with timings"))
    p.add_argument("question")
    p.add_argument("--claim", help="scope to one claim, which is what a real request does")
    p.add_argument("--configuration", default="hybrid+rerank")
    p.add_argument(
        "--min-score",
        type=float,
        help="override the calibrated dense-cosine threshold; the default comes from "
        "`claims calibrate` and is recorded in GroundingPolicy",
    )
    p.add_argument("--provenance", default="overridden on the command line")
    p.set_defaults(func=cmd_ask)

    p = common(sub.add_parser("ocr-report", help="the OCR and layout measurements"))
    p.set_defaults(func=cmd_ocr_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
