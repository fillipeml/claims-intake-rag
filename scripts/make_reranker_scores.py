"""Records what a cross-encoder thinks of each query's shortlist.

The reranker is the expensive, accurate step: it reads the query and the chunk together
instead of comparing two vectors computed apart. Too slow to run over a corpus, which is why
it runs over the shortlist the cheap retrievers produced.

Scoring every (query, chunk) pair the ablation can reach, once, and committing the result is
what lets the evaluation replay the reranker without a model — and what makes the reported
table reproducible rather than dependent on which revision of a model hub was current.

    uv run python scripts/make_reranker_scores.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_queries import load_chunks  # noqa: E402

from claims_rag.embedding import RecordedEmbedder  # noqa: E402
from claims_rag.retrieval import HybridRetriever, RetrievalConfig  # noqa: E402
from claims_rag.segmentation import WordSegmenter  # noqa: E402

FIXTURES = ROOT / "fixtures"
MODEL = "BAAI/bge-reranker-base"

#: How deep the shortlist goes. Everything at or above this rank in any configuration gets a
#: score, so the recording covers whatever the ablation asks for later.
SHORTLIST = 20


def main() -> None:
    from sentence_transformers import CrossEncoder

    chunks = load_chunks()
    segmenter = WordSegmenter.from_texts([c.text for c in chunks])
    indexed = [
        replace(
            c,
            text=segmenter.resegment(c.text),
            headings=tuple(segmenter.resegment(h) for h in c.headings),
        )
        for c in chunks
    ]
    by_id = {c.id: c for c in indexed}

    embedder = RecordedEmbedder.load(FIXTURES / "vectors.json")
    vectors = {c.id: embedder.embed_passages([c.embedding_text])[0] for c in indexed}
    retriever = HybridRetriever(indexed, vectors)

    payload = json.loads((FIXTURES / "queries.json").read_text(encoding="utf-8"))
    rows = payload["queries"]

    configs = [
        RetrievalConfig("lexical-only", use_dense=False, candidates=SHORTLIST),
        RetrievalConfig("dense-only", use_lexical=False, candidates=SHORTLIST),
        RetrievalConfig("hybrid-rrf", candidates=SHORTLIST),
    ]

    # The union of what every configuration would put in front of the reranker, scoped and
    # unscoped, so one recording serves the whole ablation.
    wanted: dict[str, set[str]] = {}
    for row in rows:
        shortlist: set[str] = set()
        for config in configs:
            query_vector = embedder.embed_query(row["text"]) if config.use_dense else None
            for claim in (row["claim_id"], None):
                hits = retriever.search(
                    row["text"], config, query_vector=query_vector, limit=SHORTLIST, claim_id=claim
                )
                shortlist.update(h.chunk_id for h in hits)
        # Accumulated, not assigned. With the claim identifiers removed the questions are
        # identical across claims — "quanto vai custar o conserto do veículo" is asked of all
        # eight — so assigning would keep only the last claim's shortlist and the recorded
        # reranker would raise on every earlier one.
        wanted.setdefault(row["text"], set()).update(shortlist)

    # The unanswerable questions need scores too, or `claims calibrate` cannot run: it has to
    # see what a bad retrieval scores in order to choose where to refuse. They are scored
    # against every chunk, because the calibration is free to scope them to any claim and a
    # recording that covered only one would couple the two.
    for question in payload.get("unanswerable", []):
        wanted.setdefault(question, set()).update(c.id for c in indexed)

    pairs = [(question, cid) for question, ids in wanted.items() for cid in sorted(ids)]
    print(f"{len(pairs)} (query, chunk) pairs to score with {MODEL}")

    model = CrossEncoder(MODEL)
    scores = model.predict(
        [(question, by_id[cid].embedding_text) for question, cid in pairs],
        show_progress_bar=True,
    )

    recorded: dict[str, dict[str, float]] = {}
    for (question, cid), score in zip(pairs, scores, strict=True):
        recorded.setdefault(question, {})[cid] = round(float(score), 6)

    (FIXTURES / "reranker.json").write_text(
        json.dumps({"model": MODEL, "scores": recorded}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote scores for {len(recorded)} queries")


if __name__ == "__main__":
    main()
