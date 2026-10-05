"""The HTTP surface: one streaming endpoint, and the latency it actually delivers.

**Why Server-Sent Events and not a plain JSON response.** The number a person feels is time to
first token, not total time, and a mean latency hides it completely: a pipeline that retrieves
for 900ms then streams fast reads as broken, while one that starts at 200ms and streams slowly
reads as working. A response that arrives whole can only report the second number. So the
answer is streamed, and the server stamps the first byte.

**Why SSE and not WebSockets.** The traffic is one request and one stream of text back. SSE is
a `text/event-stream` body over ordinary HTTP: it survives proxies that do not know about
upgrade handshakes, it reconnects by itself, and there is no connection state to manage. A
WebSocket would buy bidirectional traffic this does not have and cost the operational weight of
a protocol upgrade on every hop.

**Why the claim is a path parameter and not a filter in the body.** A claims question is always
about one claim. Making the scope part of the address means a request that forgets it cannot be
formed, rather than quietly searching everything.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from .answering import Answer, ExtractiveAnswerer, GroundingPolicy, stream_answer
from .cli import CONFIGURATIONS
from .pipeline import DEFAULT_FIXTURES, Corpus, build

#: What the reader of a stream gets, in the order it arrives.
#:
#: `meta` first so a client can render provenance before any text; `token` repeatedly; then
#: exactly one of `done` or `refused`. A client that handles only `token` and `done` still
#: works — it just shows nothing when the system declines.
EVENT_META = "meta"
EVENT_TOKEN = "token"
EVENT_DONE = "done"
EVENT_REFUSED = "refused"


class Question(BaseModel):
    """The request body.

    Declared at module scope on purpose. With `from __future__ import annotations` every
    annotation is a string, and FastAPI resolves them against the module globals — a model
    defined inside the factory is invisible there, so the parameter silently becomes a query
    string and every request comes back 422 for a field it says is missing.
    """

    question: str = Field(min_length=1, max_length=500)


def sse(event: str, payload: dict[str, Any]) -> str:
    """One Server-Sent Event.

    The double newline is the frame delimiter and is not optional: without it the client
    buffers for ever and the stream looks like a hang rather than an error.
    """
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(
    corpus: Corpus | None = None,
    fixtures: str = str(DEFAULT_FIXTURES),
    policy: GroundingPolicy | None = None,
):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    loaded = corpus if corpus is not None else build(fixtures)
    grounding = policy or GroundingPolicy()
    answerer = ExtractiveAnswerer()
    ranking_config = next(c for c in CONFIGURATIONS if c.name == "dense+rerank")
    gating_config = next(c for c in CONFIGURATIONS if c.name == "dense-only")

    app = FastAPI(
        title="claims-intake-rag",
        summary="Questions about one claim, answered from its scanned documents or refused.",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "chunks": len(loaded.chunks),
            "claims": len(set(c.claim_id for c in loaded.chunks)),
            "encoder": loaded.embedder.name,
            "grounding": {
                "min_score": grounding.min_score,
                "provenance": grounding.provenance,
            },
        }

    @app.get("/claims/{claim_id}/documents")
    def documents(claim_id: str) -> dict[str, Any]:
        chunks = [c for c in loaded.chunks if c.claim_id == claim_id]
        if not chunks:
            raise HTTPException(404, f"no documents for claim {claim_id}")
        by_document: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            row = by_document.setdefault(
                chunk.document_id, {"kind": chunk.kind, "chunks": 0, "pages": set()}
            )
            row["chunks"] += 1
            row["pages"].add(chunk.page)
        return {
            "claim_id": claim_id,
            "documents": [
                {"id": k, "kind": v["kind"], "chunks": v["chunks"], "pages": sorted(v["pages"])}
                for k, v in sorted(by_document.items())
            ],
        }

    @app.post("/claims/{claim_id}/ask")
    async def ask(claim_id: str, body: Question) -> StreamingResponse:
        if not any(c.claim_id == claim_id for c in loaded.chunks):
            raise HTTPException(404, f"no documents for claim {claim_id}")

        async def events() -> AsyncIterator[str]:
            started = time.perf_counter()
            try:
                vector = loaded.embedder.embed_query(body.question)
            except KeyError:
                # Offline, only committed questions have vectors. Saying so is the whole
                # difference between a demo and a demo that lies about what it can do.
                yield sse(
                    EVENT_REFUSED,
                    {
                        "reason": "this deployment replays committed vectors and has none for "
                        "this question; install the models extra to embed it for real",
                        "ttft_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
                return

            hits = loaded.retriever.search(
                body.question, ranking_config, query_vector=vector, limit=5, claim_id=claim_id
            )
            # The gate reads the dense cosine; the reranker above only decides the order.
            gating = loaded.retriever.search(
                body.question, gating_config, query_vector=vector, limit=1, claim_id=claim_id
            )
            retrieval_ms = (time.perf_counter() - started) * 1000

            yield sse(
                EVENT_META,
                {
                    "claim_id": claim_id,
                    "retrieval_ms": round(retrieval_ms, 2),
                    "considered": len(hits),
                    "configuration": ranking_config.name,
                },
            )

            ttft: float | None = None
            pieces: list[str] = []
            for event in stream_answer(
                body.question,
                hits,
                loaded.by_id,
                answerer,
                grounding,
                started,
                gating[0].score if gating else 0.0,
            ):
                if isinstance(event, Answer):
                    yield sse(
                        EVENT_REFUSED,
                        {
                            "reason": event.reason,
                            "text": event.text,
                            "ttft_ms": round(event.ttft_ms, 2),
                        },
                    )
                    return
                if ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
                pieces.append(event)
                yield sse(EVENT_TOKEN, {"text": event})

            total = (time.perf_counter() - started) * 1000
            yield sse(
                EVENT_DONE,
                {
                    "ttft_ms": round(ttft if ttft is not None else total, 2),
                    "total_ms": round(total, 2),
                    "characters": sum(len(p) for p in pieces),
                    "citations": [
                        loaded.by_id[h.chunk_id].cite()
                        for h in hits[:4]
                        if h.chunk_id in loaded.by_id
                    ],
                },
            )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                # Without this an nginx in front buffers the whole stream and delivers it at
                # the end, which turns a 4ms first token into a 400ms one and is invisible in
                # any test that does not go through the proxy.
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    return app
