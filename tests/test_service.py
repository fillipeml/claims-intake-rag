"""The HTTP surface and the vector store — the two things that were written and never run.

Both had defects that only running them could find. The store called two client methods that
no longer exist in the version it pins; the API declared its request model inside the factory,
which with `from __future__ import annotations` makes FastAPI treat the body as a query string
and answer 422 to every request. Neither is visible in a review.
"""

from __future__ import annotations

import json

import pytest

from claims_rag.pipeline import build
from claims_rag.retrieval import DenseIndex
from claims_rag.store import Filter, QdrantStore

FIXTURES = "fixtures"
CLAIM = "AV-2026-4407"

# The service and the store live behind the [serve] extra, so the suite stays runnable for
# someone who installed only [dev] — it reports what it skipped instead of failing. CI
# installs [dev,serve] precisely so none of this is skipped there.
fastapi = pytest.importorskip("fastapi", reason="the [serve] extra is not installed")
pytest.importorskip("qdrant_client", reason="the [serve] extra is not installed")


@pytest.fixture(scope="module")
def corpus():
    return build(FIXTURES)


@pytest.fixture(scope="module")
def client(corpus):
    from fastapi.testclient import TestClient

    from claims_rag.api import create_app

    return TestClient(create_app(corpus))


def events(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, payload) pairs."""
    out: list[tuple[str, dict]] = []
    name: str | None = None
    for line in body.splitlines():
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: ") and name:
            out.append((name, json.loads(line[len("data: ") :])))
            name = None
    return out


class TestHealth:
    def test_it_reports_the_corpus_and_the_threshold_it_will_refuse_at(self, client) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["chunks"] > 0 and body["claims"] == 8
        # The gate is part of the service's contract, so it is visible without reading code.
        assert body["grounding"]["min_score"] > 0
        assert "calibrate" in body["grounding"]["provenance"]


class TestDocuments:
    def test_a_claim_lists_its_five_documents(self, client) -> None:
        body = client.get(f"/claims/{CLAIM}/documents").json()
        kinds = {d["kind"] for d in body["documents"]}
        assert kinds == {
            "aviso_de_sinistro",
            "boletim_de_ocorrencia",
            "orcamento_de_reparo",
            "laudo_medico",
            "apolice",
        }

    def test_an_unknown_claim_is_404_not_an_empty_list(self, client) -> None:
        # An empty list reads as "this claim has no documents", which is a different and much
        # more alarming statement than "there is no such claim".
        assert client.get("/claims/AV-0000-0000/documents").status_code == 404


class TestStreaming:
    def test_the_stream_is_an_event_stream_that_proxies_will_not_buffer(self, client) -> None:
        response = client.post(
            f"/claims/{CLAIM}/ask", json={"question": "quanto vai custar o conserto do veículo"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        # Without this an nginx in front delivers the whole stream at the end, turning a 3ms
        # first token into a 300ms one in a way no test behind the proxy can see.
        assert response.headers["x-accel-buffering"] == "no"

    def test_meta_arrives_before_any_text_and_done_closes_it(self, client) -> None:
        response = client.post(
            f"/claims/{CLAIM}/ask", json={"question": "quanto vai custar o conserto do veículo"}
        )
        names = [name for name, _ in events(response.text)]
        assert names[0] == "meta"
        assert names[-1] in {"done", "refused"}
        assert "token" in names

    def test_done_carries_the_timings_and_the_provenance(self, client) -> None:
        response = client.post(
            f"/claims/{CLAIM}/ask", json={"question": "quanto vai custar o conserto do veículo"}
        )
        done = [payload for name, payload in events(response.text) if name == "done"]
        assert len(done) == 1
        assert done[0]["ttft_ms"] > 0
        assert done[0]["total_ms"] >= done[0]["ttft_ms"]
        assert done[0]["citations"]
        # A citation has to say where it came from, or it is decoration.
        assert "p." in done[0]["citations"][0]

    def test_a_question_with_no_committed_vector_is_refused_explicitly(self, client) -> None:
        # Offline the embedder replays; it does not invent. Saying so is the difference
        # between a demo and a demo that lies about what it can do.
        response = client.post(
            f"/claims/{CLAIM}/ask", json={"question": "uma pergunta que nunca foi gravada"}
        )
        refused = [payload for name, payload in events(response.text) if name == "refused"]
        assert len(refused) == 1
        assert "committed vectors" in refused[0]["reason"]

    def test_an_unknown_claim_is_refused_before_any_stream_starts(self, client) -> None:
        response = client.post("/claims/AV-0000-0000/ask", json={"question": "qualquer coisa"})
        assert response.status_code == 404

    def test_an_empty_question_is_rejected_by_the_schema(self, client) -> None:
        assert client.post(f"/claims/{CLAIM}/ask", json={"question": ""}).status_code == 422

    def test_the_body_is_read_as_a_body(self, client) -> None:
        """The defect that made every request 422.

        `from __future__ import annotations` turns annotations into strings, and FastAPI
        resolves them against module globals. A model defined inside the factory is invisible
        there, so the parameter became a query string and the error blamed a missing field.
        """
        assert client.post(f"/claims/{CLAIM}/ask", json={"question": "x"}).status_code in {
            200,
            422,
        }
        bad = client.post(f"/claims/{CLAIM}/ask", json={"nao_e_o_campo": "x"})
        assert bad.status_code == 422
        assert bad.json()["detail"][0]["loc"][0] == "body"


class TestQdrant:
    def test_it_round_trips_the_corpus(self, corpus) -> None:
        store = QdrantStore(":memory:")
        try:
            store.recreate(corpus.embedder.dimension)
            vectors = [corpus.embedder.embed_passages([c.embedding_text])[0] for c in corpus.chunks]
            assert store.upsert(corpus.chunks, vectors) == len(corpus.chunks)
            assert store.count() == len(corpus.chunks)
        finally:
            store.close()

    def test_the_payload_filter_scopes_the_search(self, corpus) -> None:
        store = QdrantStore(":memory:")
        try:
            store.recreate(corpus.embedder.dimension)
            store.upsert(
                corpus.chunks,
                [corpus.embedder.embed_passages([c.embedding_text])[0] for c in corpus.chunks],
            )
            vector = corpus.embedder.embed_query(corpus.queries[0].text)
            hits = store.search(vector, limit=5, where=Filter(claim_id=CLAIM))
            assert hits
            assert all(corpus.by_id[h.chunk_id].claim_id == CLAIM for h in hits)
        finally:
            store.close()

    def test_the_approximate_index_agrees_with_the_exact_one(self, corpus) -> None:
        """Which is the reason the flat index is kept beside it.

        At this corpus size HNSW has nothing to approximate, so the two must return the same
        ranking. When they stop agreeing, that difference is what an approximate index costs
        in recall — and without the exact baseline there would be no way to measure it.
        """
        vectors = {
            c.id: corpus.embedder.embed_passages([c.embedding_text])[0] for c in corpus.chunks
        }
        store = QdrantStore(":memory:")
        try:
            store.recreate(corpus.embedder.dimension)
            store.upsert(corpus.chunks, [vectors[c.id] for c in corpus.chunks])
            vector = corpus.embedder.embed_query(corpus.queries[0].text)

            approximate = store.search(vector, limit=5, where=Filter(claim_id=CLAIM))
            exact = DenseIndex(
                {k: v for k, v in vectors.items() if corpus.by_id[k].claim_id == CLAIM}
            ).search(vector, limit=5)
            assert [h.chunk_id for h in approximate] == [h.chunk_id for h in exact]
        finally:
            store.close()


class TestManifests:
    """The container and cluster definitions, checked for the fields whose absence is silent.

    A Deployment without a readiness probe still rolls out; it just sends traffic to pods that
    cannot serve it. A container that runs as root still works; it just runs as root. These
    are the failures that never announce themselves, which is why they are asserted rather
    than reviewed.
    """

    @staticmethod
    def manifests() -> list[dict]:
        import pathlib

        import yaml

        path = pathlib.Path(__file__).resolve().parent.parent / "k8s" / "deployment.yaml"
        return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]

    def test_the_pod_cannot_run_as_root_or_escalate(self) -> None:
        deployment = next(d for d in self.manifests() if d["kind"] == "Deployment")
        pod = deployment["spec"]["template"]["spec"]
        assert pod["securityContext"]["runAsNonRoot"] is True
        container = pod["containers"][0]["securityContext"]
        assert container["allowPrivilegeEscalation"] is False
        assert container["readOnlyRootFilesystem"] is True
        assert container["capabilities"]["drop"] == ["ALL"]

    def test_readiness_and_liveness_are_separate_probes(self) -> None:
        # Sharing one means a slow dependency gets the container killed instead of taken out
        # of rotation, which turns a degradation into an outage.
        container = next(d for d in self.manifests() if d["kind"] == "Deployment")["spec"][
            "template"
        ]["spec"]["containers"][0]
        assert {"startupProbe", "readinessProbe", "livenessProbe"} <= set(container)

    def test_memory_is_guaranteed_so_the_pod_is_not_evicted_first(self) -> None:
        # The corpus lives in process, so memory is a function of the corpus rather than of
        # load. Equal request and limit puts the pod in the Guaranteed QoS class.
        resources = next(d for d in self.manifests() if d["kind"] == "Deployment")["spec"][
            "template"
        ]["spec"]["containers"][0]["resources"]
        assert resources["requests"]["memory"] == resources["limits"]["memory"]

    def test_a_rollout_cannot_cut_every_stream_at_once(self) -> None:
        # Server-Sent Events are long-lived responses; without a budget a rollout can take
        # every replica down together and cut every answer mid-sentence.
        assert any(d["kind"] == "PodDisruptionBudget" for d in self.manifests())

    def test_the_image_entry_point_names_a_module_that_exists(self) -> None:
        import pathlib

        dockerfile = (pathlib.Path(__file__).resolve().parent.parent / "Dockerfile").read_text(
            encoding="utf-8"
        )
        assert "claims_rag.app:app" in dockerfile
        from claims_rag.app import app  # noqa: F401
