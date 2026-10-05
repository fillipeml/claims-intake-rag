"""The parts added after the first measurement, and the decisions the measurement forced.

Several of these pin a behaviour that is the *opposite* of what the code did first. Those are
the ones worth having: a defect that a review would not have caught, caught once, and kept
caught.
"""

from __future__ import annotations

import pytest

from claims_rag.answering import ExtractiveAnswerer, GroundingPolicy, answer, check_grounding
from claims_rag.cli import CONFIGURATIONS, main
from claims_rag.documents import Chunk
from claims_rag.layout import _FOOTER
from claims_rag.pipeline import build
from claims_rag.retrieval import Scored

FIXTURES = "fixtures"


@pytest.fixture(scope="module")
def corpus():
    return build(FIXTURES)


def chunk(cid: str, text: str, claim: str = "AV-1") -> Chunk:
    return Chunk(
        id=cid,
        document_id="d1",
        claim_id=claim,
        kind="apolice",
        page=1,
        text=text,
    )


class TestFooterPattern:
    """The defect that cost a whole question template.

    `\\d+/\\d+` matches the `02/02` inside `02/02/2026`, so every short line carrying a date
    was classified as a footer and dropped by the chunker. The claim notification lost its
    `Data do sinistro` line, and the golden question asking when the accident happened could
    not be answered by anything in the corpus.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Data do sinistro: 02/02/2026",
            "Data do fato: 14/03/2026",
            "Data do atendimento: 08/08/2026",
        ],
    )
    def test_a_date_is_not_a_footer(self, text: str) -> None:
        assert not _FOOTER.search(text)

    @pytest.mark.parametrize("text", ["Pagina 1 de 1", "1 / 1", "Documento gerado em 02/02/2026"])
    def test_a_page_marker_still_is(self, text: str) -> None:
        assert _FOOTER.search(text)


class TestScoping:
    def test_a_search_scoped_to_a_claim_returns_only_that_claim(self, corpus) -> None:
        claim = next(iter(corpus.claim_of.values()))
        config = next(c for c in CONFIGURATIONS if c.name == "lexical-only")
        hits = corpus.retriever.search("valor total do reparo", config, limit=20, claim_id=claim)
        assert hits
        assert all(corpus.by_id[h.chunk_id].claim_id == claim for h in hits)

    def test_an_unknown_claim_returns_nothing_rather_than_everything(self, corpus) -> None:
        config = next(c for c in CONFIGURATIONS if c.name == "lexical-only")
        assert corpus.retriever.search("valor", config, claim_id="AV-nao-existe") == []

    def test_filtering_happens_before_the_cut(self, corpus) -> None:
        """Not after, which is the whole point.

        Taking the top k of the corpus and then discarding the other claims leaves a top k
        made mostly of other people's documents. Scoped, the top 20 of a claim with twenty-odd
        chunks is nearly all of it; unscoped it would be a handful.
        """
        claim = next(iter(corpus.claim_of.values()))
        config = next(c for c in CONFIGURATIONS if c.name == "lexical-only")
        scoped = corpus.retriever.search("data", config, limit=30, claim_id=claim)
        in_claim = [c for c in corpus.chunks if c.claim_id == claim]
        assert len(scoped) > 1
        assert len(scoped) <= len(in_claim)


class TestGrounding:
    def test_the_gate_reads_the_signal_it_was_calibrated_against(self) -> None:
        """Ordering and deciding-whether-to-answer are different jobs.

        After reranking the top hit carries a cross-encoder score, which `claims calibrate`
        measured to be a poor gate. The caller passes the dense cosine instead, and the policy
        must use what it is given rather than what the hit happens to carry.
        """
        chunks = {"c1": chunk("c1", "Colisao e capotamento Limite R$ 80.000,00")}
        hits = [Scored("c1", 0.0001)]  # a cross-encoder score, far below any sensible floor
        policy = GroundingPolicy(min_score=0.853)

        assert not check_grounding("qual a franquia", hits, chunks, policy).allowed
        # The same hit, gated on the dense cosine the policy was calibrated for.
        assert check_grounding("qual a franquia", hits, chunks, policy, 0.88).allowed

    def test_the_overlap_guard_is_off_by_default(self) -> None:
        """Measured: on it refuses 38 of 64 answers, 8 of them correct.

        Asked "o motorista podia dirigir", the right chunk reads "Habilitação: válida" and the
        two share no word. A lexical floor under a semantic retriever punishes it for working.
        """
        assert GroundingPolicy().min_overlap == 0
        chunks = {"c1": chunk("c1", "Habilitacao: valida")}
        hits = [Scored("c1", 0.9)]
        grounded = check_grounding("o motorista podia dirigir", hits, chunks, GroundingPolicy())
        assert grounded.allowed
        assert grounded.overlap == 0

    def test_turning_it_on_refuses_that_same_answer(self) -> None:
        chunks = {"c1": chunk("c1", "Habilitacao: valida")}
        hits = [Scored("c1", 0.9)]
        policy = GroundingPolicy(min_overlap=1)
        assert not check_grounding("o motorista podia dirigir", hits, chunks, policy).allowed

    def test_nothing_retrieved_is_a_refusal_not_a_crash(self) -> None:
        assert not check_grounding("q", [], {}, GroundingPolicy()).allowed

    def test_a_refusal_says_which_threshold_and_where_it_came_from(self) -> None:
        chunks = {"c1": chunk("c1", "texto")}
        policy = GroundingPolicy(min_score=0.9, provenance="claims calibrate, dense cosine")
        result, _ = answer(
            "pergunta", [Scored("c1", 0.1)], chunks, ExtractiveAnswerer(), policy, None, 0.1
        )
        assert result.refused
        assert "0.9000" in result.text
        assert "claims calibrate" in result.text


class TestTheCorpusLoads:
    def test_the_golden_set_covers_every_question_template(self, corpus) -> None:
        # Eight templates over eight claims. Seven would mean a template located nothing for
        # any claim, which is how the footer defect announced itself.
        assert len(corpus.queries) == 64
        assert len({q.text for q in corpus.queries}) == 8

    def test_some_questions_need_more_than_one_document(self, corpus) -> None:
        # The case Recall@k exists for: a claims question answered only by two documents
        # agreeing with each other.
        assert sum(1 for q in corpus.queries if len(q.relevant) > 1) >= 8

    def test_every_relevant_chunk_exists(self, corpus) -> None:
        known = set(corpus.by_id)
        for query in corpus.queries:
            assert query.relevant <= known, query.id

    def test_the_index_holds_re_segmented_text(self, corpus) -> None:
        """Embedding the merged text and searching the segmented text would compare two
        different corpora and report the difference as a retrieval result."""
        merged = [c for c in corpus.chunks if "Totalgeral" in c.text]
        assert not merged, f"{len(merged)} chunk(s) reached the index unsegmented"


class TestCommands:
    def test_evaluate_prints_every_configuration(self, capsys) -> None:
        assert main(["evaluate", "--fixtures", FIXTURES]) == 0
        printed = capsys.readouterr().out
        for config in CONFIGURATIONS:
            assert f"`{config.name}`" in printed
        assert "Wilson" in printed

    def test_evaluate_reports_the_paired_comparison_too(self, capsys) -> None:
        assert main(["evaluate", "--fixtures", FIXTURES]) == 0
        printed = capsys.readouterr().out
        assert "Is the difference a difference" in printed
        assert "sign test" in printed

    def test_calibrate_shows_both_distributions(self, capsys) -> None:
        assert main(["calibrate", "--fixtures", FIXTURES, "--configuration", "dense-only"]) == 0
        printed = capsys.readouterr().out
        assert "answerable:" in printed and "unanswerable:" in printed
        assert "not a fact about the world" in printed

    def test_scope_contrasts_the_two_experiments(self, capsys) -> None:
        assert main(["scope", "--fixtures", FIXTURES]) == 0
        printed = capsys.readouterr().out
        assert "whole corpus" in printed and "scoped" in printed

    def test_ask_answers_and_reports_its_timings(self, capsys) -> None:
        claim = "AV-2026-4407"
        code = main(
            [
                "ask",
                "quanto vai custar o conserto do veículo",
                "--claim",
                claim,
                "--fixtures",
                FIXTURES,
            ]
        )
        assert code == 0
        printed = capsys.readouterr().out
        assert "first token at" in printed

    def test_ask_falls_back_rather_than_inventing_a_vector(self, capsys) -> None:
        # The recorded embedder holds vectors only for committed questions. Falling back to
        # the lexical half keeps an arbitrary question answerable; doing it silently would
        # make the timings and scores mean something other than what they claim.
        code = main(["ask", "uma pergunta que nunca foi gravada", "--fixtures", FIXTURES])
        assert code == 0
        assert "falling back to" in capsys.readouterr().out
