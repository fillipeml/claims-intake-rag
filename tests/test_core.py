"""The pieces that decide whether a number in the report is trustworthy.

Expected values here are computed by hand or taken from a published worked example, never by
running the implementation and pasting what it said. A test written the second way asserts
that the code does what the code does.
"""

from __future__ import annotations

import math

import pytest

from claims_rag.documents import (
    Block,
    Box,
    Chunk,
    Document,
    Page,
    document_from_dict,
    document_to_dict,
)
from claims_rag.layout import ReadLine, classify, score_layout
from claims_rag.metrics import (
    Query,
    Ranking,
    evaluate,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from claims_rag.retrieval import (
    Bm25Index,
    DenseIndex,
    HybridRetriever,
    RecordedReranker,
    RetrievalConfig,
    Scored,
    cosine,
    reciprocal_rank_fusion,
    tokenise,
)
from claims_rag.segmentation import WordSegmenter
from claims_rag.stats import sign_test, wilson


def box(top: int = 0, height: int = 20, left: int = 100, width: int = 400) -> Box:
    return Box(left, top, left + width, top + height)


def chunk(cid: str, text: str, headings: tuple[str, ...] = ()) -> Chunk:
    return Chunk(
        id=cid,
        document_id="d1",
        claim_id="AV-1",
        kind="orcamento_de_reparo",
        page=1,
        text=text,
        headings=headings,
    )


class TestTokenisation:
    def test_ocr_damage_and_clean_text_produce_the_same_tokens(self) -> None:
        """The decision the measured OCR output justifies.

        The engine drops diacritics almost everywhere. An index that required them to match
        would miss exactly the documents this system exists to read.
        """
        assert tokenise("Valor orçado: R$ 12.480,00 — veículo") == tokenise(
            "Valor orcado: R$ 12.480,00 — veiculo"
        )

    def test_stopwords_go_and_content_words_stay(self) -> None:
        assert tokenise("o valor do reparo") == ["valor", "reparo"]

    def test_a_claim_identifier_survives_tokenisation(self) -> None:
        # The half of the corpus that dense search cannot find.
        assert "2026" in tokenise("Sinistro AV-2026-4463")


class TestBm25:
    def test_a_rare_term_outranks_a_common_one(self) -> None:
        chunks = [
            chunk("c1", "total geral do orçamento"),
            chunk("c2", "total geral do orçamento"),
            chunk("c3", "franquia para colisão e capotamento"),
        ]
        index = Bm25Index(chunks)
        hits = index.search("franquia", limit=5)
        assert [h.chunk_id for h in hits] == ["c3"]

    def test_a_query_of_only_stopwords_returns_nothing(self) -> None:
        assert Bm25Index([chunk("c1", "qualquer texto")]).search("o de a") == []

    def test_headings_are_searchable(self) -> None:
        # A row reading "Total geral R$ 1.000" means nothing without its section.
        chunks = [chunk("c1", "Total geral   R$ 1.000,00", ("ORÇAMENTO DE REPARO",))]
        assert Bm25Index(chunks).search("orçamento")


class TestFusion:
    def test_rrf_prefers_agreement_over_one_confident_list(self) -> None:
        """The property that makes hybrid search work, pinned.

        `a` is first in one list and fortieth in the other; `b` is eighth in both. With
        k=60: a scores 1/61 + 1/100 = 0.026393, b scores 2/68 = 0.029412, so b wins.
        """
        first = [Scored("a", 9.0)] + [Scored(f"x{i}", 1.0) for i in range(6)] + [Scored("b", 0.5)]
        second = [Scored(f"y{i}", 1.0) for i in range(7)] + [Scored("b", 0.4)]
        second += [Scored(f"z{i}", 0.2) for i in range(31)] + [Scored("a", 0.1)]

        fused = reciprocal_rank_fusion([first, second])
        assert fused[0].chunk_id == "b"
        assert fused[0].score == pytest.approx(2 / 68, rel=1e-9)

    def test_fusion_ignores_the_scores_entirely(self) -> None:
        # Multiplying one retriever's scores by a thousand must change nothing: that is the
        # whole reason rank fusion is used instead of a weighted sum of normalised scores.
        a = [Scored("p", 0.9), Scored("q", 0.1)]
        b = [Scored("q", 900.0), Scored("p", 100.0)]
        scaled = [Scored(s.chunk_id, s.score * 1000) for s in b]
        assert reciprocal_rank_fusion([a, b]) == reciprocal_rank_fusion([a, scaled])


class TestDense:
    def test_cosine_is_orientation_not_magnitude(self) -> None:
        assert cosine([1, 0], [5, 0]) == pytest.approx(1.0)
        assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_vectors_of_mixed_dimension_are_refused(self) -> None:
        with pytest.raises(ValueError, match="mixed dimension"):
            DenseIndex({"a": [1.0, 0.0], "b": [1.0]})


class TestRecordedReranker:
    def test_it_never_guesses(self) -> None:
        reranker = RecordedReranker("judge", {"qual a franquia": {"c1": 0.9}})
        with pytest.raises(KeyError, match="never guesses"):
            reranker.rerank("uma pergunta que não foi gravada", [chunk("c1", "x")])

    def test_a_chunk_without_a_score_is_an_error_not_a_zero(self) -> None:
        # Scoring it zero would drop it from every result and every metric, silently.
        reranker = RecordedReranker("judge", {"q": {"c1": 0.9}})
        with pytest.raises(KeyError, match="no score"):
            reranker.rerank("q", [chunk("c1", "x"), chunk("c2", "y")])


class TestRetrieverGuards:
    def test_disabling_every_retriever_is_refused(self) -> None:
        retriever = HybridRetriever([chunk("c1", "texto")])
        config = RetrievalConfig("nothing", use_dense=False, use_lexical=False)
        with pytest.raises(ValueError, match="nothing to run"):
            retriever.search("q", config)

    def test_dense_without_vectors_is_refused_rather_than_silently_lexical(self) -> None:
        retriever = HybridRetriever([chunk("c1", "texto")])
        with pytest.raises(ValueError, match="no vectors"):
            retriever.search("q", RetrievalConfig("dense", use_lexical=False))


class TestMetrics:
    """Worked by hand. ranking [a, b, c], relevant {a, c}, k=3."""

    RANKING = Ranking("q1", ("a", "b", "c"))
    RELEVANT = frozenset({"a", "c"})

    def test_hit_and_recall(self) -> None:
        assert hit_at_k(self.RANKING, self.RELEVANT, 1) is True
        assert hit_at_k(Ranking("q", ("b", "a")), self.RELEVANT, 1) is False
        assert recall_at_k(self.RANKING, self.RELEVANT, 3) == pytest.approx(1.0)
        assert recall_at_k(self.RANKING, self.RELEVANT, 2) == pytest.approx(0.5)

    def test_reciprocal_rank(self) -> None:
        assert reciprocal_rank(self.RANKING, self.RELEVANT) == pytest.approx(1.0)
        assert reciprocal_rank(Ranking("q", ("x", "y", "a")), self.RELEVANT) == pytest.approx(1 / 3)
        assert reciprocal_rank(Ranking("q", ("x", "y")), self.RELEVANT) == 0.0

    def test_ndcg_against_the_hand_computed_value(self) -> None:
        # gain  = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5          = 1.5
        # ideal = 1/log2(2) + 1/log2(3)     = 1 + 0.6309297...  = 1.6309297
        expected = 1.5 / (1 + 1 / math.log2(3))
        assert ndcg_at_k(self.RANKING, self.RELEVANT, 3) == pytest.approx(expected)
        assert expected == pytest.approx(0.919721, abs=1e-6)

    def test_the_ideal_ranking_stops_at_the_number_of_relevant_chunks(self) -> None:
        # Three relevant and k=5: perfect is three hits in the first three places, not five.
        perfect = Ranking("q", ("a", "b", "c", "x", "y"))
        assert ndcg_at_k(perfect, frozenset({"a", "b", "c"}), 5) == pytest.approx(1.0)

    def test_a_query_nothing_answers_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no relevant chunk"):
            Query(id="q", text="?", relevant=frozenset())

    def test_a_missing_ranking_is_an_error_not_a_dropped_query(self) -> None:
        # Dropping it would raise every score by removing the hardest case from the
        # denominator, which is the quiet way an evaluation flatters itself.
        queries = [
            Query("q1", "a", frozenset({"a"})),
            Query("q2", "b", frozenset({"b"})),
        ]
        with pytest.raises(ValueError, match="no ranking"):
            evaluate(queries, [Ranking("q1", ("a",))], configuration="partial")

    def test_hit_takes_a_wilson_interval_and_ndcg_a_bootstrap(self) -> None:
        queries = [Query(f"q{i}", "x", frozenset({"a"})) for i in range(10)]
        rankings = [Ranking(f"q{i}", ("a",) if i < 8 else ("z",)) for i in range(10)]
        report = evaluate(queries, rankings, configuration="c", k=5)

        # Wilson for 8/10 at 95% is the published [0.4902, 0.9433].
        assert report.hit.point == pytest.approx(0.8)
        assert report.hit.low == pytest.approx(0.4902, abs=1e-3)
        assert report.hit.high == pytest.approx(0.9433, abs=1e-3)
        assert report.ndcg.n == 10


class TestStats:
    def test_wilson_matches_the_published_interval(self) -> None:
        interval = wilson(8, 10)
        assert (interval.low, interval.high) == (
            pytest.approx(0.4902, abs=1e-3),
            pytest.approx(0.9433, abs=1e-3),
        )

    def test_no_data_is_not_a_rate_of_zero(self) -> None:
        assert wilson(0, 0).as_percent() == "no data"

    def test_the_sign_test_counts_only_the_queries_that_moved(self) -> None:
        # Nine improved, one regressed. Two-sided exact binomial over ten discordant pairs:
        # 2 * (C(10,0) + C(10,1)) / 2^10 = 2 * 11 / 1024 = 0.021484375
        before = [0.0] * 9 + [1.0] + [0.5] * 5
        after = [1.0] * 9 + [0.0] + [0.5] * 5
        result = sign_test(before, after)
        assert (result.better, result.worse, result.tied) == (9, 1, 5)
        assert result.p_value == pytest.approx(22 / 1024)
        assert result.significant

    def test_an_even_trade_is_not_called_worse(self) -> None:
        result = sign_test([0.0, 1.0], [1.0, 0.0])
        assert "unchanged on balance" in result.describe()


class TestSegmentation:
    CORPUS = [
        "Data do atendimento: 08/08/2026",
        "Nao houve necessidade de afastamento das atividades habituais.",
        "Solicitado retorno ambulatorial em quinze dias",
    ]

    def test_it_puts_the_spaces_back(self) -> None:
        segmenter = WordSegmenter.from_texts(self.CORPUS)
        assert segmenter.resegment("Datadoatendimento") == "Data do atendimento"

    def test_a_short_word_is_left_alone(self) -> None:
        segmenter = WordSegmenter.from_texts(self.CORPUS)
        assert segmenter.split("atendimento") == ["atendimento"]

    def test_an_ocr_artefact_does_not_become_vocabulary(self) -> None:
        """The failure that cost the most to find.

        One line welding "Data do" into `datado` while keeping its other spaces taught the
        vocabulary a word that then beat `data` + `do` everywhere else. A candidate better
        explained as a sequence of known words is not a word.
        """
        corpus = [*self.CORPUS, "Datado atendimento", "Datado atendimento"]
        segmenter = WordSegmenter.from_texts(corpus)
        assert "datado" not in segmenter._frequencies
        assert segmenter.resegment("Datadoatendimento") == "Data do atendimento"

    def test_the_corpus_alone_cannot_learn_a_word_it_never_shows_apart(self) -> None:
        # Why the glossary exists, as an assertion rather than a paragraph.
        merged_only = ["Naohouvenecessidadedeafastamento"]
        bare = WordSegmenter.from_texts(merged_only, seed=False)
        assert bare._frequencies.get("afastamento", 0) == 0
        seeded = WordSegmenter.from_texts(merged_only)
        assert seeded._frequencies.get("afastamento", 0) > 0

    def test_an_unknown_run_costs_more_the_longer_it_is(self) -> None:
        # Charging a flat rate per character made a fifty-character run cost about the same
        # as the eight real words inside it, and the search kept it whole.
        segmenter = WordSegmenter.from_texts(self.CORPUS)
        short = segmenter._cost("xyzxyzxy")
        long = segmenter._cost("xyzxyzxyxyzxyzxyxyzxyzxy")
        assert long > short * 2


class TestLayout:
    def test_a_short_tall_line_in_capitals_is_a_heading(self) -> None:
        lines = [
            ReadLine("ORCAMENTO DE REPARO", box(top=0, height=40), 0.99),
            ReadLine("Texto corrido da pagina com varias palavras seguidas", box(top=60), 0.98),
            ReadLine("Outro paragrafo igualmente longo e sem numeros nele", box(top=90), 0.98),
        ]
        blocks = classify(lines, page_height=1000)
        assert blocks[0].kind == "heading"
        assert blocks[1].kind == "paragraph"

    def test_a_page_marker_at_the_foot_is_a_footer(self) -> None:
        lines = [
            ReadLine("Texto normal da pagina", box(top=100), 0.98),
            ReadLine("Pagina 1 de 1", box(top=950, height=14), 0.97),
        ]
        blocks = classify(lines, page_height=1000)
        assert blocks[-1].kind == "footer"

    def test_the_confusion_matrix_counts_what_it_matched(self) -> None:
        truth = [Block("ABC", "heading", box(top=0, height=40))]
        inferred = [Block("ABC", "heading", box(top=1, height=39))]
        score = score_layout(truth, inferred)
        assert (score.total, score.correct) == (1, 1)
        assert score.accuracy == pytest.approx(1.0)

    def test_blocks_that_overlap_nothing_are_not_counted(self) -> None:
        truth = [Block("ABC", "heading", box(top=0, height=40))]
        inferred = [Block("XYZ", "paragraph", box(top=500, height=20))]
        assert score_layout(truth, inferred).total == 0


class TestDocuments:
    def test_a_chunk_carries_its_section_into_what_is_embedded(self) -> None:
        c = chunk("c1", "Total geral   R$ 12.480,00", ("ORÇAMENTO DE REPARO", "Totais"))
        assert c.embedding_text.startswith("ORÇAMENTO DE REPARO > Totais")
        assert "12.480,00" in c.embedding_text

    def test_a_document_round_trips_through_json(self) -> None:
        doc = Document(
            id="d1",
            kind="apolice",
            claim_id="AV-1",
            pages=(Page(1, (Block("ABC", "heading", box()),), 100, 200),),
        )
        assert document_from_dict(document_to_dict(doc)) == doc

    def test_an_unknown_document_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="is not one of"):
            Document(id="d1", kind="contrato", claim_id="AV-1", pages=())  # type: ignore[arg-type]
