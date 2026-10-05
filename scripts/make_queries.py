"""Builds the golden set, and the vectors the evaluation replays.

**The methodological point this file exists to get right.** Relevance is decided from the
*ground truth*, never by matching the question against the chunks. If a chunk were marked
relevant because it shares words with the query, the golden set would be a BM25 output with a
different name, and the ablation would report that BM25 wins — by construction, on every
corpus, for ever.

So each question is paired with the **line of the original document that answers it**, which is
known exactly because this corpus was generated. The relevant chunk is then the one whose OCR
text aligns best with that line. The alignment uses the truth and the OCR output; the question
never enters it.

The questions carry **no claim identifier**, because the claim is passed separately as a
scope. Having it in the text as well was double-dipping, and it dominated every scorer:
asked whether the driver was licensed, the system returned `Numero do aviso:AV-2026-4407`
at 0.88 while `Habilitacao:valida` — the line that answers the question — did not place.
A user whose claim the system already knows does not type its number into the question.

The questions are also written to *avoid* the document's wording where Portuguese allows
it — "quanto vai custar o conserto" against a document that says "Total geral" — so that dense
retrieval has something to be better at and the comparison can go either way.

    uv run python scripts/make_queries.py
"""

from __future__ import annotations

import difflib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_fixtures import SEED, ClaimFile, build_claims, money  # noqa: E402

from claims_rag.documents import Chunk  # noqa: E402
from claims_rag.embedding import (  # noqa: E402
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    SentenceTransformerEmbedder,
    save_vectors,
    text_key,
)
from claims_rag.segmentation import WordSegmenter  # noqa: E402

FIXTURES = ROOT / "fixtures"


def load_chunks() -> list[Chunk]:
    payload = json.loads((FIXTURES / "chunks.json").read_text(encoding="utf-8"))
    return [
        Chunk(
            id=row["id"],
            document_id=row["document_id"],
            claim_id=row["claim_id"],
            kind=row["kind"],
            page=row["page"],
            text=row["text"],
            headings=tuple(row["headings"]),
            confidence=row["confidence"],
        )
        for row in payload["chunks"]
    ]


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def locate(answer_line: str, document_id: str, chunks: list[Chunk]) -> str | None:
    """The chunk of this document whose text aligns best with a known original line.

    Alignment is on the ground-truth line against the OCR text, so the question plays no part
    in deciding what counts as relevant.
    """
    target = _normalise(answer_line)
    best_id: str | None = None
    best_coverage = 0.0
    for chunk in chunks:
        if chunk.document_id != document_id:
            continue
        # Coverage, not similarity. Chunking groups the header fields of a claim notification
        # into one chunk, so an answer line is a tenth of the text around it and a
        # whole-against-whole ratio reads as a non-match. The question is whether the answer
        # is *in* the chunk, which is how much of the line appears in it, in order.
        matcher = difflib.SequenceMatcher(None, target, _normalise(chunk.text), autojunk=False)
        matched = sum(block.size for block in matcher.get_matching_blocks())
        coverage = matched / max(len(target), 1)
        if coverage > best_coverage:
            best_coverage, best_id = coverage, chunk.id
    # Below this the OCR damaged the line past recognition and the pairing would be a guess.
    return best_id if best_coverage >= 0.70 else None


def answerable(claim: ClaimFile) -> list[tuple[str, list[tuple[str, str]], tuple[str, ...]]]:
    """(question, [(document_id, the original line that answers it)], tags).

    More than one line means more than one relevant chunk, which is the case Recall@k exists
    for: a claims question is often answered only by two documents agreeing.
    """
    c = claim
    licence = "válida" if c.licence_valid else "vencida há mais de 30 dias"
    off = (
        "Não houve necessidade de afastamento das atividades habituais."
        if c.days_off == 0
        else f"Recomendado afastamento das atividades habituais por {c.days_off} dias."
    )
    return [
        (
            "quanto vai custar o conserto do veículo",
            [(f"{c.id}-orcamento_de_reparo", f"Total geral   {money(c.total)}")],
            ("money", "single"),
        ),
        (
            "qual o valor que o segurado paga do próprio bolso numa batida",
            [
                (
                    f"{c.id}-apolice",
                    f"Colisão e capotamento   Limite {money(c.coverage_limit)}   Franquia {money(c.deductible)}",
                )
            ],
            ("money", "paraphrase", "single"),
        ),
        (
            "o motorista podia dirigir no dia do acidente",
            [(f"{c.id}-boletim_de_ocorrencia", f"Habilitação: {licence}")],
            ("paraphrase", "single"),
        ),
        (
            "o médico mandou a pessoa parar de trabalhar",
            [(f"{c.id}-laudo_medico", off)],
            ("paraphrase", "single"),
        ),
        (
            "em que dia aconteceu o acidente",
            [
                (f"{c.id}-aviso_de_sinistro", f"Data do sinistro: {c.date}"),
                (f"{c.id}-boletim_de_ocorrencia", f"Data do fato: {c.date}"),
            ],
            ("date", "cross-document"),
        ),
        (
            "como foi o acidente",
            [
                (f"{c.id}-aviso_de_sinistro", c.narrative),
                (f"{c.id}-boletim_de_ocorrencia", c.narrative),
            ],
            ("narrative", "cross-document"),
        ),
        (
            "qual oficina fez o orçamento",
            [(f"{c.id}-orcamento_de_reparo", f"Oficina: {c.workshop}")],
            ("identifier", "single"),
        ),
        (
            "até quanto a apólice cobre em caso de incêndio ou roubo",
            [
                (
                    f"{c.id}-apolice",
                    f"Incêndio e roubo   Limite {money(c.coverage_limit)}   Franquia {money(0.0)}",
                )
            ],
            ("money", "single"),
        ),
    ]


#: Questions this corpus cannot answer. They exist so the refusal threshold can be chosen from
#: a measured score distribution instead of by feel — a system that never sees an unanswerable
#: question has no way to know what a bad retrieval looks like.
UNANSWERABLE = [
    "qual foi a decisão do juiz nesse processo trabalhista",
    "quantas ações de cobrança a empresa tem em andamento",
    "qual o CNPJ da transportadora contratada para a mudança",
    "a aposentadoria especial foi concedida pelo INSS",
    "qual o percentual de reajuste do aluguel do imóvel comercial",
    "quando vence a licença ambiental da fábrica",
    "qual foi a nota do aluno na avaliação de matemática",
    "em que data a patente foi depositada no INPI",
]


def main() -> None:
    chunks = load_chunks()
    by_id = {c.id: c for c in chunks}
    claims = build_claims(8, random.Random(SEED))

    segmenter = WordSegmenter.from_texts([c.text for c in chunks])

    rows: list[dict] = []
    unmatched = 0
    for claim in claims:
        for question, answers, tags in answerable(claim):
            relevant: list[str] = []
            for document_id, line in answers:
                found = locate(line, document_id, chunks)
                if found is None:
                    unmatched += 1
                    continue
                relevant.append(found)
            if not relevant:
                # Every relevant line was damaged past recognition. Keeping the query with an
                # empty relevance set would drag every mean down for a reason that has nothing
                # to do with retrieval.
                continue
            rows.append(
                {
                    "id": f"q{len(rows) + 1:03d}",
                    "text": question,
                    "relevant": sorted(set(relevant)),
                    "tags": list(tags),
                    "claim_id": claim.id,
                }
            )

    (FIXTURES / "queries.json").write_text(
        json.dumps({"queries": rows, "unanswerable": UNANSWERABLE}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(rows)} answerable queries, {len(UNANSWERABLE)} unanswerable")
    if unmatched:
        print(f"  {unmatched} answer line(s) could not be located in any chunk and were dropped")
    multi = sum(1 for r in rows if len(r["relevant"]) > 1)
    print(f"  {multi} of them need more than one chunk")

    # --- vectors -----------------------------------------------------------------------
    embedder = SentenceTransformerEmbedder()
    vectors: dict[str, list[float]] = {}

    # Chunks are embedded after re-segmentation, which is what the index holds: embedding the
    # merged text and searching the segmented text would compare two different corpora.
    passages = [segmenter.resegment(c.embedding_text) for c in chunks]
    print(f"embedding {len(passages)} chunks with {embedder.name}...")
    for text, vector in zip(passages, embedder.embed_passages(passages), strict=True):
        vectors[text_key(PASSAGE_PREFIX + text)] = vector

    questions = [r["text"] for r in rows] + UNANSWERABLE
    print(f"embedding {len(questions)} queries...")
    for text in questions:
        vectors[text_key(QUERY_PREFIX + text)] = embedder.embed_query(text)

    save_vectors(FIXTURES / "vectors.json", embedder.name, embedder.dimension, vectors)
    print(f"wrote {len(vectors)} vectors of {embedder.dimension} dimensions")
    assert by_id  # the chunk index is loaded for the caller's benefit; keep the read honest


if __name__ == "__main__":
    main()
