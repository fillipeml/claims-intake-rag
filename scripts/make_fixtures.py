"""Builds the corpus: synthetic claim files, rendered as scans, then read back by OCR.

Nothing here is simulated twice. The documents are invented and their text is known exactly,
so they are rendered to an image, degraded the way a desk scanner degrades a page, and then
**actually read by an OCR engine**. What the rest of the repository indexes is the engine's
output, mistakes and all, and the character error rate it reports is measured against the text
that went in rather than against a noise model this script made up.

That matters because the error profile drives a design decision. Run it and the engine drops
diacritics almost everywhere — ``orçado`` becomes ``orcado``, ``alça`` becomes ``alca`` — which
is why the lexical index folds accents instead of matching them.

Every identifier is invalid by construction: the CPFs fail their check digits, the policy and
claim numbers belong to no scheme, and the municipality and state do not exist.

    uv run python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
import random
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claims_rag.documents import (  # noqa: E402
    Block,
    Box,
    Chunk,
    Document,
    DocumentKind,
    Page,
    save_documents,
)
from claims_rag.layout import ReadLine, classify, score_layout  # noqa: E402

SEED = 20261005
FIXTURES = ROOT / "fixtures"
PAGES_DIR = FIXTURES / "pages"

#: A4 at 300 dpi. Not an arbitrary choice: below roughly 38 pixels of glyph height the
#: recognition model stops emitting word boundaries and returns "Datadoatendimento:08/08/2026"
#: — the characters are right and every word is welded to the next, which destroys a lexical
#: index far more thoroughly than a wrong character does. 300 dpi is also what every scanning
#: guideline recommends for OCR, so the fix and the convention agree.
PAGE_W, PAGE_H = 2480, 3508
MARGIN = 192

FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"

SIZES = {"heading": 62, "field": 46, "table_row": 46, "paragraph": 46, "footer": 32}
LEADING = {"heading": 52, "field": 24, "table_row": 20, "paragraph": 20, "footer": 20}


# --------------------------------------------------------------------------------------
# Invented data
# --------------------------------------------------------------------------------------

FIRST = ["Helena", "Rafael", "Tereza", "Gustavo", "Nádia", "Otávio", "Beatriz", "Leandro"]
LAST = ["Vilarinho", "Caldeira", "Bastos", "Quintela", "Moraes", "Trindade", "Peçanha", "Arruda"]
CITIES = ["Porto das Águas", "Vila Serena", "Campo do Meio", "Nova Aurora"]
STATE = "Zetalândia"
VEHICLES = [
    ("Hatch compacto 1.0", "2021"),
    ("Sedã médio 2.0", "2019"),
    ("Utilitário leve", "2022"),
    ("SUV compacto 1.4", "2020"),
]
WORKSHOPS = ["Oficina Modelo", "Auto Center Exemplo", "Reparação Fictícia Ltda"]
CAUSES = [
    (
        "colisão traseira",
        "O condutor relatou que trafegava pela via marginal quando o veículo à frente freou "
        "bruscamente, não havendo espaço para evitar o impacto na traseira.",
    ),
    (
        "capotamento",
        "Segundo o condutor, houve perda de aderência na alça de acesso da rodovia sob chuva, "
        "seguida de capotamento lateral sobre o canteiro.",
    ),
    (
        "alagamento",
        "O veículo foi surpreendido por elevação rápida do nível da água em via pública, "
        "atingindo o compartimento do motor.",
    ),
    (
        "colisão lateral em cruzamento",
        "O condutor informou que cruzava a via preferencial com sinalização semafórica "
        "intermitente quando foi atingido na lateral direita.",
    ),
]
PARTS = [
    ("Para-choque dianteiro", 1, 2100.00),
    ("Farol direito", 1, 1480.00),
    ("Capô", 1, 3250.00),
    ("Paralama esquerdo", 1, 1690.00),
    ("Airbag do condutor", 1, 4320.00),
    ("Vidro para-brisa", 1, 980.00),
]


def invalid_cpf(rng: random.Random) -> str:
    """A CPF-shaped string whose check digits are deliberately wrong.

    The two digits are computed properly and then replaced with different ones, so the number
    has the right shape, fails every validator, and can never collide with a real person's.
    """
    body = [rng.randrange(10) for _ in range(9)]

    def digit(seq: list[int]) -> int:
        weight = len(seq) + 1
        total = sum(d * (weight - i) for i, d in enumerate(seq))
        remainder = total % 11
        return 0 if remainder < 2 else 11 - remainder

    d1 = digit(body)
    d2 = digit([*body, d1])
    wrong1 = (d1 + 1) % 10
    wrong2 = (d2 + 1) % 10
    assert (wrong1, wrong2) != (d1, d2)
    n = "".join(str(d) for d in body)
    return f"{n[:3]}.{n[3:6]}.{n[6:9]}-{wrong1}{wrong2}"


@dataclass
class ClaimFile:
    id: str
    policy: str
    insured: str
    cpf: str
    city: str
    plate: str
    vehicle: str
    model_year: str
    date: str
    cause: str
    narrative: str
    police_report: str
    workshop: str
    parts: list[tuple[str, int, float]]
    labour: float
    deductible: float
    coverage_limit: float
    days_off: int
    cid: str
    licence_valid: bool

    @property
    def parts_total(self) -> float:
        return sum(price * qty for _, qty, price in self.parts)

    @property
    def total(self) -> float:
        return self.parts_total + self.labour


def build_claims(count: int, rng: random.Random) -> list[ClaimFile]:
    claims = []
    for i in range(1, count + 1):
        vehicle, year = VEHICLES[(i - 1) % len(VEHICLES)]
        cause, narrative = CAUSES[(i - 1) % len(CAUSES)]
        parts = rng.sample(PARTS, k=rng.randint(2, 4))
        claims.append(
            ClaimFile(
                id=f"AV-2026-{4400 + i * 7}",
                policy=f"2026.{100000 + i * 137}",
                insured=f"{FIRST[(i - 1) % len(FIRST)]} {LAST[(i - 1) % len(LAST)]}",
                cpf=invalid_cpf(rng),
                city=CITIES[(i - 1) % len(CITIES)],
                plate=f"Z{chr(65 + i % 26)}{chr(65 + (i * 3) % 26)}{i % 10}{chr(65 + i % 26)}{i % 10}{(i * 7) % 10}",
                vehicle=vehicle,
                model_year=year,
                date=f"{(i % 27) + 1:02d}/0{(i % 9) + 1}/2026",
                cause=cause,
                narrative=narrative,
                police_report=f"BO-{2026}{3000 + i * 11}",
                workshop=WORKSHOPS[(i - 1) % len(WORKSHOPS)],
                parts=parts,
                labour=round(rng.uniform(600, 2400), 2),
                deductible=round(rng.choice([2800, 3400, 4200, 5100]), 2),
                coverage_limit=round(rng.choice([60000, 80000, 95000]), 2),
                days_off=rng.choice([0, 7, 15, 30, 45]),
                cid=rng.choice(["S00.0", "S13.4", "S42.2", "S82.1"]),
                licence_valid=(i % 4 != 0),
            )
        )
    return claims


# --------------------------------------------------------------------------------------
# The five documents of a motor claim file
# --------------------------------------------------------------------------------------

#: One logical line: its text and what it really is. The renderer turns these into pixels and
#: the layout classifier later has to recover the kind from the pixels alone.
Line = tuple[str, str]


def money(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")


def doc_aviso(c: ClaimFile) -> list[Line]:
    return [
        ("AVISO DE SINISTRO", "heading"),
        (f"Número do aviso: {c.id}", "field"),
        (f"Apólice: {c.policy}", "field"),
        (f"Segurado: {c.insured}", "field"),
        (f"CPF: {c.cpf}", "field"),
        (f"Data do sinistro: {c.date}", "field"),
        (f"Local: {c.city} - {STATE}", "field"),
        (f"Placa: {c.plate}", "field"),
        (f"Veículo: {c.vehicle}, ano {c.model_year}", "field"),
        (f"Natureza do evento: {c.cause}", "field"),
        ("DESCRIÇÃO DO SEGURADO", "heading"),
        (c.narrative, "paragraph"),
        (
            "O segurado declara que as informações acima são verdadeiras e autoriza a "
            "seguradora a solicitar documentos complementares para a regulação do sinistro.",
            "paragraph",
        ),
        (f"Documento gerado em {c.date} - página 1/1", "footer"),
    ]


def doc_boletim(c: ClaimFile) -> list[Line]:
    licence = "válida" if c.licence_valid else "vencida há mais de 30 dias"
    return [
        ("BOLETIM DE OCORRÊNCIA", "heading"),
        (f"Número: {c.police_report}", "field"),
        (f"Delegacia: 2ª Circunscrição de {c.city}", "field"),
        (f"Data do fato: {c.date}", "field"),
        (f"Condutor: {c.insured}", "field"),
        (f"Habilitação: {licence}", "field"),
        (f"Veículo envolvido: placa {c.plate}", "field"),
        ("RELATO", "heading"),
        (c.narrative, "paragraph"),
        (
            "Não houve vítima fatal. O condutor foi encaminhado para atendimento médico de "
            "rotina e o veículo removido por guincho credenciado.",
            "paragraph",
        ),
        ("Página 1 de 1", "footer"),
    ]


def doc_orcamento(c: ClaimFile) -> list[Line]:
    lines: list[Line] = [
        ("ORÇAMENTO DE REPARO", "heading"),
        (f"Oficina: {c.workshop}", "field"),
        (f"Sinistro: {c.id}", "field"),
        (f"Placa: {c.plate}", "field"),
        ("PEÇAS", "heading"),
    ]
    for name, qty, price in c.parts:
        lines.append((f"{name}   {qty} un   {money(price)}", "table_row"))
    lines += [
        ("MÃO DE OBRA", "heading"),
        (f"Serviços de funilaria e pintura   1 un   {money(c.labour)}", "table_row"),
        ("TOTAIS", "heading"),
        (f"Subtotal de peças   {money(c.parts_total)}", "table_row"),
        (f"Subtotal de mão de obra   {money(c.labour)}", "table_row"),
        (f"Total geral   {money(c.total)}", "table_row"),
        (
            "Orçamento válido por 15 dias. Peças sujeitas a confirmação de disponibilidade "
            "junto ao fornecedor.",
            "paragraph",
        ),
        ("Página 1 de 1", "footer"),
    ]
    return lines


def doc_laudo(c: ClaimFile) -> list[Line]:
    off = (
        "Não houve necessidade de afastamento das atividades habituais."
        if c.days_off == 0
        else f"Recomendado afastamento das atividades habituais por {c.days_off} dias."
    )
    return [
        ("LAUDO MÉDICO", "heading"),
        (f"Paciente: {c.insured}", "field"),
        (f"CPF: {c.cpf}", "field"),
        (f"Data do atendimento: {c.date}", "field"),
        (f"CID-10: {c.cid}", "field"),
        ("AVALIAÇÃO", "heading"),
        (
            "Paciente atendido após evento automobilístico, consciente e orientado, sem "
            "alterações neurológicas agudas ao exame inicial.",
            "paragraph",
        ),
        (off, "paragraph"),
        (
            "Solicitado retorno ambulatorial em quinze dias para reavaliação clínica.",
            "paragraph",
        ),
        ("Página 1 de 1", "footer"),
    ]


def doc_apolice(c: ClaimFile) -> list[Line]:
    return [
        ("APÓLICE DE SEGURO DE AUTOMÓVEL", "heading"),
        (f"Apólice: {c.policy}", "field"),
        (f"Segurado: {c.insured}", "field"),
        ("Vigência: 01/01/2026 a 31/12/2026", "field"),
        ("COBERTURAS CONTRATADAS", "heading"),
        (
            f"Colisão e capotamento   Limite {money(c.coverage_limit)}   Franquia {money(c.deductible)}",
            "table_row",
        ),
        (
            f"Incêndio e roubo   Limite {money(c.coverage_limit)}   Franquia {money(0.0)}",
            "table_row",
        ),
        (
            f"Danos a terceiros   Limite {money(c.coverage_limit * 0.5)}   Franquia {money(c.deductible / 2)}",
            "table_row",
        ),
        ("EXCLUSÕES", "heading"),
        (
            "Não estão cobertos os danos ocorridos quando o condutor estiver sem habilitação "
            "válida na data do evento, bem como os decorrentes de uso do veículo em competição.",
            "paragraph",
        ),
        ("Página 1 de 1", "footer"),
    ]


BUILDERS: dict[DocumentKind, object] = {
    "aviso_de_sinistro": doc_aviso,
    "boletim_de_ocorrencia": doc_boletim,
    "orcamento_de_reparo": doc_orcamento,
    "laudo_medico": doc_laudo,
    "apolice": doc_apolice,
}


# --------------------------------------------------------------------------------------
# Rendering: logical lines to a page that looks like it came off a scanner
# --------------------------------------------------------------------------------------


def wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.getlength(candidate) <= width or not current:
            current = candidate
        else:
            out.append(current)
            current = word
    if current:
        out.append(current)
    return out


def render_page(lines: list[Line], rng: random.Random) -> tuple[Image.Image, list[Block]]:
    """Draw the lines and return the image plus the blocks that are true by construction."""
    image = Image.new("L", (PAGE_W, PAGE_H), 255)
    draw = ImageDraw.Draw(image)
    fonts = {
        kind: ImageFont.truetype(FONT_BOLD if kind == "heading" else FONT_REGULAR, size)
        for kind, size in SIZES.items()
    }

    truth: list[Block] = []
    y = MARGIN
    usable = PAGE_W - 2 * MARGIN

    for text, kind in lines:
        font = fonts[kind]
        for piece in wrap(text, font, usable):
            left, top, right, bottom = draw.textbbox((MARGIN, y), piece, font=font)
            draw.text((MARGIN, y), piece, font=font, fill=20)
            truth.append(
                Block(
                    text=piece,
                    kind=kind,  # type: ignore[arg-type]
                    box=Box(int(left), int(top), int(right), int(bottom)),
                    confidence=1.0,
                )
            )
            y = bottom + LEADING[kind]
        y += 6

    return _scan(image, rng), truth


def _scan(image: Image.Image, rng: random.Random) -> Image.Image:
    """Make the page look like it went through a desk scanner.

    A slight skew, a little blur, sensor noise and a JPEG round-trip. Not to be adversarial —
    to stop the corpus being a rendering of a font at a size the engine finds trivial, which
    would give a character error rate of zero and tell nobody anything.
    """
    angle = rng.uniform(-0.45, 0.45)
    rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255, expand=False)
    blurred = rotated.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.1)))

    # numpy's generator, seeded from the same stream, rather than a Python loop: the loop was
    # 8.7 million gauss() calls per page and dominated the whole run.
    generator = np.random.default_rng(rng.randrange(2**32))
    array = np.asarray(blurred).astype(np.float32)
    noisy = np.clip(array + generator.normal(0, 6.0, array.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="L")


# --------------------------------------------------------------------------------------
# Reading it back
# --------------------------------------------------------------------------------------


def read_page(ocr, image: Image.Image) -> list[ReadLine]:
    result, _ = ocr(np.array(image.convert("RGB")))
    lines: list[ReadLine] = []
    for box, text, confidence in result or []:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        lines.append(
            ReadLine(
                text=str(text),
                box=Box(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                confidence=float(confidence),
            )
        )
    lines.sort(key=lambda line: (line.box.top, line.box.left))
    return lines


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def error_rates(truth: str, read: str) -> tuple[float, float, float]:
    """Character and word error rate, plus CER once accents are folded away.

    The third number is the one that decided the tokeniser. If folding accents collapses most
    of the character error, then the engine is reading the letters correctly and losing the
    diacritics — and an index that folds them loses nothing the engine actually knew.
    """
    truth_flat = " ".join(truth.split())
    read_flat = " ".join(read.split())
    cer = edit_distance(truth_flat, read_flat) / max(len(truth_flat), 1)
    wer = edit_distance_words(truth_flat.split(), read_flat.split()) / max(
        len(truth_flat.split()), 1
    )
    folded = edit_distance(_fold(truth_flat), _fold(read_flat)) / max(len(truth_flat), 1)
    return cer, wer, folded


def edit_distance_words(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, wa in enumerate(a, start=1):
        current = [i]
        for j, wb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (wa != wb)))
        previous = current
    return previous[-1]


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------


def chunk_document(doc: Document) -> list[Chunk]:
    """Cut along the layout, not every N characters.

    A heading opens a section and the blocks under it belong to it. **Fields and table rows get
    one chunk each**, because each is already the unit somebody asks about; paragraphs are
    grouped under their heading until the next one, because half a sentence retrieves badly.

    Fields were grouped at first, which put the nine header fields of a claim notification into
    a single chunk. Measured, that was the largest retrieval defect in the repository: asked
    whether the driver was licensed, the system returned the notification's header block —
    claim number, policy, insured — with a cross-encoder score of 0.78, while the police report
    line reading `Habilitação: válida` did not reach the top four at all. A chunk holding nine
    fields answers nine questions badly instead of one question well.
    """
    chunks: list[Chunk] = []
    headings: list[str] = []
    buffer: list[str] = []
    buffer_confidence = 1.0
    counter = 0

    def flush(page: int) -> None:
        nonlocal buffer, buffer_confidence, counter
        if not buffer:
            return
        counter += 1
        chunks.append(
            Chunk(
                id=f"{doc.id}#c{counter:02d}",
                document_id=doc.id,
                claim_id=doc.claim_id,
                kind=doc.kind,
                page=page,
                text="\n".join(buffer),
                headings=tuple(headings),
                confidence=buffer_confidence,
            )
        )
        buffer = []
        buffer_confidence = 1.0

    for page in doc.pages:
        for block in page.blocks:
            if block.kind == "footer":
                continue
            if block.kind == "heading":
                flush(page.number)
                # A heading in capitals opens a top-level section; keep at most two levels so
                # the trail stays readable in a citation.
                headings = [block.text] if len(headings) < 1 else [headings[0], block.text]
                continue
            if block.kind in ("table_row", "field"):
                flush(page.number)
                counter += 1
                chunks.append(
                    Chunk(
                        id=f"{doc.id}#c{counter:02d}",
                        document_id=doc.id,
                        claim_id=doc.claim_id,
                        kind=doc.kind,
                        page=page.number,
                        text=block.text,
                        headings=tuple(headings),
                        confidence=block.confidence,
                    )
                )
                continue
            buffer.append(block.text)
            buffer_confidence = min(buffer_confidence, block.confidence)
        flush(page.number)

    return chunks


def main() -> None:
    from rapidocr_onnxruntime import RapidOCR

    rng = random.Random(SEED)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    claims = build_claims(8, rng)
    ocr = RapidOCR()

    read_docs: list[Document] = []
    truth_docs: list[Document] = []
    rows: list[dict] = []

    for claim in claims:
        for kind, builder in BUILDERS.items():
            lines = builder(claim)  # type: ignore[operator]
            image, truth_blocks = render_page(lines, rng)
            doc_id = f"{claim.id}-{kind}"
            image_path = PAGES_DIR / f"{doc_id}.png"
            image.save(image_path, optimize=True)  # gitignored: regenerate, do not commit

            read_lines = read_page(ocr, image)
            inferred = classify(read_lines, page_height=PAGE_H)

            truth_page = Page(number=1, blocks=tuple(truth_blocks), width=PAGE_W, height=PAGE_H)
            read_page_obj = Page(number=1, blocks=tuple(inferred), width=PAGE_W, height=PAGE_H)

            truth_docs.append(
                Document(id=doc_id, kind=kind, claim_id=claim.id, pages=(truth_page,))
            )
            read_docs.append(
                Document(
                    id=doc_id,
                    kind=kind,
                    claim_id=claim.id,
                    pages=(read_page_obj,),
                    meta={"image": f"pages/{doc_id}.png"},
                )
            )

            cer, wer, folded = error_rates(truth_page.text, read_page_obj.text)
            layout = score_layout(truth_blocks, inferred)
            rows.append(
                {
                    "document": doc_id,
                    "kind": kind,
                    "cer": cer,
                    "wer": wer,
                    "cer_accent_folded": folded,
                    "layout_total": layout.total,
                    "layout_correct": layout.correct,
                    "mean_confidence": read_page_obj.mean_confidence,
                }
            )
            print(
                f"  {doc_id:34s} CER {cer:6.2%}  folded {folded:6.2%}  "
                f"layout {layout.correct}/{layout.total}"
            )

    save_documents(read_docs, FIXTURES / "documents.json")
    save_documents(truth_docs, FIXTURES / "truth.json")

    chunks = [c for doc in read_docs for c in chunk_document(doc)]
    (FIXTURES / "chunks.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "id": c.id,
                        "document_id": c.document_id,
                        "claim_id": c.claim_id,
                        "kind": c.kind,
                        "page": c.page,
                        "text": c.text,
                        "headings": list(c.headings),
                        "confidence": round(c.confidence, 4),
                    }
                    for c in chunks
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (FIXTURES / "ocr-measurements.json").write_text(
        json.dumps({"pages": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    mean_cer = sum(r["cer"] for r in rows) / len(rows)
    mean_folded = sum(r["cer_accent_folded"] for r in rows) / len(rows)
    mean_wer = sum(r["wer"] for r in rows) / len(rows)
    layout_total = sum(r["layout_total"] for r in rows)
    layout_correct = sum(r["layout_correct"] for r in rows)

    print()
    print(f"{len(read_docs)} documents, {len(chunks)} chunks")
    print(f"mean CER {mean_cer:.2%} | accent-folded {mean_folded:.2%} | WER {mean_wer:.2%}")
    print(f"layout inference {layout_correct}/{layout_total} = {layout_correct / layout_total:.1%}")


if __name__ == "__main__":
    main()
