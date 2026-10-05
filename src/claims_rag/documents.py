"""What a scanned document is, once a page has been read.

The shape here is the first design decision of the pipeline, and it is a decision about
**layout**. An OCR engine that returns one string per page throws away the thing that makes a
claim file navigable: that "Valor orçado: R$ 12.480,00" sat in a table under the heading
"ORÇAMENTO DE REPARO", and not in the narrative of the police report three pages earlier.

So a page is a list of :class:`Block`, each carrying its kind, its position and the confidence
the engine had in it. Chunking then cuts along those blocks rather than every 512 characters,
and every chunk remembers the headings above it. A retrieved chunk can therefore say where it
came from, which is what lets an answer be checked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

#: What a block of a claim document is. Deliberately coarse: these are the distinctions that
#: change how a block should be chunked and weighted, not a full document ontology.
BlockKind = Literal["heading", "paragraph", "field", "table_row", "footer"]

#: The documents a motor claim file actually arrives as, in Brazil.
DocumentKind = Literal[
    "aviso_de_sinistro",  # the claim notification itself
    "boletim_de_ocorrencia",  # police report
    "orcamento_de_reparo",  # repair estimate
    "laudo_medico",  # medical report
    "apolice",  # policy excerpt
]

DOCUMENT_KINDS: tuple[DocumentKind, ...] = (
    "aviso_de_sinistro",
    "boletim_de_ocorrencia",
    "orcamento_de_reparo",
    "laudo_medico",
    "apolice",
)


@dataclass(frozen=True)
class Box:
    """Where a block sat on the page, in pixels, origin top-left.

    Kept because it is the only thing that can answer "show me where you read that", and
    because reading order on a two-column page is a function of geometry, not of the order an
    engine happened to emit.
    """

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True)
class Block:
    """One laid-out piece of a page."""

    text: str
    kind: BlockKind
    box: Box
    #: The engine's own confidence, 0 to 1. Carried through to retrieval so a chunk built from
    #: barely-legible text can be told apart from a clean one, instead of both arriving as
    #: equally authoritative strings.
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")


@dataclass(frozen=True)
class Page:
    """One scanned page, read."""

    number: int
    blocks: tuple[Block, ...]
    #: Pixel size of the image the blocks were read from, so boxes can be scaled.
    width: int = 0
    height: int = 0

    @property
    def text(self) -> str:
        """The page as one string, in reading order. For OCR scoring, not for retrieval."""
        return "\n".join(b.text for b in self.blocks)

    @property
    def mean_confidence(self) -> float:
        if not self.blocks:
            return 0.0
        return sum(b.confidence for b in self.blocks) / len(self.blocks)


@dataclass(frozen=True)
class Document:
    """A claim document: what it is, which claim it belongs to, and its pages."""

    id: str
    kind: DocumentKind
    claim_id: str
    pages: tuple[Page, ...]
    #: Free metadata that survives into every chunk, so a filter can be applied before search
    #: rather than after it.
    meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("every document needs an id")
        if self.kind not in DOCUMENT_KINDS:
            raise ValueError(f"{self.kind!r} is not one of {DOCUMENT_KINDS}")

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit, and everything needed to say where it came from."""

    id: str
    document_id: str
    claim_id: str
    kind: DocumentKind
    page: int
    text: str
    #: The headings above this chunk, outermost first. Prepended to the embedded text so that
    #: a figure in a table is searchable by the section it belongs to, and shown to the reader
    #: so an answer can be traced.
    headings: tuple[str, ...] = ()
    #: The lowest block confidence that went into this chunk.
    confidence: float = 1.0

    @property
    def embedding_text(self) -> str:
        """What is actually embedded: the headings, then the text.

        A row reading "Valor orçado | R$ 12.480,00" means nothing on its own. With
        "ORÇAMENTO DE REPARO" in front of it, both a dense and a lexical index can find it.
        """
        if not self.headings:
            return self.text
        return " > ".join(self.headings) + "\n" + self.text

    def cite(self) -> str:
        """A short, human-checkable provenance string."""
        where = " > ".join(self.headings) if self.headings else self.kind
        return f"{self.document_id} p.{self.page} ({where})"


# --------------------------------------------------------------------------------------
# Serialisation. Plain JSON, because these files are committed and read by people.
# --------------------------------------------------------------------------------------


def _block_to_dict(block: Block) -> dict:
    return {
        "text": block.text,
        "kind": block.kind,
        "box": list(block.box.as_tuple()),
        "confidence": round(block.confidence, 4),
    }


def _block_from_dict(payload: dict) -> Block:
    left, top, right, bottom = payload["box"]
    return Block(
        text=str(payload["text"]),
        kind=payload["kind"],
        box=Box(int(left), int(top), int(right), int(bottom)),
        confidence=float(payload.get("confidence", 1.0)),
    )


def document_to_dict(doc: Document) -> dict:
    return {
        "id": doc.id,
        "kind": doc.kind,
        "claim_id": doc.claim_id,
        "meta": doc.meta,
        "pages": [
            {
                "number": p.number,
                "width": p.width,
                "height": p.height,
                "blocks": [_block_to_dict(b) for b in p.blocks],
            }
            for p in doc.pages
        ],
    }


def document_from_dict(payload: dict) -> Document:
    return Document(
        id=str(payload["id"]),
        kind=payload["kind"],
        claim_id=str(payload["claim_id"]),
        meta=dict(payload.get("meta", {})),
        pages=tuple(
            Page(
                number=int(p["number"]),
                width=int(p.get("width", 0)),
                height=int(p.get("height", 0)),
                blocks=tuple(_block_from_dict(b) for b in p["blocks"]),
            )
            for p in payload["pages"]
        ),
    )


def load_documents(path: str | Path) -> list[Document]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = [document_from_dict(row) for row in payload["documents"]]

    seen = [d.id for d in docs]
    duplicates = {i for i in seen if seen.count(i) > 1}
    if duplicates:
        # A duplicated id makes one document unreachable and silently double-weights the
        # other in every metric computed over the corpus.
        raise ValueError(f"duplicate document ids in {path}: {sorted(duplicates)}")
    return docs


def save_documents(docs: list[Document], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"documents": [document_to_dict(d) for d in docs]}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
