"""Recovering the structure of a page from the geometry of what was read.

An OCR engine returns boxes and strings. It does not return "this is the heading of the repair
estimate and the four lines under it are table rows", and that distinction is what makes a
claim file searchable: a chunk reading `R$ 12.480,00` is noise, and the same figure under
`ORÇAMENTO DE REPARO > Total geral` answers a question.

So the structure is inferred here, from three signals that survive a scan:

* **Height.** A heading is set larger, so its box is taller than the page's median line.
* **Case.** A heading is usually set in capitals, and OCR preserves case well even when it
  mangles diacritics.
* **Column structure.** A table row has a value — money, a date, a quantity — sitting to the
  right of a label, with a gap between them that recurs down the page.

None of these is reliable alone, which is why the classifier combines them and why its output
is scored against the true layout rather than asserted. `claims layout-report` prints the
confusion matrix; a document-intelligence step nobody measured is a document-intelligence step
nobody can trust.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from .documents import Block, BlockKind, Box

#: A value that makes a line look like a row of a table rather than prose: money, a date, a
#: percentage, a plate, a bare quantity. Deliberately permissive — the column test below is
#: what stops an ordinary sentence containing a number being called a table row.
_VALUE = re.compile(
    r"(R\$\s*[\d.,]+|\d{2}/\d{2}/\d{4}|\d+[.,]\d{2}\b|\b\d+\s*(dias?|un|pç|pc)\b|\d+\s*%)",
    re.IGNORECASE,
)

#: A footer: a page marker or a short line of small type at the bottom of the page.
#:
#: The page-of-page form is anchored to the whole line. It used to be a bare
#: `\d+\s*/\s*\d+`, which matches the `02/02` inside `02/02/2026` — so every short
#: line carrying a date was classified as a footer and dropped by the chunker. The claim
#: notification lost its `Data do sinistro` line entirely, and the one question in the
#: golden set that asks when the accident happened could never be answered by anything.
_FOOTER = re.compile(r"(p[áa]gina\s+\d+|^\d{1,3}\s*/\s*\d{1,3}$|documento gerado)", re.IGNORECASE)


@dataclass(frozen=True)
class ReadLine:
    """One line as the OCR engine returned it, before any structure is assumed."""

    text: str
    box: Box
    confidence: float


def _upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def classify(lines: list[ReadLine], page_height: int = 0) -> list[Block]:
    """Turn read lines into typed blocks.

    The thresholds are relative to the page, never absolute. A fixed "a heading is taller than
    28 pixels" works on the DPI it was tuned at and silently classifies every line as a
    heading at a different one, which is the kind of failure that produces a plausible-looking
    index full of nothing.
    """
    if not lines:
        return []

    heights = [line.box.height for line in lines if line.box.height > 0]
    median_height = statistics.median(heights) if heights else 1
    # A column boundary exists when several lines put a value at a similar horizontal offset.
    # One line with a number in it is a sentence; five lines with numbers starting within the
    # same band are a table.
    value_starts = [
        line.box.left + int(line.box.width * (match.start() / max(len(line.text), 1)))
        for line in lines
        if (match := _VALUE.search(line.text))
    ]
    has_column = len(value_starts) >= 3 and (
        statistics.pstdev(value_starts)
        < max(60, 0.08 * max(1, max(line.box.right for line in lines)))
    )

    blocks: list[Block] = []
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        kind = _classify_one(
            line,
            text,
            median_height=median_height,
            page_height=page_height,
            has_column=has_column,
        )
        blocks.append(Block(text=text, kind=kind, box=line.box, confidence=line.confidence))
    return blocks


def _classify_one(
    line: ReadLine,
    text: str,
    *,
    median_height: float,
    page_height: int,
    has_column: bool,
) -> BlockKind:
    bottom_eighth = page_height and line.box.top > page_height * 0.88
    if bottom_eighth and (len(text) < 60 or _FOOTER.search(text)):
        return "footer"
    if _FOOTER.search(text) and len(text) < 60:
        return "footer"

    tall = line.box.height >= median_height * 1.22
    shouting = _upper_ratio(text) >= 0.7
    short = len(text) <= 70

    # A heading is set larger OR in capitals, and is short either way. Requiring both misses
    # the many forms that only capitalise, and requiring neither promotes every short line.
    if short and (tall or shouting) and not _VALUE.search(text):
        return "heading"

    if has_column and _VALUE.search(text) and len(text) <= 110:
        return "table_row"

    # A label-and-value line outside a table: "Data do sinistro: 14/03/2026".
    if ":" in text and len(text) <= 110 and _VALUE.search(text):
        return "field"
    if ":" in text and len(text) <= 80 and len(text.split(":", 1)[1].split()) <= 6:
        return "field"

    return "paragraph"


@dataclass(frozen=True)
class LayoutScore:
    """How well the inference matched the true layout."""

    total: int
    correct: int
    #: ``confusion[true][inferred]``
    confusion: dict[BlockKind, dict[BlockKind, int]]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def render(self) -> str:
        kinds: list[BlockKind] = ["heading", "paragraph", "field", "table_row", "footer"]
        lines = [
            "| true \\ inferred | " + " | ".join(kinds) + " |",
            "| --- " * (len(kinds) + 1) + "|",
        ]
        for true_kind in kinds:
            row = self.confusion.get(true_kind, {})
            lines.append(
                f"| **{true_kind}** | " + " | ".join(str(row.get(k, 0)) for k in kinds) + " |"
            )
        return "\n".join(lines)


def score_layout(truth: list[Block], inferred: list[Block]) -> LayoutScore:
    """Match inferred blocks to true blocks by box overlap and compare their kinds.

    Matching by overlap rather than by position in the list, because OCR merges and splits
    lines: a comparison that assumes the two lists line up is measuring the alignment, not the
    classifier.
    """
    kinds: list[BlockKind] = ["heading", "paragraph", "field", "table_row", "footer"]
    confusion: dict[BlockKind, dict[BlockKind, int]] = {t: dict.fromkeys(kinds, 0) for t in kinds}
    correct = 0
    total = 0

    for block in inferred:
        match = _best_overlap(block.box, truth)
        if match is None:
            continue
        total += 1
        confusion[match.kind][block.kind] += 1
        if match.kind == block.kind:
            correct += 1

    return LayoutScore(total=total, correct=correct, confusion=confusion)


def _best_overlap(box: Box, candidates: list[Block]) -> Block | None:
    best: Block | None = None
    best_area = 0
    for candidate in candidates:
        other = candidate.box
        width = min(box.right, other.right) - max(box.left, other.left)
        height = min(box.bottom, other.bottom) - max(box.top, other.top)
        if width <= 0 or height <= 0:
            continue
        area = width * height
        if area > best_area:
            best_area, best = area, candidate
    return best
