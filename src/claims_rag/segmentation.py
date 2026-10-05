"""Putting the spaces back.

The recognition model used here keeps characters and loses word boundaries, inconsistently:
the same page returns ``Paciente: Beatriz Pecanha`` on one line and
``Datadoatendimento:08/08/2026`` on the next. The character error rate barely notices — one
missing space is one edit — but a lexical index notices enormously, because
``datadoatendimento`` is a single token that the query "data do atendimento" can never match.
It is the difference between a 12% character error rate and an 80% word error rate on the same
page.

The fix is the classic one: choose the segmentation that maximises the probability of the
words under a unigram model, by dynamic programming. What makes it work here without shipping
a dictionary is where the model comes from — **the corpus supplies its own vocabulary**. Most
lines survive with their spaces intact, so the words seen space-separated across the corpus,
with their frequencies, are exactly the language model needed to split the lines that did not.

Unknown strings are charged a cost that grows with length, so the splitter prefers a few known
words to many invented ones and leaves genuinely unfamiliar text — a plate, a policy number —
alone rather than shattering it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .lexicon import base_vocabulary

#: Below this length a run of letters is left alone.
#:
#: It was 12, which measured well on prose and silently skipped every table row: the run in
#: `TotalgeralR$10.372,00` is `TotalgeralR`, eleven letters, so the one line of the repair
#: estimate anybody ever asks about was never re-segmented. Dropping to 8 costs little,
#: because a word already in the vocabulary is returned untouched before the search even
#: starts, and the search only splits when splitting is cheaper.
MIN_LENGTH = 8

#: No single Portuguese word in this corpus is longer than this, so the search never considers
#: candidates past it. Bounds the dynamic programme and removes a class of absurd splits.
MAX_WORD = 18

_RUN = re.compile(r"[A-Za-zÀ-ÿ]+")


@dataclass(frozen=True)
class SegmentationReport:
    """What the splitter did to a corpus, so the change can be measured rather than assumed."""

    tokens_examined: int
    tokens_split: int
    words_recovered: int

    def describe(self) -> str:
        if not self.tokens_examined:
            return "nothing to segment"
        share = self.tokens_split / self.tokens_examined
        return (
            f"{self.tokens_split} of {self.tokens_examined} long runs split "
            f"({share:.1%}), recovering {self.words_recovered} words"
        )


class WordSegmenter:
    """Maximum-likelihood word segmentation over a vocabulary learned from the corpus."""

    def __init__(self, frequencies: Counter[str], unknown_penalty: float = 12.0) -> None:
        self._total = sum(frequencies.values()) or 1
        self._frequencies = frequencies
        self._unknown_penalty = unknown_penalty
        self._cost = lru_cache(maxsize=200_000)(self._word_cost)

    @classmethod
    def from_texts(cls, texts: list[str], seed: bool = True) -> WordSegmenter:
        """Learn from the lines that kept their spaces, on top of the domain glossary.

        Only runs already short enough to be a plausible single word are counted, so a merged
        line cannot teach the model that ``datadoatendimento`` is a word and then license
        itself.

        ``seed=False`` gives the corpus-only vocabulary. It is kept because it is the thing
        that was tried first and it is worth being able to re-measure: on this corpus it
        leaves ``atendimento``, ``afastamento`` and ``habituais`` at frequency zero, because
        every line containing them is a line that merged.
        """
        seeded: Counter[str] = base_vocabulary() if seed else Counter()
        observed: Counter[str] = Counter()
        for text in texts:
            for run in _RUN.findall(text.lower()):
                if 1 <= len(run) < MIN_LENGTH:
                    observed[run] += 1

        if not seed:
            return cls(seeded + observed)

        # A unigram model learned from OCR output learns OCR's mistakes as words. On this
        # corpus the splitter had taught itself `datado`, because a line somewhere welded
        # "Data do" into one token while keeping its other spaces — and `datado` then beat
        # `data`+`do` on every later line, by two characters' worth of cost.
        #
        # The rule that removes it without a list of exceptions: a candidate that is better
        # explained as a sequence of two or more words already in the glossary is not a word.
        glossary_only = cls(seeded)
        kept = Counter(seeded)
        for word, count in observed.items():
            if word in seeded or len(glossary_only._explain(word)) < 2:
                kept[word] += count
        return cls(kept)

    def _explain(self, word: str) -> list[str]:
        """The cheapest decomposition of a word under this vocabulary, ignoring MIN_LENGTH.

        Used to decide whether an observed string deserves to be a vocabulary entry at all.
        """
        n = len(word)
        best = [0.0] + [math.inf] * n
        back = [0] * (n + 1)
        for end in range(1, n + 1):
            for start in range(max(0, end - MAX_WORD), end):
                piece = word[start:end]
                # Only a word the glossary already knows may be used as evidence here, or the
                # test would explain everything by inventing pieces.
                if piece not in self._frequencies:
                    continue
                candidate = best[start] + self._cost(piece)
                if candidate < best[end]:
                    best[end] = candidate
                    back[end] = start
        if best[n] == math.inf:
            return [word]
        pieces: list[str] = []
        end = n
        while end > 0:
            start = back[end]
            pieces.append(word[start:end])
            end = start
        return pieces[::-1]

    def _word_cost(self, word: str) -> float:
        frequency = self._frequencies.get(word, 0)
        if frequency:
            return -math.log(frequency / self._total)
        # Unknown words are charged a cost that grows with length at ln(10) per character:
        # an unseen string of n letters is treated as roughly ten times less likely for every
        # character it adds. The first version charged 0.6 per character, which made a
        # fifty-character run cost about the same as the eight real words inside it, so the
        # search was indifferent and kept it whole.
        return self._unknown_penalty + math.log(10) * len(word)

    def split(self, run: str) -> list[str]:
        """The best segmentation of one run of letters, by total cost."""
        lowered = run.lower()
        n = len(lowered)
        if n < MIN_LENGTH or lowered in self._frequencies:
            return [run]

        # best[i] is the cost of the cheapest segmentation of the first i characters.
        best = [0.0] + [math.inf] * n
        back = [0] * (n + 1)
        for end in range(1, n + 1):
            for start in range(max(0, end - MAX_WORD), end):
                candidate = best[start] + self._cost(lowered[start:end])
                if candidate < best[end]:
                    best[end] = candidate
                    back[end] = start

        pieces: list[str] = []
        end = n
        while end > 0:
            start = back[end]
            pieces.append(run[start:end])
            end = start
        pieces.reverse()
        return pieces

    def resegment(self, text: str) -> str:
        """Rewrite a line, splitting every long unknown run of letters."""

        def replace(match: re.Match[str]) -> str:
            return " ".join(self.split(match.group(0)))

        return _RUN.sub(replace, text)

    def resegment_all(self, texts: list[str]) -> tuple[list[str], SegmentationReport]:
        examined = 0
        split = 0
        recovered = 0
        out: list[str] = []
        for text in texts:
            for run in _RUN.findall(text):
                if len(run) >= MIN_LENGTH and run.lower() not in self._frequencies:
                    examined += 1
                    pieces = self.split(run)
                    if len(pieces) > 1:
                        split += 1
                        recovered += len(pieces) - 1
            out.append(self.resegment(text))
        return out, SegmentationReport(examined, split, recovered)
