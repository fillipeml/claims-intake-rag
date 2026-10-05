# claims-intake-rag

Retrieval over **scanned Portuguese insurance-claim documents**: OCR with a measured error
rate, layout recovered from geometry and scored against the truth, word boundaries put back by
a segmenter that learns its own vocabulary, hybrid search fused with RRF, cross-encoder
reranking, and an answer the system **refuses to give** when retrieval is weak.

> **Status: in progress.** The document-intelligence half is built and measured; the numbers
> below were produced by the commands shown and are written here only after being run. The
> retrieval ablation — the table this repository exists to produce — is **not yet measured**,
> and this README will say so until it is. Nothing here is estimated or predicted.

---

## Why this exists

An insurer's claims intake is a pile of scans: a notification, a police report, a repair
estimate, a medical report, a policy excerpt. The questions asked of that pile are specific —
*what was the total estimate*, *was the licence valid on the day*, *what is the deductible for
collision* — and the answers sit in one line of one document. Getting them out means reading
badly-scanned Portuguese, recovering enough structure to know which section a figure belongs
to, and retrieving the right line among thousands.

The part most RAG projects skip is the part that decides whether any of it works: **measuring
retrieval**. A pipeline that looks right on five hand-picked queries and has no golden set
cannot tell you whether a change helped the queries you looked at and hurt the ones you did
not.

---

## What is measured so far

### Reading the page

The corpus is forty synthetic claim documents, rendered to A4 at 300 dpi, degraded the way a
desk scanner degrades a page — a slight skew, a little blur, sensor noise — and then **read
back by a real OCR engine**. Nothing is simulated twice: the error rates below are measured
against the text that went into the renderer.

```bash
uv run python scripts/make_fixtures.py
```

|  | CER | WER |
| --- | --- | --- |
| Raw OCR output | 12.07% | 80.60% |
| After word re-segmentation | 8.21% | 56.22% |
| After folding accents — what the lexical index actually sees | **5.03%** | **44.22%** |

Two findings came out of running it rather than assuming it.

**The engine stops emitting word boundaries below about 38 pixels of glyph height.** At 32px it
returns `Datadoatendimento:08/08/2026`; at 38px it returns `Data do atendimento: 08/08/2026`.
The character error rate barely notices — a missing space is one edit — while the word error
rate explodes, because `datadoatendimento` is a token no query can match. That is why the
corpus renders at 300 dpi, which is also what every scanning guideline recommends for OCR; the
fix and the convention agree, but only one of them was measured.

**A vocabulary learned from OCR output learns OCR's mistakes as words.** The segmenter taught
itself `datado`, because one line somewhere welded *Data do* into a single token while keeping
its other spaces — and `datado` then beat `data` + `do` on every later line. The rule that
removes it without a list of exceptions: *a candidate better explained as a sequence of words
already known is not a word.*

### Putting the spaces back

`segmentation.py` chooses the segmentation maximising the probability of the words under a
unigram model, by dynamic programming. The vocabulary comes from the lines that kept their
spaces — which is elegant and, measured, insufficient:

| Vocabulary | CER | WER |
| --- | --- | --- |
| None (raw) | 12.07% | 80.60% |
| Learned from the corpus only | 10.24% | 70.10% |
| Corpus + domain glossary, artefacts rejected | **8.21%** | **56.22%** |

The corpus-only vocabulary left `atendimento`, `afastamento` and `habituais` at frequency
zero, because every line containing them is a line that merged. **A corpus cannot teach itself
a word it only ever shows welded to its neighbours.** So the vocabulary is seeded from a
glossary of the few hundred words a motor claim is made of — the sort any claims operation
already maintains — and corpus frequencies are added on top.

One thing I got wrong first: an unknown word was charged 0.6 per character, which made a
fifty-character run cost about the same as the eight real words inside it, so the search was
indifferent and kept it whole. Charging ln(10) per character — each extra character makes an
unseen string ten times less likely — fixed it.

### Recovering the layout

An OCR engine returns boxes and strings, not "this is a heading and the four lines under it
are table rows". `layout.py` infers the kind from three signals that survive a scan: height
relative to the page median, capitalisation, and whether values recur in a column. It is
**scored against the true layout**, because a document-intelligence step nobody measured is
one nobody can trust.

**448 of 542 blocks correct — 82.7%.**

---

## Not yet measured

- **The retrieval ablation.** Lexical only, dense only, hybrid, hybrid + reranking, over a
  golden set, with Hit@k, Recall@k, MRR and nDCG@k and their intervals. The code is written;
  the golden queries and the real embeddings are not in place yet.
- **The refusal threshold.** `claims calibrate` will choose it from the score distributions of
  answerable and unanswerable queries. Until then `GroundingPolicy` reports itself as *not
  calibrated*, and says so in the refusal message.
- **Time to first token**, through the streaming endpoint.
- **The Qdrant path.** Written against the real client in local mode; not yet exercised.

It is possible that on a corpus this size, with vocabulary this repetitive, BM25 alone beats
the hybrid. If that is what the measurement says, that is what this README will say.

---

## How it is put together

```
scanned page  ->  OCR  ->  layout inference  ->  re-segmentation  ->  layout-aware chunking
                                                                            |
answer (or refusal)  <-  rerank  <-  RRF fusion  <-  [ BM25 | dense vectors ]
```

**Hybrid, because either half alone has a recall ceiling.** A claim corpus contains both
`AV-2026-4463` and *o carro capotou na alça de acesso*. Dense search matches meaning and misses
identifiers; BM25 matches identifiers and misses paraphrase.

**RRF rather than a weighted sum of normalised scores.** A cosine similarity is bounded and
clusters tightly; a BM25 score is unbounded and depends on corpus statistics. Min-max
normalising them makes the numbers the same shape without making them mean the same thing, and
the weight that results is tuned on one query set and silently wrong on the next. RRF fuses
ranks and ignores scores entirely.

**The lexical index folds accents**, because the engine drops them almost everywhere —
`orçado` comes back as `orcado`, `alça` as `alca`. An index that required the diacritic would
miss exactly the documents this system exists to read. Verified: the tokeniser produces
identical tokens for the clean text and the OCR'd text.

**The flat index is kept beside the vector database.** `DenseIndex` compares against every
vector: exact, instant at this size, and the only baseline that can say what the approximate
index costs in recall.

---

## Data and privacy

Nothing here came from a real claim.

Every person, vehicle, workshop and municipality is invented and the state does not exist.
Every CPF is generated with **deliberately incorrect check digits** — the two digits are
computed properly and then replaced — so each has the right shape, fails every validator, and
cannot collide with a real person's. Claim and policy numbers belong to no scheme in use.

The rendered pages are not committed: they are a deterministic function of the seed, and what
is committed is what the OCR engine read back from them.

---

## Running it

```bash
uv sync
uv run python scripts/make_fixtures.py   # needs the [ocr] extra; regenerates and re-measures
uv run pytest
```

The demo and the tests run offline, with no model, no download and no GPU: the embedder and
the reranker are interfaces, and what is committed are the vectors and scores they produced.

---

## Licence

MIT — see [LICENSE](LICENSE).
