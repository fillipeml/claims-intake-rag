# claims-intake-rag

Retrieval over **scanned Portuguese insurance-claim documents**: OCR with a measured error
rate, layout recovered from geometry and scored against the truth, word boundaries put back by
a segmenter that learns its own vocabulary, hybrid search fused with RRF, cross-encoder
reranking, and an answer the system **refuses to give** when retrieval is weak.

Every number below was produced by the command shown next to it and written here afterwards.
Nothing is estimated, and the results that went the wrong way are in here too — including four
defects in my own work that the evaluation found and a code review would not have.

> **Status: the pipeline is complete and measured.** OCR, layout, retrieval, the ablation, the
> refusal threshold, the streaming endpoint and the containers are all exercised, and CI starts
> the image and asks it a question rather than only building it. Still missing a README
> screenshot and a published demo.

> **Companion repository.** [`qlora-serving-lab`](https://github.com/fillipeml/qlora-serving-lab)
> takes the other half of the same problem: not retrieving from claim documents but turning a short
> Portuguese claim notice into a structured English record, by fine-tuning a 0.5B model on the same
> consumer GPU. Where this repository measures retrieval and serving, that one measures training,
> quantisation and what the hardware actually does — including an fp16 matrix multiply that runs at
> one eighth of fp32 on the card both repositories run on.

---

## Why this exists

An insurer's claims intake is a pile of scans: a notification, a police report, a repair
estimate, a medical report, a policy excerpt. The questions asked of that pile are specific —
*what was the total estimate*, *was the licence valid on the day*, *what is the deductible for
collision* — and the answer sits in one line of one document.

The part most RAG projects skip is the part that decides whether any of it works: **measuring
retrieval**. A pipeline that looks right on five hand-picked queries cannot tell you whether a
change helped the queries you looked at and hurt the ones you did not. This repository is built
the other way round — the harness came first, and it spent most of its time finding things
wrong with the system around it.

---

## Reading the page

Forty synthetic claim documents, rendered to A4 at 300 dpi, degraded the way a desk scanner
degrades a page, and **read back by a real OCR engine**. Nothing is simulated twice: the error
rates are measured against the text that went into the renderer.

```bash
uv run python scripts/make_fixtures.py   # 40 documents, 299 chunks
uv run claims ocr-report
```

| | CER | WER |
| --- | --- | --- |
| Raw OCR output | 12.07% | 80.60% |
| + re-segmentation, corpus vocabulary only | 11.58% | 74.63% |
| + domain glossary under it | 7.86% | 54.42% |
| + accents folded — what the lexical index sees | **4.67%** | **41.92%** |

**Layout inference: 464 of 542 blocks — 85.6%.** The kind of each block is recovered from
geometry that survives a scan (height relative to the page median, capitalisation, whether
values recur in a column) and scored against the truth, because a document-intelligence step
nobody measured is one nobody can trust.

### Two things that only came from running it

**The engine stops emitting word boundaries below about 38 pixels of glyph height.** At 32px it
returns `Datadoatendimento:08/08/2026`; at 38px, `Data do atendimento: 08/08/2026`. The
character error rate barely notices — a missing space is one edit — while the word error rate
explodes, because `datadoatendimento` is a token no query can match. The corpus renders at
300 dpi for that reason, which is also what every scanning guideline recommends; the fix and
the convention agree, but only one of them was measured.

**A vocabulary learned from OCR output learns OCR's mistakes as words.** The segmenter taught
itself `datado`, because one line somewhere welded *Data do* into a single token while keeping
its other spaces — and `datado` then beat `data` + `do` on every later line. The rule that
removes it without a list of exceptions: *a candidate better explained as a sequence of words
already known is not a word.*

The glossary exists for a measured reason too. Learning the vocabulary from the corpus alone
is elegant and insufficient: `atendimento`, `afastamento` and `habituais` all came back at
frequency zero, because every line containing them is a line that merged. **A corpus cannot
teach itself a word it only ever shows welded to its neighbours.**

---

## Retrieval, measured

Sixty-four questions, each paired with the chunks that answer it. Relevance is decided from the
**ground truth** — the line of the original document that answers the question — never by
matching the question against the chunks, which would make the golden set a BM25 output under
another name and guarantee that BM25 wins.

```bash
uv run claims evaluate      # the table
uv run claims scope         # what filtering by claim is worth
```

Search scoped to the claim the question is about, k=5:

| Configuration | Hit@5 | Recall@5 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- |
| `lexical-only` | 37.5% [26.7, 49.7] | 0.375 [0.250, 0.500] | 0.286 | 0.308 |
| `dense-only` | 71.9% [59.9, 81.4] | 0.695 [0.586, 0.797] | 0.500 | 0.533 |
| `hybrid-rrf` | 64.1% [51.8, 74.7] | 0.617 [0.500, 0.734] | 0.449 | 0.479 |
| `dense+rerank` | **75.0% [63.2, 84.0]** | **0.727 [0.617, 0.828]** | **0.520** | **0.565** |
| `hybrid+rerank` | 75.0% [63.2, 84.0] | 0.727 [0.617, 0.828] | 0.517 | 0.563 |

Hit@5 is a proportion over queries and carries a Wilson interval. Recall, MRR and nDCG are
means of per-query values and carry a seeded bootstrap. Using Wilson on a mean would compute
cleanly and be wrong, which is the only reason the distinction is worth a sentence.

**Paired, by exact sign test over the queries whose score moved:**

| | |
| --- | --- |
| `lexical-only` → `hybrid-rrf` | 20 improved, 0 regressed — **p < 0.001** |
| `dense-only` → `hybrid-rrf` | 2 improved, 12 regressed — **p = 0.013** |
| `dense-only` → `dense+rerank` | 14 improved, 8 regressed — p = 0.286 |
| `hybrid-rrf` → `hybrid+rerank` | 17 improved, 8 regressed — p = 0.108 |

### What that says, including the part that contradicts the usual advice

**Adding BM25 to a strong dense retriever makes it worse, and the difference is real.** Twelve
queries regressed and two improved. Every guide says hybrid search is the single highest-impact
upgrade; on this corpus it is a downgrade, because RRF is unweighted and fusing a strong
retriever with a weak one drags the strong one down. Hybrid is not free.

**Adding BM25 to a weak retriever helps enormously** — the first row, 20 improved and 0
regressed. Which is the same fact: fusion moves both retrievers toward each other.

**Reranking helps both and the difference is not significant on 64 queries.** The point estimate
moves in the right direction each time and the intervals overlap. Saying more than that would
be reading the table past what it supports.

### Scoping is the largest single effect

| Configuration | Hit@5, whole corpus | Hit@5, scoped to the claim |
| --- | --- | --- |
| `lexical-only` | 15.6% [8.7, 26.4] | 37.5% [26.7, 49.7] |
| `dense-only` | 23.4% [14.7, 35.1] | 71.9% [59.9, 81.4] |
| `hybrid-rrf` | 21.9% [13.5, 33.4] | 64.1% [51.8, 74.7] |

A claims question is always about one claim, and eight repair estimates differ only in their
totals. Searching the whole corpus is not a slower route to the same answer — it is being asked
a question the corpus cannot answer. The metadata filter is semantics, not optimisation, which
is why it is applied **before** scoring rather than after: taking the top 20 of the corpus and
then discarding the other claims leaves a top 20 made mostly of other people's documents.

---

## Refusing to answer

A retrieval system asked something its corpus cannot answer will return its five least-bad
chunks, and a model handed those will write a fluent answer from them. In a claim file that is
the worst available outcome: the adjuster gets a number with a citation that does not support
it.

```bash
uv run claims calibrate --configuration dense-only
```

The threshold is measured, not chosen. Running the 64 answerable and 8 unanswerable questions
through retrieval and reading the score distributions:

| Signal | Answerable | Unanswerable | At the chosen threshold |
| --- | --- | --- | --- |
| Dense cosine | [0.840, 0.916], median 0.864 | [0.828, 0.847] | **0.853** — refuses 8/8, keeps 51/64 |
| Cross-encoder score | median 0.016 | max 0.0066 | refusing 8/8 costs half the answerable |

**The better ranker is the worse gate.** The cross-encoder wins the ablation and separates
answerable from unanswerable badly; the dense cosine ranks worse and separates cleanly. So the
reranker does the ordering and the dense score decides whether to answer at all. Ordering and
deciding-whether-to-answer are different jobs and the best signal for one is not the best for
the other.

### A guard I built, measured, and turned off

`min_overlap` required at least one content word of the question to appear in the chunk being
quoted — cheap insurance against a dense hit that is topically adjacent and lexically
unrelated. Measured, it refuses 38 of 64 answers and **8 of those were the correct chunk**, a
third of every correct answer the system produces.

The reason is the reason dense retrieval exists. Asked *o motorista podia dirigir no dia do
acidente*, the right chunk reads `Habilitação: válida` — not one word in common. A lexical
floor under a semantic retriever punishes it for working.

It does raise top-1 precision from 41% to 69%, so it is kept and configurable for work where a
wrong answer costs more than a missing one. It is off by default.

---

## Four defects the harness found in my own work

None of these would have come out of reading the code. Each came from a number that did not
match what I expected.

1. **The footer pattern ate every date.** `\d+/\d+` matches the `02/02` inside `02/02/2026`, so
   any short line carrying a date was classified as a footer and dropped by the chunker. The
   claim notification lost its `Data do sinistro` line entirely, and the one golden question
   asking *when* the accident happened could not be answered by anything. Found because 56
   queries over 8 claims is seven templates, not eight. Fixing it took layout accuracy from
   82.7% to 85.6% and the golden set from 56 questions to 64.
2. **The claim identifier was in the question and in the scope.** Asked whether the driver was
   licensed, the system returned `Numero do aviso:AV-2026-4407` at 0.88 while
   `Habilitacao:valida` did not place — the identifier dominated every scorer. Removing it from
   the question text, since the claim is passed separately, took Hit@5 from 35.7% to 71.4%.
3. **One chunk held nine fields.** A chunk answering nine questions badly instead of one
   question well. Splitting fields out is what made the identifier problem visible.
4. **The segmenter skipped every table row.** `MIN_LENGTH` was 12, and the letters in
   `TotalgeralR$10.372,00` are eleven — so the one line of the repair estimate anybody asks
   about was never re-segmented.

---

## How it is put together

```
scanned page  ->  OCR  ->  layout inference  ->  re-segmentation  ->  layout-aware chunking
                                                                            |
answer (or refusal)  <-  rerank  <-  RRF fusion  <-  [ BM25 | dense vectors ]
```

**RRF rather than a weighted sum of normalised scores.** A cosine is bounded and clusters
tightly; a BM25 score is unbounded and depends on corpus statistics. Min-max normalising them
makes the numbers the same shape without making them mean the same thing, and the weight that
results is tuned on one query set and silently wrong on the next. RRF fuses ranks and ignores
scores entirely.

**The lexical index folds accents**, because the engine drops them almost everywhere —
`orçado` comes back as `orcado`. Verified: the tokeniser produces identical tokens for the
clean text and the OCR output.

**The flat index is kept beside the vector database.** `DenseIndex` compares against every
vector: exact, instant at this size, and the only baseline that can say what an approximate
index costs in recall.

**BM25 is written here rather than imported.** It has to tokenise Portuguese the way this
corpus needs, and a retrieval repository that cannot show its lexical scoring is not showing
very much.

---

## Serving it

One streaming endpoint, because the number a person feels is **time to first token** and a
response that arrives whole can only report total time.

```bash
uv run uvicorn claims_rag.app:app
curl -N -X POST localhost:8000/claims/AV-2026-4407/ask   -H 'content-type: application/json'   -d '{"question":"quanto vai custar o conserto do veículo"}'
```

```
event: meta
data: {"claim_id": "AV-2026-4407", "retrieval_ms": 2.88, "considered": 5, "configuration": "dense+rerank"}
event: token
data: {"text": "Segundo AV-2026-4407-apolice (p. 1, APOLICE DE SEGURO DE AUTOMOVEL > COBERTURAS CONTRATADAS):

"}
event: done
data: {"ttft_ms": 2.98, "total_ms": 3.04, "characters": 146, "citations": [...]}
```

**Median TTFT over 20 requests: 3.02ms** (2.72 to 3.66), in process, on committed fixtures. A
deployment embedding live questions pays the encoder's forward pass on top of that; this
figure is the pipeline's own cost and is reported as such.

**SSE rather than WebSockets**, because the traffic is one request and one stream of text
back. SSE is a `text/event-stream` body over ordinary HTTP: it survives proxies that do not
know about upgrade handshakes and there is no connection state to manage. The response carries
`X-Accel-Buffering: no`, without which an nginx in front delivers the whole stream at the end
and turns a 3ms first token into a 300ms one — invisible to any test that does not go through
the proxy.

**The claim is in the path, not in the body.** A request that forgets the scope cannot be
formed, rather than quietly searching every claim in the corpus.

### Containers

`docker compose up` brings the service and a real Qdrant. The image is multi-stage, runs as
uid 10001 with a read-only root filesystem, and its healthcheck hits `/health` rather than the
port — a container that is listening but could not load its corpus is not healthy, and a TCP
check would call it healthy for ever.

`k8s/` has a Deployment, a Service and a PodDisruptionBudget, written for a cluster enforcing
the restricted Pod Security Standard. Three things in there are deliberate and asserted by the
test suite: readiness and liveness are **separate** probes, because sharing one means a slow
dependency gets the container killed instead of taken out of rotation; memory request equals
memory limit, so the pod is Guaranteed rather than the first thing evicted; and the disruption
budget exists because Server-Sent Events are long-lived responses and a rollout without one
cuts every answer mid-sentence.

CI does not stop at `docker build`. It starts the container and asks it a question, because
what breaks is an entry point naming a module nobody imported, or a corpus that is not in the
image — neither of which a build catches.

### What the vector store is for

`store.py` is exercised against a real Qdrant in local mode, with payload filtering. On this
corpus its ranking is **identical** to the exact flat index, which is the point: HNSW has
nothing to approximate at 299 chunks, and the exact baseline is the only thing that could
measure what an approximate index costs in recall once there is enough corpus to need one.

Writing it was not enough. Running it found two defects a review would not: the client
deprecated `recreate_collection` and **removed** `search` outright in the version this pins.

## Data and privacy

Nothing here came from a real claim.

Every person, vehicle, workshop and municipality is invented and the state does not exist.
Every CPF is generated with **deliberately incorrect check digits** — the right ones are
computed and then replaced — so each has the right shape, fails every validator, and cannot
collide with a real person's. Claim and policy numbers belong to no scheme in use. The test
suite recomputes the check digits and asserts every one is wrong.

The rendered pages are not committed: they are a deterministic function of the seed, and what
is committed is what the OCR engine read back from them.

---

## Running it

```bash
uv sync
uv run pytest
uv run claims evaluate
uv run claims ask "quanto vai custar o conserto do veículo" --claim AV-2026-4407
```

The demo, the tests and CI run offline — no model, no download, no GPU. The embedder and the
reranker are interfaces; what is committed are the vectors and scores they produced, so the
ablation replays exactly rather than depending on which revision of a model hub is current.

Regenerating the corpus needs the extras: `uv sync --extra ocr --extra models`.

---

## Licence

MIT — see [LICENSE](LICENSE).
