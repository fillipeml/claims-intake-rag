# CLAUDE.md

Working rules for AI-assisted changes in this repository. They mirror the README; the README
wins on conflict.

## Non-negotiable rules

1. **Nothing is simulated that can be run.** The corpus is rendered to images and read by a
   real OCR engine, and every error rate is measured against the text that went in. A
   degradation model standing in for OCR would be measuring itself, which is the one result
   this repository must never report.
2. **No retrieval number without its interval.** Hit@k is a proportion over queries and takes
   a Wilson interval; Recall, MRR and nDCG are means of per-query values and take a seeded
   bootstrap. Using Wilson on a mean computes cleanly and is wrong.
3. **Two configurations are compared as paired data.** The same queries run through both, so
   the ones that scored the same carry no information. Comparing two mean nDCGs as independent
   samples throws the pairing away.
4. **The answerer refuses when retrieval is weak**, and the threshold it refuses at is measured
   by `claims calibrate`, never chosen by feel. A claim answered from the five least-bad chunks
   is worse than no answer, because it arrives with a citation that does not support it.
5. **A recorded adapter never guesses.** The recorded embedder and the recorded reranker raise
   on input they have no recording for. Returning a zero vector would make a chunk
   unretrievable and quietly lower every recall figure with nothing reporting an error.
6. **Fictional data only.** Invented people, municipality and state; CPFs whose check digits
   are deliberately wrong; policy and claim numbers belonging to no scheme. No real claim
   document is committed, ever.
7. **The flat index stays.** `DenseIndex` is exact and is the baseline that says what the
   approximate index in `store.py` costs in recall. Deleting it would leave the repository
   unable to answer that question.

## Conventions

- Python 3.12, `uv`, `ruff`, `pytest`. The core depends on Pydantic and the standard library;
  everything needing a model, a GPU or a server is an optional extra behind an interface.
- The demo, the tests and CI run offline on committed fixtures, with no model and no download.
- `fixtures/pages/` is generated, not committed: it is a deterministic function of the seed and
  weighs megabytes. What is committed is what the OCR engine read back from it.
- `segmentation.py` holds the algorithm and knows nothing about insurance; `lexicon.py` holds
  the domain and knows nothing about dynamic programming. Changing domain means replacing the
  second file only.
- Commits: English, Conventional Commits, one logical change each, no AI attribution trailers.
