# Data Validator MCP — Progress

## Status: DONE — pipeline runs end-to-end

## What's Built

Full validation pipeline:
1. Input: CSV dataset with a text column and a label column
2. Embeddings generation (`src/embeddings.py`)
3. KNN agreement scoring — per-row cosine similarity to its K nearest
   neighbours, flags rows whose neighbours mostly disagree with the
   assigned label (`src/pipeline.py`, `_compute_knn_flags`)
4. LLM judge pass — flags suspicious rows, gives verdict + confidence + suggested correction (`src/llm_judge.py`)
5. Report export — flagged CSV + full JSON (`src/report.py`)
6. MCP server exposing 4 tools: `validate_dataset`, `get_flagged_rows`, `get_knn_stats`, `export_report`

Tests (11/11 passing) and sample datasets are in place.

## What Was Fixed (this session)

The earlier "embeddings → clustering connection" issue had already been
resolved by an earlier rewrite that replaced the HiPart `dePDDP` clustering
step with KNN-based agreement scoring (`src/clustering.py`/HiPart is no
longer used by the pipeline).

The remaining blocker was that `src/llm_judge.py` pointed at LM Studio's
OpenAI-compatible endpoint (`http://localhost:1234/v1`, model
`meta-llama-3.1-8b-instruct`), but only Ollama is installed/running locally
(port 11434, models `llama3.1:8b` and `llama3.2` pulled). This caused
`validate_dataset` to fail with `{"error": "Connection error."}` as soon as
it tried to judge the first suspicious row.

Fixed by pointing `LLMJudge` at Ollama's OpenAI-compatible endpoint
(`http://localhost:11434/v1`) and changing the default judge model to
`llama3.1:8b` in `src/llm_judge.py`, `src/pipeline.py`, and `src/server.py`.
Also rewrote two stale tests in `tests/test_pipeline.py` that referenced the
removed clustering-era API (`_detect_conflicts`, `cluster_assignments`,
`cluster_embeddings`, `summary["n_flagged"]`).

Verified end-to-end on `data/sample_dataset.csv` (60 rows): 32 rows flagged
as suspicious by KNN, all 32 confirmed "bad" by the LLM judge
(`estimated_mislabel_pct: 53.3`).

## Known follow-up (not addressed this session)

`README.md` still documents the old HiPart/clustering architecture and
report shape (`n_clusters`, `cluster_id`, etc.), which no longer matches the
current KNN-based pipeline output (`n_suspicious`, `k_neighbors`, etc.).
Needs a documentation pass to bring it in line with the current code.

## Neighbour-search algorithm benchmark (this session)

Compared 5 candidate algorithms for turning "K nearest neighbours" into a
per-row suspicion score, scored against `data/large_dataset.csv` (75 rows,
30 deliberately mislabeled, ground truth in the `true_label` column):

1. **plain_knn (original)** — unweighted majority vote among top-K neighbours.
2. **distance_weighted_knn** — same K neighbours, but each vote is weighted
   by its cosine similarity instead of counted equally.
3. **mutual_knn** — only counts a neighbour's vote if the relationship is
   reciprocal (i is also in j's top-K).
4. **local_outlier_knn** — LOF-flavoured: flags rows whose own-class
   neighbours are much farther away than their nearest neighbours overall.
5. **centroid_margin** — no neighbour search at all; compares each row's
   similarity to its own label's centroid vs. the best other-label centroid.

Harness: `scripts/evaluate_neighbor_algorithms.py` (sweeps K ∈ {5,10,15,20}
and the suspicion threshold per algorithm, reports precision/recall/F1
against the `true_label` ground truth).

**Result: `distance_weighted_knn` won outright** — best F1 at every K tested,
and meaningfully better precision than the original at the same recall
(e.g. K=15: F1 0.704 vs. 0.600, precision 0.610 vs. 0.450). Intuition: a
near-duplicate row with a *different* label is a much stronger conflict
signal than several only-loosely-similar same-label neighbours, and plain
majority vote treats them identically. `mutual_knn` and `local_outlier_knn`
did not clearly beat the original; `centroid_margin` was competitive on
recall but weaker on precision and discards the actual neighbour evidence
the LLM judge prompt relies on.

**Caveat:** this run used a pure-numpy TF-IDF embedding
(`scripts/_tfidf_fallback.py`) because the sandbox used to write this had no
internet access to fetch sentence-transformers. The relative ranking between
algorithms should hold (every algorithm sees the same similarity matrix),
but re-run `scripts/evaluate_neighbor_algorithms.py` on a machine with the
real `.venv` (has sentence-transformers cached) before trusting the exact
F1/precision/threshold numbers for production tuning.

**Implemented:** `distance_weighted_knn` is now the default in
`ValidationPipeline` (`weighted_agreement=True`), exposed as a toggle on
`validate_dataset` in `src/server.py` so the original behaviour is one
parameter away if a future dataset disagrees. `agreement_threshold` default
(0.5) was left unchanged for backward compatibility — the benchmark's
optimal threshold for weighted agreement was higher (~0.7-0.8), but that's
embedding-space-dependent and worth re-checking with real embeddings before
changing the default. Added `tests/test_pipeline.py::TestWeightedAgreement`
(3 tests, including the near-duplicate-conflict case that motivated the
change) — all passing against the real implementation.

**Next step:** re-run the benchmark with the real BiomedBERT/MiniLM
embeddings and, if the optimal threshold holds at ~0.7-0.8, bump
`agreement_threshold`'s default to match.

## Goal

Present to Prof. Plagianakos as a training data quality tool for clinical AI use cases.
