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

## Known follow-up (Resolved)

✅ `README.md` now documents the new KNN-based architecture and correctly reflects the latest JSON output and defaults.

## Neighbour-search algorithm benchmark (Resolved)

✅ Re-ran the benchmark with the real `all-MiniLM-L6-v2` embeddings. The optimal threshold held true (0.767 at k=15). 
✅ `agreement_threshold` default has been bumped to `0.75` and `k_neighbors` to `15` across the pipeline and MCP server.

## Testing & Visualizations (Resolved)

✅ Added a `mock_llm` toggle to bypass the local Ollama judge (useful for rapid testing or GitHub demonstrations without the compute overhead).
✅ Created a `plot_knn_concept.py` script that uses UMAP to project embeddings to 2D and visually demonstrates how the distance-weighted KNN algorithm outperforms plain KNN at calculating neighbors.

## Goal

Present to Prof. Plagianakos as a training data quality tool for clinical AI use cases.
