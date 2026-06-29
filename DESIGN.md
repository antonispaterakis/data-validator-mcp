# data-validator-mcp — Design Journal

## Purpose
Detect mislabeled rows in ML training datasets efficiently — using semantic embeddings
to pre-filter suspicious rows, then an LLM to confirm/dismiss each flag. The key
argument: LLM calls only on suspicious rows, not the full dataset.

---

## 2026-04-08 — Initial working version

**Architecture:**
1. CSV → sentence-transformers embeddings (all-MiniLM-L6-v2)
2. HiPart dePDDP clustering (max_clusters=12, min_sample_split=3)
3. Purity filter (threshold=0.60) — skip mixed clusters, flag outliers in pure ones
4. Ollama LLM judge (llama3.2) — confirm/dismiss each flag
5. `suggested_label` added to high-confidence bad rows
6. report.py — exports CSV + JSON

**MCP tools:** validate_dataset, get_flagged_rows, get_cluster_overview, export_report

**Performance on 60-row sample:**
- 4 flagged, 3 confirmed bad (~30% recall), 12.9s total
- Known bottleneck: large mixed cluster (36 rows) absorbs most mislabels — falls
  below purity threshold so gets skipped entirely

**Connected to Claude Desktop. Pitched to Plagianakos at iSL lab.**

---

## 2026-04-19 — Major refactor session

### Phase 1: Pipeline quality improvements

**Change 1 — Anchor selection (`embeddings.select_anchors`)**
Before clustering, find the most representative row per label: the one whose
embedding is closest to the label's centroid. Anchors serve as verified reference
examples passed to the LLM judge in both passes, replacing the weaker
cluster-distribution-only context from the original blob pass.

**Change 2 — Pass 1 judge upgrade**
`judge_row()` now receives `anchor_text` for the dominant cluster label.
The prompt leads with the anchor before the cluster peer examples.

**Change 3 — Pass 2 (blob) judge upgrade**
Replaced `judge_blob_row()` with a stronger version:
- Finds K=3 nearest neighbours in embedding space among rows in pure clusters
- Passes those as relative context + per-label anchors as absolute context
- Medium-confidence bad verdicts now kept with `needs_review=True` instead of discarded
- `needs_review` field added to all flagged rows

**Change 4 — Token tracking**
Single `LLMJudge` instance shared across both passes accumulates
`response.usage.prompt_tokens` + `completion_tokens` from LM Studio responses.
After the run, `token_stats` is added to the summary:
```
tokens_used, tokens_if_brute_force, efficiency_ratio, rows_scanned_by_llm
```
`efficiency_ratio` = tokens saved vs scanning every row individually.
This is the central argument for Plagianakos: the tool costs Nx fewer tokens
than brute-force LLM scanning.

**Other changes in this session:**
- Switched LLM backend: Ollama → LM Studio (http://localhost:1234/v1)
- Removed `response_format={"type": "json_object"}` (LM Studio doesn't support it)
- Added `temperature=0` for deterministic verdicts
- `detection_source` field added to flagged rows ("clustering" or "blob_pass")

### Phase 2: Real dataset test (123rc/medical_text)

Downloaded 200 rows from HuggingFace `123rc/medical_text` (5 balanced classes,
40 rows each): neoplasms, digestive_diseases, nervous_system_diseases,
cardiovascular_diseases, general_pathological. Saved as `test_dataset.csv`.

**Clustering results (all-MiniLM-L6-v2, max_clusters=20):**
- Cluster 0: 79 rows, purity 44% — giant mixed blob
- Cluster 18: 61 rows, purity 48% — giant mixed blob
- 165/200 rows in blob clusters → near-brute-force LLM scanning
- efficiency_ratio: 1.2x (essentially no gain)
- 29 confirmed bad rows, est. mislabel 14.5%

**Root cause:** The 5 medical sub-specialties share vocabulary, structure, and
clinical framing. all-MiniLM-L6-v2 embeddings don't geometrically separate them.

### Phase 3: BiomedBERT + UMAP experiment

**Hypothesis:** Domain-specific embeddings + UMAP pre-reduction will create
cleaner clusters.

**Changes:**
- Embedding model: all-MiniLM-L6-v2 → microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
- `use_umap=True` set as default

**Results (max_clusters=20):**
- Worse than before: 175/200 rows in blob clusters, efficiency_ratio 1.1x
- Cluster purities: mostly 21-48%, pure clusters are all singletons/doublets

**Why it failed:** BiomedBERT is a masked LM fine-tuned for token-level tasks
(NER, QA), not sentence similarity. When used via SentenceTransformers mean
pooling, the resulting sentence vectors have less similarity signal than
all-MiniLM, which was explicitly optimised for cosine similarity tasks.

### Phase 4: KNN-based detection (replace clustering entirely)

**Motivation:** Clustering depends on finding geometrically separable label
regions. For semantically overlapping labels, this never works regardless of
model or parameters. KNN agreement scoring sidesteps cluster formation entirely.

**New pipeline:**
1. Encode texts
2. Select anchors (unchanged)
3. Compute full N×N cosine similarity matrix (dot product on L2-norm vectors)
4. For each row: find K nearest neighbours, compute `neighbor_label_agreement`
   = fraction of neighbours sharing the row's label
5. Flag rows with agreement < 0.5
6. Single unified LLM judge pass with anchor + top-5 neighbour context

**Implementation:**
- Removed: HiPart clustering, `max_clusters`, `min_sample_split`, `purity_threshold`,
  `use_umap`, `umap_components`, `blob_pass`, Pass 1/Pass 2 split
- Added: `k_neighbors=5`, `agreement_threshold=0.5`, `_compute_knn_flags()`,
  `_judge_suspicious()`, `judge_knn_row()` in LLMJudge
- MCP tool `get_cluster_overview` replaced by `get_knn_stats`

**LLM model switch:** gemma-4-e4b-it → meta-llama-3.1-8b-instruct (lighter, faster)

**KNN test results on test_dataset.csv:**

| K | Suspicious | LLM bad | LLM good | efficiency_ratio |
|---|---|---|---|---|
| 15 | 186/200 | — | — | (killed early) |
| 5 | 153/200 | 153 | 0 | 1.3x |

**KNN also failed for the same root reason:**
With 5 balanced classes and K=5, expected random agreement = 1/5 = 20%.
Getting ≥3/5 matching neighbours (the 0.5 threshold) is rare even for correctly
labelled rows in an overlapping embedding space → most rows flagged regardless
of true label quality.

Additionally, the LLM rubber-stamped 100% of flags as bad (153/153 bad, 0 good).
Diagnosis: the prompt framing ("this row is flagged") combined with weak geometric
signal primes the LLM to always agree with the flag.

### Phase 4 diagnosis: fundamental mismatch

**The tool works when:** label categories correspond to distinct semantic regions
in embedding space (e.g., cardiology vs. orthopedics — confirmed working in
sanity_check.py with 100% recall on intentional mislabels).

**The tool breaks when:** label categories are semantically overlapping
sub-specialties within the same domain (e.g., 5 medical sub-specialties that
all use similar vocabulary). No embedding model or distance-based detection
method can reliably distinguish these.

---

## Open questions / next steps

1. **Zero-shot LLM classification** — skip geometry entirely. For each row, ask
   the LLM "what label does this text belong to?" and flag disagreements with
   the assigned label. No KNN, no clustering. Higher token cost but no false
   signal from geometry.

2. **Better sentence encoder** — use a model fine-tuned specifically for medical
   document classification (not just PubMed pre-training). Something like a
   SBERT model trained on medical classification NLI.

3. **Different test dataset** — validate the tool on a dataset where labels *do*
   correspond to distinct semantic domains. The sanity check confirmed it works
   for clear cases. A good real-world test would be customer support tickets,
   news categories, or product reviews.

4. **LLM bias fix** — the judge prompt should be more balanced. Currently "this
   row is flagged as suspicious" primes it toward "bad". Should present it
   neutrally: "Does this text match its label? Answer honestly."

---

## File map

```
src/
  embeddings.py    — encode_texts(), select_anchors()
  pipeline.py      — ValidationPipeline (KNN-based, current)
  llm_judge.py     — LLMJudge: judge_row(), judge_blob_row(), judge_knn_row()
  clustering.py    — cluster_embeddings(), reduce_dimensions() [kept, unused]
  report.py        — ReportWriter (CSV + JSON export)
  server.py        — FastMCP server, 4 tools
sanity_check.py    — 12-row synthetic test (works correctly)
test_dataset.csv   — 200 rows from 123rc/medical_text (problematic for this tool)
data/
  sample_dataset.csv
  large_dataset.csv
```

## Current defaults

| Parameter | Value |
|---|---|
| embedding_model | microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract |
| k_neighbors | 5 |
| agreement_threshold | 0.5 |
| llm_model | meta-llama-3.1-8b-instruct |
| LM Studio base URL | http://localhost:1234/v1 |
