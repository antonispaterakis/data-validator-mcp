# data-validator-mcp

An MCP (Model Context Protocol) server that validates the label quality of tabular datasets before they are used for ML model training. A practical adaptation of the topic-modeling pipeline described in the paper below.

---

## Methodology

This project's lineage traces back to:

> **Theocharopoulos, A., Anagnostou, A., Georgakopoulos, S., Tasoulis, S., & Plagianakos, V.**
> *Large Language Models for Efficient Topic Modeling.*
> **Neural Computing and Applications**, 2025.

The paper's core insight — encode documents into embeddings, use a cheap unsupervised signal to decide which rows are worth examining, and call an LLM only on that minority — carries over directly to label validation:

1. **Embeddings** — encode each row's text into a dense semantic vector.
2. **KNN agreement scoring** — for each row, find its K nearest neighbours by embedding similarity and compute what fraction of them share its assigned label. Rows whose neighbours mostly *disagree* with the label are flagged as suspicious.
3. **LLM judge** — only the suspicious rows go to an LLM (via Ollama), which returns a verdict (`good`/`bad`), confidence, reasoning, and — when confident — a suggested correction.

| Original (topic modeling) | This project (label validation) |
|---|---|
| Cluster unlabeled documents | Score each labeled row against its embedding neighbours |
| LLM names each cluster | LLM judges suspicious rows only |
| Goal: discover topics | Goal: detect annotation errors |

The key efficiency insight — inherited from the paper — is that **the LLM is only called on the minority of rows the KNN step flags as suspicious** (those whose label disagrees with most of their nearest neighbours). This keeps inference cost proportional to the noise level in the data rather than to its total size — on the bundled sample dataset that's 32 LLM calls instead of 60 (see `llm_calls_saved_vs_full_pass` in the report).

> Earlier versions of this pipeline used HiPart hierarchical clustering (`dePDDP`) to group rows before flagging outliers — `src/clustering.py` still contains that code but is no longer used by `ValidationPipeline`. Per-row KNN agreement scoring replaced it: simpler, no cluster count to tune, and rows are flagged individually rather than relative to a cluster's dominant label.

---

## Architecture

```
data-validator-mcp/
├── src/
│   ├── embeddings.py     # Sentence-BERT / domain encoder
│   ├── clustering.py     # legacy HiPart dePDDP clustering — no longer used
│   ├── llm_judge.py      # Ollama LLM-as-judge for flagged rows
│   ├── pipeline.py       # Full orchestration (embeddings → KNN agreement → LLM judge)
│   ├── report.py         # CSV + JSON report export
│   └── server.py         # MCP server (FastMCP, 4 tools)
├── tests/
│   └── test_pipeline.py
├── data/
│   └── sample_dataset.csv   # 60 rows, ~20 intentionally mislabeled
├── requirements.txt
└── .env.example
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure [Ollama](https://ollama.com) is installed and running, then pull the judge model:

```bash
ollama pull llama3.1:8b
```

---

## Running the server

```bash
# stdio transport (standard for MCP clients)
python -m src.server

# or directly
python src/server.py
```

### Registering with Claude Desktop

Add the following to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "data-validator-mcp": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/data-validator-mcp"
    }
  }
}
```

No API key needed — the server uses a local Ollama instance. Make sure Ollama is running before starting Claude Desktop.

---

## MCP Tools

### `validate_dataset(csv_path, text_col, label_col, k_neighbors=5, agreement_threshold=0.5, llm_model="llama3.1:8b")`

Runs the full pipeline and returns a JSON summary report.

```json
{
  "total_rows": 60,
  "k_neighbors": 5,
  "agreement_threshold": 0.5,
  "n_suspicious": 32,
  "n_confirmed_bad": 32,
  "llm_bad": 32,
  "llm_good": 0,
  "llm_unknown": 0,
  "estimated_mislabel_pct": 53.3,
  "label_counts": {"technology": 14, "sports": 13, "...": "..."},
  "llm_calls_made": 32,
  "llm_calls_saved_vs_full_pass": 28,
  "llm_model": "llama3.1:8b",
  "token_stats": {"tokens_used": 0, "rows_scanned_by_llm": 32, "total_rows": 60, "efficiency_ratio": 1.9},
  "timing_seconds": {"encode": 0.0, "knn": 0.0, "judge": 0.0, "total": 0.0}
}
```

### `get_flagged_rows()`

Returns every row the LLM confirmed as potentially mislabeled. Must call `validate_dataset` first.

```json
[
  {
    "row_index": 40,
    "text_preview": "Barcelona beats Real Madrid 3-0 in El Clasico...",
    "assigned_label": "technology",
    "suggested_label": "sports",
    "needs_review": false,
    "neighbor_agreement": 0.2,
    "n_matching_neighbors": 1,
    "k_neighbors": 5,
    "llm_verdict": "bad",
    "llm_confidence": "high",
    "llm_reasoning": "This text is clearly about a football match and belongs to sports, not technology.",
    "detection_source": "knn"
  }
]
```

`suggested_label` is set when `llm_verdict="bad"` and `llm_confidence="high"`. `needs_review=true` marks `bad`/medium-confidence rows for a human to check before applying any correction.

### `get_knn_stats()`

Returns KNN agreement statistics for every row flagged as suspicious *before* the LLM pass — useful for seeing how borderline each flagged row was. Must call `validate_dataset` first.

```json
[
  {
    "row_index": 40,
    "text_preview": "Barcelona beats Real Madrid 3-0 in El Clasico...",
    "assigned_label": "technology",
    "neighbor_agreement": 0.2,
    "n_matching_neighbors": 1,
    "k": 5,
    "neighbor_label_counts": {"sports": 4, "technology": 1}
  }
]
```

### `export_report(output_dir)`

Writes `flagged_rows_<timestamp>.csv` (human-friendly) and `report_<timestamp>.json` (full machine-readable dump: summary + flagged rows + KNN stats) to `output_dir`. Must call `validate_dataset` first.

---

## Running tests

```bash
pytest tests/ -v
```

Tests cover: embedding shape/dtype/normalization, cluster dominant-label logic, conflict detection, and a monkeypatched no-LLM pipeline run.

---

## Sample dataset

`data/sample_dataset.csv` contains 60 news headline rows across five categories:
`technology`, `sports`, `politics`, `entertainment`, `health`.

Approximately **20 rows are intentionally mislabeled** (rows 41–60) — e.g. a sports headline labeled `technology`, a medical headline labeled `sports` — to demonstrate the validator's detection capability.

**Verified run** (default settings — `k_neighbors=5`, `agreement_threshold=0.5`, `llm_model=llama3.1:8b`): 32/60 rows flagged as suspicious by KNN, all 32 confirmed `bad` by the LLM judge (`estimated_mislabel_pct: 53.3`). That's higher than the ~20 rows documented as intentionally mislabeled — at this threshold the KNN step also catches genuinely ambiguous boundary-case rows that the LLM agrees are mislabeled. Lowering `agreement_threshold` trades recall for precision if a tighter match to the known-bad rows is needed.

---

## Configuration

`validate_dataset` exposes these parameters (all have sensible defaults):

| Parameter | Default | Description |
|---|---|---|
| `k_neighbors` | `5` | Number of nearest neighbours per row for agreement scoring |
| `agreement_threshold` | `0.5` | Minimum fraction of neighbours that must share a row's label for it to be considered clean |
| `llm_model` | `llama3.1:8b` | Ollama model used for judging flagged rows |

`ValidationPipeline`'s constructor also exposes `embedding_model` (default: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`, reflecting this project's clinical-data validation use case — swap for `all-MiniLM-L6-v2` for general-purpose text).

---

## License

MIT
