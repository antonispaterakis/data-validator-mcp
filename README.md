# data-validator-mcp

An MCP (Model Context Protocol) server that validates the label quality of tabular datasets before they are used for ML model training. A practical adaptation of the topic-modeling pipeline described in the paper below.

---

## Methodology

The validation approach is directly derived from:

> **Theocharopoulos, A., Anagnostou, A., Georgakopoulos, S., Tasoulis, S., & Plagianakos, V.**
> *Large Language Models for Efficient Topic Modeling.*
> **Neural Computing and Applications**, 2025.

The paper proposes a three-stage pipeline for topic discovery in text corpora:

1. **Sentence-BERT embeddings** — encode documents into dense semantic vectors.
2. **HiPart hierarchical divisive clustering** — partition the embedding space recursively using the `dePDDP` algorithm, which applies PCA-based principal direction splitting.
3. **LLM labeling** — use an LLM to name the discovered clusters, but crucially only on cluster representatives — not on the whole corpus.

**This project adapts that pipeline for label validation instead of topic discovery:**

| Original (topic modeling) | This project (label validation) |
|---|---|
| Cluster unlabeled documents | Cluster already-labeled documents |
| LLM names each cluster | LLM judges suspicious rows only |
| Goal: discover topics | Goal: detect annotation errors |

The key efficiency insight — inherited from the paper — is that **the LLM is only called on the minority of rows that the clustering step flags as suspicious** (those whose assigned label disagrees with their cluster's dominant label). This keeps API cost proportional to the noise level in the data rather than to its total size.

---

## Architecture

```
data-validator-mcp/
├── src/
│   ├── embeddings.py     # Sentence-BERT encoding (all-MiniLM-L6-v2)
│   ├── clustering.py     # HiPart dePDDP clustering + cluster stats
│   ├── llm_judge.py      # Ollama LLM-as-judge for flagged rows
│   ├── pipeline.py       # Full orchestration
│   └── server.py         # MCP server (FastMCP, 3 tools)
├── tests/
│   └── test_pipeline.py
├── data/
│   └── sample_dataset.csv   # 60 rows, ~10 intentionally mislabeled
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
ollama pull llama3.2
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

### `validate_dataset(csv_path, text_col, label_col)`

Runs the full pipeline and returns a JSON summary report.

```json
{
  "total_rows": 60,
  "n_clusters": 7,
  "n_flagged": 12,
  "flagged_pct": 20.0,
  "llm_bad": 9,
  "llm_good": 3,
  "llm_unknown": 0,
  "estimated_mislabel_pct": 15.0,
  "label_counts": {"technology": 14, "sports": 13, ...}
}
```

### `get_flagged_rows()`

Returns the full list of rows flagged as potentially mislabeled, each with the LLM verdict and reasoning. Must call `validate_dataset` first.

```json
[
  {
    "row_index": 40,
    "text_preview": "Barcelona beats Real Madrid 3-0 in El Clasico...",
    "assigned_label": "technology",
    "cluster_id": 2,
    "cluster_dominant_label": "sports",
    "cluster_size": 11,
    "llm_verdict": "bad",
    "llm_confidence": "high",
    "llm_reasoning": "This text is clearly about a football match and belongs to sports, not technology."
  }
]
```

### `get_cluster_overview()`

Shows every cluster with its size, dominant label, and full label distribution. Must call `validate_dataset` first.

```json
[
  {
    "cluster_id": 0,
    "size": 9,
    "dominant_label": "politics",
    "label_distribution": {"politics": 7, "sports": 1, "entertainment": 1}
  }
]
```

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

---

## Configuration

The `ValidationPipeline` constructor exposes these parameters (all have sensible defaults):

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-BERT model name |
| `max_clusters` | `12` | Upper bound on clusters for HiPart |
| `min_sample_split` | `3` | Minimum rows to split a cluster further |
| `purity_threshold` | `0.60` | Min fraction of dominant label in a cluster before its outliers are flagged |
| `llm_model` | `llama3.1:8b` | Ollama model used for judging |
| `blob_pass` | `True` | Second LLM pass on mixed-cluster rows for higher recall |

---

## License

MIT
