"""
MCP server for data-validator-mcp.

Exposes four tools:
  • validate_dataset      — run the full KNN pipeline on a CSV
  • get_flagged_rows      — retrieve rows the pipeline confirmed as mislabeled
  • get_knn_stats         — inspect per-row KNN agreement scores for suspicious rows
  • export_report         — write flagged CSV + full JSON report to disk

State is held in a module-level object so results persist across tool calls
within a single server session.
"""

from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from .pipeline import ValidationPipeline
from .report import ReportWriter

mcp = FastMCP(
    "data-validator-mcp",
    instructions=(
        "Training data label quality validator — detects mislabeled rows in a CSV "
        "using KNN agreement scoring + a local LLM judge (Ollama). No API key needed.\n\n"
        "Step 1 — Call validate_dataset with the path to your CSV, the name of the "
        "text column, and the name of the label column. Optionally tune k_neighbors "
        "and agreement_threshold.\n\n"
        "Step 2 — Call get_flagged_rows to see every suspicious row: what label it "
        "has, neighbour agreement score, LLM verdict (good/bad), confidence, "
        "reasoning, and a suggested correction where the model is confident.\n\n"
        "Step 3 — Call get_knn_stats to inspect the KNN agreement distribution "
        "across all rows flagged as suspicious before the LLM pass.\n\n"
        "Step 4 — Call export_report with an output directory to save a "
        "flagged_rows CSV and a full JSON report to disk for sharing or review."
    ),
)

_pipeline = ValidationPipeline()
_ran = False


@mcp.tool(
    title="Validate Dataset",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
    ),
)
def validate_dataset(
    csv_path: Annotated[str, Field(description="Absolute path to the CSV file to validate.")],
    text_col: Annotated[str, Field(description="Name of the column containing the text samples.")],
    label_col: Annotated[str, Field(description="Name of the column containing the assigned labels.")],
    k_neighbors: Annotated[
        int,
        Field(
            ge=3,
            le=100,
            description=(
                "Number of nearest neighbours to consider per row. "
                "Higher K = more stable agreement scores but slower. "
                "Default of 15 works well for datasets up to ~1000 rows."
            ),
        ),
    ] = 15,
    agreement_threshold: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Fraction of neighbours that must share a row's label for it to be "
                "considered clean. Rows below this threshold are sent to the LLM. "
                "Lower values (e.g. 0.3) flag fewer rows; higher (e.g. 0.7) flags more."
            ),
        ),
    ] = 0.5,
    llm_model: Annotated[
        str,
        Field(
            description="Ollama model name to use for judging. Must be pulled and available in Ollama (e.g. `ollama pull llama3.1:8b`).",
        ),
    ] = "llama3.1:8b",
    weighted_agreement: Annotated[
        bool,
        Field(
            description=(
                "Weight each neighbour's vote by its cosine similarity instead of "
                "counting all K neighbours equally. Benchmarked against the original "
                "unweighted majority vote on a ground-truth-labelled dataset "
                "(see scripts/evaluate_neighbor_algorithms.py) and won on F1 and "
                "precision at every K tested. Set False to restore the original "
                "unweighted KNN behaviour."
            ),
        ),
    ] = True,
) -> dict:
    """
    Run the full KNN validation pipeline on a CSV file and return a summary report.

    Pipeline stages (all timed):
      1. Encode each text row into a semantic embedding vector
      2. Select one anchor (most representative example) per label
      3. Compute full pairwise cosine similarity matrix
      4. For each row, find K nearest neighbours and compute label agreement
      5. Flag rows with agreement below threshold as suspicious
      6. Send each suspicious row to the LLM with anchor + neighbour context

    Returns a summary dict with total rows, suspicious count, LLM verdict
    breakdown, estimated mislabel %, token_stats, and per-stage timing.
    """
    global _pipeline, _ran
    try:
        _pipeline = ValidationPipeline(
            k_neighbors=k_neighbors,
            agreement_threshold=agreement_threshold,
            llm_model=llm_model,
            weighted_agreement=weighted_agreement,
        )
        summary = _pipeline.run(csv_path, text_col, label_col)
        _ran = True
        return summary
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool(
    title="Get Flagged Rows",
    annotations=ToolAnnotations(readOnlyHint=True),
)
def get_flagged_rows() -> list | dict:
    """
    Return all rows confirmed as potentially mislabeled by the LLM.

    Each entry includes:
      - row_index              original row number in the CSV
      - text_preview           first 200 characters of the text
      - assigned_label         the label currently in the dataset
      - suggested_label        recommended correction (set when verdict=bad and
                               confidence=high; null otherwise)
      - needs_review           True when verdict=bad but confidence is medium —
                               human review advised before applying the correction
      - neighbor_agreement     fraction of K nearest neighbours sharing this label
      - n_matching_neighbors   count of neighbours with the same label
      - k_neighbors            K used in the KNN pass
      - llm_verdict            "good" (label is correct) or "bad" (mislabeled)
      - llm_confidence         "high" or "medium" (low-confidence bad are discarded)
      - llm_reasoning          one or two sentence explanation from the LLM
      - detection_source       always "knn"

    Must call validate_dataset first.
    """
    if not _ran:
        return {"error": "No dataset has been validated yet. Call validate_dataset first."}
    return _pipeline.flagged_rows


@mcp.tool(
    title="Get KNN Stats",
    annotations=ToolAnnotations(readOnlyHint=True),
)
def get_knn_stats() -> list | dict:
    """
    Return KNN agreement statistics for every row flagged as suspicious
    before the LLM pass (i.e. all rows with agreement < threshold).

    Each entry includes:
      - row_index              original row number in the CSV
      - text_preview           first 200 characters of the text
      - assigned_label         the label currently in the dataset
      - neighbor_agreement     fraction of K neighbours sharing this label
      - n_matching_neighbors   count of neighbours with the same label
      - k                      K used
      - neighbor_label_counts  full label distribution among the K neighbours

    Useful for understanding the difficulty of each flagged row before
    looking at the LLM verdicts.

    Must call validate_dataset first.
    """
    if not _ran:
        return {"error": "No dataset has been validated yet. Call validate_dataset first."}
    return _pipeline.knn_stats


@mcp.tool(
    title="Export Report",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
    ),
)
def export_report(
    output_dir: Annotated[
        str,
        Field(
            description=(
                "Directory where report files will be written. Created automatically "
                "if it does not exist. Two timestamped files are generated: "
                "flagged_rows_<timestamp>.csv (human-friendly, opens in Excel) and "
                "report_<timestamp>.json (full machine-readable dump with summary, "
                "flagged rows, and KNN stats)."
            )
        ),
    ],
) -> dict:
    """
    Save the validation results to disk as a CSV and a JSON report.

    The CSV is the human artifact — easy to share, annotate, or load into a notebook.
    The JSON is the machine artifact — preserves every field for downstream tooling.

    Must call validate_dataset first.
    """
    if not _ran:
        return {"error": "No dataset has been validated yet. Call validate_dataset first."}
    try:
        writer = ReportWriter(output_dir)
        paths = writer.write(
            summary=_pipeline.summary,
            flagged_rows=_pipeline.flagged_rows,
            cluster_info=_pipeline.knn_stats,
        )
        return {"status": "ok", "files": paths}
    except Exception as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    mcp.run()
