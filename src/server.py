"""
MCP server for data-validator-mcp.

Exposes four tools:
  • validate_dataset      — run the full pipeline on a CSV
  • get_flagged_rows      — retrieve rows the pipeline flagged as mislabeled
  • get_cluster_overview  — inspect the discovered clusters
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
        "using semantic clustering + a local LLM judge (Ollama). No API key needed.\n\n"
        "Step 1 — Call validate_dataset with the path to your CSV, the name of the "
        "text column, and the name of the label column. Optionally tune max_clusters "
        "and purity_threshold.\n\n"
        "Step 2 — Call get_flagged_rows to see every suspicious row: what label it "
        "has, what the cluster suggests, the LLM verdict (good/bad), confidence, "
        "reasoning, and a suggested correction where the model is confident.\n\n"
        "Step 3 — Call get_cluster_overview to inspect how the data was partitioned "
        "and check cluster purity across all discovered groups.\n\n"
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
    max_clusters: Annotated[
        int,
        Field(
            ge=1,
            le=50,
            description=(
                "Upper bound on the number of clusters HiPart may create. "
                "Lower values produce coarser, more reliable groupings. "
                "Default of 12 works well for datasets up to ~500 rows."
            ),
        ),
    ] = 12,
    purity_threshold: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Minimum fraction of one label in a cluster before its outliers are flagged. "
                "Raise (e.g. 0.80) to reduce false positives; lower (e.g. 0.50) to catch more "
                "potential errors at the cost of more noise."
            ),
        ),
    ] = 0.60,
    use_umap: Annotated[
        bool,
        Field(
            description=(
                "Apply UMAP dimensionality reduction before clustering. "
                "Helps on small or semantically dense datasets where HiPart struggles "
                "to find clean separations in raw embedding space. Adds ~2-5s."
            ),
        ),
    ] = False,
    llm_model: Annotated[
        str,
        Field(
            description=(
                "Ollama model used for judging. Larger models give better precision. "
                "llama3.1:8b is recommended. llama3.2 is faster but less accurate."
            ),
        ),
    ] = "llama3.1:8b",
    blob_pass: Annotated[
        bool,
        Field(
            description=(
                "Run a second LLM pass on rows in low-purity (mixed) clusters. "
                "Improves recall by covering rows that clustering alone cannot judge. "
                "Increases LLM calls proportional to blob cluster size, but stays "
                "well below a full dataset pass."
            ),
        ),
    ] = True,
) -> dict:
    """
    Run the full validation pipeline on a CSV file and return a summary report.

    Pipeline stages (all timed):
      1. Encode each text row into a semantic embedding vector
      2. Cluster embeddings with HiPart dePDDP
      3. Flag rows whose label disagrees with their cluster's dominant label
         (only in clusters above purity_threshold — mixed clusters are skipped)
      4. Send each flagged row to a local Ollama LLM for verdict confirmation

    Returns a summary dict with total rows, cluster count, flag count,
    LLM verdict breakdown, estimated mislabel %, and per-stage timing.
    """
    global _pipeline, _ran
    try:
        _pipeline = ValidationPipeline(
            max_clusters=max_clusters,
            purity_threshold=purity_threshold,
            use_umap=use_umap,
            blob_pass=blob_pass,
            llm_model=llm_model,
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
    Return all rows flagged as potentially mislabeled.

    Each entry includes:
      - row_index          original row number in the CSV
      - text_preview       first 200 characters of the text
      - assigned_label     the label currently in the dataset
      - suggested_label    recommended correction (set when verdict=bad and confidence=high,
                           null otherwise — human review required for uncertain cases)
      - cluster_id         which cluster this row belongs to
      - cluster_dominant_label  the most common label in that cluster
      - cluster_size       number of rows in the cluster
      - llm_verdict        "good" (label is correct), "bad" (mislabeled), or "unknown"
      - llm_confidence     "high", "medium", or "low"
      - llm_reasoning      one or two sentence explanation from the LLM

    Must call validate_dataset first.
    """
    if not _ran:
        return {"error": "No dataset has been validated yet. Call validate_dataset first."}
    return _pipeline.flagged_rows


@mcp.tool(
    title="Get Cluster Overview",
    annotations=ToolAnnotations(readOnlyHint=True),
)
def get_cluster_overview() -> list | dict:
    """
    Return a summary of every cluster discovered during validation.

    Each entry includes:
      - cluster_id           integer cluster identifier
      - size                 number of rows assigned to this cluster
      - dominant_label       the most frequent label in the cluster
      - label_distribution   full count of every label present in the cluster

    High-purity clusters (one label dominates) are reliable signal.
    Low-purity clusters are semantically mixed — flagging is skipped for these.

    Must call validate_dataset first.
    """
    if not _ran:
        return {"error": "No dataset has been validated yet. Call validate_dataset first."}
    return _pipeline.cluster_info


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
                "flagged rows, and cluster overview)."
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
            cluster_info=_pipeline.cluster_info,
        )
        return {"status": "ok", "files": paths}
    except Exception as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    mcp.run()
