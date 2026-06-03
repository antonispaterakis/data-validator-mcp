"""
Report writer: exports pipeline results to disk.

Writes two files into the given output directory:
  - flagged_rows_<timestamp>.csv   one row per flagged sample, openable in Excel/pandas
  - report_<timestamp>.json        full machine-readable dump (summary + flagged + clusters)

The CSV is the "human" artifact — easy to share, annotate, or load into a notebook.
The JSON is the "machine" artifact — preserves every field for downstream tooling.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime


_CSV_FIELDS = [
    "row_index",
    "text_preview",
    "assigned_label",
    "suggested_label",
    "cluster_id",
    "cluster_dominant_label",
    "cluster_size",
    "llm_verdict",
    "llm_confidence",
    "llm_reasoning",
]


class ReportWriter:
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def write(
        self,
        summary: dict,
        flagged_rows: list[dict],
        cluster_info: list[dict],
    ) -> dict[str, str]:
        """
        Write all report files to output_dir.

        Returns a dict mapping file role to absolute path:
          {"flagged_csv": "...", "report_json": "..."}
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = os.path.join(self.output_dir, f"flagged_rows_{ts}.csv")
        json_path = os.path.join(self.output_dir, f"report_{ts}.json")

        self._write_csv(flagged_rows, csv_path)
        self._write_json(summary, flagged_rows, cluster_info, json_path)

        return {"flagged_csv": csv_path, "report_json": json_path}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_csv(self, flagged_rows: list[dict], path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flagged_rows)

    def _write_json(
        self,
        summary: dict,
        flagged_rows: list[dict],
        cluster_info: list[dict],
        path: str,
    ) -> None:
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "flagged_rows": flagged_rows,
            "cluster_overview": cluster_info,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
