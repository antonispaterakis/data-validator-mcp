"""
Core validation pipeline.

Orchestrates: load CSV → encode → (UMAP) → cluster → detect conflicts → LLM judge.

Two-pass LLM strategy:
  Pass 1 — clustering-flagged rows: rows whose label disagrees with their cluster's
            dominant label (only in pure clusters). Strong cluster context.
  Pass 2 — blob rows: rows in low-purity clusters where clustering can't decide.
            Uses full label taxonomy + cluster distribution as context instead.
            LLM also suggests the correct label directly.

Together, the two passes cover the full dataset while keeping total LLM calls
well below N (a naive full pass), following the efficiency principle from:
  Theocharopoulos et al., "Large Language Models for Efficient Topic Modeling",
  Neural Computing and Applications, 2025.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from .clustering import cluster_embeddings, dominant_label_for_cluster, get_cluster_overview, reduce_dimensions
from .embeddings import encode_texts
from .llm_judge import LLMJudge


class ValidationPipeline:
    """
    Stateful pipeline — call `run()`, then query `flagged_rows` and
    `cluster_info` from the same instance.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_clusters: int = 12,
        min_sample_split: int = 3,
        purity_threshold: float = 0.60,
        use_umap: bool = False,
        umap_components: int = 15,
        blob_pass: bool = True,
        llm_model: str = "llama3.1:8b",
    ):
        self.embedding_model = embedding_model
        self.max_clusters = max_clusters
        self.min_sample_split = min_sample_split
        self.purity_threshold = purity_threshold
        self.use_umap = use_umap
        self.umap_components = umap_components
        self.blob_pass = blob_pass
        self.llm_model = llm_model

        # Populated by run()
        self.df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.cluster_assignments: Optional[np.ndarray] = None
        self.flagged_rows: list[dict] = []
        self.cluster_info: list[dict] = []
        self.summary: dict = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        csv_path: str,
        text_col: str,
        label_col: str,
    ) -> dict:
        """
        Execute the full pipeline and return a summary report dict.
        """
        t_total_start = time.perf_counter()

        # 1. Load
        self.df = self._load_csv(csv_path, text_col, label_col)
        texts = self.df[text_col].tolist()
        labels = self.df[label_col]
        n_rows = len(self.df)
        all_labels = sorted(labels.unique().tolist())

        # 2. Encode
        print(f"[pipeline] Encoding {n_rows} rows with {self.embedding_model}…", file=sys.stderr)
        t0 = time.perf_counter()
        self.embeddings = encode_texts(texts, model_name=self.embedding_model)
        t_encode = time.perf_counter() - t0

        # 3. (Optional) UMAP dimensionality reduction
        cluster_input = self.embeddings
        if self.use_umap:
            print(f"[pipeline] Reducing to {self.umap_components}d with UMAP…", file=sys.stderr)
            cluster_input = reduce_dimensions(self.embeddings, n_components=self.umap_components)

        # 4. Cluster
        print("[pipeline] Clustering with HiPart dePDDP…", file=sys.stderr)
        t0 = time.perf_counter()
        self.cluster_assignments = cluster_embeddings(
            cluster_input,
            max_clusters=self.max_clusters,
            min_sample_split=self.min_sample_split,
        )
        t_cluster = time.perf_counter() - t0
        n_clusters = len(np.unique(self.cluster_assignments))

        # 5. Detect conflicts — returns clustering flags + blob indices separately
        print(f"[pipeline] Detected {n_clusters} clusters. Scanning for label conflicts…", file=sys.stderr)
        conflict_indices, blob_indices = self._detect_conflicts(labels)
        n_flagged_clustering = len(conflict_indices)
        n_blob = len(blob_indices)
        print(
            f"[pipeline] {n_flagged_clustering} rows flagged by clustering, "
            f"{n_blob} rows in low-purity clusters (blob pass).",
            file=sys.stderr,
        )

        # 6. LLM judge — pass 1: clustering-flagged rows
        t0 = time.perf_counter()
        judged = self._judge_flagged(conflict_indices, texts, labels)

        # 7. LLM judge — pass 2: blob rows (if enabled)
        blob_judged = []
        if self.blob_pass and blob_indices:
            print(f"[pipeline] Blob pass: judging {n_blob} rows in mixed clusters…", file=sys.stderr)
            blob_judged = self._judge_blob(blob_indices, texts, labels, all_labels)

        t_judge = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        # Combine: clustering flags first, then blob confirmations.
        # Blob pass only keeps high-confidence bad verdicts — the LLM has weak
        # context for blob rows, so medium/low confidence flags are too noisy.
        all_judged = judged + [
            r for r in blob_judged
            if r["llm_verdict"] == "bad" and r["llm_confidence"] == "high"
        ]

        # Store for later retrieval via MCP tools
        self.cluster_info = get_cluster_overview(self.cluster_assignments, labels)
        self.flagged_rows = all_judged

        # 8. Summary
        verdicts = Counter(r["llm_verdict"] for r in all_judged)
        n_confirmed_bad = verdicts.get("bad", 0)
        n_llm_calls = len(judged) + len(blob_judged)

        self.summary = {
            "csv_path": csv_path,
            "text_col": text_col,
            "label_col": label_col,
            "total_rows": n_rows,
            "n_clusters": n_clusters,
            "purity_threshold": self.purity_threshold,
            "n_flagged_by_clustering": n_flagged_clustering,
            "n_blob_rows_judged": n_blob if self.blob_pass else 0,
            "n_confirmed_bad": n_confirmed_bad,
            "llm_bad": verdicts.get("bad", 0),
            "llm_good": verdicts.get("good", 0),
            "llm_unknown": verdicts.get("unknown", 0),
            "estimated_mislabel_pct": (
                round(100 * n_confirmed_bad / n_rows, 1) if n_rows else 0
            ),
            "label_counts": dict(Counter(labels.tolist())),
            "llm_calls_made": n_llm_calls,
            "llm_calls_saved_vs_full_pass": n_rows - n_llm_calls,
            "llm_model": self.llm_model,
            "timing_seconds": {
                "encode": round(t_encode, 3),
                "cluster": round(t_cluster, 3),
                "judge": round(t_judge, 3),
                "total": round(t_total, 3),
            },
        }
        return self.summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_csv(self, csv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        missing = [c for c in (text_col, label_col) if c not in df.columns]
        if missing:
            raise ValueError(
                f"Column(s) not found in CSV: {missing}. "
                f"Available columns: {df.columns.tolist()}"
            )
        df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
        df[text_col] = df[text_col].astype(str)
        df[label_col] = df[label_col].astype(str)
        return df

    def _detect_conflicts(self, labels: pd.Series) -> tuple[list[int], list[int]]:
        """
        Returns (conflict_indices, blob_indices).

        conflict_indices — rows in pure clusters whose label ≠ dominant label.
        blob_indices     — ALL rows in low-purity clusters (sent to blob pass).
        """
        conflict_indices = []
        blob_indices = []
        seen_blob_clusters: set[int] = set()

        for idx in range(len(labels)):
            cluster_id = self.cluster_assignments[idx]
            mask = self.cluster_assignments == cluster_id
            cluster_labels = labels[mask]
            purity = cluster_labels.value_counts().iloc[0] / len(cluster_labels)

            if purity < self.purity_threshold:
                # Collect blob cluster rows once per cluster
                if cluster_id not in seen_blob_clusters:
                    seen_blob_clusters.add(cluster_id)
                    blob_indices.extend(int(j) for j in np.where(mask)[0])
            else:
                dominant = cluster_labels.value_counts().idxmax()
                if labels.iloc[idx] != dominant:
                    conflict_indices.append(idx)

        return conflict_indices, blob_indices

    def _judge_flagged(
        self,
        conflict_indices: list[int],
        texts: list[str],
        labels: pd.Series,
    ) -> list[dict]:
        """Pass 1: judge rows flagged by clustering. Strong cluster context."""
        if not conflict_indices:
            return []

        judge = LLMJudge(model=self.llm_model)
        results = []

        for i, idx in enumerate(conflict_indices, 1):
            cluster_id = self.cluster_assignments[idx]
            dominant = dominant_label_for_cluster(
                cluster_id, self.cluster_assignments, labels
            )
            cluster_mask = self.cluster_assignments == cluster_id
            cluster_texts = [
                texts[j]
                for j in np.where(cluster_mask)[0]
                if j != idx
            ][:5]

            print(
                f"[pipeline] Pass 1 — {i}/{len(conflict_indices)}: "
                f"row {idx} label='{labels.iloc[idx]}' dominant='{dominant}'",
                file=sys.stderr,
            )

            judgement = judge.judge_row(
                text=texts[idx],
                label=labels.iloc[idx],
                cluster_dominant=dominant,
                cluster_examples=cluster_texts,
            )

            verdict = judgement["verdict"]
            confidence = judgement["confidence"]
            suggested_label = (
                dominant if verdict == "bad" and confidence == "high" else None
            )

            results.append({
                "row_index": idx,
                "text_preview": texts[idx][:200],
                "assigned_label": labels.iloc[idx],
                "suggested_label": suggested_label,
                "cluster_id": int(cluster_id),
                "cluster_dominant_label": dominant,
                "cluster_size": int(cluster_mask.sum()),
                "llm_verdict": verdict,
                "llm_confidence": confidence,
                "llm_reasoning": judgement["reasoning"],
                "detection_source": "clustering",
            })

        return results

    def _judge_blob(
        self,
        blob_indices: list[int],
        texts: list[str],
        labels: pd.Series,
        all_labels: list[str],
    ) -> list[dict]:
        """
        Pass 2: judge every row in low-purity clusters.
        No dominant label — uses full taxonomy + cluster distribution.
        Only confirmed bad rows are added to flagged_rows.
        """
        judge = LLMJudge(model=self.llm_model)
        results = []

        for i, idx in enumerate(blob_indices, 1):
            cluster_id = self.cluster_assignments[idx]
            cluster_mask = self.cluster_assignments == cluster_id
            cluster_dist = dict(Counter(labels[cluster_mask].tolist()))

            print(
                f"[pipeline] Pass 2 — {i}/{len(blob_indices)}: "
                f"row {idx} label='{labels.iloc[idx]}'",
                file=sys.stderr,
            )

            judgement = judge.judge_blob_row(
                text=texts[idx],
                label=labels.iloc[idx],
                all_labels=all_labels,
                cluster_distribution=cluster_dist,
            )

            results.append({
                "row_index": idx,
                "text_preview": texts[idx][:200],
                "assigned_label": labels.iloc[idx],
                "suggested_label": judgement.get("suggested_label"),
                "cluster_id": int(cluster_id),
                "cluster_dominant_label": None,  # meaningless in blob clusters
                "cluster_size": int(cluster_mask.sum()),
                "llm_verdict": judgement["verdict"],
                "llm_confidence": judgement["confidence"],
                "llm_reasoning": judgement["reasoning"],
                "detection_source": "blob_pass",
            })

        return results
