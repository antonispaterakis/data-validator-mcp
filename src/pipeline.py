"""
Core validation pipeline — KNN-based label conflict detection.

Replaces the earlier HiPart clustering approach with K-nearest-neighbour
agreement scoring, which works regardless of whether the label classes
form geometrically separable clusters.

Pipeline:
  1. Encode texts with a sentence-transformer model.
  2. Select one anchor (most representative embedding) per label.
  3. Compute full N×N cosine similarity matrix
     (dot product on L2-normalised vectors — no separate normalisation needed).
  4. For each row, find its K=15 nearest neighbours (excluding itself) and
     compute neighbour_label_agreement = fraction sharing the row's label.
  5. Flag rows with agreement < 0.5 as suspicious.
  6. Send each suspicious row to the LLM with:
       - anchor text for the assigned label (absolute reference)
       - top-5 nearest neighbour texts + labels (relative context)
       - agreement score (quantitative signal)
  7. Keep bad verdicts: high-confidence → suggested_label set;
     medium-confidence → needs_review=True; low-confidence → discarded.

Token tracking:
  A single LLMJudge instance accumulates token counts. After the run,
  token_stats is added to the summary with efficiency_ratio showing
  pipeline cost vs hypothetical full-dataset scan.

Efficiency principle from:
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

from .embeddings import encode_texts, select_anchors
from .llm_judge import LLMJudge


class ValidationPipeline:
    """
    Stateful pipeline — call `run()`, then query `flagged_rows` and
    `knn_stats` from the same instance.
    """

    def __init__(
        self,
        embedding_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        k_neighbors: int = 5,
        agreement_threshold: float = 0.5,
        llm_model: str = "meta-llama-3.1-8b-instruct",
    ):
        self.embedding_model = embedding_model
        self.k_neighbors = k_neighbors
        self.agreement_threshold = agreement_threshold
        self.llm_model = llm_model

        # Populated by run()
        self.df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.flagged_rows: list[dict] = []
        self.knn_stats: list[dict] = []   # per-suspicious-row KNN metadata
        self.summary: dict = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, csv_path: str, text_col: str, label_col: str) -> dict:
        """Execute the full pipeline and return a summary report dict."""
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

        # 3. Select anchors — one representative row per label
        print("[pipeline] Selecting anchors…", file=sys.stderr)
        anchors = select_anchors(texts, labels, self.embeddings)
        print(f"[pipeline] Anchors selected for: {list(anchors.keys())}", file=sys.stderr)

        # 4. KNN agreement scoring
        print(f"[pipeline] Computing KNN (K={self.k_neighbors}) agreement scores…", file=sys.stderr)
        t0 = time.perf_counter()
        suspicious_indices, self.knn_stats, neighbor_lookup = self._compute_knn_flags(texts, labels)
        t_knn = time.perf_counter() - t0
        n_suspicious = len(suspicious_indices)
        print(
            f"[pipeline] {n_suspicious}/{n_rows} rows suspicious "
            f"(agreement < {self.agreement_threshold})",
            file=sys.stderr,
        )

        # 5. LLM judge — single unified pass
        judge = LLMJudge(model=self.llm_model)
        knn_lookup = {s["row_index"]: s for s in self.knn_stats}

        t0 = time.perf_counter()
        self.flagged_rows = self._judge_suspicious(
            suspicious_indices, texts, labels, all_labels,
            judge, anchors, neighbor_lookup, knn_lookup,
        )
        t_judge = time.perf_counter() - t0
        t_total = time.perf_counter() - t_total_start

        # 6. Token stats
        ts = judge.token_stats
        n_llm_calls = n_suspicious
        avg_tokens = ts["total_tokens"] / max(n_llm_calls, 1)
        tokens_brute_force = int(avg_tokens * n_rows)
        token_stats = {
            "tokens_used": ts["total_tokens"],
            "tokens_input": ts["total_input_tokens"],
            "tokens_output": ts["total_output_tokens"],
            "rows_scanned_by_llm": n_llm_calls,
            "total_rows": n_rows,
            "tokens_if_brute_force": tokens_brute_force,
            "efficiency_ratio": round(tokens_brute_force / max(ts["total_tokens"], 1), 1),
        }

        # 7. Summary
        verdicts = Counter(r["llm_verdict"] for r in self.flagged_rows)
        n_confirmed_bad = verdicts.get("bad", 0)

        self.summary = {
            "csv_path": csv_path,
            "text_col": text_col,
            "label_col": label_col,
            "total_rows": n_rows,
            "k_neighbors": self.k_neighbors,
            "agreement_threshold": self.agreement_threshold,
            "n_suspicious": n_suspicious,
            "n_confirmed_bad": n_confirmed_bad,
            "llm_bad": verdicts.get("bad", 0),
            "llm_good": verdicts.get("good", 0),
            "llm_unknown": verdicts.get("unknown", 0),
            "estimated_mislabel_pct": round(100 * n_confirmed_bad / n_rows, 1) if n_rows else 0,
            "label_counts": dict(Counter(labels.tolist())),
            "llm_calls_made": n_llm_calls,
            "llm_calls_saved_vs_full_pass": n_rows - n_llm_calls,
            "llm_model": self.llm_model,
            "token_stats": token_stats,
            "timing_seconds": {
                "encode": round(t_encode, 3),
                "knn": round(t_knn, 3),
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

    def _compute_knn_flags(
        self,
        texts: list[str],
        labels: pd.Series,
    ) -> tuple[list[int], list[dict], dict[int, list[dict]]]:
        """
        Compute K nearest neighbours for every row and flag those with
        low label agreement among neighbours.

        Returns:
          suspicious_indices — row indices where agreement < threshold
          knn_stats          — list of per-suspicious-row dicts (for MCP tool)
          neighbor_lookup    — {idx: [{text, label, similarity}, ...]} top-5
                               neighbours for each suspicious row (for LLM context)
        """
        N = len(labels)
        k = min(self.k_neighbors, N - 1)

        # Full cosine similarity matrix — O(N²·D) but embeddings are already
        # L2-normalised so cosine sim = dot product; no extra work needed.
        sim_matrix: np.ndarray = self.embeddings @ self.embeddings.T  # (N, N)
        np.fill_diagonal(sim_matrix, -1.0)  # exclude each row's self-similarity

        # Sort all neighbours once (descending sim) — O(N² log N)
        sorted_neighbors: np.ndarray = np.argsort(sim_matrix, axis=1)[:, ::-1]  # (N, N)

        suspicious_indices: list[int] = []
        knn_stats: list[dict] = []
        neighbor_lookup: dict[int, list[dict]] = {}

        for idx in range(N):
            top_k = sorted_neighbors[idx, :k]
            neighbor_labels = [labels.iloc[int(j)] for j in top_k]

            n_matching = sum(1 for l in neighbor_labels if l == labels.iloc[idx])
            agreement = n_matching / k

            if agreement < self.agreement_threshold:
                suspicious_indices.append(idx)

                knn_stats.append({
                    "row_index": idx,
                    "text_preview": texts[idx][:200],
                    "assigned_label": labels.iloc[idx],
                    "neighbor_agreement": round(agreement, 3),
                    "n_matching_neighbors": n_matching,
                    "k": k,
                    "neighbor_label_counts": dict(Counter(neighbor_labels)),
                })

                # Top-5 neighbours for LLM prompt context
                neighbor_lookup[idx] = [
                    {
                        "text": texts[int(j)],
                        "label": labels.iloc[int(j)],
                        "similarity": round(float(sim_matrix[idx, int(j)]), 3),
                    }
                    for j in top_k[:5]
                ]

        return suspicious_indices, knn_stats, neighbor_lookup

    def _judge_suspicious(
        self,
        suspicious_indices: list[int],
        texts: list[str],
        labels: pd.Series,
        all_labels: list[str],
        judge: LLMJudge,
        anchors: dict[str, dict],
        neighbor_lookup: dict[int, list[dict]],
        knn_lookup: dict[int, dict],
    ) -> list[dict]:
        """
        Single unified LLM pass: judge every suspicious row using
        anchor + KNN context. Keeps high-confidence bad (with suggested_label)
        and medium-confidence bad (with needs_review=True). Discards low-confidence.
        """
        results = []

        for i, idx in enumerate(suspicious_indices, 1):
            row_label = labels.iloc[idx]
            anchor_text = anchors.get(row_label, {}).get("text", "")
            neighbors = neighbor_lookup.get(idx, [])
            stats = knn_lookup[idx]

            print(
                f"[pipeline] {i}/{len(suspicious_indices)}: "
                f"row {idx} label='{row_label}' agreement={stats['neighbor_agreement']:.0%}",
                file=sys.stderr,
            )

            judgement = judge.judge_knn_row(
                text=texts[idx],
                label=row_label,
                anchor_text=anchor_text,
                neighbor_contexts=neighbors,
                agreement=stats["neighbor_agreement"],
                n_matching=stats["n_matching_neighbors"],
                k=stats["k"],
                all_labels=all_labels,
            )

            verdict = judgement["verdict"]
            confidence = judgement["confidence"]

            # Discard non-bad verdicts and low-confidence bad ones
            if verdict != "bad" or confidence == "low":
                continue

            needs_review = confidence == "medium"
            suggested_label = (
                judgement.get("suggested_label") if confidence == "high" else None
            )

            results.append({
                "row_index": idx,
                "text_preview": texts[idx][:200],
                "assigned_label": row_label,
                "suggested_label": suggested_label,
                "needs_review": needs_review,
                "neighbor_agreement": stats["neighbor_agreement"],
                "n_matching_neighbors": stats["n_matching_neighbors"],
                "k_neighbors": stats["k"],
                "llm_verdict": verdict,
                "llm_confidence": confidence,
                "llm_reasoning": judgement["reasoning"],
                "detection_source": "knn",
            })

        return results
