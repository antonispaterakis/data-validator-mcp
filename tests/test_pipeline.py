"""
Unit and integration tests for data-validator-mcp.

Run with:  pytest tests/ -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.clustering import (
    dominant_label_for_cluster,
    get_cluster_overview,
)
from src.embeddings import encode_texts
from src.pipeline import ValidationPipeline


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TestEncodeTexts:
    def test_shape(self):
        texts = ["Hello world", "Another sentence", "Third sample"]
        embeddings = encode_texts(texts, show_progress=False)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0

    def test_dtype(self):
        embeddings = encode_texts(["test"], show_progress=False)
        assert embeddings.dtype == np.float64

    def test_normalized(self):
        embeddings = encode_texts(["unit vector test"], show_progress=False)
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------


class TestDominantLabel:
    def _make_data(self):
        # cluster_assignments: 0,0,0,1,1,  labels: A,A,B,B,B
        assignments = np.array([0, 0, 0, 1, 1])
        labels = pd.Series(["A", "A", "B", "B", "B"])
        return assignments, labels

    def test_dominant_majority(self):
        assignments, labels = self._make_data()
        assert dominant_label_for_cluster(0, assignments, labels) == "A"

    def test_dominant_single_class_cluster(self):
        assignments, labels = self._make_data()
        assert dominant_label_for_cluster(1, assignments, labels) == "B"


class TestGetClusterOverview:
    def test_returns_correct_structure(self):
        assignments = np.array([0, 0, 1, 1, 1])
        labels = pd.Series(["X", "X", "Y", "Y", "X"])
        overview = get_cluster_overview(assignments, labels)

        assert len(overview) == 2
        ids = {entry["cluster_id"] for entry in overview}
        assert ids == {0, 1}

    def test_sizes_sum_to_total(self):
        n = 20
        assignments = np.random.randint(0, 3, size=n)
        labels = pd.Series(np.random.choice(["cat", "dog"], size=n))
        overview = get_cluster_overview(assignments, labels)
        assert sum(entry["size"] for entry in overview) == n


# ---------------------------------------------------------------------------
# Pipeline (no LLM calls — patches the judge to avoid real API usage)
# ---------------------------------------------------------------------------


class TestValidationPipeline:
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_dataset.csv")

    def test_csv_loads(self):
        pipe = ValidationPipeline()
        df = pipe._load_csv(self.DATA_PATH, "text", "label")
        assert len(df) > 0
        assert "text" in df.columns
        assert "label" in df.columns

    def test_missing_column_raises(self):
        pipe = ValidationPipeline()
        with pytest.raises(ValueError, match="Column\\(s\\) not found"):
            pipe._load_csv(self.DATA_PATH, "nonexistent", "label")

    def _one_hot_embeddings(self, labels: pd.Series) -> np.ndarray:
        """Embeddings where every row is a one-hot vector for its label —
        guarantees each row's nearest neighbours all share its label."""
        unique_labels = labels.unique().tolist()
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        emb = np.zeros((len(labels), len(unique_labels)))
        for i, lbl in enumerate(labels):
            emb[i, label_to_idx[lbl]] = 1.0
        return emb

    def test_compute_knn_flags_returns_expected_structure(self):
        """When every row's neighbours all share its label, nothing is flagged."""
        pipe = ValidationPipeline(k_neighbors=5, agreement_threshold=0.5)
        pipe.df = pd.read_csv(self.DATA_PATH)
        pipe.df = pipe.df.dropna(subset=["text", "label"]).reset_index(drop=True)
        texts = pipe.df["text"].tolist()
        labels = pipe.df["label"]

        pipe.embeddings = self._one_hot_embeddings(labels)
        suspicious, knn_stats, neighbor_lookup = pipe._compute_knn_flags(texts, labels)

        assert suspicious == []
        assert knn_stats == []
        assert isinstance(neighbor_lookup, dict)

    def test_run_skips_llm_when_no_conflicts(self, monkeypatch):
        """
        If every row's nearest neighbours all share its label, no row is
        suspicious, so no LLM calls should happen and the summary should
        report 0 flagged rows.
        """
        import src.pipeline as pl

        df = pd.read_csv(self.DATA_PATH).dropna(subset=["text", "label"]).reset_index(drop=True)
        embeddings = self._one_hot_embeddings(df["label"])

        # Intercept encode_texts to return one-hot-per-label embeddings
        monkeypatch.setattr(
            pl,
            "encode_texts",
            lambda texts, **kwargs: embeddings,
        )

        pipe = ValidationPipeline(k_neighbors=5)
        summary = pipe.run(self.DATA_PATH, "text", "label")

        assert summary["n_suspicious"] == 0
        assert summary["llm_bad"] == 0
        assert summary["total_rows"] > 0
