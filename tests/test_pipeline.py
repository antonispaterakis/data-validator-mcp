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


class TestWeightedAgreement:
    """
    Distance-weighted KNN agreement (default since the
    scripts/evaluate_neighbor_algorithms.py benchmark showed it beats plain
    majority-vote KNN on F1/precision at every K tested). Covers both modes
    and the case that motivated the change: a near-duplicate with a
    different label should be a *stronger* signal than several
    loosely-similar same-label neighbours.
    """

    def _mixed_similarity_embeddings(self) -> tuple[list[str], pd.Series, np.ndarray]:
        # Row 0 ("target"): label A.
        # Neighbours: one near-duplicate with label B (sim ~0.95), and three
        # loosely-similar same-label-A neighbours (sim ~0.2).
        # Plain majority vote (3 vs 1) says "agrees with its label" (0.75).
        # Weighted-by-similarity should flag this as suspicious instead,
        # since the one disagreeing neighbour dominates by similarity.
        texts = ["target", "near-dup-diff-label", "loose-a1", "loose-a2", "loose-a3"]
        labels = pd.Series(["A", "B", "A", "A", "A"])

        # Build 2D embeddings by hand with known cosine similarities to row 0.
        def vec(angle_deg: float) -> np.ndarray:
            theta = np.radians(angle_deg)
            return np.array([np.cos(theta), np.sin(theta)])

        embeddings = np.array([
            vec(0),      # target itself, angle 0
            vec(18),     # near-duplicate, diff label -> cos(18°) ≈ 0.95
            vec(78),     # loose same-label -> cos(78°) ≈ 0.21
            vec(80),
            vec(82),
        ])
        return texts, labels, embeddings

    def test_weighted_flags_near_duplicate_conflict(self):
        texts, labels, embeddings = self._mixed_similarity_embeddings()

        weighted = ValidationPipeline(k_neighbors=4, agreement_threshold=0.5, weighted_agreement=True)
        weighted.embeddings = embeddings
        w_suspicious, w_stats, _ = weighted._compute_knn_flags(texts, labels)

        plain = ValidationPipeline(k_neighbors=4, agreement_threshold=0.5, weighted_agreement=False)
        plain.embeddings = embeddings
        p_suspicious, p_stats, _ = plain._compute_knn_flags(texts, labels)

        # Plain majority vote: 3/4 same-label neighbours -> agreement 0.75 -> NOT flagged.
        assert 0 not in p_suspicious

        # Weighted: the near-duplicate (sim ~0.95, different label) should
        # dominate the three loosely-similar same-label neighbours (sim ~0.2
        # each), pulling weighted agreement below the threshold.
        assert 0 in w_suspicious
        flagged_stat = next(s for s in w_stats if s["row_index"] == 0)
        assert flagged_stat["neighbor_agreement"] < 0.5
        p_stat_0 = next((s for s in p_stats if s["row_index"] == 0), None)
        if p_stat_0:
            assert flagged_stat["neighbor_agreement"] < p_stat_0["neighbor_agreement"]

    def test_weighted_agreement_default_is_true(self):
        assert ValidationPipeline().weighted_agreement is True

    def test_weighted_and_unweighted_agree_when_all_neighbors_equidistant(self):
        """When all neighbours are equally similar, weighting changes nothing."""
        labels = pd.Series(["A", "A", "A", "B", "B"])
        texts = [f"row{i}" for i in range(5)]
        # All pairwise cosine similarities equal -> identical unit vectors
        # except tiny perturbation to keep argsort stable/deterministic.
        base = np.array([1.0, 0.0])
        embeddings = np.tile(base, (5, 1)) + np.random.RandomState(0).normal(0, 1e-9, (5, 2))
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        weighted = ValidationPipeline(k_neighbors=4, agreement_threshold=0.5, weighted_agreement=True)
        weighted.embeddings = embeddings
        w_suspicious, _, _ = weighted._compute_knn_flags(texts, labels)

        plain = ValidationPipeline(k_neighbors=4, agreement_threshold=0.5, weighted_agreement=False)
        plain.embeddings = embeddings
        p_suspicious, _, _ = plain._compute_knn_flags(texts, labels)

        assert sorted(w_suspicious) == sorted(p_suspicious)
