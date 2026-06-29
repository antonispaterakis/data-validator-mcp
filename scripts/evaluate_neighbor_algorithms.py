"""
Evaluation harness for comparing neighbour-search / agreement-scoring
algorithms for mislabel detection.

Uses data/large_dataset.csv, which has a `true_label` column alongside the
(possibly corrupted) `label` column, so we can score each algorithm against
ground truth: precision / recall / F1 for "is this row mislabeled?".

Run with the project's venv:
    .venv/bin/python scripts/evaluate_neighbor_algorithms.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.embeddings import encode_texts  # real sentence-transformer encoder
except ImportError:
    from _tfidf_fallback import encode_texts_tfidf as encode_texts
    print(
        "[evaluate] sentence-transformers not installed — falling back to a "
        "pure-numpy TF-IDF embedding for this run. The algorithm comparison "
        "is still valid (same similarity matrix feeds every algorithm below), "
        "but re-run with the real encoder for production numbers.\n",
        file=sys.stderr,
    )

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "large_dataset.csv")


# ---------------------------------------------------------------------------
# Algorithms — each takes (sim_matrix, labels, k) and returns a per-row
# "suspicion score" in [0, 1] (higher = more likely mislabeled).
# Thresholding + scoring happens once, uniformly, after.
# ---------------------------------------------------------------------------

def algo_plain_knn(sim: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Current production algorithm: unweighted majority vote among top-k."""
    N = sim.shape[0]
    order = np.argsort(sim, axis=1)[:, ::-1]
    scores = np.zeros(N)
    for i in range(N):
        top_k = order[i, :k]
        agreement = np.mean(labels[top_k] == labels[i])
        scores[i] = 1 - agreement
    return scores


def algo_distance_weighted_knn(sim: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Weight each neighbour's vote by its similarity, not just count.
    A close same-label neighbour should count more than a barely-similar one."""
    N = sim.shape[0]
    order = np.argsort(sim, axis=1)[:, ::-1]
    scores = np.zeros(N)
    for i in range(N):
        top_k = order[i, :k]
        sims = sim[i, top_k]
        sims = np.clip(sims, 0, None)  # negative cosine sims shouldn't subtract
        same = (labels[top_k] == labels[i]).astype(float)
        denom = sims.sum() + 1e-8
        weighted_agreement = (sims * same).sum() / denom
        scores[i] = 1 - weighted_agreement
    return scores


def algo_mutual_knn(sim: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Shared/mutual-KNN agreement: only count a neighbour's vote if the
    relationship is reciprocal (i is also in j's top-k). This down-weights
    points that sit in a dense 'wrong-label' region but aren't reciprocally
    close to anything specific — reduces sensitivity to local density quirks."""
    N = sim.shape[0]
    order = np.argsort(sim, axis=1)[:, ::-1]
    topk_sets = [set(order[i, :k].tolist()) for i in range(N)]
    scores = np.zeros(N)
    for i in range(N):
        top_k = order[i, :k]
        mutual = [j for j in top_k if i in topk_sets[j]]
        if not mutual:
            mutual = list(top_k)  # fallback if nothing reciprocal
        agreement = np.mean(labels[np.array(mutual)] == labels[i])
        scores[i] = 1 - agreement
    return scores


def make_centroid_margin(embeddings: np.ndarray):
    """Class-centroid margin (no neighbour search at all — baseline to beat):
    compare each row's similarity to its own label's centroid vs. the best
    other-label centroid. Needs the raw embeddings (not just the similarity
    matrix), so it's built as a closure factory rather than fitting the
    `(sim, labels, k) -> scores` signature the other algorithms use."""
    def _algo(sim: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        unique_labels = np.unique(labels)
        centroids = {}
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            c = embeddings[idx].mean(axis=0)
            centroids[lbl] = c / (np.linalg.norm(c) + 1e-8)
        N = embeddings.shape[0]
        scores = np.zeros(N)
        for i in range(N):
            own_sim = float(embeddings[i] @ centroids[labels[i]])
            other_sims = [
                float(embeddings[i] @ centroids[lbl])
                for lbl in unique_labels
                if lbl != labels[i]
            ]
            best_other = max(other_sims) if other_sims else -1.0
            margin = own_sim - best_other  # positive = correctly closer to own class
            # squash margin into a suspicion score in [0, 1]
            scores[i] = 1 / (1 + np.exp(4 * margin))
        return scores
    return _algo


def algo_local_outlier_knn(sim: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """LOF-flavoured variant: a row's suspicion is high if its own-label
    neighbours are, on average, much farther away than its nearest neighbours
    overall (i.e. it sits far from its own class but close to others)."""
    N = sim.shape[0]
    order = np.argsort(sim, axis=1)[:, ::-1]
    scores = np.zeros(N)
    for i in range(N):
        top_k = order[i, :k]
        nearest_sim = sim[i, top_k].mean()

        same_label_idx = np.where(labels == labels[i])[0]
        same_label_idx = same_label_idx[same_label_idx != i]
        if len(same_label_idx) == 0:
            scores[i] = 0.0
            continue
        own_class_sim = sim[i, same_label_idx]
        own_class_top = np.sort(own_class_sim)[::-1][: min(k, len(own_class_sim))].mean()

        # if own-class similarity is much lower than general nearest-neighbour
        # similarity, this row doesn't fit its class well.
        gap = nearest_sim - own_class_top
        scores[i] = 1 / (1 + np.exp(-6 * gap))
    return scores


ALGORITHMS = {
    "plain_knn (current)": algo_plain_knn,
    "distance_weighted_knn": algo_distance_weighted_knn,
    "mutual_knn": algo_mutual_knn,
    "local_outlier_knn": algo_local_outlier_knn,
}


def precision_recall_f1(scores: np.ndarray, ground_truth: np.ndarray, threshold: float):
    pred = scores >= threshold
    tp = int(np.sum(pred & ground_truth))
    fp = int(np.sum(pred & ~ground_truth))
    fn = int(np.sum(~pred & ground_truth))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


def best_threshold(scores: np.ndarray, ground_truth: np.ndarray):
    """Sweep thresholds, return the one maximising F1."""
    candidates = np.unique(np.round(scores, 4))
    best = (0.0, 0.0, 0.0, 0.5)
    for t in candidates:
        p, r, f1, *_ = precision_recall_f1(scores, ground_truth, t)
        if f1 > best[2]:
            best = (p, r, f1, t)
    return best


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label", "true_label"]).reset_index(drop=True)
    texts = df["text"].tolist()
    labels = df["label"].to_numpy()
    ground_truth = (df["label"] != df["true_label"]).to_numpy()
    print(f"{len(df)} rows, {ground_truth.sum()} actually mislabeled ({ground_truth.mean():.1%})\n")

    print("Encoding with all-MiniLM-L6-v2 ...")
    t0 = time.perf_counter()
    embeddings = encode_texts(texts, show_progress=False)
    print(f"  done in {time.perf_counter() - t0:.2f}s\n")

    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -1.0)

    algos = dict(ALGORITHMS)
    algos["centroid_margin"] = make_centroid_margin(embeddings)

    print(f"{'algorithm':<24} {'k':>3}  {'precision':>9} {'recall':>7} {'f1':>6}  {'thr':>6}")
    print("-" * 70)

    results = []
    for name, fn in algos.items():
        for k in (5, 10, 15, 20):
            if k >= len(df):
                continue
            scores = fn(sim, labels, k)
            p, r, f1, thr = best_threshold(scores, ground_truth)
            results.append((name, k, p, r, f1, thr))
            print(f"{name:<24} {k:>3}  {p:>9.3f} {r:>7.3f} {f1:>6.3f}  {thr:>6.3f}")

    results.sort(key=lambda x: -x[4])
    print("\nTop 5 by F1:")
    for name, k, p, r, f1, thr in results[:5]:
        print(f"  {name:<24} k={k:<3} F1={f1:.3f}  P={p:.3f} R={r:.3f}  threshold={thr:.3f}")


if __name__ == "__main__":
    main()
