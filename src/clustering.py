from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 15,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce embedding dimensionality with UMAP before clustering.

    UMAP finds non-linear structure that PCA-based HiPart misses, producing
    tighter, better-separated clusters — especially useful for small or
    semantically dense datasets where raw embeddings cluster poorly.

    n_components: target dimensions (15 works well before HiPart)
    n_neighbors: local neighbourhood size — lower = more local structure
    """
    from umap import UMAP
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, len(embeddings) - 1),
        random_state=random_state,
        metric="cosine",
    )
    return reducer.fit_transform(embeddings).astype(np.float64)


def cluster_embeddings(
    embeddings: np.ndarray,
    max_clusters: int = 15,
    min_sample_split: int = 5,
) -> np.ndarray:
    """
    Cluster N embeddings with HiPart dePDDP.

    Returns an integer array of length N with a cluster id per row.
    The number of clusters discovered is at most `max_clusters` but may
    be fewer depending on the data distribution.
    """
    from HiPart.clustering import DePDDP

    model = DePDDP(
        decomposition_method="pca",
        max_clusters_number=max_clusters,
        min_sample_split=min_sample_split,
    )
    model.fit(embeddings)
    return np.asarray(model.labels_, dtype=int)


def dominant_label_for_cluster(
    cluster_id: int,
    cluster_assignments: np.ndarray,
    row_labels: pd.Series,
) -> str:
    """Return the most frequent label among rows assigned to `cluster_id`."""
    mask = cluster_assignments == cluster_id
    counter: Counter = Counter(row_labels[mask].tolist())
    return counter.most_common(1)[0][0]


def get_cluster_overview(
    cluster_assignments: np.ndarray,
    row_labels: pd.Series,
) -> list[dict]:
    """
    Summarise every discovered cluster.

    Returns a list of dicts (one per cluster) with:
      cluster_id, size, dominant_label, label_distribution
    """
    overview = []
    for cluster_id in sorted(np.unique(cluster_assignments)):
        mask = cluster_assignments == cluster_id
        counter: Counter = Counter(row_labels[mask].tolist())
        dominant = counter.most_common(1)[0][0]
        overview.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(mask.sum()),
                "dominant_label": dominant,
                "label_distribution": dict(counter),
            }
        )
    return overview