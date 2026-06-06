"""
Sentence-BERT text encoding.
Caches the model in memory so repeated calls within a session are fast.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_encoder_cache: dict[str, SentenceTransformer] = {}


def get_encoder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    if model_name not in _encoder_cache:
        _encoder_cache[model_name] = SentenceTransformer(model_name)
    return _encoder_cache[model_name]


def encode_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    show_progress: bool = True,
) -> np.ndarray:
    """Return (N, D) float32 embedding matrix for a list of text strings."""
    encoder = get_encoder(model_name)
    embeddings = encoder.encode(
        texts,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine-friendly unit vectors
    )
    return embeddings.astype(np.float64)


def select_anchors(
    texts: list[str],
    labels: "list[str] | object",  # list or pd.Series
    embeddings: np.ndarray,
) -> dict[str, dict]:
    """
    For each unique label, find the most representative (anchor) row.

    The anchor is the row whose embedding has the highest cosine similarity
    to the centroid of all rows sharing that label. Since embeddings are
    L2-normalised, cosine similarity reduces to a dot product.

    Returns:
        {label: {"text": str, "embedding": np.ndarray, "index": int}}
    """
    labels_list = list(labels)
    unique_labels: list[str] = list(dict.fromkeys(labels_list))  # order-preserving dedup
    labels_arr = np.array(labels_list)

    anchors: dict[str, dict] = {}
    for lbl in unique_labels:
        indices = np.where(labels_arr == lbl)[0]
        lbl_embs = embeddings[indices]  # (k, d)

        centroid = lbl_embs.mean(axis=0)
        centroid_unit = centroid / (np.linalg.norm(centroid) + 1e-8)

        sims = lbl_embs @ centroid_unit        # cosine similarity (dot product on unit vecs)
        best_local = int(np.argmax(sims))
        best_global = int(indices[best_local])

        anchors[lbl] = {
            "text": texts[best_global],
            "embedding": embeddings[best_global],
            "index": best_global,
        }

    return anchors
