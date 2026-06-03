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
