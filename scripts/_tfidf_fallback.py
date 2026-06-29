"""
Pure-numpy TF-IDF embedding fallback — used ONLY when sentence-transformers
isn't installed (e.g. running the algorithm-comparison harness in an
environment without internet access to fetch the real model).

Not used by the production pipeline. src/embeddings.py (sentence-transformers)
remains the real encoder — swap evaluate_neighbor_algorithms.py back to it
when running on a machine that has the model cached / internet access.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def encode_texts_tfidf(texts: list[str], **_kwargs) -> np.ndarray:
    docs_tokens = [_tokenize(t) for t in texts]
    df_counts: Counter = Counter()
    for tokens in docs_tokens:
        df_counts.update(set(tokens))

    n_docs = len(texts)
    vocab = sorted(df_counts.keys())
    vocab_index = {w: i for i, w in enumerate(vocab)}
    idf = np.array([np.log((1 + n_docs) / (1 + df_counts[w])) + 1 for w in vocab])

    X = np.zeros((n_docs, len(vocab)))
    for i, tokens in enumerate(docs_tokens):
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        for w, c in tf.items():
            X[i, vocab_index[w]] = (c / total) * idf[vocab_index[w]]

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    return X
