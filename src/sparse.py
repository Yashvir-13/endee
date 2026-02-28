"""
TF-IDF sparse encoder for hybrid search.

Generates sparse vector representations (indices + values) compatible
with Endee's hybrid search. The fitted vectorizer is saved to disk so
query-time encoding uses the same vocabulary as ingestion.
"""

import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

# Default paths and config
DEFAULT_VECTORIZER_PATH = "data/tfidf_vectorizer.pkl"
MAX_FEATURES = 30_000  # vocabulary cap = Endee sparse_dim


class TFIDFSparseEncoder:
    """
    TF-IDF based sparse encoder for Endee hybrid search.

    At ingestion time: fit on the full corpus, save the vectorizer.
    At query time:     load the saved vectorizer, encode the query.
    """

    def __init__(self, max_features: int = MAX_FEATURES):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,       # apply log normalization to TF
            strip_accents="unicode",
            stop_words="english",
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        """Fit the vectorizer on the corpus."""
        self.vectorizer.fit(texts)
        self._fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"  [Sparse] Fitted TF-IDF vectorizer (vocab_size={vocab_size})")

    def save(self, path: str = DEFAULT_VECTORIZER_PATH) -> None:
        """Save the fitted vectorizer to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"  [Sparse] Saved vectorizer to {path}")

    @classmethod
    def load(cls, path: str = DEFAULT_VECTORIZER_PATH) -> "TFIDFSparseEncoder":
        """Load a previously fitted vectorizer from disk."""
        encoder = cls()
        encoder.vectorizer = joblib.load(path)
        encoder._fitted = True
        return encoder

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        """
        Encode a single text into sparse (indices, values) for Endee.

        Returns:
            (indices, values) — both as lists for Endee API
        """
        assert self._fitted, "Vectorizer not fitted. Call fit() or load() first."
        sparse_matrix = self.vectorizer.transform([text])
        coo = sparse_matrix.tocoo()
        indices = coo.col.tolist()
        values = coo.data.tolist()
        return indices, values

    def encode_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Encode a batch of texts into sparse representations."""
        assert self._fitted, "Vectorizer not fitted. Call fit() or load() first."
        sparse_matrix = self.vectorizer.transform(texts)

        results = []
        for i in range(sparse_matrix.shape[0]):
            row = sparse_matrix.getrow(i).tocoo()
            indices = row.col.tolist()
            values = row.data.tolist()
            results.append((indices, values))
        return results

    @property
    def dim(self) -> int:
        """Return the sparse dimensionality (vocabulary size)."""
        if self._fitted:
            return len(self.vectorizer.vocabulary_)
        return self.max_features
