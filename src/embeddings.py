"""
Embedding abstraction layer.

Provides a swappable interface for embedding text into dense vectors.
Default model: intfloat/e5-large-v2 (1024-dim).

E5 models use asymmetric prefixes for better retrieval:
  - "query: " for questions (used during search)
  - "passage: " for documents (used during ingestion)

The model is auto-downloaded from HuggingFace on first run.
Override with EMBEDDING_MODEL env var to use a local path or different model.
"""

import os
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a query/question into a vector."""
        ...

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document passages into vectors."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class SentenceTransformerEmbedder(EmbeddingProvider):
    """
    Local embedding using sentence-transformers.

    Default model: intfloat/e5-large-v2 (1024 dimensions).
    E5 models require prefixes for asymmetric retrieval:
      - queries:   "query: <text>"
      - passages:  "passage: <text>"

    You can override the model by setting the EMBEDDING_MODEL env var
    to either a HuggingFace model ID or a local path.
    """

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name_or_path)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the 'query: ' prefix."""
        return self.model.encode(f"query: {text}").tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed passages with the 'passage: ' prefix."""
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(prefixed, show_progress_bar=True).tolist()

    def dimension(self) -> int:
        return self._dimension
