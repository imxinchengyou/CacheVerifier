from abc import ABC, abstractmethod

import numpy as np

from cacheverifier.data.schema import QueryRecord


class Embedder(ABC):
    """Maps records to L2-normalized embedding vectors.

    Takes full `QueryRecord`s rather than raw strings so that
    `PrecomputedEmbedder` can read a record's own pre-computed `embedding`
    field instead of encoding `query` text — needed to reproduce vCache's
    reported numbers, which were generated from embeddings baked into their
    benchmark datasets rather than re-encoded on the fly.

    Vectors are expected to be L2-normalized so that cosine similarity
    reduces to a dot product in `cacheverifier.cache.store.VectorCacheStore`.
    """

    @abstractmethod
    def embed(self, records: list[QueryRecord]) -> np.ndarray:
        """Return an (len(records), dim) float32 array of unit-norm vectors."""
        raise NotImplementedError

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
