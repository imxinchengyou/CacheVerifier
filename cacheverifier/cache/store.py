from dataclasses import dataclass

import hnswlib
import numpy as np


@dataclass(frozen=True)
class CacheEntry:
    query_id: str
    query: str
    answer: str
    equivalence_id: str


@dataclass(frozen=True)
class NeighborMatch:
    entry: CacheEntry
    similarity: float
    index: int = -1
    """Position of `entry` in insertion order (i.e. into whatever list of
    records was streamed to build this store) — -1 for matches not produced
    by `VectorCacheStore.query`. Lets `cacheverifier.experiments.verified_sweep`
    cache a match trace by index instead of duplicating entry text."""


class VectorCacheStore:
    """Approximate cosine-similarity nearest-neighbor cache backed by HNSW
    (hnswlib) — the same ANN index vCache's own benchmark harness uses
    (`HNSWLibVectorDB` in `vcache-project/vCache`'s `benchmarks/benchmark.py`),
    rather than a brute-force scan.

    This isn't a micro-optimization: a brute-force O(n)-per-query
    implementation was measured at ~56s for just 20,000 records at dim=1024,
    which extrapolates to ~19 hours of pure search time to sweep Group A/B's
    threshold grids over the real ~60k/150k-record SemCacheLMArena /
    SemCacheSearchQueries traces (proposal Section 4.1). HNSW turns each
    query into ~O(log n), which is what makes those runs practical at all.

    Embeddings must be L2-normalized (see `cacheverifier.embeddings.base.Embedder`).
    HNSW is approximate: `ef` is kept >= the current entry count for small
    caches (so the handful of entries used in unit tests are retrieved
    exactly) and capped at `ef_base * 4` as the cache grows, trading a small
    amount of recall for bounded per-query cost at real-experiment scale.
    """

    def __init__(
        self,
        dim: int,
        initial_capacity: int = 1024,
        M: int = 16,
        ef_construction: int = 200,
        ef: int = 64,
    ) -> None:
        self.dim = dim
        self._capacity = max(initial_capacity, 1)
        self._ef_base = ef
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(max_elements=self._capacity, M=M, ef_construction=ef_construction)
        self._index.set_ef(ef)
        self._entries: list[CacheEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def insert(self, entry: CacheEntry, embedding: np.ndarray) -> None:
        if embedding.shape != (self.dim,):
            raise ValueError(f"expected embedding shape ({self.dim},), got {embedding.shape}")
        size = len(self._entries)
        if size == self._capacity:
            self._capacity *= 2
            self._index.resize_index(self._capacity)

        self._index.add_items(embedding.reshape(1, -1), np.array([size]))
        self._entries.append(entry)

        new_size = size + 1
        if new_size > self._ef_base:
            self._index.set_ef(min(new_size, self._ef_base * 4))

    def query(self, embedding: np.ndarray) -> NeighborMatch | None:
        """Return the approximate single nearest neighbor, or None if the cache is empty."""
        size = len(self._entries)
        if size == 0:
            return None
        labels, distances = self._index.knn_query(embedding.reshape(1, -1), k=1)
        best_idx = int(labels[0, 0])
        similarity = 1.0 - float(distances[0, 0])
        return NeighborMatch(entry=self._entries[best_idx], similarity=similarity, index=best_idx)
