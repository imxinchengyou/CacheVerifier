import numpy as np

from cacheverifier.cache.policy import CachePolicy, Decision
from cacheverifier.cache.store import NeighborMatch
from cacheverifier.data.schema import QueryRecord


class StaticThresholdPolicy(CachePolicy):
    """Group A baseline: GPTCache-style single global similarity threshold.

    hit iff sim(query, nearest_neighbor) >= threshold. No verifier, no
    per-embedding adaptation. `threshold` is expected to be swept over a grid
    (see `cacheverifier.experiments.run_baselines`) to trace out a hit-rate /
    error-rate curve comparable to the GPTCache configuration reported in the
    vCache paper.
    """

    name = "static_threshold"

    def __init__(self, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

    def decide(
        self,
        query: QueryRecord,
        query_embedding: np.ndarray,
        match: NeighborMatch | None,
    ) -> Decision:
        if match is None:
            return Decision(action="miss", branch="empty_cache")
        if match.similarity >= self.threshold:
            return Decision(action="hit", branch="above_threshold")
        return Decision(action="miss", branch="below_threshold")
