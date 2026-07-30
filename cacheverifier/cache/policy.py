from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cacheverifier.cache.store import NeighborMatch
from cacheverifier.data.schema import QueryRecord


@dataclass(frozen=True)
class Decision:
    """Outcome of a policy evaluating one incoming request against its
    nearest cache neighbor (if any).

    `branch` documents *why* the decision was made (e.g. "above_threshold",
    "gray_zone_verifier_approved") — this is what the metrics module needs to
    separate, say, a Group D "verifier approved" hit from a Group A "above
    static threshold" hit even though both report action="hit".

    `verifier_invoked` / `verifier_latency_ms` are populated by Group C/D
    policies in week 2; week 1's Group A/B policies never call a verifier and
    leave them at their defaults.
    """

    action: Literal["hit", "miss"]
    branch: str
    verifier_invoked: bool = False
    verifier_latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class CachePolicy(ABC):
    """Decides, for one request, whether to serve from cache (hit) or fall
    through to the source LLM (miss).

    This follows vCache's own online-evaluation protocol (see
    `vcache-project/vCache`'s `benchmarks/benchmark.py`): there is no offline
    "history" split used to pre-calibrate a policy — every request is decided
    and learned from in one continuous pass over the trace, cache starting
    empty. Subclasses only need to implement `decide`; `observe` is an
    optional hook for online-learning policies (Group B, and later Group C/D)
    to update internal state after the runner has computed ground truth for
    this request — static policies (Group A) leave it as a no-op.
    """

    name: str

    @abstractmethod
    def decide(
        self,
        query: QueryRecord,
        query_embedding: np.ndarray,
        match: NeighborMatch | None,
    ) -> Decision:
        raise NotImplementedError

    def observe(
        self,
        query: QueryRecord,
        match: NeighborMatch | None,
        would_be_correct: bool | None,
        action: Literal["hit", "miss"],
    ) -> None:
        """Called by the runner after ground truth is known for this request,
        for every request regardless of the decision. `would_be_correct` is
        whether `match` (the nearest neighbor at decide-time, if any) actually
        shared the query's `equivalence_id` — i.e. whether accepting it would
        have been correct. `None` iff `match` is None (empty cache). `action`
        is the decision `decide()` actually returned for this same request.

        Passing `action` lets a policy only learn from misses, matching
        vCache's own update rule (`VerifiedDecisionPolicy.__update_cache` only
        ever runs on the EXPLORE/miss branch — hits are trusted without
        verification). Static policies (Group A) leave this as a no-op.
        """
        return None
