import numpy as np

from cacheverifier.cache.policy import CachePolicy, Decision
from cacheverifier.cache.store import NeighborMatch
from cacheverifier.data.schema import QueryRecord
from cacheverifier.verifiers.base import Verifier


class SynchronousVerifiedPolicy(CachePolicy):
    """Shared decision mechanism for Groups C and D (proposal Section 4.4):

        sim >= tau_high        -> hit directly (high confidence)
        tau_low <= sim < tau_high -> call `verifier` SYNCHRONOUSLY; hit iff approved
        sim < tau_low           -> miss directly (low confidence)

    Group C and D are the SAME class with a different `Verifier` plugged
    in — that's the point: Group C uses `OracleVerifier` to isolate the
    sync-vs-async question at Krites' own (oracle) verifier-fidelity
    assumption; Group D swaps in real lightweight verifiers (cross-encoder,
    NLI, mini-LLM, ...) to measure what changes once the verifier is real.
    Both put the verifier call on the serving path — the blocking
    alternative Krites' own paper discusses ("Blocking verified caching")
    but never implements or measures.
    """

    def __init__(self, tau_low: float, tau_high: float, verifier: Verifier) -> None:
        if not 0.0 <= tau_low <= tau_high <= 1.0:
            raise ValueError(f"expected 0 <= tau_low <= tau_high <= 1, got tau_low={tau_low}, tau_high={tau_high}")
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.verifier = verifier
        self.name = f"synchronous_verified[{verifier.name}]"

    def decide(
        self,
        query: QueryRecord,
        query_embedding: np.ndarray,
        match: NeighborMatch | None,
    ) -> Decision:
        if match is None:
            return Decision(action="miss", branch="empty_cache")

        if match.similarity >= self.tau_high:
            return Decision(action="hit", branch="above_tau_high")
        if match.similarity < self.tau_low:
            return Decision(action="miss", branch="below_tau_low")

        result = self.verifier.verify(query, match.entry)
        action = "hit" if result.approved else "miss"
        return Decision(
            action=action,
            branch=f"gray_zone_verifier_{'approved' if result.approved else 'rejected'}",
            verifier_invoked=True,
            verifier_latency_ms=result.latency_ms,
        )
