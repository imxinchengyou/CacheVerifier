from abc import ABC, abstractmethod
from dataclasses import dataclass

from cacheverifier.cache.store import CacheEntry
from cacheverifier.data.schema import QueryRecord


@dataclass(frozen=True)
class VerificationResult:
    approved: bool
    latency_ms: float


class Verifier(ABC):
    """Synchronous gray-zone verifier interface.

    Groups C (synchronous Krites) and D (this proposal's method) both call a
    `Verifier` synchronously, on the serving path, for matches whose
    similarity falls in [tau_low, tau_high) — see
    `cacheverifier.cache.synchronous_verified.SynchronousVerifiedPolicy`, which
    both groups share; only the `Verifier` implementation differs between
    them.

    Subclasses implement `score` (raw, unbounded — higher means more likely
    correct) rather than `verify` directly, so `threshold` is a plain mutable
    attribute: `cacheverifier.experiments.verified_sweep` computes each
    candidate's score exactly once and cheaply re-thresholds it for every
    point in a threshold sweep, instead of re-running the model per point.

    `candidate` is the full matched `CacheEntry`, not just its answer text,
    so `OracleVerifier` can read `candidate.equivalence_id` for ground-truth
    judging; real verifiers (cross-encoder, NLI, mini-LLM) only ever look at
    `candidate.answer`.
    """

    name: str
    threshold: float

    @abstractmethod
    def score(self, query: QueryRecord, candidate: CacheEntry) -> tuple[float, float]:
        """Return (raw_score, latency_ms). Higher raw_score means more
        likely to be an acceptable answer; not assumed to be a probability
        or bounded to [0, 1]."""
        raise NotImplementedError

    def verify(self, query: QueryRecord, candidate: CacheEntry) -> VerificationResult:
        score, latency_ms = self.score(query, candidate)
        return VerificationResult(approved=score >= self.threshold, latency_ms=latency_ms)
