from cacheverifier.cache.store import CacheEntry
from cacheverifier.data.schema import QueryRecord
from cacheverifier.verifiers.base import Verifier


class OracleVerifier(Verifier):
    """Perfect-judgment verifier: approves iff `equivalence_id` matches.

    This is deliberately the SAME verifier-fidelity assumption Krites' own
    paper uses in simulation (Section 4 of their paper: "we do not run the
    LLM judge in simulation, but instantiate J directly from the benchmark's
    ground truth equivalence-class relation"). Using it for Group C isolates
    the one variable Group C is meant to test — synchronous vs. asynchronous
    verification — by holding verifier correctness fixed at Krites' own
    (unrealistic) oracle assumption. Group D's real verifiers are what
    actually measure the gap this oracle papers over.

    `latency_ms` is NOT zero by default: Group C exists specifically to
    measure the cost of putting a verifier call on the synchronous serving
    path, so the oracle still needs a plausible stand-in latency for
    "whatever judge Krites would have called" (they use GPT-4.1-nano, an
    API-latency-class model) even though its *correctness* is oracle-perfect.
    Default of 70ms matches the proposal's own estimate for a hosted API
    judge (Section 4.3's "小型商用API模型...~70ms+").

    `score` is binary (1.0/0.0), so any `threshold` in (0, 1) is equivalent —
    sweeping it (as Group D does for real verifiers) adds nothing here.
    """

    name = "oracle"

    def __init__(self, latency_ms: float = 70.0, threshold: float = 0.5) -> None:
        self.latency_ms = latency_ms
        self.threshold = threshold

    def score(self, query: QueryRecord, candidate: CacheEntry) -> tuple[float, float]:
        return (1.0 if query.equivalence_id == candidate.equivalence_id else 0.0), self.latency_ms
