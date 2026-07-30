import numpy as np

from cacheverifier.cache.policy import CachePolicy
from cacheverifier.cache.store import CacheEntry, VectorCacheStore
from cacheverifier.data.schema import QueryRecord
from cacheverifier.embeddings.base import Embedder
from cacheverifier.metrics.core import RequestOutcome


class ExperimentRunner:
    """Drives the online cache simulation the way vCache's own benchmark
    harness does (`vcache-project/vCache`'s `benchmarks/benchmark.py`): the
    cache starts empty and every record in the trace is streamed through
    exactly once, in order — there is no separate offline "history" split
    used to pre-seed the cache or pre-calibrate a policy. Online-learning
    policies (Group B's `AdaptiveThresholdPolicy`) build up their state from
    this same single pass via `CachePolicy.observe`.

    Every request, hit or miss, ends up inserted into the cache afterward
    with its own ground-truth answer, so the cache keeps growing over the
    whole stream.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def run(
        self, records: list[QueryRecord], policy: CachePolicy, embeddings: np.ndarray | None = None
    ) -> list[RequestOutcome]:
        """`embeddings` lets a caller sweeping many policies over the same
        `records` (e.g. Group A's threshold grid) embed once and reuse the
        result, instead of paying encoding cost again per grid point — a
        real cost for `SentenceTransformerEmbedder`, not just a style nit."""
        if not records:
            return []

        if embeddings is None:
            embeddings = self.embedder.embed(records)
        store = VectorCacheStore(dim=embeddings.shape[1])
        outcomes: list[RequestOutcome] = []

        for record, embedding in zip(records, embeddings):
            match = store.query(embedding)
            decision = policy.decide(record, embedding, match)

            would_be_correct = None if match is None else (match.entry.equivalence_id == record.equivalence_id)
            correct = would_be_correct if decision.action == "hit" else True
            policy.observe(record, match, would_be_correct, decision.action)

            outcomes.append(
                RequestOutcome(
                    action=decision.action,
                    correct=correct,
                    would_be_correct=would_be_correct,
                    verifier_invoked=decision.verifier_invoked,
                    verifier_latency_ms=decision.verifier_latency_ms,
                )
            )

            store.insert(
                CacheEntry(
                    query_id=record.query_id,
                    query=record.query,
                    answer=record.answer,
                    equivalence_id=record.equivalence_id,
                ),
                embedding,
            )

        return outcomes
